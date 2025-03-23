"""
Query router.

This module provides endpoints for querying the document database.
"""

from fastapi import APIRouter, HTTPException, Depends, Query as QueryParam
from typing import List, Dict, Any, Optional

from core.dependencies import get_query_service
from models.schemas import ChatMessage, ChatRequest

router = APIRouter(tags=["query"])

@router.get("/query", summary="Retrieve relevant documents", 
           description="Query for the most relevant document based on input text.")
async def query_documents(
    query: str,
    n_results: int = QueryParam(3, description="Number of results to return"),
    combine_chunks: bool = QueryParam(True, description="Whether to combine chunks from the same document"),
    web_search: bool = QueryParam(None, description="Whether to augment with web search results (auto if None)"),
    web_results_count: int = QueryParam(5, description="Number of web search results to include"),
    explain_classification: bool = QueryParam(False, description="Whether to include query classification explanation"),
    enhance_query: bool = QueryParam(True, description="Whether to enhance the query for better retrieval"),
    use_elasticsearch: bool = QueryParam(None, description="Whether to use Elasticsearch (auto if None)"),
    hybrid_search: bool = QueryParam(True, description="Whether to combine results from ChromaDB and Elasticsearch"),
    apply_reranking: bool = QueryParam(True, description="Whether to apply reranking to improve document relevance")
):
    """Query for relevant documents based on input text."""
    query_service = get_query_service()
    
    try:
        result = query_service.process_query(
            query=query,
            n_results=n_results,
            combine_chunks=combine_chunks,
            web_search=web_search,
            web_results_count=web_results_count,
            explain_classification=explain_classification,
            enhance_query=enhance_query,
            use_elasticsearch=use_elasticsearch,
            hybrid_search=hybrid_search,
            apply_reranking=apply_reranking
        )
        
        return result
        
    except Exception as e:
        # Log the error
        print(f"Error in query_documents: {e}")
        
        # Return a helpful error response
        error_response = {
            "query": query,
            "status": "error",
            "error": str(e)
        }
        
        # Add more context based on the type of error
        if "embed" in str(e).lower():
            error_response["response"] = "Error generating embeddings. The Ollama model may not support the embedding API."
            error_response["suggestion"] = "Check if the model supports embeddings or try a different model."
        elif "chroma" in str(e).lower():
            error_response["response"] = "Error connecting to the vector database."
            error_response["suggestion"] = "Verify ChromaDB is running and accessible."
        else:
            error_response["response"] = f"An error occurred while processing your query: {str(e)}"
            
        error_response["sources"] = {"documents": [], "ids": [], "metadatas": []}
        
        return error_response
        
@router.post("/chat", summary="Chat with contextual memory", 
           description="Query with chat history and RAG for a conversational experience.")
async def chat_query(
    chat_request: ChatRequest
):
    """Process a chat query with conversation history."""
    query_service = get_query_service()
    
    try:
        # Get the latest user message (should be the last message in the list)
        if not chat_request.messages or len(chat_request.messages) == 0:
            return {
                "status": "error",
                "response": "No messages provided in the request.",
                "sources": {"documents": [], "ids": [], "metadatas": []}
            }
            
        # Extract the latest user message to use as query
        latest_message = None
        for msg in reversed(chat_request.messages):
            if msg.role == "user":
                latest_message = msg.content
                break
                
        if not latest_message:
            return {
                "status": "error",
                "response": "No user message found in the conversation history.",
                "sources": {"documents": [], "ids": [], "metadatas": []}
            }
            
        # First, do a regular query to get relevant documents
        rag_result = query_service.process_query(
            query=latest_message,
            n_results=chat_request.n_results,
            combine_chunks=chat_request.combine_chunks,
            web_search=chat_request.web_search,
            web_results_count=chat_request.web_results_count,
            explain_classification=False,  # Always false for chat
            enhance_query=chat_request.enhance_query,
            use_elasticsearch=chat_request.use_elasticsearch,
            hybrid_search=chat_request.hybrid_search,
            apply_reranking=chat_request.apply_reranking  # Use reranking if enabled
        )
        
        # Check if we got a valid result
        if rag_result.get("status") == "error" or rag_result.get("status") == "not_found":
            return rag_result
            
        # Convert messages to the format expected by the Ollama API
        ollama_messages = [{"role": msg.role, "content": msg.content} for msg in chat_request.messages]
        
        # Get the context from the RAG result
        context = None
        if rag_result.get("sources") and rag_result.get("sources").get("documents"):
            documents = rag_result.get("sources").get("documents")
            if documents:
                # Combine all retrieved documents as context
                context = "\n\n".join(documents)
        
        # Generate a chat response with the context
        response = query_service.process_chat(
            messages=ollama_messages,
            context=context
        )
        
        # Add the sources from the RAG query to the chat response
        response["sources"] = rag_result.get("sources", {"documents": [], "ids": [], "metadatas": []})
        
        # Add the source_type if available
        if "source_type" in rag_result:
            response["source_type"] = rag_result["source_type"]
            
        # Add web_search_used if available
        if "web_search_used" in rag_result:
            response["web_search_used"] = rag_result["web_search_used"]
        
        return response
        
    except Exception as e:
        # Log the error
        print(f"Error in chat_query: {e}")
        
        # Return a helpful error response
        error_response = {
            "status": "error",
            "error": str(e),
            "response": f"An error occurred while processing your chat query: {str(e)}",
            "sources": {"documents": [], "ids": [], "metadatas": []}
        }
        
        return error_response