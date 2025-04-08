"""
Query router.

This module provides endpoints for querying the document database.
"""

from fastapi import APIRouter, HTTPException, Depends, Query as QueryParam
from typing import List, Dict, Any, Optional
import logging
import os
import requests

from core.dependencies import get_query_service, get_ollama_client
from core.config import get_settings
from models.schemas import ChatMessage, ChatRequest
from utils.ollama_client import OllamaClient

router = APIRouter(tags=["query"])

@router.get("/models", summary="Get available Ollama models", 
            description="Retrieves the list of available models from the Ollama server")
async def get_models():
    """Get the list of available models from the Ollama server."""
    ollama_client = get_ollama_client()
    
    try:
        models = ollama_client.get_available_models()
        
        # Extract and organize model information
        result = {
            "status": "success",
            "models": [model["name"] for model in models],
            "model_info": models,
            "default_model": ollama_client.model,
            "default_embedding_model": ollama_client.embedding_model
        }
        
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving models: {str(e)}"
        }

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
    apply_reranking: bool = QueryParam(True, description="Whether to apply reranking to improve document relevance"),
    check_question_matches: bool = QueryParam(True, description="Whether to check for question matches"),
    query_model: str = QueryParam(None, description="Model to use for generating the response"),
    embedding_model: str = QueryParam(None, description="Model to use for generating embeddings")
):
    """Query for relevant documents based on input text."""
    query_service = get_query_service()
    
    try:
        # If custom models are specified, use them
        ollama_client = None
        if query_model or embedding_model:
            ollama_client = get_ollama_client()
            # Create a temporary client with the specified models
            if query_model or embedding_model:
                ollama_client = OllamaClient(
                    model=query_model or ollama_client.model,
                    embedding_model=embedding_model or ollama_client.embedding_model,
                    base_url=ollama_client.base_url
                )
        
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
            apply_reranking=apply_reranking,
            check_question_matches=check_question_matches,
            custom_ollama_client=ollama_client
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
    settings = get_settings()
    
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
        
        # Handle different model options
        if chat_request.use_openai and settings.get("openai_api_key"):
            # Use OpenAI API for chat completion
            if chat_request.model == 'assistant' and chat_request.assistant_id:
                logging.info(f"Using OpenAI Assistant: {chat_request.assistant_id}")
            else:
                logging.info(f"Using OpenAI model: {chat_request.model}")
                
            # Initialize with empty context and sources
            context = ""
            rag_result = {"status": "success", "sources": {"documents": [], "ids": [], "metadatas": []}, "web_search_used": chat_request.web_search or False}
            
            # Check if this is an OpenAI assistant - if so, skip all RAG processing for local docs
            if chat_request.model == 'assistant' and chat_request.assistant_id and not chat_request.use_local_docs:
                logging.info(f"Using OpenAI Assistant without local document retrieval")
                # For OpenAI assistants without local docs, just pass the query directly without any RAG processing
                # Since assistant has its own files/retrieval capabilities
            # Only perform RAG if local docs are enabled
            elif chat_request.use_local_docs:
                # Do a regular query to get relevant documents (using Ollama for embeddings)
                ollama_client = get_ollama_client()
                
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
                    apply_reranking=chat_request.apply_reranking,
                    check_question_matches=chat_request.check_question_matches,
                    custom_ollama_client=ollama_client
                )
            elif chat_request.web_search:
                # Only do web search if local docs are disabled but web search is enabled
                ollama_client = get_ollama_client()
                rag_result = query_service.process_query(
                    query=latest_message,
                    n_results=0,
                    combine_chunks=False,
                    web_search=True,
                    web_results_count=chat_request.web_results_count,
                    explain_classification=False,
                    enhance_query=chat_request.enhance_query,
                    use_elasticsearch=False,
                    hybrid_search=False,
                    apply_reranking=False,
                    check_question_matches=False,
                    custom_ollama_client=ollama_client
                )
            
            # Check if we got a valid result
            if rag_result.get("status") == "error" or rag_result.get("status") == "not_found":
                return rag_result
            
            # Get the context from the RAG result
            context = ""
            if rag_result.get("sources") and rag_result.get("sources").get("documents"):
                documents = rag_result.get("sources").get("documents")
                if documents:
                    # Combine all retrieved documents as context
                    context = "\n\n".join(documents)
            
            # Format the OpenAI messages
            openai_messages = []
            
            # Add system message with context
            if context:
                system_message = f"You are a helpful assistant. Answer the user's questions based on the following information:\n\n{context}\n\nIf the information provided doesn't answer the question, say so clearly."
                openai_messages.append({"role": "system", "content": system_message})
            else:
                openai_messages.append({"role": "system", "content": "You are a helpful assistant."})
            
            # Add the conversation history
            for msg in chat_request.messages:
                openai_messages.append({"role": msg.role, "content": msg.content})
                
            # Call OpenAI API
            try:
                import openai
                openai.api_key = settings.get("openai_api_key")
                
                # Handle Assistants API vs Chat Completion API
                if chat_request.model == 'assistant' and chat_request.assistant_id:
                    # Use the Assistants API
                    try:
                        # Create a thread
                        thread = openai.beta.threads.create()
                        
                        # Add user messages to the thread
                        for msg in chat_request.messages:
                            if msg.role == "user":
                                openai.beta.threads.messages.create(
                                    thread_id=thread.id,
                                    role="user",
                                    content=msg.content
                                )
                        
                        # Add context as a user message if available
                        if context:
                            context_msg = f"Here's some relevant information that might help answer my question:\n\n{context}"
                            openai.beta.threads.messages.create(
                                thread_id=thread.id,
                                role="user",
                                content=context_msg
                            )
                        
                        # Run the assistant on the thread
                        run = openai.beta.threads.runs.create(
                            thread_id=thread.id,
                            assistant_id=chat_request.assistant_id
                        )
                        
                        # Poll for completion - in a real app, you'd use a webhook
                        import time
                        run_status = run.status
                        max_attempts = 30  # 30 seconds max wait time
                        attempts = 0
                        
                        while run_status in ["queued", "in_progress", "cancelling"] and attempts < max_attempts:
                            time.sleep(1)  # Wait for 1 second
                            run = openai.beta.threads.runs.retrieve(
                                thread_id=thread.id,
                                run_id=run.id
                            )
                            run_status = run.status
                            attempts += 1
                        
                        # Get the assistant's messages
                        messages = openai.beta.threads.messages.list(
                            thread_id=thread.id
                        )
                        
                        # Get the last assistant message
                        assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
                        if assistant_messages:
                            latest_message = assistant_messages[0]
                            response_content = latest_message.content[0].text.value
                        else:
                            response_content = "The assistant did not provide a response."
                            
                        # Format response
                        response = {
                            "status": "success",
                            "response": response_content,
                            "model_used": "assistant",
                            "provider": "openai",
                            "assistant_id": chat_request.assistant_id
                        }
                    except Exception as e:
                        logging.error(f"Error using OpenAI Assistant: {e}")
                        response = {
                            "status": "error",
                            "response": f"Error using OpenAI Assistant: {str(e)}",
                            "model_used": "assistant",
                            "provider": "openai"
                        }
                else:
                    # Use the Chat Completion API
                    model_name = chat_request.model if chat_request.model else "gpt-3.5-turbo"
                    openai_response = openai.chat.completions.create(
                        model=model_name,
                        messages=openai_messages,
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    response_content = openai_response.choices[0].message.content
                    
                    # Format response
                    response = {
                        "status": "success",
                        "response": response_content,
                        "model_used": model_name,
                        "provider": "openai"
                    }
                
            except Exception as e:
                logging.error(f"Error with OpenAI API: {e}")
                response = {
                    "status": "error",
                    "response": f"Error with OpenAI API: {str(e)}",
                    "model_used": chat_request.model,
                    "provider": "openai"
                }
        else:
            # Use Ollama for chat completion
            logging.info(f"Using Ollama model: {chat_request.model or 'default'}")
            
            # Get the Ollama client
            ollama_client = get_ollama_client()
            
            # If custom models are specified, use them
            if chat_request.model:
                try:
                    # Create a temporary client with the specified models
                    ollama_client = OllamaClient(
                        model=chat_request.model,
                        embedding_model=ollama_client.embedding_model,
                        base_url=ollama_client.base_url
                    )
                except Exception as e:
                    logging.error(f"Error creating custom Ollama client: {e}")
                    # Continue with default client
            
            # Initialize rag_result with default empty values
            rag_result = {
                "status": "success",
                "sources": {"documents": [], "ids": [], "metadatas": []},
                "source_type": "none",
                "web_search_used": chat_request.web_search or False
            }
            
            # First, decide whether to perform document retrieval
            if chat_request.use_local_docs:
                try:
                    # Do a regular query to get relevant documents
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
                        apply_reranking=chat_request.apply_reranking,
                        check_question_matches=chat_request.check_question_matches,
                        custom_ollama_client=ollama_client
                    )
                except Exception as e:
                    logging.error(f"Error during RAG query: {e}")
                    rag_result = {
                        "status": "success",
                        "sources": {"documents": [], "ids": [], "metadatas": []},
                        "source_type": "documents"
                    }
            elif chat_request.web_search:
                # Only do web search if local docs are disabled but web search is enabled
                try:
                    rag_result = query_service.process_query(
                        query=latest_message,
                        n_results=0,
                        combine_chunks=False,
                        web_search=True,
                        web_results_count=chat_request.web_results_count,
                        explain_classification=False,
                        enhance_query=chat_request.enhance_query,
                        use_elasticsearch=False,
                        hybrid_search=False,
                        apply_reranking=False,
                        check_question_matches=False,
                        custom_ollama_client=ollama_client
                    )
                except Exception as e:
                    logging.error(f"Error during web search: {e}")
                    rag_result = {
                        "status": "success",
                        "sources": {"documents": [], "ids": [], "metadatas": []},
                        "source_type": "web",
                        "web_search_used": False
                    }
            else:
                # Skip all document retrieval
                logging.info("Skipping all document retrieval as requested")
            
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
            
            try:
                # Generate a chat response with the context
                response = query_service.process_chat(
                    messages=ollama_messages,
                    context=context,
                    custom_ollama_client=ollama_client
                )
            except Exception as e:
                logging.error(f"Error during chat response generation: {e}")
                # Default error response if chat processing fails
                response = {
                    "status": "error",
                    "response": f"An error occurred during chat processing: {str(e)}",
                    "error": str(e)
                }
            response["provider"] = "ollama"
        
        # Add the sources from the RAG query to the chat response
        # If local docs are disabled, ensure we don't include any sources
        if not chat_request.use_local_docs:
            response["sources"] = {"documents": [], "ids": [], "metadatas": []}
        else:
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