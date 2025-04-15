"""
Query router.

This module provides endpoints for querying the document database.
"""

from fastapi import APIRouter, HTTPException, Depends, Query as QueryParam
from typing import List, Dict, Any, Optional
import logging
import os
import requests
import time

from core.dependencies import get_query_service, get_ollama_client, get_web_search_client, get_openai_client
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
    
    # Common validation for all paths
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
    
    # FAST PATH: For OpenAI assistants with local docs disabled
    is_openai_assistant = (chat_request.use_openai and 
                          chat_request.model == 'assistant' and 
                          chat_request.assistant_id)
    
    # OpenAI assistants are pre-loaded with context, so we automatically disable local docs
    # when an OpenAI assistant is specified, unless explicitly requested
    if is_openai_assistant and chat_request.use_local_docs is True:
        logging.info("OpenAI Assistant specified with explicit use_local_docs=true. Will retrieve local documents as supplementary context.")
    elif is_openai_assistant:
        logging.info("OpenAI Assistant specified. Automatically disabling local document retrieval.")
        chat_request.use_local_docs = False
    
    # Check if we should use the fast path - completely bypass RAG
    if is_openai_assistant and not chat_request.use_local_docs and settings.get("openai_api_key"):
        logging.info(f"FAST PATH: Using OpenAI Assistant without local document retrieval")
        
        # Initialize with empty context and sources
        rag_result = {"status": "success", "sources": {"documents": [], "ids": [], "metadatas": []}, "web_search_used": False}
        
        # Only do web search if explicitly set to True (not None or False)
        web_results = None
        if chat_request.web_search is True:  # Explicitly check for True to handle None case
            logging.info(f"Performing web search only for OpenAI Assistant")
            try:
                web_search_client = get_web_search_client()
                if web_search_client:
                    web_results = web_search_client.search_with_serper(latest_message, num_results=chat_request.web_results_count)
                    if web_results:
                        rag_result["sources"]["web_results"] = web_results
                        rag_result["web_search_used"] = True
                else:
                    logging.warning("Web search client not available - SERPER_API_KEY might be missing")
            except Exception as e:
                logging.error(f"Error during web search for OpenAI Assistant: {e}")
        else:
            logging.info("Web search disabled for OpenAI Assistant fast path")
                
        # Direct OpenAI assistant call, skipping all embedding generation and RAG
        try:
            # Get OpenAI client from dependencies
            openai_client = get_openai_client()
            
            if not openai_client or not openai_client.is_available:
                return {
                    "status": "error",
                    "response": "OpenAI API is not available. Please check your API key configuration.",
                    "model_used": "assistant",
                    "provider": "openai",
                    "sources": {"documents": [], "ids": [], "metadatas": []}
                }
            
            # Prepare messages
            chat_messages = [{'role': msg.role, 'content': msg.content} for msg in chat_request.messages]
            
            # Prepare additional messages (web search results)
            additional_messages = []
            
            # Add web search results if available
            if web_results:
                web_results_text = web_search_client.format_results_as_context(web_results)
                web_context_msg = f"Here are some web search results that might help:\n\n{web_results_text}"
                additional_messages.append({
                    'role': 'user',
                    'content': web_context_msg
                })
            
            # Call the OpenAI client to create a thread and run it with the assistant
            response_content = openai_client.create_thread_with_assistant(
                messages=chat_messages,
                assistant_id=chat_request.assistant_id,
                additional_messages=additional_messages,
                function_responses=chat_request.function_responses
            )
                
            # Format response
            response = {
                "status": "success",
                "response": response_content,
                "model_used": "assistant",
                "provider": "openai",
                "assistant_id": chat_request.assistant_id,
                "sources": rag_result["sources"],
                "web_search_used": rag_result["web_search_used"]
            }
            
            # Add function call information to response if applicable
            if chat_request.function_responses and any(func_name in response_content for func_name in chat_request.function_responses.keys()):
                logging.info("Function response was used in reply")
                response["function_call_used"] = True
            
            return response
            
        except Exception as e:
            logging.error(f"Error using OpenAI Assistant: {e}")
            error_response = {
                "status": "error",
                "response": f"Error using OpenAI Assistant: {str(e)}",
                "model_used": "assistant",
                "provider": "openai",
                "sources": {"documents": [], "ids": [], "metadatas": []}
            }
            return error_response
    
    # STANDARD PATH: For all other cases
    try:
        # Process RAG based on conditions
        context = ""
        rag_result = {
            "status": "success", 
            "sources": {"documents": [], "ids": [], "metadatas": []}, 
            "web_search_used": chat_request.web_search or False
        }
        
        # Get or create appropriate Ollama client
        ollama_client = get_ollama_client()
        if chat_request.model and not is_openai_assistant:
            try:
                # Create a temporary client with the specified model
                ollama_client = OllamaClient(
                    model=chat_request.model,
                    embedding_model=ollama_client.embedding_model,
                    base_url=ollama_client.base_url
                )
            except Exception as e:
                logging.error(f"Error creating custom Ollama client: {e}")
                # Continue with default client
        
        # Handle document retrieval based on settings
        if chat_request.use_local_docs:
            # Do full RAG query with document retrieval
            try:
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
                    "source_type": "documents",
                    "web_search_used": False
                }
        elif chat_request.web_search is True:  # Explicitly check for True to handle None case
            # Web search only with no document retrieval
            logging.info("Using web search only with no document retrieval")
            try:
                rag_result = query_service.process_query(
                    query=latest_message,
                    n_results=0,  # Skip document retrieval
                    combine_chunks=False,
                    web_search=True,  # Force web search to be True
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
            # No document retrieval, no web search
            logging.info("Skipping all document retrieval and web search as requested")
            rag_result = {
                "status": "success",
                "sources": {"documents": [], "ids": [], "metadatas": []},
                "web_search_used": False
            }
        
        # Check if RAG result indicates an error
        if rag_result.get("status") == "error" or rag_result.get("status") == "not_found":
            return rag_result
            
        # Process any retrieved documents as context
        if rag_result.get("sources") and rag_result.get("sources").get("documents"):
            documents = rag_result.get("sources").get("documents")
            if documents:
                context = "\n\n".join(documents)
        
        # OPENAI API PATH
        if chat_request.use_openai and settings.get("openai_api_key"):
            # Using our OpenAIClient from dependencies
            
            if is_openai_assistant:
                # Get web search client for formatting
                web_search_client = get_web_search_client()
                
                # OpenAI Assistant API
                try:
                    # Get OpenAI client from dependencies
                    openai_client = get_openai_client()
                    
                    if not openai_client or not openai_client.is_available:
                        return {
                            "status": "error",
                            "response": "OpenAI API is not available. Please check your API key configuration.",
                            "model_used": "assistant",
                            "provider": "openai",
                            "sources": {"documents": [], "ids": [], "metadatas": []}
                        }
                    
                    # Format messages
                    chat_messages = [{'role': msg.role, 'content': msg.content} for msg in chat_request.messages]
                    
                    # Prepare additional messages (context and web search results)
                    additional_messages = []
                    
                    # Add context as a user message if available
                    if context:
                        context_msg = f"Here's some relevant information that might help answer my question:\n\n{context}"
                        additional_messages.append({
                            'role': 'user',
                            'content': context_msg
                        })
                    
                    # Add web search results if available
                    if rag_result.get("web_search_used") and rag_result.get("sources", {}).get("web_results"):
                        web_results = rag_result["sources"]["web_results"]
                        if web_search_client:
                            web_results_text = web_search_client.format_results_as_context(web_results)
                            web_context_msg = f"Here are some web search results that might help:\n\n{web_results_text}"
                            additional_messages.append({
                                'role': 'user',
                                'content': web_context_msg
                            })
                    
                    # Call the OpenAI client to create a thread and run it with the assistant
                    response_content = openai_client.create_thread_with_assistant(
                        messages=chat_messages,
                        assistant_id=chat_request.assistant_id,
                        additional_messages=additional_messages
                    )
                        
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
                # OpenAI Chat Completion API
                try:
                    # Get OpenAI client from dependencies
                    openai_client = get_openai_client()
                    
                    if not openai_client or not openai_client.is_available:
                        return {
                            "status": "error",
                            "response": "OpenAI API is not available. Please check your API key configuration.",
                            "model_used": chat_request.model or "gpt-3.5-turbo",
                            "provider": "openai",
                            "sources": {"documents": [], "ids": [], "metadatas": []}
                        }
                    
                    # Format messages
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
                    
                    # Call the API through our client
                    model_name = chat_request.model if chat_request.model else "gpt-3.5-turbo"
                    response_content = openai_client.create_chat_completion(
                        messages=openai_messages,
                        model=model_name,
                        temperature=0.7,
                        max_tokens=1000,
                        function_responses=chat_request.function_responses
                    )
                    
                    # Log if function responses were provided
                    if chat_request.function_responses:
                        logging.info(f"Function responses provided for: {', '.join(chat_request.function_responses.keys())}")
                    
                    # Format response
                    response = {
                        "status": "success",
                        "response": response_content,
                        "model_used": model_name,
                        "provider": "openai"
                    }
                    
                    # Add function call information to response if applicable
                    if chat_request.function_responses and any(func_name in response_content for func_name in chat_request.function_responses.keys()):
                        logging.info("Function response was used in reply")
                        response["function_call_used"] = True
                except Exception as e:
                    logging.error(f"Error with OpenAI Chat API: {e}")
                    response = {
                        "status": "error",
                        "response": f"Error with OpenAI API: {str(e)}",
                        "model_used": chat_request.model,
                        "provider": "openai"
                    }
        else:
            # OLLAMA PATH
            try:
                # Convert messages to the format expected by the Ollama API
                ollama_messages = [{"role": msg.role, "content": msg.content} for msg in chat_request.messages]
                
                # Generate a chat response with the context
                response = query_service.process_chat(
                    messages=ollama_messages,
                    context=context,
                    custom_ollama_client=ollama_client
                )
                response["provider"] = "ollama"
            except Exception as e:
                logging.error(f"Error during chat response generation: {e}")
                response = {
                    "status": "error",
                    "response": f"An error occurred during chat processing: {str(e)}",
                    "error": str(e),
                    "provider": "ollama"
                }
        
        # Add sources and metadata to the response
        response["sources"] = rag_result.get("sources", {"documents": [], "ids": [], "metadatas": []}) if chat_request.use_local_docs else {"documents": [], "ids": [], "metadatas": []}
        
        if "source_type" in rag_result:
            response["source_type"] = rag_result["source_type"]
            
        if "web_search_used" in rag_result:
            response["web_search_used"] = rag_result["web_search_used"]
        
        return response
        
    except Exception as e:
        # Log the error
        logging.error(f"Error in chat_query: {e}")
        
        # Return a helpful error response
        error_response = {
            "status": "error",
            "error": str(e),
            "response": f"An error occurred while processing your chat query: {str(e)}",
            "sources": {"documents": [], "ids": [], "metadatas": []}
        }
        
        return error_response