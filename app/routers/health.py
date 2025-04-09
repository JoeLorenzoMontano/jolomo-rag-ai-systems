"""
Health check router.

This module provides endpoints for checking system health.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Dict, Any, List

from core.dependencies import (
    get_db_service, 
    get_ollama_client,
    get_query_classifier,
    get_elasticsearch_service,
    get_openai_client
)
from core.config import get_settings
import os

router = APIRouter(tags=["health"])

@router.get("/health", summary="Health check", description="Checks if all components are operational.")
async def health_check():
    """Check the health of the system."""
    db_service = get_db_service()
    ollama_client = get_ollama_client()
    query_classifier = get_query_classifier()
    elasticsearch_service = get_elasticsearch_service()
    
    health_status = {
        "api": "healthy",
        "chroma": "unknown",
        "elasticsearch": "disabled",
        "ollama": "unknown",
        "models": {
            "response_model": "unknown",
            "embedding_model": "unknown"
        },
        "collection": {
            "status": "unknown",
            "document_count": 0
        }
    }
    
    # Check ChromaDB
    try:
        is_healthy, status = db_service.is_healthy()
        health_status["chroma"] = "healthy" if is_healthy else status
        
        # Check collection status
        try:
            doc_count = db_service.get_document_count()
            health_status["collection"]["status"] = "healthy"
            health_status["collection"]["document_count"] = doc_count
        except Exception as e:
            health_status["collection"]["status"] = f"error: {str(e)}"
    except Exception as e:
        health_status["chroma"] = f"unhealthy: {str(e)}"
        
    # Check Elasticsearch if available
    if elasticsearch_service:
        try:
            is_healthy, status = elasticsearch_service.is_healthy()
            health_status["elasticsearch"] = "healthy" if is_healthy else status
            
            # Add Elasticsearch document count
            try:
                es_doc_count = elasticsearch_service.get_document_count()
                health_status["collection"]["es_document_count"] = es_doc_count
            except Exception as e:
                health_status["collection"]["es_status"] = f"error: {str(e)}"
        except Exception as e:
            health_status["elasticsearch"] = f"unhealthy: {str(e)}"
    else:
        health_status["elasticsearch"] = "disabled"
    
    # Check Ollama
    try:
        import requests
        response = requests.get(f"{ollama_client.base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            health_status["ollama"] = "healthy"
            
            # Check available models
            model_data = response.json()
            available_models = []
            
            # Handle different response formats
            if "models" in model_data:
                available_models = [m["name"] for m in model_data["models"]]
            elif isinstance(model_data, list):
                available_models = [m["name"] for m in model_data]
                
            # Check if our models are available
            response_model = ollama_client.model
            embedding_model = ollama_client.embedding_model
            
            if response_model in available_models:
                health_status["models"]["response_model"] = f"available ({response_model})"
            else:
                health_status["models"]["response_model"] = f"not found (looking for {response_model})"
                
            if embedding_model in available_models:
                health_status["models"]["embedding_model"] = f"available ({embedding_model})"
            else:
                health_status["models"]["embedding_model"] = f"not found (looking for {embedding_model})"
        else:
            health_status["ollama"] = f"unhealthy: status code {response.status_code}"
    except Exception as e:
        health_status["ollama"] = f"unhealthy: {str(e)}"
        
    # Check OpenAI integration if configured
    settings = get_settings()
    openai_client = get_openai_client()
    
    if openai_client and openai_client.is_available:
        health_status["openai"] = "healthy"
        if settings.get("openai_assistant_ids"):
            health_status["openai_assistants"] = settings["openai_assistant_ids"]
    elif settings.get("openai_api_key"):
        health_status["openai"] = "configured but not available"
    else:
        health_status["openai"] = "not configured"
    
    # Test embedding functionality
    if health_status["ollama"] == "healthy":
        try:
            # Quick test of embedding generation with a simple string
            embedding = ollama_client.generate_embedding("test health check")
            if embedding and len(embedding) > 0:
                health_status["models"]["embedding_model"] += f" - working (dimension: {len(embedding)})"
                
                # Initialize domain terms if collection has documents but terms aren't initialized
                if (health_status["collection"]["document_count"] > 0 and 
                    len(query_classifier.product_terms) <= 3):  # Only default terms
                    try:
                        print("Initializing domain terms using statistical extraction during health check")
                        query_classifier.update_terms_from_db(db_service.collection)
                        print(f"Domain terms initialized: {len(query_classifier.product_terms)} terms extracted using statistical approach")
                    except Exception as e:
                        print(f"Failed to initialize domain terms: {e}")
        except Exception as e:
            health_status["models"]["embedding_model"] += f" - error: {str(e)}"
    
    return health_status
    
@router.post("/api-settings", summary="Update API settings", description="Update external API configuration")
async def update_api_settings(settings: Dict[str, Any] = Body(...)):
    """Update API settings for external services like OpenAI."""
    current_settings = get_settings()
    update_success = True
    update_message = "Settings updated successfully."
    
    try:
        # Handle OpenAI API Key
        openai_updated = False
        if "openai_api_key" in settings:
            os.environ["OPENAI_API_KEY"] = settings["openai_api_key"]
            openai_updated = True
            
        # Handle OpenAI Assistant IDs
        if "openai_assistant_ids" in settings:
            assistant_ids = settings["openai_assistant_ids"]
            if isinstance(assistant_ids, list):
                os.environ["OPENAI_ASSISTANT_IDS"] = ",".join(assistant_ids)
                openai_updated = True
        
        # Refresh OpenAI client if settings changed
        if openai_updated:
            from core.dependencies import _create_services
            _create_services()  # Reinitialize services including OpenAIClient
            
        # Handle Serper API Key (for web search)
        if "serper_api_key" in settings:
            os.environ["SERPER_API_KEY"] = settings["serper_api_key"]
            
        # Handle Elasticsearch settings
        if "elasticsearch_enabled" in settings:
            os.environ["ELASTICSEARCH_ENABLED"] = str(settings["elasticsearch_enabled"]).lower()
            
        if "elasticsearch_url" in settings:
            os.environ["ELASTICSEARCH_URL"] = settings["elasticsearch_url"]
            
    except Exception as e:
        update_success = False
        update_message = f"Error updating settings: {str(e)}"
    
    return {
        "status": "success" if update_success else "error",
        "message": update_message
    }

@router.get("/current-settings", summary="Get current settings", description="Retrieve current API configuration settings")
async def get_current_settings():
    """Get current API configuration settings."""
    settings = get_settings()
    
    # Filter out sensitive information
    filtered_settings = {
        "elasticsearch_enabled": settings.get("elasticsearch_enabled", False),
        "elasticsearch_url": settings.get("elasticsearch_url", ""),
        "has_openai_key": bool(settings.get("openai_api_key")),
        "openai_assistant_ids": settings.get("openai_assistant_ids", []),
        "has_serper_key": bool(settings.get("serper_api_key")),
        "max_chunk_size": settings.get("max_chunk_size", 1000),
        "min_chunk_size": settings.get("min_chunk_size", 200),
        "chunk_overlap": settings.get("chunk_overlap", 100),
        "enable_chunking": settings.get("enable_chunking", True)
    }
    
    return {
        "status": "success",
        "settings": filtered_settings
    }