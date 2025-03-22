"""
Health check router.

This module provides endpoints for checking system health.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from core.dependencies import (
    get_db_service, 
    get_ollama_client,
    get_query_classifier
)

router = APIRouter(tags=["health"])

@router.get("/health", summary="Health check", description="Checks if all components are operational.")
async def health_check():
    """Check the health of the system."""
    db_service = get_db_service()
    ollama_client = get_ollama_client()
    query_classifier = get_query_classifier()
    
    health_status = {
        "api": "healthy",
        "chroma": "unknown",
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