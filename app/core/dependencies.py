"""
Dependency injection container.

This module sets up all the services and their dependencies.
"""

import os
from typing import Dict, Any
import logging

from core.config import (
    CHROMA_HOST, CHROMA_PORT, MAX_RETRIES, RETRY_DELAY,
    DOCS_FOLDER, MAX_CHUNK_SIZE, MIN_CHUNK_SIZE, CHUNK_OVERLAP, 
    ENABLE_CHUNKING, SERPER_API_KEY
)
from services.database_service import DatabaseService
from services.job_service import JobService
from services.content_processing_service import ContentProcessingService
from services.query_service import QueryService
from utils.ollama_client import OllamaClient
from utils.query_classifier import QueryClassifier
from utils.web_search import WebSearchClient

# Create a container for services
_services = {}
logger = logging.getLogger(__name__)

def _create_services() -> None:
    """Initialize all services."""
    # Initialize ChromaDB service
    logger.info(f"Initializing ChromaDB connection to {CHROMA_HOST}:{CHROMA_PORT}")
    db_service = DatabaseService(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        max_retries=MAX_RETRIES,
        retry_delay=RETRY_DELAY
    )
    _services["db_service"] = db_service
    
    # Initialize job tracking service
    job_service = JobService()
    _services["job_service"] = job_service
    
    # Initialize Ollama client
    ollama_client = OllamaClient()
    _services["ollama_client"] = ollama_client
    
    # Initialize query classifier
    query_classifier = QueryClassifier(confidence_threshold=0.6)
    # Set default terms - these will be updated from the database when documents are processed
    query_classifier.product_terms = ["duplocloud", "tenant", "infrastructure"]
    _services["query_classifier"] = query_classifier
    
    # Initialize web search client if API key is available
    web_search_client = None
    if SERPER_API_KEY:
        web_search_client = WebSearchClient(serper_api_key=SERPER_API_KEY)
        logger.info("Web search client initialized with Serper API key")
    else:
        logger.info("No Serper API key provided, web search will be unavailable")
    _services["web_search_client"] = web_search_client
    
    # Initialize content processing service
    content_processing_service = ContentProcessingService(
        db_service=db_service,
        job_service=job_service,
        ollama_client=ollama_client,
        query_classifier=query_classifier,
        docs_folder=DOCS_FOLDER,
        max_chunk_size=MAX_CHUNK_SIZE,
        min_chunk_size=MIN_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        enable_chunking=ENABLE_CHUNKING
    )
    _services["content_processing_service"] = content_processing_service
    
    # Initialize query service
    query_service = QueryService(
        db_service=db_service,
        ollama_client=ollama_client,
        query_classifier=query_classifier,
        web_search_client=web_search_client
    )
    _services["query_service"] = query_service
    
    # Make sure documents directory exists
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    
    logger.info("All services initialized successfully")

def get_service(service_name: str) -> Any:
    """
    Get a service by name.
    
    Args:
        service_name: Name of the service to get
        
    Returns:
        The requested service instance
    """
    if not _services:
        _create_services()
        
    return _services.get(service_name)

def get_db_service() -> DatabaseService:
    """Get the database service."""
    return get_service("db_service")

def get_job_service() -> JobService:
    """Get the job tracking service."""
    return get_service("job_service")

def get_document_service() -> ContentProcessingService:
    """Get the content processing service. 
    
    Note: This function is kept for backward compatibility but uses the content processing service.
    """
    return get_service("content_processing_service")

def get_content_processing_service() -> ContentProcessingService:
    """Get the content processing service."""
    return get_service("content_processing_service")

def get_query_service() -> QueryService:
    """Get the query processing service."""
    return get_service("query_service")

def get_ollama_client() -> OllamaClient:
    """Get the Ollama client."""
    return get_service("ollama_client")

def get_query_classifier() -> QueryClassifier:
    """Get the query classifier."""
    return get_service("query_classifier")

def get_web_search_client() -> WebSearchClient:
    """Get the web search client."""
    return get_service("web_search_client")

def get_all_services() -> Dict[str, Any]:
    """Get all initialized services."""
    if not _services:
        _create_services()
        
    return _services