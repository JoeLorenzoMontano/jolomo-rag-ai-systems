"""
Application configuration.

This module handles loading configuration from environment variables.
"""

import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ChromaDB connection settings
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# Chunking configuration from environment variables with defaults
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
ENABLE_CHUNKING = os.getenv("ENABLE_CHUNKING", "true").lower() == "true"

# Web search API key
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Folder to store raw documents
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "./data")

# Connection retry settings
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "3"))  # seconds

def get_settings() -> Dict[str, Any]:
    """
    Get all configuration settings as a dictionary.
    
    Returns:
        Dictionary of configuration settings
    """
    return {
        "chroma_host": CHROMA_HOST,
        "chroma_port": CHROMA_PORT,
        "max_chunk_size": MAX_CHUNK_SIZE,
        "min_chunk_size": MIN_CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "enable_chunking": ENABLE_CHUNKING,
        "serper_api_key": SERPER_API_KEY,
        "docs_folder": DOCS_FOLDER,
        "max_retries": MAX_RETRIES,
        "retry_delay": RETRY_DELAY
    }

def log_config() -> None:
    """Log the current configuration settings."""
    logging.info("Document Processing API Configuration:")
    logging.info(f"  ChromaDB: {CHROMA_HOST}:{CHROMA_PORT}")
    logging.info(f"  Document chunking settings:")
    logging.info(f"    ENABLE_CHUNKING: {ENABLE_CHUNKING}")
    logging.info(f"    MAX_CHUNK_SIZE: {MAX_CHUNK_SIZE} chars")
    logging.info(f"    MIN_CHUNK_SIZE: {MIN_CHUNK_SIZE} chars")
    logging.info(f"    CHUNK_OVERLAP: {CHUNK_OVERLAP} chars")
    logging.info(f"  DOCS_FOLDER: {DOCS_FOLDER}")
    logging.info(f"  Web search enabled: {bool(SERPER_API_KEY)}")
    logging.info(f"  DB Connection: max_retries={MAX_RETRIES}, retry_delay={RETRY_DELAY}s")