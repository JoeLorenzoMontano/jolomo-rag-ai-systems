"""
ChromaDB database service.

This module provides functionality to interact with ChromaDB for document storage and retrieval.
"""

import chromadb
from chromadb.config import Settings
from typing import Dict, List, Any, Optional, Tuple
import time
import logging

class DatabaseService:
    """Service for interacting with ChromaDB."""
    
    def __init__(self, host: str, port: int, max_retries: int = 5, retry_delay: int = 3):
        """
        Initialize the database service with connection settings.
        
        Args:
            host: ChromaDB host address
            port: ChromaDB port number
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retries in seconds
        """
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None
        self.collection = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize the connection
        self._connect()
        
    def _connect(self) -> None:
        """Establish connection to ChromaDB with retry logic."""
        # Try connection with retry logic
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Connection attempt {attempt + 1} to ChromaDB...")
                self.client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    ssl=False,
                )
                # Test connection with heartbeat
                self.client.heartbeat()
                self.logger.info(f"Successfully connected to ChromaDB on attempt {attempt + 1}")
                
                # Get server version
                try:
                    server_info = self.client._server_state()
                    self.logger.info(f"ChromaDB server version: {server_info.get('version', 'unknown')}")
                except:
                    self.logger.warning("Could not get ChromaDB server version")
                    
                # Initialize the default collection
                self._init_collection()
                break
                
            except Exception as e:
                self.logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.warning("All connection attempts failed. Falling back to in-memory database.")
                    self.client = chromadb.EphemeralClient()
                    self.logger.info("Using in-memory ChromaDB")
                    self._init_collection()
    
    def _init_collection(self) -> None:
        """Initialize or get the default collection."""
        try:
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"description": "Main document collection for RAG processing"}
            )
            self.logger.info(f"Collection 'documents' ready with {self.collection.count()} documents")
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            raise RuntimeError(f"Failed to create ChromaDB collection: {e}")
    
    def add_documents(self, documents: List[str], 
                     embeddings: List[List[float]], 
                     ids: List[str], 
                     metadatas: List[Dict[str, Any]]) -> None:
        """
        Add documents to the collection.
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            ids: List of unique IDs for the documents
            metadatas: List of metadata dictionaries for each document
        """
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
    
    def query_documents(self, query_embedding: List[float], n_results: int = 3) -> Dict[str, Any]:
        """
        Query the collection for similar documents.
        
        Args:
            query_embedding: Embedding vector for the query
            n_results: Number of results to return
            
        Returns:
            Dictionary with query results
        """
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
    
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the collection.
        
        Returns:
            Number of documents
        """
        return self.collection.count()
    
    def get_all_documents(self, include_embeddings: bool = False) -> Dict[str, Any]:
        """
        Get all documents from the collection.
        
        Args:
            include_embeddings: Whether to include embedding vectors
            
        Returns:
            Dictionary with all documents and their metadata
        """
        includes = ["documents", "metadatas"]
        if include_embeddings:
            includes.append("embeddings")
            
        return self.collection.get(include=includes)
    
    def get_documents_by_filter(self, 
                               filter_dict: Optional[Dict[str, Any]] = None, 
                               limit: int = 100, 
                               offset: int = 0) -> Dict[str, Any]:
        """
        Get documents with optional filtering.
        
        Args:
            filter_dict: Dictionary with filter conditions
            limit: Maximum number of documents to return
            offset: Offset for pagination
            
        Returns:
            Dictionary with filtered documents
        """
        return self.collection.get(
            where=filter_dict,
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"]
        )
    
    def delete_all_documents(self) -> int:
        """
        Delete all documents from the collection.
        
        Returns:
            Number of documents deleted
        """
        try:
            # Get count before deletion
            count = self.collection.count()
            
            if count > 0:
                # Get all document IDs
                results = self.collection.get(include=[])
                
                if results and "ids" in results and results["ids"]:
                    # Delete all documents by ID
                    self.collection.delete(ids=results["ids"])
                    self.logger.info(f"Deleted {len(results['ids'])} documents")
                else:
                    # Fallback method: recreate the collection
                    try:
                        # First try to delete the entire collection
                        self.client.delete_collection("documents")
                        self.logger.info("Collection deleted")
                        
                        # Then recreate it
                        self.collection = self.client.create_collection(
                            name="documents",
                            metadata={"description": "Main document collection for RAG processing"}
                        )
                        self.logger.info("Collection recreated")
                    except Exception as inner_e:
                        self.logger.error(f"Error during collection recreation: {inner_e}")
                        raise
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error deleting documents: {e}")
            raise
    
    def is_healthy(self) -> Tuple[bool, str]:
        """
        Check if the database connection is healthy.
        
        Returns:
            Tuple of (is_healthy, status_message)
        """
        try:
            self.client.heartbeat()
            doc_count = self.collection.count()
            return True, f"healthy with {doc_count} documents"
        except Exception as e:
            return False, f"unhealthy: {str(e)}"