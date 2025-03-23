"""
Elasticsearch service.

This module provides functionality to interact with Elasticsearch for document storage and retrieval.
"""

from elasticsearch import Elasticsearch, NotFoundError, ConnectionError
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

class ElasticsearchService:
    """Service for interacting with Elasticsearch."""
    
    def __init__(self, url: str, index_name: str, max_retries: int = 5, retry_delay: int = 3):
        """
        Initialize the Elasticsearch service with connection settings.
        
        Args:
            url: Elasticsearch URL
            index_name: Name of the index to use
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retries in seconds
        """
        self.url = url
        self.index_name = index_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize the connection
        self._connect()
        
    def _connect(self) -> None:
        """Establish connection to Elasticsearch with retry logic."""
        # Try connection with retry logic
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Connection attempt {attempt + 1} to Elasticsearch...")
                self.client = Elasticsearch(self.url)
                
                # Test connection with info request
                info = self.client.info()
                self.logger.info(f"Successfully connected to Elasticsearch on attempt {attempt + 1}")
                self.logger.info(f"Elasticsearch version: {info.get('version', {}).get('number', 'unknown')}")
                
                # Initialize the index
                self._init_index()
                break
                
            except ConnectionError as e:
                self.logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error("All connection attempts failed.")
                    raise RuntimeError(f"Failed to connect to Elasticsearch: {e}")
    
    def _init_index(self) -> None:
        """Initialize or get the index."""
        try:
            # Check if index exists
            if not self.client.indices.exists(index=self.index_name):
                # Create index with mappings for text search
                mappings = {
                    "mappings": {
                        "properties": {
                            "text": {
                                "type": "text",
                                "analyzer": "standard"
                            },
                            "embedding": {
                                "type": "dense_vector",
                                "dims": 384,  # Adjust based on your embedding model
                                "index": True,
                                "similarity": "cosine"
                            },
                            "metadata": {
                                "type": "object",
                                "dynamic": True
                            }
                        }
                    },
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    }
                }
                
                # Create the index with mappings
                self.client.indices.create(index=self.index_name, body=mappings)
                self.logger.info(f"Created index '{self.index_name}'")
            else:
                self.logger.info(f"Index '{self.index_name}' already exists")
                
            # Get document count
            count = self.get_document_count()
            self.logger.info(f"Index '{self.index_name}' has {count} documents")
            
        except Exception as e:
            self.logger.error(f"Error creating index: {e}")
            raise RuntimeError(f"Failed to create Elasticsearch index: {e}")
    
    def add_documents(self, documents: List[str], 
                     embeddings: List[List[float]], 
                     ids: List[str], 
                     metadatas: List[Dict[str, Any]]) -> None:
        """
        Add documents to the index.
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            ids: List of unique IDs for the documents
            metadatas: List of metadata dictionaries for each document
        """
        if not documents:
            return
            
        # Prepare bulk indexing operations
        operations = []
        for i, doc_id in enumerate(ids):
            # Create the document with text, embedding, and metadata
            doc = {
                "text": documents[i],
                "embedding": embeddings[i],
                "metadata": metadatas[i]
            }
            
            # Add to bulk operations
            operations.append({"index": {"_index": self.index_name, "_id": doc_id}})
            operations.append(doc)
        
        # Execute bulk operation
        if operations:
            response = self.client.bulk(body=operations, refresh=True)
            if response.get("errors", False):
                self.logger.error(f"Errors during bulk indexing: {response}")
            else:
                self.logger.info(f"Indexed {len(ids)} documents")
    
    def query_documents_by_vector(self, query_embedding: List[float], n_results: int = 3) -> Dict[str, Any]:
        """
        Query the index for similar documents using vector similarity.
        
        Args:
            query_embedding: Embedding vector for the query
            n_results: Number of results to return
            
        Returns:
            Dictionary with query results
        """
        query = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": n_results,
                "num_candidates": n_results * 2
            },
            "_source": ["text", "metadata"]
        }
        
        response = self.client.search(index=self.index_name, body=query)
        
        # Format response similar to ChromaDB
        results = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": []
        }
        
        for hit in response["hits"]["hits"]:
            results["ids"].append(hit["_id"])
            results["documents"].append(hit["_source"]["text"])
            results["metadatas"].append(hit["_source"]["metadata"])
            # For KNN search, the score is relevance (higher is better), 
            # but we need distance (lower is better) to match ChromaDB
            # Convert to a distance-like metric (1 - score/max_score)
            if response["hits"]["max_score"] > 0:
                distance = 1.0 - (hit["_score"] / response["hits"]["max_score"])
            else:
                distance = 1.0
            results["distances"].append(distance)
            
        return results
    
    def query_documents_by_text(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Query the index for documents matching text query using BM25.
        
        Args:
            query_text: Text query
            n_results: Number of results to return
            
        Returns:
            Dictionary with query results
        """
        query = {
            "query": {
                "match": {
                    "text": {
                        "query": query_text,
                        "operator": "and"
                    }
                }
            },
            "size": n_results
        }
        
        response = self.client.search(index=self.index_name, body=query)
        
        # Format response similar to ChromaDB
        results = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": []
        }
        
        for hit in response["hits"]["hits"]:
            results["ids"].append(hit["_id"])
            results["documents"].append(hit["_source"]["text"])
            results["metadatas"].append(hit["_source"]["metadata"])
            # For text search, convert score to a distance-like metric
            if response["hits"]["max_score"] > 0:
                distance = 1.0 - (hit["_score"] / response["hits"]["max_score"])
            else:
                distance = 1.0
            results["distances"].append(distance)
            
        return results
    
    def hybrid_search(self, query_text: str, query_embedding: List[float], 
                    n_results: int = 3, vector_weight: float = 0.7) -> Dict[str, Any]:
        """
        Perform hybrid search combining vector similarity and text relevance.
        
        Args:
            query_text: Text query for BM25 search
            query_embedding: Embedding vector for semantic search
            n_results: Number of results to return
            vector_weight: Weight given to vector search vs. text search (0-1)
            
        Returns:
            Dictionary with hybrid query results
        """
        # Perform hybrid search using Elasticsearch's rank_features
        query = {
            "query": {
                "function_score": {
                    "query": {
                        "match": {
                            "text": {
                                "query": query_text,
                                "operator": "and"
                            }
                        }
                    },
                    "functions": [
                        {
                            "script_score": {
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                    "params": {
                                        "query_vector": query_embedding
                                    }
                                }
                            },
                            "weight": vector_weight
                        },
                        {
                            "weight": 1.0 - vector_weight
                        }
                    ],
                    "score_mode": "sum",
                    "boost_mode": "multiply"
                }
            },
            "size": n_results
        }
        
        response = self.client.search(index=self.index_name, body=query)
        
        # Format response similar to ChromaDB
        results = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": []
        }
        
        for hit in response["hits"]["hits"]:
            results["ids"].append(hit["_id"])
            results["documents"].append(hit["_source"]["text"])
            results["metadatas"].append(hit["_source"]["metadata"])
            # For hybrid search, convert score to a distance-like metric
            if response["hits"]["max_score"] > 0:
                distance = 1.0 - (hit["_score"] / response["hits"]["max_score"])
            else:
                distance = 1.0
            results["distances"].append(distance)
            
        return results
    
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the index.
        
        Returns:
            Number of documents
        """
        try:
            response = self.client.count(index=self.index_name)
            return response.get("count", 0)
        except NotFoundError:
            # Index doesn't exist yet
            return 0
    
    def get_all_documents(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """
        Get all documents from the index with pagination.
        
        Args:
            limit: Maximum number of documents to return
            offset: Offset for pagination
            
        Returns:
            Dictionary with all documents and their metadata
        """
        query = {
            "query": {
                "match_all": {}
            },
            "from": offset,
            "size": limit
        }
        
        response = self.client.search(index=self.index_name, body=query)
        
        # Format response similar to ChromaDB
        results = {
            "ids": [],
            "documents": [],
            "metadatas": []
        }
        
        for hit in response["hits"]["hits"]:
            results["ids"].append(hit["_id"])
            results["documents"].append(hit["_source"]["text"])
            results["metadatas"].append(hit["_source"]["metadata"])
            
        return results
    
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
        if filter_dict is None or not filter_dict:
            return self.get_all_documents(limit, offset)
            
        # Convert filter_dict to Elasticsearch query
        filter_terms = []
        for key, value in filter_dict.items():
            if key == "$and" or key == "$or":
                # Handle complex filters later
                continue
            
            # For simple filters, use metadata.key: value format
            filter_terms.append({
                "term": {
                    f"metadata.{key}": value
                }
            })
        
        query = {
            "query": {
                "bool": {
                    "must": filter_terms
                }
            },
            "from": offset,
            "size": limit
        }
        
        response = self.client.search(index=self.index_name, body=query)
        
        # Format response similar to ChromaDB
        results = {
            "ids": [],
            "documents": [],
            "metadatas": []
        }
        
        for hit in response["hits"]["hits"]:
            results["ids"].append(hit["_id"])
            results["documents"].append(hit["_source"]["text"])
            results["metadatas"].append(hit["_source"]["metadata"])
            
        return results
    
    def delete_all_documents(self) -> int:
        """
        Delete all documents from the index.
        
        Returns:
            Number of documents deleted
        """
        try:
            # Get count before deletion
            count = self.get_document_count()
            
            if count > 0:
                # Delete by query to remove all documents
                response = self.client.delete_by_query(
                    index=self.index_name,
                    body={"query": {"match_all": {}}},
                    refresh=True
                )
                
                self.logger.info(f"Deleted {response.get('deleted', 0)} documents")
            
            return count
            
        except Exception as e:
            self.logger.error(f"Error deleting documents: {e}")
            raise
    
    def is_healthy(self) -> Tuple[bool, str]:
        """
        Check if the Elasticsearch connection is healthy.
        
        Returns:
            Tuple of (is_healthy, status_message)
        """
        try:
            # Check cluster health
            health = self.client.cluster.health()
            status = health.get("status", "unknown")
            
            # Get document count
            doc_count = self.get_document_count()
            
            if status in ["green", "yellow"]:
                return True, f"healthy ({status}) with {doc_count} documents"
            else:
                return False, f"unhealthy: cluster status is {status}"
                
        except Exception as e:
            return False, f"unhealthy: {str(e)}"