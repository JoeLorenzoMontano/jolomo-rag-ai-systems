"""
Reranking utility for improving retrieval quality.

This module provides a reranking utility that uses LangChain's
reranking functionality to improve the ordering of retrieved documents.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document

class Reranker:
    """
    A utility for reranking retrieved documents based on relevance to the query.
    
    This reranker uses cross-encoder scoring to rerank documents, which produces
    more accurate relevance judgments than the initial vector similarity retrieval.
    """
    
    def __init__(self):
        """Initialize the reranker with optional configuration."""
        self.logger = logging.getLogger(__name__)
        
        try:
            # Import the cross-encoder reranker from sentence-transformers
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
            self.initialized = True
        except ImportError:
            self.logger.warning("Could not import sentence-transformers. Reranking will not be available.")
            self.initialized = False
    
    def rerank(self, 
               query: str, 
               documents: List[str], 
               ids: List[str], 
               metadatas: List[Dict[str, Any]],
               distances: Optional[List[float]] = None) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query: User query
            documents: List of document contents
            ids: List of document IDs
            metadatas: List of document metadata
            distances: Original similarity scores (lower is better)
            
        Returns:
            Tuple of reordered (docs, ids, metadatas, distances)
        """
        # No need to rerank if we have 0 or 1 document, or if reranker wasn't initialized
        if not documents or len(documents) <= 1 or not self.initialized:
            return documents, ids, metadatas, distances or [0.0] * len(documents)
            
        try:
            # Create pairs of (query, document) for each document
            query_doc_pairs = [(query, doc) for doc in documents]
            
            # Score the pairs with the cross-encoder model
            scores = self.model.predict(query_doc_pairs)
            
            # Create items with scores for sorting
            items = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                items.append({
                    'document': doc,
                    'id': ids[i] if i < len(ids) else f"doc-{i}",
                    'metadata': metadatas[i] if i < len(metadatas) else {},
                    'distance': distances[i] if distances and i < len(distances) else 0.5,
                    'score': score  # Higher is better for cross-encoder scores
                })
                
            # Sort by score (descending)
            sorted_items = sorted(items, key=lambda x: x['score'], reverse=True)
            
            # Rebuild lists in new order
            reranked_docs = [item['document'] for item in sorted_items]
            reranked_ids = [item['id'] for item in sorted_items]
            reranked_metadatas = [item['metadata'] for item in sorted_items]
            
            # Generate new distances where lower values (closer to 0) are better
            # by inverting and normalizing the scores
            max_score = max(scores) if scores else 1.0
            reranked_distances = [1.0 - (item['score'] / max_score) for item in sorted_items]
            
            self.logger.info(f"Reranked {len(documents)} documents for query: '{query}'")
            
            return reranked_docs, reranked_ids, reranked_metadatas, reranked_distances
            
        except Exception as e:
            self.logger.error(f"Error during reranking: {e}")
            # Return original ordering if reranking fails
            return documents, ids, metadatas, distances or [0.0] * len(documents)

    def rerank_fallback(self, 
                        query: str, 
                        documents: List[str], 
                        ids: List[str], 
                        metadatas: List[Dict[str, Any]],
                        distances: Optional[List[float]] = None) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
        """
        Fallback reranking that uses BM25 in case the main model isn't available.
        
        Args:
            query: User query
            documents: List of document contents
            ids: List of document IDs
            metadatas: List of document metadata
            distances: Original similarity scores
            
        Returns:
            Tuple of reordered (docs, ids, metadatas, distances)
        """
        if not documents or len(documents) <= 1:
            return documents, ids, metadatas, distances or [0.0] * len(documents)
            
        try:
            from rank_bm25 import BM25Okapi
            import nltk
            from nltk.tokenize import word_tokenize
            
            # Download necessary NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            # Tokenize the documents
            tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
            
            # Create a BM25 object
            bm25 = BM25Okapi(tokenized_docs)
            
            # Tokenize the query
            tokenized_query = word_tokenize(query.lower())
            
            # Get scores for the query
            scores = bm25.get_scores(tokenized_query)
            
            # Create pairs of documents and scores
            doc_score_pairs = list(zip(documents, ids, metadatas, scores))
            
            # Sort by score (descending)
            sorted_pairs = sorted(doc_score_pairs, key=lambda x: x[3], reverse=True)
            
            # Unpack sorted results
            reranked_docs, reranked_ids, reranked_metadatas, reranked_scores = zip(*sorted_pairs) if sorted_pairs else ([], [], [], [])
            
            # Convert BM25 scores to distances (lower is better)
            max_score = max(reranked_scores) if reranked_scores else 1.0
            reranked_distances = [1.0 - (score / max_score) for score in reranked_scores]
            
            self.logger.info(f"Fallback reranked {len(documents)} documents using BM25")
            
            return list(reranked_docs), list(reranked_ids), list(reranked_metadatas), reranked_distances
            
        except ImportError:
            self.logger.warning("BM25 reranking failed due to missing dependencies")
            return documents, ids, metadatas, distances or [0.0] * len(documents)
        except Exception as e:
            self.logger.error(f"Error during BM25 reranking: {e}")
            return documents, ids, metadatas, distances or [0.0] * len(documents)