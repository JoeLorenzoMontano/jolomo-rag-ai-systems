"""
Query processing service.

This module handles query processing, document retrieval, and response generation.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional, Union

from utils.ollama_client import OllamaClient
from utils.query_classifier import QueryClassifier
from utils.web_search import WebSearchClient
from utils.reranker import Reranker
from services.database_service import DatabaseService
from services.elasticsearch_service import ElasticsearchService

class QueryService:
    """Service for processing queries and generating responses."""
    
    def __init__(self, 
                db_service: DatabaseService,
                elasticsearch_service: Optional[ElasticsearchService] = None,
                ollama_client: OllamaClient = None,
                query_classifier: QueryClassifier = None,
                web_search_client: Optional[WebSearchClient] = None):
        """
        Initialize the query processing service.
        
        Args:
            db_service: Database service for retrieval
            elasticsearch_service: Optional Elasticsearch service for hybrid search
            ollama_client: Ollama client for embeddings and responses
            query_classifier: Query classifier for routing
            web_search_client: Optional web search client
        """
        self.db_service = db_service
        self.elasticsearch_service = elasticsearch_service
        self.ollama_client = ollama_client
        self.query_classifier = query_classifier
        self.web_search_client = web_search_client
        
        # Initialize reranker
        self.reranker = Reranker()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def process_query(self, 
                     query: str, 
                     n_results: int = 3, 
                     combine_chunks: bool = True,
                     web_search: Optional[bool] = None,
                     web_results_count: int = 5,
                     explain_classification: bool = False,
                     enhance_query: bool = True,
                     use_elasticsearch: Optional[bool] = None,
                     hybrid_search: bool = True,
                     apply_reranking: bool = True) -> Dict[str, Any]:
        """
        Process a query and generate a response.
        
        Args:
            query: The user's query
            n_results: Number of results to return
            combine_chunks: Whether to combine chunks from the same document
            web_search: Whether to use web search (None for auto)
            web_results_count: Number of web search results to include
            explain_classification: Whether to include classification explanation
            enhance_query: Whether to enhance the query for better retrieval (default: True)
            use_elasticsearch: Whether to use Elasticsearch (None for auto based on availability)
            hybrid_search: Whether to combine ChromaDB and Elasticsearch results for better retrieval
            
        Returns:
            Dictionary with query response and sources
        """
        try:
            # Check if ChromaDB has any documents at all
            doc_count = self.db_service.get_document_count()
            if doc_count == 0:
                return {
                    "query": query,
                    "response": "No documents have been processed yet. Please use the /process endpoint first.",
                    "sources": {"documents": [], "ids": [], "metadatas": []},
                    "status": "error",
                    "error": "Empty collection"
                }
            
            # Store the original query for response
            original_query = query
            
            # Enhance the query if enabled
            search_query = query
            enhanced_query_text = None
            
            if enhance_query:
                try:
                    self.logger.info(f"Enhancing query: '{query}'")
                    enhanced_query_text = self.ollama_client.enhance_query(query)
                    
                    if enhanced_query_text and enhanced_query_text != query:
                        self.logger.info(f"Enhanced query: '{enhanced_query_text}'")
                        search_query = enhanced_query_text
                except Exception as e:
                    self.logger.warning(f"Query enhancement failed: {e}, using original query")
            
            # Generate embedding for the query using Ollama
            query_embedding = self.ollama_client.generate_embedding(search_query)
            
            # Log the embedding dimension for debugging
            self.logger.info(f"Generated query embedding with dimension: {len(query_embedding)}")
            
            # Determine if we should use Elasticsearch
            elasticsearch_available = self.elasticsearch_service is not None
            should_use_elasticsearch = False
            
            if use_elasticsearch is not None:
                # User explicitly specified whether to use Elasticsearch
                should_use_elasticsearch = use_elasticsearch and elasticsearch_available
            else:
                # Auto-determine based on availability
                should_use_elasticsearch = elasticsearch_available
            
            search_engine = "chromadb"
            results = None
            
            try:
                # Using more results since we might combine chunks
                retrieve_count = n_results * 3 if combine_chunks else n_results
                
                if should_use_elasticsearch and hybrid_search:
                    # Perform hybrid search with both engines
                    self.logger.info(f"Performing hybrid search using both ChromaDB and Elasticsearch")
                    search_engine = "hybrid"
                    
                    # Get results from both engines
                    chroma_results = self.db_service.query_documents(query_embedding, n_results=retrieve_count)
                    
                    # Use Elasticsearch hybrid search (text + vector)
                    es_results = self.elasticsearch_service.hybrid_search(
                        query_text=search_query,
                        query_embedding=query_embedding,
                        n_results=retrieve_count,
                        vector_weight=0.7  # Favor vector similarity over text matching
                    )
                    
                    # Improved hybrid merging algorithm
                    # Start with both result sets
                    self.logger.info(f"Merging results from ChromaDB ({len(chroma_results['ids'][0])} results) and Elasticsearch ({len(es_results['ids'])} results)")
                    
                    # Initialize unified results
                    all_ids = []
                    all_docs = []
                    all_metadatas = []
                    all_scores = []  # We'll use scores instead of distances for merging (higher is better)
                    
                    # Process ChromaDB results
                    for i, chroma_id in enumerate(chroma_results['ids'][0]):
                        # Convert distance to score (1.0 - distance)
                        chroma_score = 1.0 - chroma_results['distances'][0][i]
                        all_ids.append(chroma_id)
                        all_docs.append(chroma_results['documents'][0][i])
                        all_metadatas.append(chroma_results['metadatas'][0][i])
                        # Add source information to metadata
                        all_metadatas[-1]['search_source'] = 'vector'
                        # Store score with a multiplier for ChromaDB results
                        all_scores.append((chroma_score * 0.7, i, 'vector'))  # (score, original_index, source)
                    
                    # Process Elasticsearch results
                    for i, es_id in enumerate(es_results['ids']):
                        # Check if this document is already in results from ChromaDB
                        if es_id in all_ids:
                            # If already included, update the score by adding the ES score
                            idx = all_ids.index(es_id)
                            es_score = 1.0 - es_results['distances'][i]
                            # Update score tuple with combined sources
                            current_score, original_idx, source = all_scores[idx]
                            # Add the Elasticsearch score with its weight
                            all_scores[idx] = (current_score + (es_score * 0.3), original_idx, 'hybrid')
                            # Update metadata to indicate it was found by both
                            all_metadatas[idx]['search_source'] = 'hybrid'
                        else:
                            # Not in results yet, add it
                            es_score = 1.0 - es_results['distances'][i]
                            all_ids.append(es_id)
                            all_docs.append(es_results['documents'][i])
                            all_metadatas.append(es_results['metadatas'][i])
                            # Add source information
                            all_metadatas[-1]['search_source'] = 'text'
                            # Store score with original index and source
                            all_scores.append((es_score * 0.3, i + len(chroma_results['ids'][0]), 'text'))
                    
                    # Sort by score (descending)
                    combined_results = list(zip(all_ids, all_docs, all_metadatas, all_scores))
                    sorted_results = sorted(combined_results, key=lambda x: x[3][0], reverse=True)
                    
                    # Log the combined results count
                    self.logger.info(f"Combined results: {len(sorted_results)} items after merging")
                    
                    # Limit to requested number and unpack
                    top_results = sorted_results[:retrieve_count]
                    ids, docs, metadatas, score_tuples = zip(*top_results) if top_results else ([], [], [], [])
                    
                    # Convert scores back to distances (lower is better)
                    distances = [1.0 - score_tuple[0] for score_tuple in score_tuples]
                    
                    # Format results to match ChromaDB structure
                    results = {
                        "ids": [list(ids)],
                        "documents": [list(docs)],
                        "metadatas": [list(metadatas)],
                        "distances": [list(distances)]
                    }
                    
                    # Log sources in final results
                    sources = [score[2] for score in score_tuples]
                    source_counts = {
                        'vector': sources.count('vector'),
                        'text': sources.count('text'),
                        'hybrid': sources.count('hybrid')
                    }
                    self.logger.info(f"Final hybrid results sources: {source_counts}")
                    
                    # Add metadata about the hybrid search sources
                    for i, metadata in enumerate(results["metadatas"][0]):
                        source = score_tuples[i][2]
                        metadata['search_source'] = source
                    
                elif should_use_elasticsearch:
                    # Use only Elasticsearch
                    self.logger.info(f"Using Elasticsearch for document retrieval")
                    search_engine = "elasticsearch"
                    
                    # Decide between vector search, text search, or hybrid based on query
                    if len(search_query.split()) <= 3:
                        # Short queries work better with vector search
                        results = self.elasticsearch_service.query_documents_by_vector(
                            query_embedding=query_embedding,
                            n_results=retrieve_count
                        )
                    else:
                        # For longer queries, use hybrid search
                        results = self.elasticsearch_service.hybrid_search(
                            query_text=search_query,
                            query_embedding=query_embedding,
                            n_results=retrieve_count
                        )
                    
                    # Format results to match ChromaDB structure for consistency
                    results = {
                        "ids": [results["ids"]],
                        "documents": [results["documents"]],
                        "metadatas": [results["metadatas"]],
                        "distances": [results["distances"]]
                    }
                else:
                    # Use ChromaDB (default)
                    self.logger.info(f"Using ChromaDB for document retrieval")
                    search_engine = "chromadb"
                    results = self.db_service.query_documents(query_embedding, n_results=retrieve_count)
                
            except Exception as e:
                # Handle potential embedding dimension mismatch
                error_msg = str(e)
                if "dimension" in error_msg.lower():
                    self.logger.error(f"Embedding dimension error: {error_msg}")
                    return {
                        "query": query,
                        "response": "Error: Embedding dimension mismatch. Please reprocess documents with the current embedding model.",
                        "sources": {"documents": [], "ids": [], "metadatas": []},
                        "status": "error",
                        "error": f"Embedding dimension mismatch. Documents in DB have different dimensions than current model output. {error_msg}"
                    }
                else:
                    # Re-raise other exceptions
                    raise
            
            # Handle the case where no documents are found
            if not results["documents"] or len(results["documents"][0]) == 0:
                return {
                    "query": query,
                    "response": "No relevant documents found in the database.",
                    "sources": {"documents": [], "ids": [], "metadatas": []},
                    "status": "not_found"
                }
            
            # Prepare document data
            docs = results["documents"][0]
            ids = results["ids"][0]
            metadatas = results["metadatas"][0] if "metadatas" in results else [{}] * len(ids)
            distances = results["distances"][0] if "distances" in results else [0] * len(ids)
            
            # Combine chunks from the same document if requested
            if combine_chunks:
                docs, ids, metadatas, distances = self._combine_chunks(docs, ids, metadatas, distances, n_results)
            
            # Apply reranking if enabled
            reranked = False
            if apply_reranking and len(docs) > 1:
                try:
                    self.logger.info(f"Applying reranking to {len(docs)} documents")
                    reranked_docs, reranked_ids, reranked_metadatas, reranked_distances = self.reranker.rerank(
                        query=query, 
                        documents=docs, 
                        ids=ids, 
                        metadatas=metadatas,
                        distances=distances
                    )
                    
                    # Only update if reranking was successful
                    if reranked_docs and len(reranked_docs) == len(docs):
                        docs = reranked_docs
                        ids = reranked_ids
                        metadatas = reranked_metadatas
                        distances = reranked_distances
                        reranked = True
                        self.logger.info("Successfully reranked documents")
                except Exception as e:
                    self.logger.warning(f"Primary reranking failed: {e}, trying fallback reranker")
                    
                    # Try fallback BM25 reranking
                    try:
                        reranked_docs, reranked_ids, reranked_metadatas, reranked_distances = self.reranker.rerank_fallback(
                            query=query, 
                            documents=docs, 
                            ids=ids, 
                            metadatas=metadatas,
                            distances=distances
                        )
                        
                        # Only update if fallback reranking was successful
                        if reranked_docs and len(reranked_docs) == len(docs):
                            docs = reranked_docs
                            ids = reranked_ids
                            metadatas = reranked_metadatas
                            distances = reranked_distances
                            reranked = True
                            self.logger.info("Successfully reranked documents using fallback BM25")
                    except Exception as fallback_error:
                        self.logger.warning(f"Fallback reranking also failed, using original order: {fallback_error}")
            
            # Get the best matching document (first result)
            best_match = docs[0] if docs else ""
            
            # Generate a response based on the best match using Ollama
            context = best_match
            
            # If the context is too short and we have multiple results, add more context
            # This improves answer quality by providing more information
            if len(context.split()) < 100 and len(docs) > 1:
                context = docs[0] + "\n\n" + docs[1]
            
            # Classify the query to determine if we should use web search
            source_type = "documents"
            confidence = 1.0
            classification_metadata = {}
            
            if web_search is None:  # Auto-classify if not explicitly set
                # Extract document scores for better classification
                doc_distance_scores = distances if distances else []
                # Convert distances to similarity scores (lower distance = higher similarity)
                doc_scores = [1.0 - min(d, 1.0) for d in doc_distance_scores] if doc_distance_scores else []
                
                # Classify the query
                source_type, confidence, classification_metadata = self.query_classifier.classify(
                    query=query, 
                    doc_scores=doc_scores
                )
                self.logger.info(f"Query classified as '{source_type}' with {confidence:.2f} confidence")
            
            # Decide whether to use web search based on classification or explicit setting
            should_use_web = web_search if web_search is not None else (
                source_type == "web" or source_type == "hybrid"
            )
            
            # Add web search results if enabled/auto-determined
            web_results = []
            if should_use_web and self.web_search_client:
                try:
                    self.logger.info(f"Performing web search for query: {query}")
                    web_results = self.web_search_client.search_with_serper(query, num_results=web_results_count)
                    
                    if web_results:
                        # Format web results and add to context
                        web_context = self.web_search_client.format_results_as_context(web_results)
                        context = web_context + "\n\n" + context
                        self.logger.info(f"Added {len(web_results)} web search results to context")
                except Exception as e:
                    self.logger.error(f"Error during web search: {e}")
                    # Continue with only vector DB results
            
            response = self.ollama_client.generate_response(context=context, query=query)
            
            # Clean up the response for better frontend rendering
            cleaned_results = {
                "documents": docs,
                "ids": ids,
                "metadatas": metadatas,
                "distances": distances,
                "combined_chunks": combine_chunks,
                "reranked": reranked,
                "web_results": web_results if web_results else []
            }
            
            # Prepare response
            response_data = {
                "query": original_query,  # Always return the original query
                "response": response, 
                "sources": cleaned_results,
                "status": "success",
                "web_search_used": len(web_results) > 0,  # Only true if actual web results were found and used
                "source_type": source_type,
                "search_engine": search_engine,
                "reranking_applied": reranked
            }
            
            # Add enhanced query information
            response_data["query_enhanced"] = False
            if enhanced_query_text and enhanced_query_text != original_query:
                response_data["enhanced_query"] = enhanced_query_text
                response_data["query_enhanced"] = True
            
            # Include classification details if requested
            if explain_classification and web_search is None:
                response_data["classification"] = {
                    "source_type": source_type,
                    "confidence": confidence,
                    "explanations": classification_metadata.get("explanations", []),
                    "matched_terms": classification_metadata.get("matched_terms", []),
                    "scores": classification_metadata.get("scores", {})
                }
                
            return response_data
            
        except Exception as e:
            self.logger.error(f"Error in process_query: {e}")
            
            # Provide a more helpful error response
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

    def _combine_chunks(self, 
                       docs: List[str], 
                       ids: List[str], 
                       metadatas: List[Dict[str, Any]], 
                       distances: List[float],
                       n_results: int) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
        """
        Combine chunks from the same document for better context.
        
        Args:
            docs: List of document texts
            ids: List of document IDs
            metadatas: List of document metadata
            distances: List of similarity distances
            n_results: Number of combined results to return
            
        Returns:
            Tuple of combined (docs, ids, metadatas, distances)
        """
        # Group by source document
        doc_groups = {}
        
        for i, doc_id in enumerate(ids):
            # Extract source filename from chunk ID or use full ID if not chunked
            source_file = doc_id
            if "#chunk-" in doc_id:
                source_file = doc_id.split("#chunk-")[0]
            
            # Create or update the document group
            if source_file not in doc_groups:
                doc_groups[source_file] = {
                    "content": [],
                    "ids": [],
                    "metadata": [],
                    "distances": [],
                    "avg_distance": 0
                }
            
            doc_groups[source_file]["content"].append(docs[i])
            doc_groups[source_file]["ids"].append(doc_id)
            doc_groups[source_file]["metadata"].append(metadatas[i])
            doc_groups[source_file]["distances"].append(distances[i])
        
        # Calculate average distance for sorting
        for source, group in doc_groups.items():
            group["avg_distance"] = sum(group["distances"]) / len(group["distances"])
        
        # Sort groups by average distance and limit to n_results
        sorted_groups = sorted(doc_groups.items(), key=lambda x: x[1]["avg_distance"])
        top_groups = sorted_groups[:n_results]
        
        # Combine chunks within each group and create the final result
        combined_docs = []
        combined_ids = []
        combined_metadatas = []
        combined_distances = []
        
        for source_file, group in top_groups:
            # Combine all chunks from this document
            combined_text = "\n\n".join(group["content"])
            combined_docs.append(combined_text)
            combined_ids.append(source_file)
            
            # Combine metadata - use the first chunk's metadata as base
            base_meta = group["metadata"][0].copy() if group["metadata"] else {}
            base_meta["chunk_count"] = len(group["content"])
            base_meta["chunks"] = group["ids"]
            combined_metadatas.append(base_meta)
            
            # Use average distance
            combined_distances.append(group["avg_distance"])
            
        return combined_docs, combined_ids, combined_metadatas, combined_distances
        
    def process_chat(self, messages: List[Dict[str, str]], context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a chat query with conversation history.
        
        Args:
            messages: List of message objects with 'role' and 'content' keys
            context: Optional context from RAG to use for grounding responses
            
        Returns:
            Dictionary with the response and other metadata
        """
        try:
            # Get the latest user message
            latest_user_message = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    latest_user_message = msg.get("content", "")
                    break
                    
            if not latest_user_message:
                return {
                    "status": "error",
                    "response": "No user message found in conversation history",
                    "error": "Missing user query"
                }
                
            # Classify the query to determine if it needs new RAG context or just conversation history
            source_type, confidence, classification_metadata = self.query_classifier.classify(
                query=latest_user_message,
                conversation_history=messages
            )
            
            self.logger.info(f"Query classified as '{source_type}' with {confidence:.2f} confidence")
            
            # Decide how to handle based on classification
            if source_type == "conversation":
                # This is a follow-up that doesn't need new RAG context
                self.logger.info("Using conversation history without new RAG context")
                # Use the existing conversation but add a system message to handle follow-ups better
                system_message = {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. The user is asking a follow-up question to your previous response.\n"
                        "- Answer based on your previous responses in this conversation\n"
                        "- If asked about specific numbered items, points, or details from your previous response, refer to them directly\n"
                        "- If you didn't previously mention numbered items/points and the user asks about them, politely explain you didn't provide a numbered list\n"
                        "- If you can't answer the follow-up from the conversation history, say: 'I don't have enough information to answer that specific question.'\n"
                        "- Be helpful and conversational\n"
                    )
                }
                
                # Create a copy of messages and insert system message at beginning
                chat_messages = list(messages)
                chat_messages.insert(0, system_message)
                
                response = self.ollama_client.generate_chat_response(
                    messages=chat_messages,
                    context=None
                )
            elif source_type == "hybrid_conversation":
                # Use both conversation history and RAG context, but with conversation taking priority
                self.logger.info("Using conversation history with light RAG context integration")
                # Only include essential RAG context to avoid overriding conversation flow
                if context:
                    # Use context but reduce its importance
                    response = self.ollama_client.generate_chat_response(
                        messages=messages,
                        context=context
                    )
                else:
                    response = self.ollama_client.generate_chat_response(messages=messages)
            else:
                # Standard RAG approach with full context
                self.logger.info("Using full RAG context with conversation history")
                response = self.ollama_client.generate_chat_response(
                    messages=messages,
                    context=context
                )
            
            # Prepare the response object
            response_data = {
                "status": "success",
                "response": response,
                "messages": messages,
                "source_type": source_type,
                "confidence": confidence,
                "classification": classification_metadata
            }
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Error in process_chat: {e}")
            
            # Return an error response
            error_response = {
                "status": "error",
                "error": str(e),
                "response": f"An error occurred while processing your chat: {str(e)}"
            }
            
            return error_response