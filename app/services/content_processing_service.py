"""
Content processing service.

This module handles document processing, chunking, and embedding operations.
"""

import os
import logging
from typing import Dict, List, Any, Tuple, Optional, Generator
import shutil
import io

from utils.text_chunker import TextChunker
from utils.ollama_client import OllamaClient
from utils.pdf_extractor import PDFExtractor
from utils.query_classifier import QueryClassifier
from services.database_service import DatabaseService
from services.elasticsearch_service import ElasticsearchService
from services.job_service import JobService, JOB_STATUS_PROCESSING, JOB_STATUS_COMPLETED, JOB_STATUS_FAILED
from core.utils import clean_filename
from core.config import ELASTICSEARCH_ENABLED

class ContentProcessingService:
    """Service for processing and managing document content."""
    
    def __init__(self, 
                db_service: DatabaseService,
                job_service: JobService,
                ollama_client: OllamaClient,
                query_classifier: QueryClassifier,
                docs_folder: str = "./data",
                max_chunk_size: int = 1000,
                min_chunk_size: int = 200,
                chunk_overlap: int = 100,
                enable_chunking: bool = True,
                elasticsearch_service: ElasticsearchService = None,
                generate_questions: bool = True,
                max_questions_per_chunk: int = 5):
        """
        Initialize the content processing service.
        
        Args:
            db_service: Database service for storage
            job_service: Job tracking service
            ollama_client: Ollama client for AI operations
            query_classifier: Query classifier for term extraction
            docs_folder: Folder to store documents
            max_chunk_size: Maximum chunk size for document splitting
            min_chunk_size: Minimum chunk size
            chunk_overlap: Overlap between chunks
            enable_chunking: Whether to enable chunking
            elasticsearch_service: Optional Elasticsearch service
            generate_questions: Whether to generate questions for each chunk
            max_questions_per_chunk: Maximum number of questions to generate per chunk
        """
        self.db_service = db_service
        self.job_service = job_service
        self.ollama_client = ollama_client
        self.query_classifier = query_classifier
        self.docs_folder = docs_folder
        self.elasticsearch_service = elasticsearch_service
        self.elasticsearch_enabled = ELASTICSEARCH_ENABLED and elasticsearch_service is not None
        self.generate_questions = generate_questions
        self.max_questions_per_chunk = max_questions_per_chunk
        
        # Initialize text chunker with default settings
        self.text_chunker = TextChunker(
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            chunk_overlap=chunk_overlap,
            enable_chunking=enable_chunking
        )
        
        # Initialize PDF extractor
        self.pdf_extractor = PDFExtractor(temp_dir=docs_folder)
        
        # Ensure the docs directory exists
        os.makedirs(self.docs_folder, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def process_documents_task(self, job_id: str, 
                              chunk_size: Optional[int] = None,
                              min_size: Optional[int] = None,
                              overlap: Optional[int] = None,
                              enable_chunking: Optional[bool] = None,
                              enhance_chunks: bool = True,
                              generate_questions: Optional[bool] = None,
                              max_questions_per_chunk: Optional[int] = None) -> None:
        """
        Process all documents in the docs folder as a background task.
        
        Args:
            job_id: ID of the job for tracking
            chunk_size: Optional override for max chunk size
            min_size: Optional override for min chunk size
            overlap: Optional override for chunk overlap
            enable_chunking: Optional override for chunking enabled
            enhance_chunks: Whether to enhance chunks with semantic enrichment
            generate_questions: Whether to generate questions for each chunk
            max_questions_per_chunk: Maximum number of questions to generate per chunk
        """
        # Keep track of results
        successful = 0
        failed = 0
        failed_files = []
        all_chunks = []
        all_chunk_ids = []
        source_files = []
        
        try:
            # Update job status to processing
            self.job_service.update_job_status(job_id, JOB_STATUS_PROCESSING)
            
            # Apply overrides if provided without modifying globals
            temp_max_chunk_size = chunk_size if chunk_size is not None else self.text_chunker.max_chunk_size
            temp_min_chunk_size = min_size if min_size is not None else self.text_chunker.min_chunk_size
            temp_chunk_overlap = overlap if overlap is not None else self.text_chunker.chunk_overlap
            temp_enable_chunking = enable_chunking if enable_chunking is not None else self.text_chunker.enable_chunking
            temp_generate_questions = generate_questions if generate_questions is not None else self.generate_questions
            temp_max_questions_per_chunk = max_questions_per_chunk if max_questions_per_chunk is not None else self.max_questions_per_chunk
            
            # Log if semantic enrichment is enabled
            if enhance_chunks:
                self.logger.info(f"Job {job_id}: Semantic chunk enrichment is ENABLED")
                
            # Log if question generation is enabled
            if temp_generate_questions:
                self.logger.info(f"Job {job_id}: Question generation is ENABLED (max {temp_max_questions_per_chunk} per chunk)")
            
            # Log chunking settings for this run
            self.logger.info(f"Job {job_id}: Using chunking settings:")
            self.logger.info(f"  ENABLE_CHUNKING: {temp_enable_chunking}")
            self.logger.info(f"  MAX_CHUNK_SIZE: {temp_max_chunk_size} chars")
            self.logger.info(f"  MIN_CHUNK_SIZE: {temp_min_chunk_size} chars")
            self.logger.info(f"  CHUNK_OVERLAP: {temp_chunk_overlap} chars")
            
            # Process files recursively and collect results
            source_files, all_chunks, all_chunk_ids, successful, failed, failed_files = self._process_directory(
                self.docs_folder, 
                job_id,
                temp_max_chunk_size,
                temp_min_chunk_size,
                temp_chunk_overlap,
                temp_enable_chunking,
                enhance_chunks,
                temp_generate_questions,
                temp_max_questions_per_chunk
            )
            
            # Update total files count
            self.job_service.update_job_status(job_id, 
                                          JOB_STATUS_PROCESSING, 
                                          progress=90, 
                                          total_files=len(source_files))
            
            if not all_chunks:
                self.job_service.mark_job_failed(job_id, "No documents to process")
                return
            
            self.logger.info(f"Job {job_id}: Processed {len(all_chunks)} chunks from {len(source_files)} source files")
            
            # After processing documents, refresh the domain terms using statistical approach
            term_update_status = None
            try:
                # Get document count before refresh
                prev_term_count = len(self.query_classifier.product_terms)
                
                # Use statistical extraction with NLTK
                self.logger.info(f"Job {job_id}: Refreshing domain terms using statistical extraction")
                self.query_classifier.update_terms_from_db(self.db_service.collection)
                
                # Get updated term count
                new_term_count = len(self.query_classifier.product_terms)
                term_update_status = {
                    "previous_term_count": prev_term_count,
                    "new_term_count": new_term_count,
                    "extraction_method": "statistical",
                    "terms_updated": True,
                    "sample_terms": self.query_classifier.product_terms[:10] if self.query_classifier.product_terms else []
                }
            except Exception as e:
                self.logger.error(f"Job {job_id}: Error refreshing domain terms: {e}")
                term_update_status = {
                    "terms_updated": False,
                    "error": str(e)
                }
            
            # Prepare enrichment status information
            enrichment_status = {
                "enabled": enhance_chunks,
                "chunks_processed": successful
            }
            
            # Prepare question generation status information
            question_generation_status = {
                "enabled": temp_generate_questions,
                "max_questions_per_chunk": temp_max_questions_per_chunk,
                "chunks_processed": successful
            }
            
            # Prepare result for completion
            result = {
                "message": "All documents processed successfully" if failed == 0 else "Documents processed with some errors",
                "source_files": len(source_files),
                "total_chunks": len(all_chunks),
                "successful_chunks": successful,
                "failed_chunks": failed,
                "failed_items": failed_files if failed > 0 else None,
                "chunking_enabled": temp_enable_chunking,
                "chunk_size": temp_max_chunk_size,
                "term_extraction": term_update_status,
                "semantic_enrichment": enrichment_status,
                "question_generation": question_generation_status
            }
            
            # Update job status to completed
            self.job_service.mark_job_completed(job_id, result)
                
            self.logger.info(f"Job {job_id}: Processing completed - {successful} chunks processed, {failed} failed")
                
        except Exception as e:
            error_message = f"Error in document processing: {str(e)}"
            self.logger.error(f"Job {job_id}: {error_message}")
            
            # Update job status to failed
            result = {
                "message": "Document processing failed",
                "error": error_message,
                "source_files": len(source_files),
                "total_chunks": len(all_chunks),
                "successful_chunks": successful,
                "failed_chunks": failed
            }
            self.job_service.mark_job_failed(job_id, error_message)
            self.job_service.update_job_status(job_id, JOB_STATUS_FAILED, result=result)
    
    def _process_directory(self, directory: str, job_id: str,
                          max_chunk_size: int, min_chunk_size: int, 
                          chunk_overlap: int, enable_chunking: bool,
                          enhance_chunks: bool, generate_questions: bool = False,
                          max_questions_per_chunk: int = 5) -> Tuple[List[str], List[str], List[str], int, int, List[str]]:
        """
        Process files in a directory recursively.
        
        Args:
            directory: Directory to process
            job_id: Job ID for tracking
            max_chunk_size: Maximum chunk size
            min_chunk_size: Minimum chunk size
            chunk_overlap: Chunk overlap
            enable_chunking: Whether chunking is enabled
            enhance_chunks: Whether to add semantic enrichment
            generate_questions: Whether to generate questions for each chunk
            max_questions_per_chunk: Maximum number of questions to generate per chunk
            
        Returns:
            Tuple of (source_files, all_chunks, all_chunk_ids, successful, failed, failed_files)
        """
        source_files = []
        all_chunks = []
        all_chunk_ids = []
        successful = 0
        failed = 0
        failed_files = []
        
        # Create a temporary text chunker with custom settings
        temp_chunker = TextChunker(
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            chunk_overlap=chunk_overlap,
            enable_chunking=enable_chunking
        )
        
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if not filename.endswith('.md'):
                    continue
                    
                file_path = os.path.join(root, filename)
                source_files.append(file_path)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Use relative path as identifier
                        rel_path = os.path.relpath(file_path, self.docs_folder)
                        
                        # Apply chunking with temp settings
                        chunks = temp_chunker.chunk_text(content, rel_path)
                        
                        # Process this file's chunks
                        self.logger.info(f"Job {job_id}: Processing {len(chunks)} chunks from file {rel_path}")
                        
                        # First collect all chunks to allow for context passing
                        chunk_list = list(chunks)
                        
                        file_successful, file_failed = self._process_chunks(
                            chunk_list, all_chunks, all_chunk_ids, job_id, 
                            enhance_chunks, failed_files, generate_questions,
                            max_questions_per_chunk
                        )
                        
                        successful += file_successful
                        failed += file_failed
                        
                        # Update progress for this file
                        processed_files = len([f for f in source_files if os.path.isfile(f)])
                        progress = int(processed_files / max(1, len([f for f in source_files if f.endswith('.md')])) * 80)
                        self.job_service.update_job_status(job_id, 
                                                      JOB_STATUS_PROCESSING, 
                                                      progress=progress,
                                                      processed_files=processed_files,
                                                      successful_chunks=successful,
                                                      failed_chunks=failed)
                        
                except Exception as e:
                    self.logger.error(f"Job {job_id}: Error reading file {file_path}: {e}")
                    failed_files.append(f"{file_path} ({str(e)})")
                    failed += 1
        
        return source_files, all_chunks, all_chunk_ids, successful, failed, failed_files
    
    def _process_chunks(self, chunk_list: List[Tuple[str, str]], 
                      all_chunks: List[str], all_chunk_ids: List[str],
                      job_id: str, enhance_chunks: bool, 
                      failed_files: List[str],
                      generate_questions: bool = False,
                      max_questions_per_chunk: int = 5) -> Tuple[int, int]:
        """
        Process and embed a list of document chunks.
        
        Args:
            chunk_list: List of (text, id) chunk tuples
            all_chunks: List to append chunk texts to
            all_chunk_ids: List to append chunk IDs to
            job_id: Job ID for tracking
            enhance_chunks: Whether to add semantic enrichment
            failed_files: List to append failure records to
            generate_questions: Whether to generate questions for each chunk
            max_questions_per_chunk: Maximum number of questions to generate per chunk
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        successful = 0
        failed = 0
        
        for i, (chunk_text, chunk_id) in enumerate(chunk_list):
            try:
                # Skip empty chunks
                if not chunk_text.strip():
                    self.logger.info(f"Job {job_id}: Skipping empty chunk: {chunk_id}")
                    failed += 1
                    failed_files.append(f"{chunk_id} (empty)")
                    continue
                    
                # Store chunk for tracking
                all_chunks.append(chunk_text)
                all_chunk_ids.append(chunk_id)
                
                # Generate embedding for this chunk
                processing_text = chunk_text
                metadata = {"original_text": chunk_text}
                
                # Generate semantic enrichment if enabled
                if enhance_chunks:
                    try:
                        # Get previous and next chunks for context if they exist
                        prev_chunk_text = None
                        next_chunk_text = None
                        
                        if i > 0:
                            prev_chunk_text = chunk_list[i-1][0]
                            
                        if i < len(chunk_list) - 1:
                            next_chunk_text = chunk_list[i+1][0]
                            
                        # Generate context-aware summary
                        enrichment = self.ollama_client.generate_semantic_enrichment(
                            chunk_text, 
                            chunk_id,
                            prev_chunk_text,
                            next_chunk_text
                        )
                        
                        if enrichment.strip():
                            enhanced_text = f"{chunk_text}\n\nENRICHMENT:\n{enrichment}"
                            processing_text = enhanced_text
                            metadata["has_enrichment"] = True
                            metadata["enrichment"] = enrichment
                            self.logger.info(f"Job {job_id}: Enhanced chunk {chunk_id} with contextual summary")
                    except Exception as e:
                        self.logger.error(f"Job {job_id}: Error generating enrichment for {chunk_id}: {e}")
                        metadata["has_enrichment"] = False
                        metadata["enrichment_error"] = str(e)
                else:
                    metadata["has_enrichment"] = False
                    
                # Generate questions from chunk if enabled
                if generate_questions:
                    try:
                        # Generate questions and answers
                        self.logger.info(f"Job {job_id}: Generating questions for chunk {chunk_id}")
                        questions_answers = self.ollama_client.generate_questions_from_chunk(
                            chunk_text, 
                            chunk_id,
                            max_questions_per_chunk
                        )
                        
                        if questions_answers and len(questions_answers) > 0:
                            metadata["has_questions"] = True
                            metadata["questions"] = questions_answers
                            # Log the questions generated
                            self.logger.info(f"Job {job_id}: Generated {len(questions_answers)} questions for chunk {chunk_id}")
                            question_texts = [qa["question"] for qa in questions_answers]
                            self.logger.info(f"Job {job_id}: Sample questions: {', '.join(question_texts[:2])}...")
                        else:
                            metadata["has_questions"] = False
                    except Exception as e:
                        self.logger.error(f"Job {job_id}: Error generating questions for {chunk_id}: {e}")
                        metadata["has_questions"] = False
                        metadata["questions_error"] = str(e)
                else:
                    metadata["has_questions"] = False
                
                # Generate embedding
                embedding = self.ollama_client.generate_embedding(processing_text)
                
                # Add file information to metadata
                if "#chunk-" in chunk_id:
                    source_file = chunk_id.split("#chunk-")[0]
                    chunk_num = chunk_id.split("#chunk-")[1]
                    metadata["filename"] = source_file
                    metadata["chunk_id"] = chunk_id
                    metadata["chunk_num"] = chunk_num
                else:
                    metadata["filename"] = chunk_id
                
                # Add to ChromaDB immediately
                self.db_service.add_documents(
                    documents=[processing_text],
                    embeddings=[embedding],
                    ids=[chunk_id],
                    metadatas=[metadata]
                )
                
                # Add to Elasticsearch if enabled
                if self.elasticsearch_enabled and self.elasticsearch_service:
                    try:
                        self.elasticsearch_service.add_documents(
                            documents=[processing_text],
                            embeddings=[embedding],
                            ids=[chunk_id],
                            metadatas=[metadata]
                        )
                        self.logger.info(f"Job {job_id}: Added chunk {chunk_id} to Elasticsearch")
                    except Exception as es_error:
                        self.logger.error(f"Job {job_id}: Error adding to Elasticsearch: {es_error}")
                        # Continue with ChromaDB only
                
                successful += 1
                
            except Exception as e:
                self.logger.error(f"Job {job_id}: Error processing chunk {chunk_id}: {e}")
                failed += 1
                failed_files.append(f"{chunk_id} ({str(e)})")
        
        return successful, failed
    
    def process_single_file_task(self, job_id: str, file_path: str, 
                             chunk_size: Optional[int] = None,
                             min_size: Optional[int] = None,
                             overlap: Optional[int] = None,
                             enable_chunking: Optional[bool] = None,
                             enhance_chunks: bool = True,
                             generate_questions: Optional[bool] = None,
                             max_questions_per_chunk: Optional[int] = None) -> None:
        """
        Process a single uploaded file as a background task.
        
        Args:
            job_id: ID of the job for tracking
            file_path: Path to the file to process
            chunk_size: Optional override for max chunk size
            min_size: Optional override for min chunk size
            overlap: Optional override for chunk overlap
            enable_chunking: Optional override for chunking enabled
            enhance_chunks: Whether to enhance chunks with semantic enrichment
            generate_questions: Whether to generate questions for each chunk
            max_questions_per_chunk: Maximum number of questions to generate per chunk
        """
        try:
            # Update job status to processing
            self.job_service.update_job_status(job_id, JOB_STATUS_PROCESSING, progress=10)
            
            # Check if file exists
            if not os.path.exists(file_path):
                self.job_service.mark_job_failed(job_id, f"File not found: {file_path}")
                return
                
            # Track results
            successful = 0
            failed = 0
            all_chunks = []
            all_chunk_ids = []
            
            self.logger.info(f"Job {job_id}: Processing single file: {file_path}")
            
            try:
                # Read the file
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                    # Use relative path as identifier
                    rel_path = os.path.relpath(file_path, self.docs_folder)
                    
                    # Apply chunking with overrides if provided
                    temp_max_chunk_size = chunk_size if chunk_size is not None else self.text_chunker.max_chunk_size
                    temp_min_chunk_size = min_size if min_size is not None else self.text_chunker.min_chunk_size
                    temp_chunk_overlap = overlap if overlap is not None else self.text_chunker.chunk_overlap
                    temp_enable_chunking = enable_chunking if enable_chunking is not None else self.text_chunker.enable_chunking
                    
                    # Create a temporary text chunker with custom settings
                    temp_chunker = TextChunker(
                        max_chunk_size=temp_max_chunk_size,
                        min_chunk_size=temp_min_chunk_size,
                        chunk_overlap=temp_chunk_overlap,
                        enable_chunking=temp_enable_chunking
                    )
                    
                    # Log chunking settings
                    self.logger.info(f"Job {job_id}: Using chunking settings:")
                    self.logger.info(f"  ENABLE_CHUNKING: {temp_enable_chunking}")
                    self.logger.info(f"  MAX_CHUNK_SIZE: {temp_max_chunk_size} chars")
                    self.logger.info(f"  MIN_CHUNK_SIZE: {temp_min_chunk_size} chars")
                    self.logger.info(f"  CHUNK_OVERLAP: {temp_chunk_overlap} chars")
                    
                    # Apply chunking with custom settings
                    chunks = temp_chunker.chunk_text(content, rel_path)
                    
                    # Add chunks to our collection
                    for chunk_text, chunk_id in chunks:
                        all_chunks.append(chunk_text)
                        all_chunk_ids.append(chunk_id)
                    
                    # Update progress
                    self.job_service.update_job_status(job_id, 
                                                  JOB_STATUS_PROCESSING, 
                                                  progress=30,
                                                  processed_files=1)
                        
            except Exception as e:
                self.logger.error(f"Job {job_id}: Error reading file {file_path}: {e}")
                self.job_service.mark_job_failed(job_id, f"Error reading file: {str(e)}")
                return
                    
            if not all_chunks:
                self.job_service.mark_job_failed(job_id, "No content could be chunked from the file")
                return
            
            self.logger.info(f"Job {job_id}: Created {len(all_chunks)} chunks from file")
            self.job_service.update_job_status(job_id, JOB_STATUS_PROCESSING, progress=50)
            
            # Initialize counters
            successful = 0
            failed = 0
            failed_files = []
                    
            # Process chunks with context
            chunk_pairs = list(zip(all_chunks, all_chunk_ids))
            
            # Process chunks with semantic enrichment
            for i, (chunk_text, chunk_id) in enumerate(chunk_pairs):
                try:
                    # Skip empty chunks
                    if not chunk_text.strip():
                        self.logger.info(f"Job {job_id}: Skipping empty chunk: {chunk_id}")
                        failed += 1
                        failed_files.append(f"{chunk_id} (empty)")
                        continue
                    
                    # Determine whether to use original or enriched text for embedding
                    processing_text = chunk_text
                    metadata = {"original_text": chunk_text}
                    
                    # Generate semantic enrichment with context
                    if enhance_chunks:
                        try:
                            # Get previous and next chunks for context if they exist
                            prev_chunk_text = None
                            next_chunk_text = None
                            
                            if i > 0:
                                prev_chunk_text = chunk_pairs[i-1][0]
                                
                            if i < len(chunk_pairs) - 1:
                                next_chunk_text = chunk_pairs[i+1][0]
                            
                            # Generate enrichment with context
                            enrichment = self.ollama_client.generate_semantic_enrichment(
                                chunk_text, 
                                chunk_id,
                                prev_chunk_text,
                                next_chunk_text
                            )
                            
                            if enrichment.strip():
                                # Create the enhanced text by combining original with enrichment
                                enhanced_text = f"{chunk_text}\n\nENRICHMENT:\n{enrichment}"
                                
                                # Use the enhanced text for embedding
                                processing_text = enhanced_text
                                
                                # Store both original and enrichment in metadata
                                metadata["has_enrichment"] = True
                                metadata["enrichment"] = enrichment
                                
                                self.logger.info(f"Job {job_id}: Enhanced chunk {chunk_id} with contextual summary")
                        except Exception as e:
                            self.logger.error(f"Job {job_id}: Error generating enrichment: {e}")
                            metadata["has_enrichment"] = False
                            metadata["enrichment_error"] = str(e)
                    else:
                        metadata["has_enrichment"] = False
                    
                    # Generate questions from chunk if enabled
                    temp_generate_questions = generate_questions if generate_questions is not None else self.generate_questions
                    temp_max_questions = max_questions_per_chunk if max_questions_per_chunk is not None else self.max_questions_per_chunk
                    
                    # Log question generation settings
                    if temp_generate_questions:
                        self.logger.info(f"Job {job_id}: Question generation is ENABLED (max {temp_max_questions} per chunk)")
                    
                    if temp_generate_questions:
                        try:
                            # Generate questions and answers
                            self.logger.info(f"Job {job_id}: Generating questions for chunk {chunk_id}")
                            questions_answers = self.ollama_client.generate_questions_from_chunk(
                                chunk_text, 
                                chunk_id,
                                temp_max_questions
                            )
                            
                            if questions_answers and len(questions_answers) > 0:
                                metadata["has_questions"] = True
                                metadata["questions"] = questions_answers
                                # Log the questions generated
                                self.logger.info(f"Job {job_id}: Generated {len(questions_answers)} questions for chunk {chunk_id}")
                                question_texts = [qa["question"] for qa in questions_answers]
                                self.logger.info(f"Job {job_id}: Sample questions: {', '.join(question_texts[:2])}...")
                            else:
                                metadata["has_questions"] = False
                        except Exception as e:
                            self.logger.error(f"Job {job_id}: Error generating questions for {chunk_id}: {e}")
                            metadata["has_questions"] = False
                            metadata["questions_error"] = str(e)
                    else:
                        metadata["has_questions"] = False
                    
                    # Generate embedding
                    self.logger.info(f"Job {job_id}: Processing chunk {i+1}/{len(all_chunks)}")
                    embedding = self.ollama_client.generate_embedding(processing_text)
                    
                    # Add file information to metadata
                    if "#chunk-" in all_chunk_ids[i]:
                        source_file = all_chunk_ids[i].split("#chunk-")[0]
                        chunk_num = all_chunk_ids[i].split("#chunk-")[1]
                        metadata["filename"] = source_file
                        metadata["chunk_id"] = all_chunk_ids[i]
                        metadata["chunk_num"] = chunk_num
                    else:
                        metadata["filename"] = all_chunk_ids[i]
                    
                    # Add to ChromaDB
                    self.db_service.add_documents(
                        documents=[processing_text],
                        embeddings=[embedding],
                        ids=[all_chunk_ids[i]],
                        metadatas=[metadata]
                    )
                    
                    # Add to Elasticsearch if enabled
                    if self.elasticsearch_enabled and self.elasticsearch_service:
                        try:
                            self.elasticsearch_service.add_documents(
                                documents=[processing_text],
                                embeddings=[embedding],
                                ids=[all_chunk_ids[i]],
                                metadatas=[metadata]
                            )
                            self.logger.info(f"Job {job_id}: Added chunk {all_chunk_ids[i]} to Elasticsearch")
                        except Exception as es_error:
                            self.logger.error(f"Job {job_id}: Error adding to Elasticsearch: {es_error}")
                            # Continue with ChromaDB only
                    
                    successful += 1
                    progress = 50 + int((i + 1) / len(all_chunks) * 40)  # Progress from 50% to 90%
                    self.job_service.update_job_status(job_id, 
                                              JOB_STATUS_PROCESSING, 
                                              progress=progress,
                                              successful_chunks=successful)
                    
                except Exception as e:
                    self.logger.error(f"Job {job_id}: Error processing chunk {i+1}: {e}")
                    failed += 1
                    failed_files.append(f"{chunk_id} ({str(e)})")
                    self.job_service.update_job_status(job_id, 
                                              JOB_STATUS_PROCESSING, 
                                              failed_chunks=failed)
            
            # Update domain terms using statistical approach
            term_update_status = None
            try:
                # Get document count before refresh
                prev_term_count = len(self.query_classifier.product_terms)
                
                # Use statistical extraction with NLTK
                self.logger.info(f"Job {job_id}: Refreshing domain terms using statistical extraction")
                self.query_classifier.update_terms_from_db(self.db_service.collection)
                
                # Get updated term count
                new_term_count = len(self.query_classifier.product_terms)
                term_update_status = {
                    "previous_term_count": prev_term_count,
                    "new_term_count": new_term_count,
                    "extraction_method": "statistical",
                    "terms_updated": True,
                    "sample_terms": self.query_classifier.product_terms[:10] if self.query_classifier.product_terms else []
                }
            except Exception as e:
                self.logger.error(f"Job {job_id}: Error refreshing domain terms: {e}")
                term_update_status = {
                    "terms_updated": False,
                    "error": str(e)
                }
            
            # Prepare enrichment status information
            enrichment_status = {
                "enabled": enhance_chunks,
                "chunks_processed": successful
            }
            
            # Prepare question generation status information
            temp_generate_questions = generate_questions if generate_questions is not None else self.generate_questions
            temp_max_questions = max_questions_per_chunk if max_questions_per_chunk is not None else self.max_questions_per_chunk
            
            question_generation_status = {
                "enabled": temp_generate_questions,
                "max_questions_per_chunk": temp_max_questions,
                "chunks_processed": successful
            }
            
            # Finalize job
            result = {
                "message": "File processed successfully" if failed == 0 else "File processed with some errors",
                "file_path": file_path,
                "total_chunks": len(all_chunks),
                "successful_chunks": successful,
                "failed_chunks": failed,
                "failed_items": failed_files if failed > 0 else None,
                "term_extraction": term_update_status,
                "semantic_enrichment": enrichment_status,
                "question_generation": question_generation_status
            }
            self.job_service.mark_job_completed(job_id, result)
            
            self.logger.info(f"Job {job_id}: Processing completed - {successful} chunks processed, {failed} failed")
            
        except Exception as e:
            error_message = f"Error processing file: {str(e)}"
            self.logger.error(f"Job {job_id}: {error_message}")
            
            result = {
                "message": "File processing failed",
                "error": error_message,
                "file_path": file_path
            }
            self.job_service.mark_job_failed(job_id, error_message)
            self.job_service.update_job_status(job_id, JOB_STATUS_FAILED, progress=100, result=result)
    
    def upload_file(self, file_content: bytes, filename: str) -> str:
        """
        Upload a file to the documents directory.
        
        Args:
            file_content: Binary content of the file
            filename: Original filename
            
        Returns:
            Path to the saved file
        """
        # Generate a safe filename to prevent path traversal
        safe_filename = clean_filename(filename)
        file_path = os.path.join(self.docs_folder, safe_filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Process PDF files by extracting text
        if filename.endswith('.pdf'):
            try:
                # Use our PDF extractor for better results
                pdf_text = self.pdf_extractor.extract_text(file_content, filename=safe_filename)
                
                # Save the extracted text to a markdown file
                md_filename = safe_filename.replace('.pdf', '.md')
                md_path = os.path.join(self.docs_folder, md_filename)
                
                with open(md_path, "w", encoding="utf-8") as md_file:
                    md_file.write(pdf_text)
                
                # Use the markdown file for processing instead
                return md_path
                
            except Exception as e:
                self.logger.error(f"Error processing PDF file: {e}")
                raise ValueError(f"Error processing PDF file: {str(e)}")
                
        # If the file is a text file, convert it to markdown
        elif filename.endswith('.txt'):
            try:
                # Read the text file
                with open(file_path, "r", encoding="utf-8") as txt_file:
                    txt_content = txt_file.read()
                
                # Save as markdown
                md_filename = safe_filename.replace('.txt', '.md')
                md_path = os.path.join(self.docs_folder, md_filename)
                
                with open(md_path, "w", encoding="utf-8") as md_file:
                    md_file.write(txt_content)
                
                # Use the markdown file for processing
                return md_path
                
            except Exception as e:
                self.logger.error(f"Error converting text file to markdown: {e}")
                raise ValueError(f"Error converting text file to markdown: {str(e)}")
        
        return file_path
    
    def refresh_domain_terms_task(self, job_id: str) -> None:
        """
        Refresh domain terms as a background task.
        
        Args:
            job_id: ID of the job for tracking
        """
        try:
            # Update job status to processing
            self.job_service.update_job_status(job_id, JOB_STATUS_PROCESSING, progress=10)
            
            # Get document count before refresh
            prev_term_count = len(self.query_classifier.product_terms)
            
            # Refresh terms from ChromaDB using statistical extraction
            self.logger.info(f"Job {job_id}: Using statistical approach for domain term extraction")
            
            # Update progress
            self.job_service.update_job_status(job_id, JOB_STATUS_PROCESSING, progress=30)
            
            # Do the actual term extraction
            self.query_classifier.update_terms_from_db(self.db_service.collection)
            
            # Update progress
            self.job_service.update_job_status(job_id, JOB_STATUS_PROCESSING, progress=80)
            
            # Get updated term count
            new_term_count = len(self.query_classifier.product_terms)
            
            # Prepare result
            result = {
                "message": "Domain terms refreshed successfully using statistical extraction",
                "extraction_method": "statistical",
                "previous_term_count": prev_term_count,
                "new_term_count": new_term_count,
                "sample_terms": self.query_classifier.product_terms[:20]  # Show first 20 terms as a sample
            }
            
            # Update job status to completed
            self.job_service.mark_job_completed(job_id, result)
            
            self.logger.info(f"Job {job_id}: Domain term extraction completed - {new_term_count} terms extracted")
            
        except Exception as e:
            error_message = f"Error in domain term extraction: {str(e)}"
            self.logger.error(f"Job {job_id}: {error_message}")
            
            # Update job status to failed
            self.job_service.mark_job_failed(job_id, error_message)