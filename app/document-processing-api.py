from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks, UploadFile, File, Form
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Tuple, Optional
import threading
import uuid
import chromadb
from chromadb.config import Settings
import os
import re
import requests
import time as import_time
import shutil
import io
from utils.pdf_extractor import PDFExtractor
from utils.ollama_client import OllamaClient
from utils.web_search import WebSearchClient
from utils.document_processor import DocumentProcessor
from utils.query_classifier import QueryClassifier

# ===============================================================
# Environment Variables and Configuration
# ===============================================================

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
DOCS_FOLDER = "./data"

# Connection retry settings
MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds

# ===============================================================
# FastAPI Initialization
# ===============================================================

# Initialize FastAPI app
app = FastAPI(
    title="Document Processing API", 
    description="API for storing and retrieving documents with embeddings.", 
    version="1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================================================
# Service Initialization
# ===============================================================

# Initialize directory for documents
os.makedirs(DOCS_FOLDER, exist_ok=True)

# Log configuration
print(f"Document chunking settings:")
print(f"  ENABLE_CHUNKING: {ENABLE_CHUNKING}")
print(f"  MAX_CHUNK_SIZE: {MAX_CHUNK_SIZE} chars")
print(f"  MIN_CHUNK_SIZE: {MIN_CHUNK_SIZE} chars")
print(f"  CHUNK_OVERLAP: {CHUNK_OVERLAP} chars")

# Initialize ChromaDB connection
print(f"Connecting to ChromaDB at http://{CHROMA_HOST}:{CHROMA_PORT}")

# Create ChromaDB connection settings
chroma_settings = Settings(
    chroma_api_impl="rest",
    chroma_server_host=CHROMA_HOST,
    chroma_server_http_port=CHROMA_PORT,
)

# Try connection with retry logic
chroma_client = None
for attempt in range(MAX_RETRIES):
    try:
        print(f"Connection attempt {attempt + 1} to ChromaDB...")
        chroma_client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            ssl=False,
        )
        # Test connection with heartbeat
        chroma_client.heartbeat()
        print(f"Successfully connected to ChromaDB on attempt {attempt + 1}")
        
        # Get server version
        try:
            server_info = chroma_client._server_state()
            print(f"ChromaDB server version: {server_info.get('version', 'unknown')}")
        except:
            print("Could not get ChromaDB server version")
            
        break
    except Exception as e:
        print(f"Connection attempt {attempt + 1} failed: {str(e)}")
        if attempt < MAX_RETRIES - 1:
            print(f"Retrying in {RETRY_DELAY} seconds...")
            import time
            time.sleep(RETRY_DELAY)
        else:
            print("All connection attempts failed. Falling back to in-memory database.")
            print("WARNING: Data will not be persisted between restarts!")
            chroma_client = chromadb.EphemeralClient()
            print("Using in-memory ChromaDB")

# Create or get collection with proper settings
try:
    db_collection = chroma_client.get_or_create_collection(
        name="documents",
        metadata={"description": "Main document collection for RAG processing"}
    )
    print(f"Collection 'documents' ready with {db_collection.count()} documents")
except Exception as e:
    print(f"Error creating collection: {e}")
    raise RuntimeError(f"Failed to create ChromaDB collection: {e}")

# Initialize Ollama Client
ollama_client = OllamaClient()

# Initialize Web Search Client
web_search_client = WebSearchClient(serper_api_key=SERPER_API_KEY)

# Initialize Document Processor with default settings
doc_processor = DocumentProcessor(
    max_chunk_size=MAX_CHUNK_SIZE,
    min_chunk_size=MIN_CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    enable_chunking=ENABLE_CHUNKING
)

# Initialize Query Classifier for determining when to use web search
# Pass the ChromaDB collection to extract domain-specific terms, but postpone term extraction until we have ollama_client
query_classifier = QueryClassifier(confidence_threshold=0.6)

# Initialize the domain terms from the database after both the classifier and Ollama client are initialized
try:
    query_classifier.product_terms = ["duplocloud", "tenant", "infrastructure"]  # Default terms
except Exception as e:
    print(f"Error initializing domain terms: {e}")
    # Continue with minimal default terms

# Initialize PDF extractor
pdf_extractor = PDFExtractor(temp_dir=DOCS_FOLDER)

# ===============================================================
# Background Processing Job Tracking
# ===============================================================

# Dictionary to track background processing jobs
processing_jobs = {}

# Job status constants
JOB_STATUS_QUEUED = "queued"
JOB_STATUS_PROCESSING = "processing"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"

# Thread-safe lock for updating job status
job_lock = threading.Lock()

# ===============================================================
# API Endpoints
# ===============================================================

@app.get("/", summary="Root endpoint", description="Returns a simple message indicating the API is running.")
async def root():
    return {"message": "Document Processing API is running"}

@app.get("/health", summary="Health check", description="Checks if all components are operational.")
async def health_check():
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
        chroma_client.heartbeat()
        health_status["chroma"] = "healthy"
        
        # Check collection status
        try:
            doc_count = db_collection.count()
            health_status["collection"]["status"] = "healthy"
            health_status["collection"]["document_count"] = doc_count
        except Exception as e:
            health_status["collection"]["status"] = f"error: {str(e)}"
    except Exception as e:
        health_status["chroma"] = f"unhealthy: {str(e)}"
    
    # Check Ollama
    try:
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
                        print("Initializing domain terms with Ollama during health check")
                        query_classifier.update_terms_from_db(db_collection, ollama_client)
                        print(f"Domain terms initialized: {len(query_classifier.product_terms)} terms extracted")
                    except Exception as e:
                        print(f"Failed to initialize domain terms: {e}")
        except Exception as e:
            health_status["models"]["embedding_model"] += f" - error: {str(e)}"
    
    return health_status

@app.post("/process", summary="Start document embedding processing in the background", description="Starts processing documents in the background and returns a job ID for tracking progress.")
async def process_documents(
    background_tasks: BackgroundTasks,
    chunk_size: int = Query(None, description="Override max chunk size (chars)"),
    min_size: int = Query(None, description="Override min chunk size (chars)"),
    overlap: int = Query(None, description="Override chunk overlap (chars)"),
    enable_chunking: bool = Query(None, description="Override chunking enabled setting"),
    enhance_chunks: bool = Query(True, description="Generate additional content with Ollama to improve retrieval")
):
    # Create a unique job ID
    job_id = str(uuid.uuid4())
    
    # Set up initial job status
    with job_lock:
        processing_jobs[job_id] = {
            "status": JOB_STATUS_QUEUED,
            "progress": 0,
            "total_files": 0,
            "processed_files": 0,
            "successful_chunks": 0,
            "failed_chunks": 0,
            "error": None,
            "settings": {
                "chunk_size": chunk_size if chunk_size is not None else MAX_CHUNK_SIZE,
                "min_size": min_size if min_size is not None else MIN_CHUNK_SIZE,
                "overlap": overlap if overlap is not None else CHUNK_OVERLAP,
                "enable_chunking": enable_chunking if enable_chunking is not None else ENABLE_CHUNKING,
                "enhance_chunks": enhance_chunks
            },
            "result": None
        }
    
    # Add the background task
    background_tasks.add_task(
        process_documents_task, 
        job_id=job_id,
        chunk_size=chunk_size, 
        min_size=min_size, 
        overlap=overlap, 
        enable_chunking=enable_chunking,
        enhance_chunks=enhance_chunks
    )
    
    # Return the job ID and initial status
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Document processing started in background. Use /job/{job_id} to check status."
    }

def process_documents_task(
    job_id: str,
    chunk_size: Optional[int] = None,
    min_size: Optional[int] = None,
    overlap: Optional[int] = None,
    enable_chunking: Optional[bool] = None,
    enhance_chunks: bool = True
):
    # Keep track of results
    successful = 0
    failed = 0
    failed_files = []
    all_chunks = []
    all_chunk_ids = []
    source_files = []
    
    try:
        # Update job status to processing
        with job_lock:
            processing_jobs[job_id]["status"] = JOB_STATUS_PROCESSING
        
        # Apply overrides if provided without modifying globals
        temp_max_chunk_size = chunk_size if chunk_size is not None else MAX_CHUNK_SIZE
        temp_min_chunk_size = min_size if min_size is not None else MIN_CHUNK_SIZE
        temp_chunk_overlap = overlap if overlap is not None else CHUNK_OVERLAP
        temp_enable_chunking = enable_chunking if enable_chunking is not None else ENABLE_CHUNKING
        
        # Log if semantic enrichment is enabled
        if enhance_chunks:
            print(f"Job {job_id}: Semantic chunk enrichment is ENABLED")
        
        # Log chunking settings for this run
        print(f"Job {job_id}: Using chunking settings:")
        print(f"  ENABLE_CHUNKING: {temp_enable_chunking}")
        print(f"  MAX_CHUNK_SIZE: {temp_max_chunk_size} chars")
        print(f"  MIN_CHUNK_SIZE: {temp_min_chunk_size} chars")
        print(f"  CHUNK_OVERLAP: {temp_chunk_overlap} chars")
        
        # Function to process files recursively
        def process_directory(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isdir(file_path):
                    # Recursively process subdirectories
                    process_directory(file_path)
                elif os.path.isfile(file_path) and file_path.endswith('.md'):
                    source_files.append(file_path)  # Track all source files for reporting
                    file_chunks = []
                    file_chunk_ids = []
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            # Use relative path as identifier
                            rel_path = os.path.relpath(file_path, DOCS_FOLDER)
                            
                            # Apply chunking with temp settings
                            try:
                                # Create a temporary document processor with custom settings
                                temp_processor = DocumentProcessor(
                                    max_chunk_size=temp_max_chunk_size,
                                    min_chunk_size=temp_min_chunk_size,
                                    chunk_overlap=temp_chunk_overlap,
                                    enable_chunking=temp_enable_chunking
                                )
                                
                                # Use the document processor with temporary settings
                                chunks = temp_processor.chunk_document(content, rel_path)
                                
                                # Process this file's chunks immediately
                                print(f"Job {job_id}: Processing {len(chunks)} chunks from file {rel_path}")
                                
                                # Initialize counters outside the try/except block
                                successful = 0  # Global counter for successful chunks
                                failed = 0      # Global counter for failed chunks
                                
                                # Process each chunk from this file
                                file_successful = 0
                                file_failed = 0
                                
                                # First collect all chunks to allow for context passing
                                chunk_list = list(chunks)
                                
                                for i, (chunk_text, chunk_id) in enumerate(chunk_list):
                                    try:
                                        # Skip empty chunks
                                        if not chunk_text.strip():
                                            print(f"Job {job_id}: Skipping empty chunk: {chunk_id}")
                                            file_failed += 1
                                            failed += 1
                                            failed_files.append(f"{chunk_id} (empty)")
                                            continue
                                            
                                        # Store chunk for tracking
                                        file_chunks.append(chunk_text)
                                        file_chunk_ids.append(chunk_id)
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
                                                enrichment = ollama_client.generate_semantic_enrichment(
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
                                                    print(f"Job {job_id}: Enhanced chunk {chunk_id} with contextual summary")
                                            except Exception as e:
                                                print(f"Job {job_id}: Error generating enrichment for {chunk_id}: {e}")
                                                metadata["has_enrichment"] = False
                                                metadata["enrichment_error"] = str(e)
                                        else:
                                            metadata["has_enrichment"] = False
                                        
                                        # Generate embedding
                                        embedding = ollama_client.generate_embedding(processing_text)
                                        
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
                                        db_collection.add(
                                            ids=[chunk_id],
                                            embeddings=[embedding],
                                            metadatas=[metadata],
                                            documents=[processing_text]
                                        )
                                        
                                        file_successful += 1
                                        successful += 1
                                        
                                        # Update job status with new successful count
                                        with job_lock:
                                            processing_jobs[job_id]["successful_chunks"] = successful
                                            
                                    except Exception as e:
                                        print(f"Job {job_id}: Error processing chunk {chunk_id}: {e}")
                                        file_failed += 1
                                        failed += 1
                                        failed_files.append(f"{chunk_id} ({str(e)})")
                                        
                                        # Update failed count
                                        with job_lock:
                                            processing_jobs[job_id]["failed_chunks"] = failed
                                
                                print(f"Job {job_id}: Completed file {rel_path} - {file_successful} chunks succeeded, {file_failed} failed")
                                
                                # Update progress
                                with job_lock:
                                    processing_jobs[job_id]["processed_files"] += 1
                                    processing_jobs[job_id]["progress"] = int(len(source_files) > 0 and (processing_jobs[job_id]["processed_files"] / len(source_files)) * 100)
                            finally:
                                # Nothing to restore as we're not modifying globals anymore
                                pass
                    except Exception as e:
                        print(f"Job {job_id}: Error reading file {file_path}: {e}")
                        failed_files.append(f"{file_path} ({str(e)})")
                        
                        # Update progress for failed file
                        with job_lock:
                            processing_jobs[job_id]["processed_files"] += 1
                            processing_jobs[job_id]["progress"] = int(len(source_files) > 0 and (processing_jobs[job_id]["processed_files"] / len(source_files)) * 100)
        
        # Process all files recursively
        process_directory(DOCS_FOLDER)
        
        # Update total files count
        with job_lock:
            processing_jobs[job_id]["total_files"] = len(source_files)
        
        if not all_chunks:
            with job_lock:
                processing_jobs[job_id]["status"] = JOB_STATUS_FAILED
                processing_jobs[job_id]["error"] = "No documents to process"
            return
        
        print(f"Job {job_id}: Processed {len(all_chunks)} chunks from {len(source_files)} source files")
        
        # After processing documents, refresh the domain terms using LLM
        term_update_status = None
        try:
            # Get document count before refresh
            prev_term_count = len(query_classifier.product_terms)
            
            # Use LLM-powered extraction via Ollama
            print(f"Job {job_id}: Refreshing domain terms using LLM extraction")
            query_classifier.update_terms_from_db(db_collection, ollama_client)
            
            # Get updated term count
            new_term_count = len(query_classifier.product_terms)
            term_update_status = {
                "previous_term_count": prev_term_count,
                "new_term_count": new_term_count,
                "extraction_method": "LLM-powered",
                "terms_updated": True,
                "sample_terms": query_classifier.product_terms[:10] if query_classifier.product_terms else []
            }
        except Exception as e:
            print(f"Job {job_id}: Error refreshing domain terms: {e}")
            term_update_status = {
                "terms_updated": False,
                "error": str(e)
            }
        
        # Prepare enrichment status information
        enrichment_status = {
            "enabled": enhance_chunks,
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
            "semantic_enrichment": enrichment_status
        }
        
        # Update job status to completed
        with job_lock:
            processing_jobs[job_id]["status"] = JOB_STATUS_COMPLETED
            processing_jobs[job_id]["progress"] = 100
            processing_jobs[job_id]["result"] = result
            
        print(f"Job {job_id}: Processing completed - {successful} chunks processed, {failed} failed")
            
    except Exception as e:
        error_message = f"Error in document processing: {str(e)}"
        print(f"Job {job_id}: {error_message}")
        
        # Update job status to failed
        with job_lock:
            processing_jobs[job_id]["status"] = JOB_STATUS_FAILED
            processing_jobs[job_id]["error"] = error_message
            processing_jobs[job_id]["result"] = {
                "message": "Document processing failed",
                "error": error_message,
                "source_files": len(source_files),
                "total_chunks": len(all_chunks),
                "successful_chunks": successful,
                "failed_chunks": failed
            }

@app.get("/query", summary="Retrieve relevant documents", description="Query ChromaDB for the most relevant document based on input text.")
async def query_documents(
    query: str,
    n_results: int = Query(3, description="Number of results to return"),
    combine_chunks: bool = Query(True, description="Whether to combine chunks from the same document"),
    web_search: bool = Query(None, description="Whether to augment with web search results (auto if None)"),
    web_results_count: int = Query(5, description="Number of web search results to include"),
    explain_classification: bool = Query(False, description="Whether to include query classification explanation")
):
    try:
        # Check if ChromaDB has any documents at all
        doc_count = db_collection.count()
        if doc_count == 0:
            return {
                "query": query,
                "response": "No documents have been processed yet. Please use the /process endpoint first.",
                "sources": {"documents": [], "ids": [], "metadatas": []},
                "status": "error",
                "error": "Empty collection"
            }
            
        # Generate embedding for the query using Ollama
        query_embedding = ollama_client.generate_embedding(query)
        
        # Log the embedding dimension for debugging
        print(f"Generated query embedding with dimension: {len(query_embedding)}")
        
        try:
            # Get relevant documents/chunks from ChromaDB
            # Using more results since we might combine chunks
            retrieve_count = n_results * 3 if combine_chunks else n_results
            
            results = db_collection.query(
                query_embeddings=[query_embedding], 
                n_results=retrieve_count,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            # Handle potential embedding dimension mismatch
            error_msg = str(e)
            if "dimension" in error_msg.lower():
                print(f"Embedding dimension error: {error_msg}")
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
                
            # Replace the results with the combined chunks
            docs = combined_docs
            ids = combined_ids
            metadatas = combined_metadatas
            distances = combined_distances
        
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
            doc_distance_scores = distances if 'distances' in locals() else []
            # Convert distances to similarity scores (lower distance = higher similarity)
            doc_scores = [1.0 - min(d, 1.0) for d in doc_distance_scores] if doc_distance_scores else []
            
            # Classify the query
            source_type, confidence, classification_metadata = query_classifier.classify(
                query=query, 
                doc_scores=doc_scores
            )
            print(f"Query classified as '{source_type}' with {confidence:.2f} confidence")
        
        # Decide whether to use web search based on classification or explicit setting
        should_use_web = web_search if web_search is not None else (
            source_type == "web" or source_type == "hybrid"
        )
        
        # Add web search results if enabled/auto-determined
        web_results = []
        if should_use_web and SERPER_API_KEY:
            try:
                print(f"Performing web search for query: {query}")
                web_results = web_search_client.search_with_serper(query, num_results=web_results_count)
                
                if web_results:
                    # Format web results and add to context
                    web_context = web_search_client.format_results_as_context(web_results)
                    context = web_context + "\n\n" + context
                    print(f"Added {len(web_results)} web search results to context")
            except Exception as e:
                print(f"Error during web search: {e}")
                # Continue with only vector DB results
        
        response = ollama_client.generate_response(context=context, query=query)
        
        # Clean up the response for better frontend rendering
        cleaned_results = {
            "documents": docs,
            "ids": ids,
            "metadatas": metadatas,
            "distances": distances,
            "combined_chunks": combine_chunks,
            "web_results": web_results if web_results else []
        }
        
        # Prepare response
        response_data = {
            "query": query, 
            "response": response, 
            "sources": cleaned_results,
            "status": "success",
            "web_search_used": len(web_results) > 0,  # Only true if actual web results were found and used
            "source_type": source_type
        }
        
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
        print(f"Error in query_documents: {e}")
        
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

@app.post("/refresh-terms", summary="Refresh domain terms", description="Refreshes the domain-specific terms used for query classification based on current document content.")
async def refresh_domain_terms(background_tasks: BackgroundTasks):
    """Refresh the domain-specific terms used in query classification using LLM extraction"""
    try:
        # Get document count before refresh
        prev_term_count = len(query_classifier.product_terms)
        
        # Start a background task for term extraction
        job_id = str(uuid.uuid4())
        
        # Set up a job to track term extraction progress
        with job_lock:
            processing_jobs[job_id] = {
                "status": JOB_STATUS_QUEUED,
                "progress": 0,
                "type": "term_extraction",
                "error": None,
                "result": None
            }
        
        # Add the background task
        background_tasks.add_task(
            refresh_domain_terms_background_task,
            job_id=job_id
        )
        
        return {
            "status": "success",
            "message": "Domain term extraction started in the background",
            "job_id": job_id,
            "previous_term_count": prev_term_count,
            "current_terms_sample": query_classifier.product_terms[:10]  # Show first 10 terms as a sample
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error refreshing domain terms: {str(e)}"
        }

def refresh_domain_terms_background_task(job_id: str):
    """Background task for refreshing domain terms using LLM"""
    try:
        # Update job status to processing
        with job_lock:
            processing_jobs[job_id]["status"] = JOB_STATUS_PROCESSING
            processing_jobs[job_id]["progress"] = 10
        
        # Get document count before refresh
        prev_term_count = len(query_classifier.product_terms)
        
        # Refresh terms from ChromaDB using Ollama
        print(f"Job {job_id}: Using Ollama for domain term extraction")
        
        # Update progress
        with job_lock:
            processing_jobs[job_id]["progress"] = 30
        
        # Do the actual term extraction
        query_classifier.update_terms_from_db(db_collection, ollama_client)
        
        # Update progress
        with job_lock:
            processing_jobs[job_id]["progress"] = 80
        
        # Get updated term count
        new_term_count = len(query_classifier.product_terms)
        
        # Prepare result
        result = {
            "message": "Domain terms refreshed successfully using LLM-powered extraction",
            "extraction_method": "LLM-powered",
            "previous_term_count": prev_term_count,
            "new_term_count": new_term_count,
            "sample_terms": query_classifier.product_terms[:20]  # Show first 20 terms as a sample
        }
        
        # Update job status to completed
        with job_lock:
            processing_jobs[job_id]["status"] = JOB_STATUS_COMPLETED
            processing_jobs[job_id]["progress"] = 100
            processing_jobs[job_id]["result"] = result
        
        print(f"Job {job_id}: Domain term extraction completed - {new_term_count} terms extracted")
        
    except Exception as e:
        error_message = f"Error in domain term extraction: {str(e)}"
        print(f"Job {job_id}: {error_message}")
        
        # Update job status to failed
        with job_lock:
            processing_jobs[job_id]["status"] = JOB_STATUS_FAILED
            processing_jobs[job_id]["error"] = error_message

@app.get("/job/{job_id}", summary="Get job status", description="Check the status of a document processing job.")
async def get_job_status(job_id: str):
    """Get the status of a document processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
    return processing_jobs[job_id]

@app.get("/jobs", summary="List all jobs", description="List all document processing jobs.")
async def list_jobs():
    """List all document processing jobs"""
    return {
        "total_jobs": len(processing_jobs),
        "jobs": processing_jobs
    }

@app.get("/terms", summary="List domain terms", description="List the domain-specific terms used for query classification.")
async def list_domain_terms():
    """List the domain-specific terms used in query classification"""
    try:
        return {
            "status": "success",
            "term_count": len(query_classifier.product_terms),
            "terms": query_classifier.product_terms
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error listing domain terms: {str(e)}"
        }

@app.post("/upload-file", summary="Upload a file", description="Upload a file (txt or PDF) to be processed.")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    process_immediately: bool = Form(False)
):
    """Upload a file (txt or PDF) to be processed"""
    try:
        # Check if file is empty
        contents = await file.read()
        if not contents:
            return {
                "status": "error",
                "message": "File is empty"
            }
        
        # Reset file position after reading
        await file.seek(0)
        
        # Check file type
        if not (file.filename.endswith('.txt') or file.filename.endswith('.pdf')):
            return {
                "status": "error",
                "message": "Only .txt and .pdf files are supported"
            }
            
        # Create the docs directory if it doesn't exist
        os.makedirs(DOCS_FOLDER, exist_ok=True)
        
        # Generate a safe filename to prevent path traversal
        safe_filename = re.sub(r'[^\w\-\.]', '_', file.filename)
        file_path = os.path.join(DOCS_FOLDER, safe_filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            # Reset the file pointer to the beginning
            await file.seek(0)
            # Copy the contents to the destination file
            shutil.copyfileobj(file.file, buffer)
        
        # Process PDF files by extracting text
        if file.filename.endswith('.pdf'):
            try:
                # Use our PDF extractor for better results
                try:
                    pdf_text = pdf_extractor.extract_text(contents, filename=safe_filename)
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Error extracting text from PDF: {str(e)}"
                    }
                
                # Save the extracted text to a markdown file
                md_filename = safe_filename.replace('.pdf', '.md')
                md_path = os.path.join(DOCS_FOLDER, md_filename)
                
                with open(md_path, "w", encoding="utf-8") as md_file:
                    md_file.write(pdf_text)
                
                # Use the markdown file for processing instead
                file_path = md_path
                
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error processing PDF file: {str(e)}"
                }
                
        # If the file is a text file, convert it to markdown
        elif file.filename.endswith('.txt'):
            try:
                # Read the text file
                with open(file_path, "r", encoding="utf-8") as txt_file:
                    txt_content = txt_file.read()
                
                # Save as markdown
                md_filename = safe_filename.replace('.txt', '.md')
                md_path = os.path.join(DOCS_FOLDER, md_filename)
                
                with open(md_path, "w", encoding="utf-8") as md_file:
                    md_file.write(txt_content)
                
                # Use the markdown file for processing
                file_path = md_path
                
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error converting text file to markdown: {str(e)}"
                }
        
        result = {
            "status": "success",
            "message": f"File uploaded successfully: {file.filename}",
            "file_path": file_path
        }
        
        # Process the file immediately if requested
        if process_immediately and background_tasks:
            # Create a unique job ID
            job_id = str(uuid.uuid4())
            
            # Set up initial job status
            with job_lock:
                processing_jobs[job_id] = {
                    "status": JOB_STATUS_QUEUED,
                    "progress": 0,
                    "total_files": 1,
                    "processed_files": 0,
                    "successful_chunks": 0,
                    "failed_chunks": 0,
                    "error": None,
                    "settings": {
                        "chunk_size": MAX_CHUNK_SIZE,
                        "min_size": MIN_CHUNK_SIZE,
                        "overlap": CHUNK_OVERLAP,
                        "enable_chunking": ENABLE_CHUNKING,
                        "enhance_chunks": True,
                        "file_path": file_path
                    },
                    "result": None
                }
            
            # Add the background task to process just this file
            background_tasks.add_task(
                process_single_file_task,
                job_id=job_id,
                file_path=file_path
            )
            
            # Add job information to the result
            result["job_id"] = job_id
            result["processing_status"] = "queued"
            result["message"] += " and queued for processing"
        
        return result
        
    except Exception as e:
        print(f"Error uploading file: {e}")
        return {
            "status": "error",
            "message": f"Error uploading file: {str(e)}"
        }

def process_single_file_task(job_id: str, file_path: str):
    """Background task to process a single uploaded file"""
    try:
        # Update job status to processing
        with job_lock:
            processing_jobs[job_id]["status"] = JOB_STATUS_PROCESSING
            processing_jobs[job_id]["progress"] = 10
        
        # Check if file exists
        if not os.path.exists(file_path):
            with job_lock:
                processing_jobs[job_id]["status"] = JOB_STATUS_FAILED
                processing_jobs[job_id]["error"] = f"File not found: {file_path}"
            return
            
        # Track results
        successful = 0
        failed = 0
        all_chunks = []
        all_chunk_ids = []
        
        print(f"Job {job_id}: Processing single file: {file_path}")
        
        try:
            # Read the file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
                # Use relative path as identifier
                rel_path = os.path.relpath(file_path, DOCS_FOLDER)
                
                # Apply chunking
                chunks = doc_processor.chunk_document(content, rel_path)
                
                # Add chunks to our collection
                for chunk_text, chunk_id in chunks:
                    all_chunks.append(chunk_text)
                    all_chunk_ids.append(chunk_id)
                
                # Update progress
                with job_lock:
                    processing_jobs[job_id]["processed_files"] = 1
                    processing_jobs[job_id]["progress"] = 30
                    
        except Exception as e:
            print(f"Job {job_id}: Error reading file {file_path}: {e}")
            with job_lock:
                processing_jobs[job_id]["status"] = JOB_STATUS_FAILED
                processing_jobs[job_id]["error"] = f"Error reading file: {str(e)}"
            return
                
        if not all_chunks:
            with job_lock:
                processing_jobs[job_id]["status"] = JOB_STATUS_FAILED
                processing_jobs[job_id]["error"] = "No content could be chunked from the file"
            return
        
        print(f"Job {job_id}: Created {len(all_chunks)} chunks from file")
        with job_lock:
            processing_jobs[job_id]["progress"] = 50
        
        # Initialize counters
        successful = 0
        failed = 0
                
        # Process chunks with context
        chunk_pairs = list(zip(all_chunks, all_chunk_ids))
        
        for i, (chunk_text, chunk_id) in enumerate(chunk_pairs):
            try:
                # Skip empty chunks
                if not chunk_text.strip():
                    print(f"Job {job_id}: Skipping empty chunk: {chunk_id}")
                    failed += 1
                    continue
                
                # Determine whether to use original or enriched text for embedding
                processing_text = chunk_text
                metadata = {"original_text": chunk_text}
                
                # Generate semantic enrichment with context
                try:
                    # Get previous and next chunks for context if they exist
                    prev_chunk_text = None
                    next_chunk_text = None
                    
                    if i > 0:
                        prev_chunk_text = chunk_pairs[i-1][0]
                        
                    if i < len(chunk_pairs) - 1:
                        next_chunk_text = chunk_pairs[i+1][0]
                    
                    # Generate enrichment with context
                    enrichment = ollama_client.generate_semantic_enrichment(
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
                        
                        print(f"Job {job_id}: Enhanced chunk {chunk_id} with contextual summary")
                except Exception as e:
                    print(f"Job {job_id}: Error generating enrichment: {e}")
                    metadata["has_enrichment"] = False
                    metadata["enrichment_error"] = str(e)
                
                # Generate embedding
                print(f"Job {job_id}: Processing chunk {i+1}/{len(all_chunks)}")
                embedding = ollama_client.generate_embedding(processing_text)
                
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
                db_collection.add(
                    ids=[all_chunk_ids[i]],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[processing_text]
                )
                
                successful += 1
                with job_lock:
                    processing_jobs[job_id]["successful_chunks"] = successful
                    progress = 50 + int((i + 1) / len(all_chunks) * 40)  # Progress from 50% to 90%
                    processing_jobs[job_id]["progress"] = progress
                
            except Exception as e:
                print(f"Job {job_id}: Error processing chunk {i+1}: {e}")
                failed += 1
                with job_lock:
                    processing_jobs[job_id]["failed_chunks"] = failed
        
        # Update domain terms using LLM
        term_update_status = None
        try:
            # Get document count before refresh
            prev_term_count = len(query_classifier.product_terms)
            
            # Use LLM-powered extraction via Ollama
            print(f"Job {job_id}: Refreshing domain terms using LLM extraction")
            query_classifier.update_terms_from_db(db_collection, ollama_client)
            
            # Get updated term count
            new_term_count = len(query_classifier.product_terms)
            term_update_status = {
                "previous_term_count": prev_term_count,
                "new_term_count": new_term_count,
                "extraction_method": "LLM-powered",
                "terms_updated": True,
                "sample_terms": query_classifier.product_terms[:10] if query_classifier.product_terms else []
            }
        except Exception as e:
            print(f"Job {job_id}: Error refreshing domain terms: {e}")
            term_update_status = {
                "terms_updated": False,
                "error": str(e)
            }
        
        # Finalize job
        with job_lock:
            processing_jobs[job_id]["status"] = JOB_STATUS_COMPLETED
            processing_jobs[job_id]["progress"] = 100
            processing_jobs[job_id]["result"] = {
                "message": "File processed successfully" if failed == 0 else "File processed with some errors",
                "file_path": file_path,
                "total_chunks": len(all_chunks),
                "successful_chunks": successful,
                "failed_chunks": failed,
                "term_extraction": term_update_status
            }
            
        print(f"Job {job_id}: Processing completed - {successful} chunks processed, {failed} failed")
        
    except Exception as e:
        error_message = f"Error processing file: {str(e)}"
        print(f"Job {job_id}: {error_message}")
        
        with job_lock:
            processing_jobs[job_id]["status"] = JOB_STATUS_FAILED
            processing_jobs[job_id]["error"] = error_message
            processing_jobs[job_id]["progress"] = 100
            processing_jobs[job_id]["result"] = {
                "message": "File processing failed",
                "error": error_message,
                "file_path": file_path
            }

@app.post("/clear-db", summary="Clear the database", description="Clear all documents from ChromaDB.")
async def clear_database():
    """Clear all documents from the database"""
    global db_collection
    try:
        # Get current document count for reporting
        doc_count = db_collection.count()
        
        if doc_count > 0:
            # Get all document IDs
            results = db_collection.get(include=[])
            
            if results and "ids" in results and results["ids"]:
                # Delete all documents by ID
                db_collection.delete(ids=results["ids"])
                print(f"Deleted {len(results['ids'])} documents")
            else:
                # Fallback method: recreate the collection
                try:
                    # First try to delete the entire collection
                    chroma_client.delete_collection("documents")
                    print("Collection deleted")
                    
                    # Then recreate it
                    db_collection = chroma_client.create_collection(
                        name="documents",
                        metadata={"description": "Main document collection for RAG processing"}
                    )
                    print("Collection recreated")
                except Exception as inner_e:
                    print(f"Error during collection recreation: {inner_e}")
                    return {
                        "status": "error",
                        "message": f"Error clearing database: {str(inner_e)}"
                    }
        
        # Refresh the domain terms to reflect the empty database
        try:
            query_classifier.update_terms_from_db(db_collection)
        except Exception as e:
            print(f"Error refreshing domain terms after clearing DB: {e}")
        
        return {
            "status": "success",
            "message": f"Database cleared successfully. Removed {doc_count} documents.",
            "documents_removed": doc_count
        }
    except Exception as e:
        print(f"Error clearing database: {e}")
        return {
            "status": "error",
            "message": f"Error clearing database: {str(e)}"
        }

@app.get("/chunks", summary="List document chunks", description="Retrieve chunks stored in ChromaDB with optional filtering.")
async def list_document_chunks(
    limit: int = Query(20, description="Limit the number of chunks returned"),
    offset: int = Query(0, description="Starting offset for pagination"),
    filename: str = Query(None, description="Filter by filename (partial match)"),
    content: str = Query(None, description="Filter by content (partial match)")
):
    """List document chunks stored in the database with filtering options"""
    try:
        # Get collection count to verify it's accessible
        doc_count = db_collection.count()
        if doc_count == 0:
            return {
                "status": "empty",
                "message": "No documents in the database",
                "chunks": [],
                "total": 0
            }
            
        # Create the query to get chunks - we'll filter programmatically instead of using where clause
        # because ChromaDB doesn't support partial matching with $like operator
        query_results = db_collection.get(
            limit=limit if not filename else doc_count,  # Get all if filtering by filename
            offset=offset if not filename else 0,       # Start from beginning if filtering
            include=["documents", "metadatas", "embeddings"]
        )
        
        # Extract the results
        chunks = []
        for i, doc in enumerate(query_results["documents"]):
            metadata = query_results["metadatas"][i] if "metadatas" in query_results else {}
            file_name = metadata.get("filename", "unknown")
            
            # Skip if filename filter is provided and doesn't match (case-insensitive partial match)
            if filename and filename.lower() not in file_name.lower():
                continue
                
            # Skip if content filter is provided and doesn't match
            if content and content.lower() not in doc.lower():
                continue
                
            # Extract original text if it exists
            original_text = metadata.get("original_text", doc)
            
            # Get enrichment if it exists
            enrichment = metadata.get("enrichment", "")
            has_enrichment = metadata.get("has_enrichment", False)
            
            chunks.append({
                "id": query_results["ids"][i],
                "text": original_text,
                "filename": file_name,
                "has_enrichment": has_enrichment,
                "enrichment": enrichment if has_enrichment else "",
                "embedding_dimension": len(query_results["embeddings"][i]) if "embeddings" in query_results else 0
            })
        
        # Apply pagination after filtering if needed
        # Only needed when we're doing client-side filtering by filename
        paginated_chunks = chunks
        if filename and len(chunks) > limit:
            start_idx = min(offset, len(chunks))
            end_idx = min(start_idx + limit, len(chunks))
            paginated_chunks = chunks[start_idx:end_idx]
        
        # Return the results
        return {
            "status": "success",
            "total_in_db": doc_count,
            "total_matching": len(chunks),
            "chunks_returned": len(paginated_chunks),
            "chunks": paginated_chunks
        }
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return {
            "status": "error",
            "message": f"Error retrieving chunks: {str(e)}",
            "chunks": []
        }

@app.get("/openapi.json", include_in_schema=False)
def custom_openapi():
    return get_openapi(
        title="Document Processing API",
        version="1.0",
        description="API for processing and retrieving documents using embeddings.",
        routes=app.routes,
    )

# ===============================================================
# Main Entry Point
# ===============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)