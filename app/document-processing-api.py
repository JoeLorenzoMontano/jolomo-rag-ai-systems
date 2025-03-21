from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Tuple, Optional
import chromadb
import os
import re
import requests
import time as import_time
from utils.ollama_client import OllamaClient
from utils.web_search import WebSearchClient

# Initialize FastAPI app
app = FastAPI(title="Document Processing API", description="API for storing and retrieving documents with embeddings.", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize ChromaDB client
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# Connect to ChromaDB with simple connection and retry logic
max_retries = 5
retry_delay = 3  # seconds

print(f"Connecting to ChromaDB at http://{CHROMA_HOST}:{CHROMA_PORT}")

# Create connection settings 
from chromadb.config import Settings

settings = Settings(
    chroma_api_impl="rest",
    chroma_server_host=CHROMA_HOST,
    chroma_server_http_port=CHROMA_PORT,
)

# Try connection with retry logic
for attempt in range(max_retries):
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
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            import time
            time.sleep(retry_delay)
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

# Initialize Web Search Client with Serper API key from environment
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
web_search_client = WebSearchClient(serper_api_key=SERPER_API_KEY)

# Health check endpoints
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
        except Exception as e:
            health_status["models"]["embedding_model"] += f" - error: {str(e)}"
    
    return health_status

# Chunking configuration from environment variables with defaults
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))  # Maximum number of characters per chunk
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "200"))   # Minimum size of a chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))     # Overlap between chunks to maintain context
ENABLE_CHUNKING = os.getenv("ENABLE_CHUNKING", "true").lower() == "true"  # Enable/disable chunking

# Print chunking settings
print(f"Document chunking settings:")
print(f"  ENABLE_CHUNKING: {ENABLE_CHUNKING}")
print(f"  MAX_CHUNK_SIZE: {MAX_CHUNK_SIZE} chars")
print(f"  MIN_CHUNK_SIZE: {MIN_CHUNK_SIZE} chars")
print(f"  CHUNK_OVERLAP: {CHUNK_OVERLAP} chars")

def chunk_document(document: str, file_path: str) -> List[Tuple[str, str]]:
    """
    Splits a document into chunks for better embedding and retrieval.
    Returns a list of (chunk_text, chunk_id) tuples.
    
    Args:
        document: The document text to chunk
        file_path: The path of the original document (used for chunk IDs)
    
    Returns:
        List of tuples with (chunk_text, chunk_id)
    """
    # Skip empty documents
    if not document.strip():
        return []
    
    # If chunking is disabled or document is small, return as single chunk
    if not ENABLE_CHUNKING or len(document) < MAX_CHUNK_SIZE:
        return [(document, file_path)]
    
    chunks = []
    
    # Split document into paragraphs based on double newlines
    # This preserves natural document structure
    paragraphs = re.split(r'\n\s*\n', document)
    
    current_chunk = []
    current_length = 0
    chunk_index = 0
    
    # Process each paragraph
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        para_length = len(para)
        
        # If this paragraph alone exceeds max chunk size, split it further
        if para_length > MAX_CHUNK_SIZE:
            # If we have a current chunk, finalize it first
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunk_id = f"{file_path}#chunk-{chunk_index}"
                chunks.append((chunk_text, chunk_id))
                chunk_index += 1
                current_chunk = []
                current_length = 0
            
            # Split long paragraphs by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sentence_chunk = []
            sentence_length = 0
            
            for sentence in sentences:
                if sentence_length + len(sentence) > MAX_CHUNK_SIZE and sentence_length > MIN_CHUNK_SIZE:
                    # Finalize this sentence chunk
                    chunk_text = " ".join(sentence_chunk)
                    chunk_id = f"{file_path}#chunk-{chunk_index}"
                    chunks.append((chunk_text, chunk_id))
                    chunk_index += 1
                    
                    # Start a new chunk with overlap
                    retain_sentences = []
                    retain_length = 0
                    
                    # Keep some sentences for context overlap
                    for prev_sent in reversed(sentence_chunk):
                        if retain_length + len(prev_sent) <= CHUNK_OVERLAP:
                            retain_sentences.insert(0, prev_sent)
                            retain_length += len(prev_sent) + 1  # +1 for space
                        else:
                            break
                    
                    sentence_chunk = retain_sentences
                    sentence_length = retain_length
                
                sentence_chunk.append(sentence)
                sentence_length += len(sentence) + 1  # +1 for space
            
            # Add the remaining sentences as a chunk
            if sentence_chunk:
                chunk_text = " ".join(sentence_chunk)
                chunk_id = f"{file_path}#chunk-{chunk_index}"
                chunks.append((chunk_text, chunk_id))
                chunk_index += 1
            
        # If adding this paragraph would exceed the limit, finalize the current chunk
        elif current_length + para_length > MAX_CHUNK_SIZE and current_length > MIN_CHUNK_SIZE:
            chunk_text = "\n\n".join(current_chunk)
            chunk_id = f"{file_path}#chunk-{chunk_index}"
            chunks.append((chunk_text, chunk_id))
            chunk_index += 1
            
            # For overlap, keep some content from the previous chunk
            overlap_paras = []
            overlap_length = 0
            
            # Find paragraphs to retain for overlap
            for prev_para in reversed(current_chunk):
                if overlap_length + len(prev_para) <= CHUNK_OVERLAP:
                    overlap_paras.insert(0, prev_para)
                    overlap_length += len(prev_para) + 2  # +2 for newlines
                else:
                    break
            
            current_chunk = overlap_paras
            current_length = overlap_length
            
            # Add the current paragraph to the new chunk
            current_chunk.append(para)
            current_length += para_length
            
        # Otherwise add the paragraph to the current chunk
        else:
            current_chunk.append(para)
            current_length += para_length + 2  # +2 for the paragraph separator
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunk_id = f"{file_path}#chunk-{chunk_index}"
        chunks.append((chunk_text, chunk_id))
    
    print(f"Split document '{file_path}' into {len(chunks)} chunks")
    return chunks

# Folder to store raw documents
DOCS_FOLDER = "./data"
os.makedirs(DOCS_FOLDER, exist_ok=True)

@app.post("/process", summary="Process and store document embeddings", description="Processes documents and stores their embeddings in ChromaDB.")
async def process_documents(
    chunk_size: int = Query(None, description="Override max chunk size (chars)"),
    min_size: int = Query(None, description="Override min chunk size (chars)"),
    overlap: int = Query(None, description="Override chunk overlap (chars)"),
    enable_chunking: bool = Query(None, description="Override chunking enabled setting")
):
    # Apply overrides if provided without modifying globals
    temp_max_chunk_size = chunk_size if chunk_size is not None else MAX_CHUNK_SIZE
    temp_min_chunk_size = min_size if min_size is not None else MIN_CHUNK_SIZE
    temp_chunk_overlap = overlap if overlap is not None else CHUNK_OVERLAP
    temp_enable_chunking = enable_chunking if enable_chunking is not None else ENABLE_CHUNKING
    
    # Log chunking settings for this run
    if (chunk_size is not None or min_size is not None or 
        overlap is not None or enable_chunking is not None):
        print(f"Using custom chunking settings for this run:")
        print(f"  ENABLE_CHUNKING: {temp_enable_chunking}")
        print(f"  MAX_CHUNK_SIZE: {temp_max_chunk_size} chars")
        print(f"  MIN_CHUNK_SIZE: {temp_min_chunk_size} chars")
        print(f"  CHUNK_OVERLAP: {temp_chunk_overlap} chars")
    
    # Document containers
    all_chunks = []
    all_chunk_ids = []
    source_files = []
    
    # Function to process files recursively
    def process_directory(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isdir(file_path):
                # Recursively process subdirectories
                process_directory(file_path)
            elif os.path.isfile(file_path) and file_path.endswith('.md'):
                source_files.append(file_path)  # Track all source files for reporting
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Use relative path as identifier
                        rel_path = os.path.relpath(file_path, DOCS_FOLDER)
                        
                        # Apply chunking with temp settings
                        try:
                            # Pass the temporary settings directly to the function arguments
                            # instead of modifying the globals
                            def chunk_with_settings(doc_text, doc_path, max_size, min_size, overlap, enable):
                                """Chunk document with explicit settings instead of globals"""
                                # Skip empty documents
                                if not doc_text.strip():
                                    return []
                                
                                # If chunking is disabled or document is small, return as single chunk
                                if not enable or len(doc_text) < max_size:
                                    return [(doc_text, doc_path)]
                                
                                chunks = []
                                
                                # Split document into paragraphs based on double newlines
                                paragraphs = re.split(r'\n\s*\n', doc_text)
                                
                                current_chunk = []
                                current_length = 0
                                chunk_index = 0
                                
                                # Process each paragraph
                                for para in paragraphs:
                                    para = para.strip()
                                    if not para:
                                        continue
                                        
                                    para_length = len(para)
                                    
                                    # If this paragraph alone exceeds max chunk size, split it further
                                    if para_length > max_size:
                                        # If we have a current chunk, finalize it first
                                        if current_chunk:
                                            chunk_text = "\n\n".join(current_chunk)
                                            chunk_id = f"{doc_path}#chunk-{chunk_index}"
                                            chunks.append((chunk_text, chunk_id))
                                            chunk_index += 1
                                            current_chunk = []
                                            current_length = 0
                                        
                                        # Split long paragraphs by sentences
                                        sentences = re.split(r'(?<=[.!?])\s+', para)
                                        sentence_chunk = []
                                        sentence_length = 0
                                        
                                        for sentence in sentences:
                                            if sentence_length + len(sentence) > max_size and sentence_length > min_size:
                                                # Finalize this sentence chunk
                                                chunk_text = " ".join(sentence_chunk)
                                                chunk_id = f"{doc_path}#chunk-{chunk_index}"
                                                chunks.append((chunk_text, chunk_id))
                                                chunk_index += 1
                                                
                                                # Start a new chunk with overlap
                                                retain_sentences = []
                                                retain_length = 0
                                                
                                                # Keep some sentences for context overlap
                                                for prev_sent in reversed(sentence_chunk):
                                                    if retain_length + len(prev_sent) <= overlap:
                                                        retain_sentences.insert(0, prev_sent)
                                                        retain_length += len(prev_sent) + 1  # +1 for space
                                                    else:
                                                        break
                                                
                                                sentence_chunk = retain_sentences
                                                sentence_length = retain_length
                                            
                                            sentence_chunk.append(sentence)
                                            sentence_length += len(sentence) + 1  # +1 for space
                                        
                                        # Add the remaining sentences as a chunk
                                        if sentence_chunk:
                                            chunk_text = " ".join(sentence_chunk)
                                            chunk_id = f"{doc_path}#chunk-{chunk_index}"
                                            chunks.append((chunk_text, chunk_id))
                                            chunk_index += 1
                                        
                                    # If adding this paragraph would exceed the limit, finalize the current chunk
                                    elif current_length + para_length > max_size and current_length > min_size:
                                        chunk_text = "\n\n".join(current_chunk)
                                        chunk_id = f"{doc_path}#chunk-{chunk_index}"
                                        chunks.append((chunk_text, chunk_id))
                                        chunk_index += 1
                                        
                                        # For overlap, keep some content from the previous chunk
                                        overlap_paras = []
                                        overlap_length = 0
                                        
                                        # Find paragraphs to retain for overlap
                                        for prev_para in reversed(current_chunk):
                                            if overlap_length + len(prev_para) <= overlap:
                                                overlap_paras.insert(0, prev_para)
                                                overlap_length += len(prev_para) + 2  # +2 for newlines
                                            else:
                                                break
                                        
                                        current_chunk = overlap_paras
                                        current_length = overlap_length
                                        
                                        # Add the current paragraph to the new chunk
                                        current_chunk.append(para)
                                        current_length += para_length
                                        
                                    # Otherwise add the paragraph to the current chunk
                                    else:
                                        current_chunk.append(para)
                                        current_length += para_length + 2  # +2 for the paragraph separator
                                
                                # Don't forget the last chunk
                                if current_chunk:
                                    chunk_text = "\n\n".join(current_chunk)
                                    chunk_id = f"{doc_path}#chunk-{chunk_index}"
                                    chunks.append((chunk_text, chunk_id))
                                
                                print(f"Split document '{doc_path}' into {len(chunks)} chunks")
                                return chunks
                            
                            # Use our local function with passed parameters
                            chunks = chunk_with_settings(
                                content, 
                                rel_path,
                                temp_max_chunk_size,
                                temp_min_chunk_size,
                                temp_chunk_overlap,
                                temp_enable_chunking
                            )
                            
                            # Add chunks to our collection
                            for chunk_text, chunk_id in chunks:
                                all_chunks.append(chunk_text)
                                all_chunk_ids.append(chunk_id)
                        finally:
                            # Nothing to restore as we're not modifying globals anymore
                            pass
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    
    # Process all files recursively
    process_directory(DOCS_FOLDER)
    
    if not all_chunks:
        raise HTTPException(status_code=400, detail="No documents to process.")
    
    print(f"Processing {len(all_chunks)} chunks from {len(source_files)} source files")
    
    # Process chunks in batches to avoid memory issues
    batch_size = 5
    successful = 0
    failed = 0
    failed_files = []
    
    for i in range(0, len(all_chunks), batch_size):
        batch_docs = all_chunks[i:i+batch_size]
        batch_ids = all_chunk_ids[i:i+batch_size]
        
        try:
            # Generate embeddings for the current batch - use a safer approach to handle errors
            batch_embeddings = []
            valid_docs = []
            valid_ids = []
            
            for j, doc in enumerate(batch_docs):
                try:
                    # Skip empty documents
                    if not doc.strip():
                        print(f"Skipping empty chunk: {batch_ids[j]}")
                        failed += 1
                        failed_files.append(f"{batch_ids[j]} (empty)")
                        continue
                        
                    # Attempt to generate embedding
                    print(f"Processing chunk {batch_ids[j]} ({len(doc)} chars)")
                    embedding = ollama_client.generate_embedding(doc)
                    
                    # Verify embedding is valid (not None and has values)
                    if embedding is None or len(embedding) == 0:
                        raise ValueError("Empty embedding returned")
                        
                    batch_embeddings.append(embedding)
                    valid_docs.append(doc)
                    valid_ids.append(batch_ids[j])
                except Exception as e:
                    print(f"Error generating embedding for {batch_ids[j]}: {e}")
                    failed += 1
                    failed_files.append(f"{batch_ids[j]} ({str(e)})")
            
            # Update our batch to only include documents with valid embeddings
            batch_docs = valid_docs
            batch_ids = valid_ids
            
            # Skip to next batch if all embeddings failed
            if not batch_embeddings:
                continue
                
            # Create metadata with source file information
            metadatas = []
            for chunk_id in batch_ids:
                # Split the ID to extract source file path
                if "#chunk-" in chunk_id:
                    source_file = chunk_id.split("#chunk-")[0]
                    chunk_num = chunk_id.split("#chunk-")[1]
                    metadatas.append({
                        "filename": source_file,
                        "chunk_id": chunk_id,
                        "chunk_num": chunk_num
                    })
                else:
                    # No chunks, just the file
                    metadatas.append({
                        "filename": chunk_id
                    })
            
            # Add to ChromaDB
            db_collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=metadatas,
                documents=batch_docs
            )
            
            successful += len(batch_embeddings)
            print(f"Added batch: {len(batch_embeddings)} chunks")
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            failed += len(batch_docs)
            failed_files.extend(batch_ids)
    
    # Return detailed status
    if failed > 0:
        return {
            "message": "Documents processed with some errors",
            "source_files": len(source_files),
            "total_chunks": len(all_chunks),
            "successful_chunks": successful,
            "failed_chunks": failed,
            "failed_items": failed_files,
            "chunking_enabled": temp_enable_chunking,
            "chunk_size": temp_max_chunk_size
        }
    else:
        return {
            "message": "All documents processed successfully",
            "source_files": len(source_files),
            "total_chunks": len(all_chunks),
            "successful_chunks": successful,
            "chunking_enabled": temp_enable_chunking,
            "chunk_size": temp_max_chunk_size
        }

@app.get("/query", summary="Retrieve relevant documents", description="Query ChromaDB for the most relevant document based on input text.")
async def query_documents(
    query: str,
    n_results: int = Query(3, description="Number of results to return"),
    combine_chunks: bool = Query(True, description="Whether to combine chunks from the same document"),
    web_search: bool = Query(False, description="Whether to augment with web search results"),
    web_results_count: int = Query(5, description="Number of web search results to include")
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
        if len(context.split()) < 100 and len(docs) > 1:
            context = docs[0] + "\n\n" + docs[1]
        
        # Add web search results if enabled
        web_results = []
        if web_search and SERPER_API_KEY:
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
            "web_results": web_results if web_search and web_results else []
        }
        
        return {
            "query": query, 
            "response": response, 
            "sources": cleaned_results,
            "status": "success",
            "web_search_used": web_search and len(web_results) > 0
        }
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

# Custom OpenAPI documentation
@app.post("/test-embedding", summary="Test embedding generation", description="Tests the embedding generation with provided text.")
async def test_embedding(text: str = "This is a test of the embedding functionality"):
    """
    Endpoint to test embedding generation with custom text.
    Useful for diagnosing embedding issues.
    """
    try:
        # Generate embedding
        start_time = import_time()
        embedding = ollama_client.generate_embedding(text)
        end_time = import_time()
        
        # Return detailed information about the embedding
        return {
            "status": "success",
            "model": ollama_client.embedding_model,
            "text_length": len(text),
            "embedding_length": len(embedding),
            "embedding_sample": embedding[:5],  # Just show the first few elements
            "processing_time_ms": round((end_time - start_time) * 1000, 2),
            "response_format": "embeddings array" if isinstance(embedding, list) else type(embedding).__name__
        }
    except Exception as e:
        return {
            "status": "error",
            "model": ollama_client.embedding_model,
            "error": str(e)
        }

@app.get("/openapi.json", include_in_schema=False)
def custom_openapi():
    return get_openapi(
        title="Document Processing API",
        version="1.0",
        description="API for processing and retrieving documents using embeddings.",
        routes=app.routes,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
