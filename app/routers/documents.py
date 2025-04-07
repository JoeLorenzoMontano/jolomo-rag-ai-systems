"""
Document processing router.

This module provides endpoints for document processing and management.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import os

from core.dependencies import (
    get_db_service,
    get_document_service,
    get_job_service
)
from core.utils import clean_filename, filter_chunks_by_filename
from models.schemas import FileUploadResponse, ChunkListResponse, ChunkInfo, DeleteDocumentResponse

router = APIRouter(tags=["documents"])

@router.post("/process", summary="Start document embedding processing in the background", 
            description="Starts processing documents in the background and returns a job ID for tracking progress.")
async def process_documents(
    background_tasks: BackgroundTasks,
    chunk_size: int = Query(None, description="Override max chunk size (chars)"),
    min_size: int = Query(None, description="Override min chunk size (chars)"),
    overlap: int = Query(None, description="Override chunk overlap (chars)"),
    enable_chunking: bool = Query(None, description="Override chunking enabled setting"),
    enhance_chunks: bool = Query(True, description="Generate additional content with Ollama to improve retrieval")
):
    """Start processing all documents in the background."""
    job_service = get_job_service()
    document_service = get_document_service()
    
    # Create a job to track progress
    job_id = job_service.create_job(
        job_type="document_processing",
        settings={
            "chunk_size": chunk_size,
            "min_size": min_size,
            "overlap": overlap,
            "enable_chunking": enable_chunking,
            "enhance_chunks": enhance_chunks
        }
    )
    
    # Add the background task
    background_tasks.add_task(
        document_service.process_documents_task,
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

@router.post("/upload-file", summary="Upload a file", 
            description="Upload a file (txt or PDF) to be processed.",
            response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    process_immediately: bool = Form(False),
    chunk_size: Optional[int] = Form(None, description="Override max chunk size (chars)"),
    min_size: Optional[int] = Form(None, description="Override min chunk size (chars)"),
    overlap: Optional[int] = Form(None, description="Override chunk overlap (chars)"),
    enable_chunking: Optional[bool] = Form(None, description="Override chunking enabled setting"),
    enhance_chunks: Optional[bool] = Form(True, description="Generate additional content with Ollama to improve retrieval"),
    generate_questions: Optional[bool] = Form(True, description="Generate questions for each chunk"),
    max_questions_per_chunk: Optional[int] = Form(5, description="Maximum number of questions to generate per chunk")
):
    """Upload a file for processing."""
    document_service = get_document_service()
    job_service = get_job_service()
    
    try:
        # Check if file is empty
        contents = await file.read()
        if not contents:
            return {
                "status": "error",
                "message": "File is empty",
                "file_path": ""
            }
        
        # Reset file position after reading
        await file.seek(0)
        
        # Check file type
        if not (file.filename.endswith('.txt') or file.filename.endswith('.pdf')):
            return {
                "status": "error",
                "message": "Only .txt and .pdf files are supported",
                "file_path": ""
            }
        
        # Process the file
        file_path = document_service.upload_file(contents, file.filename)
        
        result = {
            "status": "success",
            "message": f"File uploaded successfully: {file.filename}",
            "file_path": file_path
        }
        
        # Process the file immediately if requested
        if process_immediately and background_tasks:
            # Create a job to track progress
            job_id = job_service.create_job(
                job_type="file_processing",
                settings={
                    "file_path": file_path,
                    "chunk_size": chunk_size,
                    "min_size": min_size,
                    "overlap": overlap,
                    "enable_chunking": enable_chunking,
                    "enhance_chunks": enhance_chunks,
                    "generate_questions": generate_questions,
                    "max_questions_per_chunk": max_questions_per_chunk
                }
            )
            
            # Add the background task
            background_tasks.add_task(
                document_service.process_single_file_task,
                job_id=job_id,
                file_path=file_path,
                chunk_size=chunk_size,
                min_size=min_size, 
                overlap=overlap,
                enable_chunking=enable_chunking,
                enhance_chunks=enhance_chunks,
                generate_questions=generate_questions,
                max_questions_per_chunk=max_questions_per_chunk
            )
            
            # Add job information to the result
            result["job_id"] = job_id
            result["processing_status"] = "queued"
            result["message"] += " and queued for processing"
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error uploading file: {str(e)}",
            "file_path": ""
        }

@router.post("/clear-db", summary="Clear the database", description="Clear all documents from ChromaDB and Elasticsearch (if enabled).")
async def clear_database():
    """Clear all documents from both ChromaDB and Elasticsearch."""
    db_service = get_db_service()
    query_classifier_service = get_document_service().query_classifier
    document_service = get_document_service()
    
    try:
        # Get current document count for reporting from ChromaDB
        chroma_doc_count = db_service.get_document_count()
        
        # Clear ChromaDB
        db_service.delete_all_documents()
        result = {
            "chroma": {
                "status": "success",
                "documents_removed": chroma_doc_count
            }
        }
        
        # Clear Elasticsearch if it's available
        es_result = {"status": "not_available", "documents_removed": 0}
        
        # Check if Elasticsearch service exists in document_service
        if hasattr(document_service, 'elasticsearch_service') and document_service.elasticsearch_service:
            try:
                # Get document count before clearing
                es_doc_count = document_service.elasticsearch_service.get_document_count()
                
                # Delete all documents
                document_service.elasticsearch_service.delete_all_documents()
                
                es_result = {
                    "status": "success",
                    "documents_removed": es_doc_count
                }
            except Exception as es_error:
                es_result = {
                    "status": "error",
                    "error": str(es_error),
                    "documents_removed": 0
                }
        
        result["elasticsearch"] = es_result
        
        # Refresh the domain terms to reflect the empty database
        try:
            query_classifier_service.update_terms_from_db(db_service.collection)
            result["terms_updated"] = True
        except Exception as e:
            print(f"Error refreshing domain terms after clearing DB: {e}")
            result["terms_updated"] = False
            result["terms_error"] = str(e)
        
        # Create a user-friendly message
        total_docs = chroma_doc_count + es_result.get("documents_removed", 0)
        message = f"Databases cleared successfully. Removed {total_docs} documents"
        if es_result["status"] == "success":
            message += f" ({chroma_doc_count} from ChromaDB, {es_result['documents_removed']} from Elasticsearch)."
        elif es_result["status"] == "error":
            message += f" from ChromaDB. Error clearing Elasticsearch: {es_result['error']}"
        else:
            message += " from ChromaDB. Elasticsearch not available."
        
        return {
            "status": "success",
            "message": message,
            "result": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error clearing database: {str(e)}"
        }

@router.delete("/document/{filename}", summary="Delete document", 
             description="Delete a specific document and all its chunks from ChromaDB and Elasticsearch (if enabled).",
             response_model=DeleteDocumentResponse)
async def delete_document(filename: str):
    """Delete a specific document and all its chunks."""
    db_service = get_db_service()
    query_classifier_service = get_document_service().query_classifier
    document_service = get_document_service()
    
    try:
        # Delete document chunks from ChromaDB
        chroma_chunks_deleted = db_service.delete_document_by_filename(filename)
        
        # Initialize Elasticsearch result
        es_chunks_deleted = None
        
        # Delete from Elasticsearch if it's available
        if hasattr(document_service, 'elasticsearch_service') and document_service.elasticsearch_service:
            try:
                es_chunks_deleted = document_service.elasticsearch_service.delete_document_by_filename(filename)
            except Exception as es_error:
                # Log the error but continue with the operation
                print(f"Error deleting from Elasticsearch: {es_error}")
        
        # Refresh domain terms if documents were deleted
        if chroma_chunks_deleted > 0:
            try:
                query_classifier_service.update_terms_from_db(db_service.collection)
            except Exception as e:
                print(f"Error refreshing domain terms after document deletion: {e}")
        
        # Create user-friendly message
        message = f"Document '{filename}' deleted successfully with {chroma_chunks_deleted} chunks from ChromaDB"
        if es_chunks_deleted is not None:
            message += f" and {es_chunks_deleted} chunks from Elasticsearch"
        
        return {
            "status": "success",
            "message": message,
            "document": filename,
            "chunks_deleted": chroma_chunks_deleted,
            "es_chunks_deleted": es_chunks_deleted
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error deleting document: {str(e)}",
            "document": filename,
            "chunks_deleted": 0,
            "es_chunks_deleted": None
        }

@router.get("/chunks", summary="List document chunks", 
          description="Retrieve chunks stored in ChromaDB with optional filtering.",
          response_model=ChunkListResponse)
async def list_document_chunks(
    limit: int = Query(20, description="Limit the number of chunks returned"),
    offset: int = Query(0, description="Starting offset for pagination"),
    filename: str = Query(None, description="Filter by filename (partial match)"),
    content: str = Query(None, description="Filter by content (partial match)")
):
    """List document chunks with optional filtering."""
    db_service = get_db_service()
    
    try:
        # Get collection count to verify it's accessible
        doc_count = db_service.get_document_count()
        if doc_count == 0:
            return {
                "status": "empty",
                "message": "No documents in the database",
                "chunks": [],
                "total_in_db": 0,
                "total_matching": 0,
                "chunks_returned": 0
            }
            
        # Get all chunks since we'll filter client-side
        query_limit = limit if not filename and not content else doc_count
        query_offset = offset if not filename and not content else 0
            
        # Get results from ChromaDB
        results = db_service.get_all_documents(include_embeddings=True)
        
        # Extract the results
        chunks = []
        for i, doc in enumerate(results["documents"]):
            metadata = results["metadatas"][i] if "metadatas" in results else {}
            embedding_dim = len(results["embeddings"][i]) if "embeddings" in results else 0
            file_name = metadata.get("filename", "unknown")
            
            # Extract original text if it exists
            original_text = metadata.get("original_text", doc)
            
            # Get enrichment if it exists
            enrichment = metadata.get("enrichment", "")
            has_enrichment = metadata.get("has_enrichment", False)
            
            # Get questions if they exist
            has_questions = metadata.get("has_questions", False)
            questions = []
            if has_questions and "questions_json" in metadata:
                try:
                    import json
                    questions_json = metadata.get("questions_json", "[]")
                    questions = json.loads(questions_json)
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Error parsing questions JSON for chunk {results['ids'][i]}: {e}")
            
            chunks.append(ChunkInfo(
                id=results["ids"][i],
                text=original_text,
                filename=file_name,
                has_enrichment=has_enrichment,
                enrichment=enrichment if has_enrichment else "",
                embedding_dimension=embedding_dim,
                has_questions=has_questions,
                questions=questions
            ))
        
        # Apply filters
        filtered_chunks = filter_chunks_by_filename(chunks, filename, content)
        
        # Apply pagination
        start_idx = min(offset, len(filtered_chunks))
        end_idx = min(start_idx + limit, len(filtered_chunks))
        paginated_chunks = filtered_chunks[start_idx:end_idx]
        
        # Return the results
        return {
            "status": "success",
            "total_in_db": doc_count,
            "total_matching": len(filtered_chunks),
            "chunks_returned": len(paginated_chunks),
            "chunks": paginated_chunks
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving chunks: {str(e)}",
            "chunks": [],
            "total_in_db": 0,
            "total_matching": 0,
            "chunks_returned": 0
        }