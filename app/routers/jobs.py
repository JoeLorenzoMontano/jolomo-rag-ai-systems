"""
Job tracking router.

This module provides endpoints for tracking background jobs.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any

from core.dependencies import (
    get_job_service,
    get_document_service
)
from models.schemas import JobResponse

router = APIRouter(tags=["jobs"])

@router.get("/job/{job_id}", summary="Get job status", 
           description="Check the status of a document processing job.")
async def get_job_status(job_id: str):
    """Get the status of a job."""
    job_service = get_job_service()
    
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
    return job

@router.get("/jobs", summary="List all jobs", 
           description="List all document processing jobs.")
async def list_jobs():
    """List all jobs."""
    job_service = get_job_service()
    
    all_jobs = job_service.get_all_jobs()
    return {
        "total_jobs": len(all_jobs),
        "jobs": all_jobs
    }

@router.post("/refresh-terms", summary="Refresh domain terms", 
            description="Refreshes the domain-specific terms used for query classification based on current document content.")
async def refresh_domain_terms(background_tasks: BackgroundTasks):
    """Refresh the domain-specific terms used in query classification using statistical extraction."""
    job_service = get_job_service()
    document_service = get_document_service()
    query_classifier = document_service.query_classifier
    
    try:
        # Get document count before refresh
        prev_term_count = len(query_classifier.product_terms)
        
        # Create a job for term extraction
        job_id = job_service.create_job(job_type="term_extraction")
        
        # Add the background task
        background_tasks.add_task(
            document_service.refresh_domain_terms_task,
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