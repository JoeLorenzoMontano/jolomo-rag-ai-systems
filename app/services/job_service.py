"""
Job tracking service for background tasks.

This module provides functionality to track and manage background processing jobs.
"""

import threading
import uuid
from typing import Dict, Any, Optional

# Job status constants
JOB_STATUS_QUEUED = "queued"
JOB_STATUS_PROCESSING = "processing"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"

class JobService:
    """Service for tracking and managing background jobs."""
    
    def __init__(self):
        """Initialize the job tracking service."""
        # Dictionary to track background processing jobs
        self.processing_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Thread-safe lock for updating job status
        self.job_lock = threading.Lock()
    
    def create_job(self, job_type: str = "document_processing", settings: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new job and return its ID.
        
        Args:
            job_type: Type of job being created
            settings: Optional settings for the job
            
        Returns:
            The UUID of the created job
        """
        job_id = str(uuid.uuid4())
        
        # Create job with default values
        job_data = {
            "status": JOB_STATUS_QUEUED,
            "progress": 0,
            "type": job_type,
            "total_files": 0,
            "processed_files": 0,
            "successful_chunks": 0,
            "failed_chunks": 0,
            "error": None,
            "settings": settings or {},
            "result": None
        }
        
        # Store the job
        with self.job_lock:
            self.processing_jobs[job_id] = job_data
        
        return job_id
    
    def update_job_status(self, job_id: str, status: str, progress: int = None, **kwargs) -> None:
        """
        Update a job's status and optional additional fields.
        
        Args:
            job_id: ID of the job to update
            status: New status value
            progress: Optional progress percentage (0-100)
            **kwargs: Additional fields to update
        """
        with self.job_lock:
            if job_id not in self.processing_jobs:
                return
                
            self.processing_jobs[job_id]["status"] = status
            
            if progress is not None:
                self.processing_jobs[job_id]["progress"] = progress
                
            # Update any additional fields
            for key, value in kwargs.items():
                self.processing_jobs[job_id][key] = value
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a job by its ID.
        
        Args:
            job_id: ID of the job to retrieve
            
        Returns:
            The job data dict or None if not found
        """
        with self.job_lock:
            return self.processing_jobs.get(job_id)
    
    def get_all_jobs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all tracked jobs.
        
        Returns:
            Dictionary of all job data
        """
        with self.job_lock:
            # Return a copy to avoid thread safety issues
            return self.processing_jobs.copy()
    
    def mark_job_completed(self, job_id: str, result: Dict[str, Any]) -> None:
        """
        Mark a job as completed with results.
        
        Args:
            job_id: ID of the job to update
            result: Result data to store
        """
        with self.job_lock:
            if job_id not in self.processing_jobs:
                return
                
            self.processing_jobs[job_id]["status"] = JOB_STATUS_COMPLETED
            self.processing_jobs[job_id]["progress"] = 100
            self.processing_jobs[job_id]["result"] = result
    
    def mark_job_failed(self, job_id: str, error: str) -> None:
        """
        Mark a job as failed with error information.
        
        Args:
            job_id: ID of the job to update
            error: Error message
        """
        with self.job_lock:
            if job_id not in self.processing_jobs:
                return
                
            self.processing_jobs[job_id]["status"] = JOB_STATUS_FAILED
            self.processing_jobs[job_id]["error"] = error