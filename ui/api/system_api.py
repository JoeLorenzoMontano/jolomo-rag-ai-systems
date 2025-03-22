from .client import APIClient

class SystemAPI(APIClient):
    """Client for system-related API endpoints."""
    
    def get_health(self, timeout=10):
        """Get system health status.
        
        Args:
            timeout (int, optional): Request timeout. Defaults to 10.
            
        Returns:
            dict: Health status information
        """
        return self.get("health", timeout=timeout)
    
    def get_terms(self, timeout=10):
        """Get domain-specific terms used by the classifier.
        
        Args:
            timeout (int, optional): Request timeout. Defaults to 10.
            
        Returns:
            dict: Terms list
        """
        return self.get("terms", timeout=timeout)
    
    def refresh_terms(self, timeout=20):
        """Refresh domain-specific terms by analyzing documents.
        
        Args:
            timeout (int, optional): Request timeout. Defaults to 20.
            
        Returns:
            dict: Result of term refresh operation
        """
        return self.post("refresh-terms", timeout=timeout)
    
    def get_job_status(self, job_id, timeout=5):
        """Get the status of a background job.
        
        Args:
            job_id (str): Job ID
            timeout (int, optional): Request timeout. Defaults to 5.
            
        Returns:
            dict: Job status information
        """
        return self.get(f"job/{job_id}", timeout=timeout)
    
    def list_jobs(self, timeout=5):
        """List all background jobs.
        
        Args:
            timeout (int, optional): Request timeout. Defaults to 5.
            
        Returns:
            dict: List of jobs
        """
        return self.get("jobs", timeout=timeout)
    
    def get_chroma_info(self, timeout=10):
        """Get information about ChromaDB.
        
        Args:
            timeout (int, optional): Request timeout. Defaults to 10.
            
        Returns:
            dict: ChromaDB information
        """
        health_data = self.get_health(timeout=timeout)
        
        # Extract ChromaDB information
        chroma_info = {
            "status": "success",
            "server_version": health_data.get("chroma", "unknown"),
            "api_status": health_data.get("api", "unknown"),
            "document_count": 0,
            "collection_count": 0
        }
        
        # Check collection information
        if "collection" in health_data and health_data["collection"]["status"] == "healthy":
            chroma_info["document_count"] = health_data["collection"]["document_count"]
            chroma_info["collection_count"] = 1  # We only have one collection in this app
            
        return chroma_info