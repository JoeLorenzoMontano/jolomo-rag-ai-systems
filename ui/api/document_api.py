from .client import APIClient

class DocumentAPI(APIClient):
    """Client for document-related API endpoints."""
    
    def process_documents(self, chunking_params=None, timeout=10):
        """Start document processing job.
        
        Args:
            chunking_params (dict, optional): Parameters for chunking. Defaults to None.
            timeout (int, optional): Request timeout. Defaults to 10.
            
        Returns:
            dict: Job information
        """
        return self.post("process", params=chunking_params, timeout=timeout)
    
    def upload_file(self, file_data, filename, content_type, process_immediately=False, timeout=60):
        """Upload a file to be processed.
        
        Args:
            file_data (bytes): File content
            filename (str): File name
            content_type (str): File content type
            process_immediately (bool, optional): Process file immediately. Defaults to False.
            timeout (int, optional): Request timeout. Defaults to 60.
            
        Returns:
            dict: Upload result
        """
        files = {'file': (filename, file_data, content_type)}
        data = {'process_immediately': str(process_immediately)}
        
        return self.post("upload-file", files=files, data=data, timeout=timeout)
    
    def clear_database(self, timeout=10):
        """Clear the document database.
        
        Args:
            timeout (int, optional): Request timeout. Defaults to 10.
            
        Returns:
            dict: Operation result
        """
        return self.post("clear-db", timeout=timeout)
    
    def get_chunks(self, limit=20, offset=0, filename=None, content=None, timeout=15):
        """Get document chunks from the database.
        
        Args:
            limit (int, optional): Maximum chunks to return. Defaults to 20.
            offset (int, optional): Pagination offset. Defaults to 0.
            filename (str, optional): Filter by filename. Defaults to None.
            content (str, optional): Filter by content. Defaults to None.
            timeout (int, optional): Request timeout. Defaults to 15.
            
        Returns:
            dict: Chunks matching the criteria
        """
        params = {
            'limit': limit,
            'offset': offset
        }
        
        if filename:
            params['filename'] = filename
            
        if content:
            params['content'] = content
            
        return self.get("chunks", params=params, timeout=timeout)