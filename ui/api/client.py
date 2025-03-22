import requests
import logging
from flask import current_app

class APIClient:
    """Base API client for making requests to the backend API."""
    
    def __init__(self, base_url=None):
        """Initialize the API client with a base URL.
        
        Args:
            base_url (str, optional): Base URL of the API. Defaults to None,
                which will use the configured API_URL from Flask application config.
        """
        self.base_url = base_url
    
    @property
    def base_url(self):
        """Get the base URL for API requests."""
        if self._base_url is None:
            return current_app.config["API_URL"]
        return self._base_url
        
    @base_url.setter
    def base_url(self, url):
        """Set the base URL for API requests."""
        self._base_url = url
    
    def get(self, endpoint, params=None, timeout=10):
        """Make a GET request to the API.
        
        Args:
            endpoint (str): API endpoint to call
            params (dict, optional): Query parameters. Defaults to None.
            timeout (int, optional): Request timeout in seconds. Defaults to 10.
            
        Returns:
            dict: Response JSON or error details
        """
        try:
            response = requests.get(
                f"{self.base_url}/{endpoint.lstrip('/')}",
                params=params,
                timeout=timeout
            )
            return response.json()
        except Exception as e:
            logging.error(f"API error in GET {endpoint}: {e}")
            return {"status": "error", "message": str(e)}
    
    def post(self, endpoint, json=None, data=None, files=None, timeout=10):
        """Make a POST request to the API.
        
        Args:
            endpoint (str): API endpoint to call
            json (dict, optional): JSON data. Defaults to None.
            data (dict, optional): Form data. Defaults to None.
            files (dict, optional): Files to upload. Defaults to None.
            timeout (int, optional): Request timeout in seconds. Defaults to 10.
            
        Returns:
            dict: Response JSON or error details
        """
        try:
            response = requests.post(
                f"{self.base_url}/{endpoint.lstrip('/')}",
                json=json,
                data=data,
                files=files,
                timeout=timeout
            )
            return response.json()
        except Exception as e:
            logging.error(f"API error in POST {endpoint}: {e}")
            return {"status": "error", "message": str(e)}