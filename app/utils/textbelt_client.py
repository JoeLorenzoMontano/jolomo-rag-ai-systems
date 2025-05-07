"""
Textbelt client for sending SMS messages.

This module provides a client for the Textbelt API to send SMS messages.
"""

import os
import requests
from typing import Dict, Any, Optional
import logging

class TextbeltClient:
    """Client for interacting with the Textbelt API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://textbelt.com"
    ):
        """
        Initialize the Textbelt client.
        
        Args:
            api_key: API key for Textbelt. If not provided, will try to get from environment.
            base_url: Base URL for the Textbelt API.
        """
        self.api_key = api_key or os.getenv("TEXTBELT_API_KEY")
        if not self.api_key:
            logging.warning("No Textbelt API key provided. SMS functionality will be limited.")
            self.is_available = False
        else:
            self.is_available = True
            
        self.base_url = base_url
        
    def send_sms(
        self,
        phone: str,
        message: str,
        retry: bool = True
    ) -> Dict[str, Any]:
        """
        Send an SMS message using the Textbelt API.
        
        Args:
            phone: Phone number to send the message to, in international format
            message: Message content to send (no links allowed)
            retry: Whether to retry if the request fails
            
        Returns:
            Dict containing the response from the Textbelt API
        """
        if not self.is_available:
            return {
                "success": False,
                "error": "Textbelt API key not available"
            }
            
        # Remove any links from the message as they're not allowed by Textbelt
        sanitized_message = self._sanitize_message(message)
        
        # Prepare the payload
        payload = {
            "phone": phone,
            "message": sanitized_message,
            "key": self.api_key
        }
        
        try:
            response = requests.post(f"{self.base_url}/text", data=payload)
            response_data = response.json()
            
            # Log warning if message was modified
            if sanitized_message != message:
                logging.warning("Message was modified to remove links or URLs")
                
            return response_data
        except Exception as e:
            logging.error(f"Error sending SMS: {str(e)}")
            
            if retry:
                logging.info("Retrying SMS send...")
                return self.send_sms(phone, message, retry=False)
                
            return {
                "success": False,
                "error": f"Error sending SMS: {str(e)}"
            }
    
    def check_quota(self) -> Dict[str, Any]:
        """
        Check the remaining quota for the API key.
        
        Returns:
            Dict containing quota information
        """
        if not self.is_available:
            return {
                "success": False,
                "error": "Textbelt API key not available"
            }
            
        try:
            response = requests.get(f"{self.base_url}/quota/{self.api_key}")
            return response.json()
        except Exception as e:
            logging.error(f"Error checking quota: {str(e)}")
            return {
                "success": False,
                "error": f"Error checking quota: {str(e)}"
            }
    
    def _sanitize_message(self, message: str) -> str:
        """
        Sanitize message to remove links which are not allowed by Textbelt.
        
        Args:
            message: Original message to sanitize
            
        Returns:
            Sanitized message with links removed or replaced
        """
        # Simple pattern to detect and remove common URL formats
        import re
        
        # Replace http/https links with [Link removed] placeholder
        sanitized = re.sub(r'https?://\S+', '[Link removed]', message)
        
        # Replace www. links with [Link removed] placeholder
        sanitized = re.sub(r'www\.\S+', '[Link removed]', sanitized)
        
        return sanitized

    def generate_sms_response(
        self,
        query: str,
        context: str = "",
        model: Optional[str] = None,
        ollama_client = None
    ) -> str:
        """
        Generate a response suitable for SMS (shorter, no links).
        
        Args:
            query: The user's query
            context: Relevant context from RAG results
            model: Model to use for generating the response
            ollama_client: Optional custom Ollama client
            
        Returns:
            SMS-friendly response text
        """
        from core.dependencies import get_ollama_client
        
        # Get or use provided Ollama client
        if not ollama_client:
            ollama_client = get_ollama_client()
        
        # Create a system message prompting for SMS-friendly responses
        system_message = (
            "You are an AI assistant responding via SMS. "
            "Keep your responses concise (under 160 characters if possible). "
            "Do not include links or URLs. "
            "Be direct and helpful."
        )
        
        # Create messages structure
        messages = [
            {"role": "system", "content": system_message}
        ]
        
        # Add context if available
        if context:
            messages.append({
                "role": "system", 
                "content": f"Use this information to answer: {context}"
            })
        
        # Add user query
        messages.append({"role": "user", "content": query})
        
        try:
            # Generate response using Ollama
            response = ollama_client.generate_chat_response(
                messages=messages,
                model=model or ollama_client.model
            )
            
            # Log raw response for debugging
            logging.info(f"Raw SMS generation response: {response}")
            
            # Extract and sanitize the response text
            # Handle both dictionary response and string response
            if isinstance(response, dict):
                if "message" in response:
                    # New chat API format
                    response_text = response["message"]["content"]
                else:
                    # Old format or custom format
                    response_text = response.get("response", "Sorry, I couldn't generate a response.")
            else:
                # Response is already a string
                response_text = response
                
            logging.info(f"Extracted response text: {response_text}")
            sanitized_response = self._sanitize_message(response_text)
            
            return sanitized_response
        except Exception as e:
            logging.error(f"Error generating SMS response: {str(e)}")
            return "Sorry, I couldn't process your request at this time."