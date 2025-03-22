import requests
import json
import logging
import time
from typing import List, Dict, Any, Optional

class WebSearchClient:
    """Client for performing web searches to augment RAG context"""
    
    def __init__(self, serper_api_key: Optional[str] = None):
        """
        Initialize the web search client
        
        Args:
            serper_api_key: API key for Serper.dev Google search API
        """
        self.serper_api_key = serper_api_key
        self.logger = logging.getLogger(__name__)
    
    def search_with_serper(self, query: str, num_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search the web using Serper.dev Google search API
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            List of search result dictionaries with title, snippet, and URL
        """
        if not self.serper_api_key:
            self.logger.warning("No Serper API key provided, skipping web search")
            return []
            
        url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": query,
            "num": num_results
        })
        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }
        
        try:
            self.logger.info(f"Performing web search via Serper API: {query}")
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            results = response.json()
            
            # Process the results into a standardized format
            processed_results = []
            
            # Get organic search results
            if "organic" in results:
                for item in results["organic"][:num_results]:
                    processed_results.append({
                        "title": item.get("title", ""),
                        "content": item.get("snippet", ""),
                        "url": item.get("link", ""),
                        "source": "Google Search"
                    })
            
            # Add knowledge graph info if available
            if "knowledgeGraph" in results and len(processed_results) < num_results:
                kg = results["knowledgeGraph"]
                if "description" in kg:
                    processed_results.append({
                        "title": kg.get("title", "Knowledge Graph"),
                        "content": kg.get("description", ""),
                        "url": kg.get("link", ""),
                        "source": "Google Knowledge Graph"
                    })
            
            # Add answer box if available
            if "answerBox" in results and len(processed_results) < num_results:
                ab = results["answerBox"]
                answer_content = ab.get("answer", ab.get("snippet", ""))
                if answer_content:
                    processed_results.append({
                        "title": ab.get("title", "Featured Snippet"),
                        "content": answer_content,
                        "url": ab.get("link", ""),
                        "source": "Google Featured Snippet"
                    })
            
            self.logger.info(f"Found {len(processed_results)} web search results")
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Error in web search: {str(e)}")
            return []
    
    def format_results_as_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results as context text for the LLM
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Formatted string with search results
        """
        if not results:
            return ""
            
        context_parts = ["# Web Search Results\n"]
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"## Result {i}: {result['title']}")
            context_parts.append(f"Source: {result['source']} - {result['url']}")
            context_parts.append(f"{result['content']}\n")
        
        return "\n".join(context_parts)