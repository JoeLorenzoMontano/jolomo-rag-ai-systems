from .client import APIClient

class QueryAPI(APIClient):
    """Client for query and chat API endpoints."""
    
    def query(self, query_text, n_results=3, combine_chunks=True, web_search=None, 
              web_results_count=5, explain_classification=False, enhance_query=True, timeout=None):
        """Query documents with advanced options.
        
        Args:
            query_text (str): Query text
            n_results (int, optional): Number of results to return. Defaults to 3.
            combine_chunks (bool, optional): Combine chunks from same document. Defaults to True.
            web_search (bool, optional): Use web search. Defaults to None (auto-classify).
            web_results_count (int, optional): Web results count. Defaults to 5.
            explain_classification (bool, optional): Include classification explanation. Defaults to False.
            enhance_query (bool, optional): Enhance query. Defaults to True.
            timeout (int, optional): Request timeout. Defaults to None.
            
        Returns:
            dict: Query results
        """
        params = {
            'query': query_text,
            'n_results': n_results,
            'combine_chunks': combine_chunks,
            'web_search': web_search,
            'web_results_count': web_results_count,
            'explain_classification': explain_classification,
            'enhance_query': enhance_query
        }
        
        return self.get("query", params=params, timeout=timeout)
    
    def chat(self, messages=None, query_text=None, n_results=3, combine_chunks=True, 
            web_search=None, web_results_count=3, enhance_query=True, timeout=None):
        """Send a chat query with conversation history.
        
        Args:
            messages (list, optional): List of message objects. Defaults to None.
            query_text (str, optional): Single query text (legacy). Defaults to None.
            n_results (int, optional): Number of results. Defaults to 3.
            combine_chunks (bool, optional): Combine chunks. Defaults to True.
            web_search (bool, optional): Use web search. Defaults to None (auto).
            web_results_count (int, optional): Web results count. Defaults to 3.
            enhance_query (bool, optional): Enhance query. Defaults to True.
            timeout (int, optional): Request timeout. Defaults to None.
            
        Returns:
            dict: Chat response
        """
        # Handle the case of a single query (legacy format)
        if messages is None and query_text:
            messages = [{"role": "user", "content": query_text}]
        
        if messages is None:
            return {"status": "error", "message": "No messages or query provided"}
        
        payload = {
            'messages': messages,
            'n_results': n_results,
            'combine_chunks': combine_chunks,
            'web_search': web_search,
            'web_results_count': web_results_count,
            'enhance_query': enhance_query
        }
        
        return self.post("chat", json=payload, timeout=timeout)