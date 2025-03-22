"""
Query router.

This module provides endpoints for querying the document database.
"""

from fastapi import APIRouter, HTTPException, Depends, Query as QueryParam
from typing import List, Dict, Any, Optional

from core.dependencies import get_query_service

router = APIRouter(tags=["query"])

@router.get("/query", summary="Retrieve relevant documents", 
           description="Query for the most relevant document based on input text.")
async def query_documents(
    query: str,
    n_results: int = QueryParam(3, description="Number of results to return"),
    combine_chunks: bool = QueryParam(True, description="Whether to combine chunks from the same document"),
    web_search: bool = QueryParam(None, description="Whether to augment with web search results (auto if None)"),
    web_results_count: int = QueryParam(5, description="Number of web search results to include"),
    explain_classification: bool = QueryParam(False, description="Whether to include query classification explanation"),
    enhance_query: bool = QueryParam(True, description="Whether to enhance the query for better retrieval")
):
    """Query for relevant documents based on input text."""
    query_service = get_query_service()
    
    try:
        result = query_service.process_query(
            query=query,
            n_results=n_results,
            combine_chunks=combine_chunks,
            web_search=web_search,
            web_results_count=web_results_count,
            explain_classification=explain_classification,
            enhance_query=enhance_query
        )
        
        return result
        
    except Exception as e:
        # Log the error
        print(f"Error in query_documents: {e}")
        
        # Return a helpful error response
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