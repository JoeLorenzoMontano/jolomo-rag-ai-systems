"""
Domain terms router.

This module provides endpoints for managing domain terms used in query classification.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List

from core.dependencies import get_query_classifier
from models.schemas import TermsListResponse

router = APIRouter(tags=["terms"])

@router.get("/terms", summary="List domain terms", 
          description="List the domain-specific terms used for query classification.",
          response_model=TermsListResponse)
async def list_domain_terms():
    """List the domain-specific terms used in query classification."""
    query_classifier = get_query_classifier()
    
    try:
        return {
            "status": "success",
            "term_count": len(query_classifier.product_terms),
            "terms": query_classifier.product_terms
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error listing domain terms: {str(e)}",
            "term_count": 0,
            "terms": []
        }