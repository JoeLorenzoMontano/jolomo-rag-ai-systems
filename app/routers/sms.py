"""
SMS router.

This module provides endpoints for sending SMS responses using Textbelt API.
"""

from fastapi import APIRouter, HTTPException, Depends
import logging
from typing import Dict, Any, Optional

from core.dependencies import get_query_service, get_ollama_client, get_textbelt_client
from core.config import get_settings
from models.schemas import SMSRequest, SMSResponse
from utils.textbelt_client import TextbeltClient

router = APIRouter(tags=["sms"])

@router.post("/sms", response_model=SMSResponse, summary="Send SMS response", 
           description="Process a query and send the response via SMS.")
async def send_sms_response(
    sms_request: SMSRequest
):
    """Process a query and send the response via SMS."""
    query_service = get_query_service()
    textbelt_client = get_textbelt_client()
    settings = get_settings()
    
    if not textbelt_client or not textbelt_client.is_available:
        return SMSResponse(
            status="error",
            message="Textbelt API key not available. SMS functionality is disabled.",
            textbelt_response={"success": False, "error": "API key not configured"}
        )
    
    try:
        # Get or create appropriate Ollama client
        ollama_client = get_ollama_client()
        if sms_request.model:
            try:
                # Create a temporary client with the specified model
                from utils.ollama_client import OllamaClient
                ollama_client = OllamaClient(
                    model=sms_request.model,
                    embedding_model=ollama_client.embedding_model,
                    base_url=ollama_client.base_url
                )
            except Exception as e:
                logging.error(f"Error creating custom Ollama client: {e}")
                # Continue with default client
        
        # Get relevant documents
        rag_result = query_service.process_query(
            query=sms_request.query,
            n_results=sms_request.n_results,
            combine_chunks=sms_request.combine_chunks,
            web_search=sms_request.web_search,
            web_results_count=sms_request.web_results_count,
            explain_classification=False,
            enhance_query=sms_request.enhance_query,
            use_elasticsearch=sms_request.use_elasticsearch,
            hybrid_search=sms_request.hybrid_search,
            apply_reranking=sms_request.apply_reranking,
            check_question_matches=sms_request.check_question_matches,
            custom_ollama_client=ollama_client
        )
        
        # Check if RAG result indicates an error
        if rag_result.get("status") == "error" or rag_result.get("status") == "not_found":
            error_message = rag_result.get("response", "Error retrieving information")
            # Send error message via SMS
            sms_response = textbelt_client.send_sms(
                phone=sms_request.phone,
                message=f"Sorry, I encountered an error: {error_message}"
            )
            
            return SMSResponse(
                status="error",
                message=error_message,
                textbelt_response=sms_response
            )
        
        # Process any retrieved documents as context
        context = ""
        if rag_result.get("sources") and rag_result.get("sources").get("documents"):
            documents = rag_result.get("sources").get("documents")
            if documents:
                context = "\n\n".join(documents)
        
        # Generate SMS-friendly response
        response_text = textbelt_client.generate_sms_response(
            query=sms_request.query,
            context=context,
            model=sms_request.model,
            ollama_client=ollama_client
        )
        
        # Send the SMS
        sms_response = textbelt_client.send_sms(
            phone=sms_request.phone,
            message=response_text
        )
        
        # Build response
        response = SMSResponse(
            status="success" if sms_response.get("success") else "error",
            message=response_text,
            textbelt_response=sms_response,
            sources=rag_result.get("sources"),
            model_used=sms_request.model or ollama_client.model
        )
        
        return response
        
    except Exception as e:
        logging.error(f"Error in send_sms_response: {e}")
        
        # Try to send error message via SMS
        try:
            sms_response = textbelt_client.send_sms(
                phone=sms_request.phone,
                message=f"Sorry, I encountered an error while processing your query."
            )
        except Exception as sms_error:
            logging.error(f"Error sending error SMS: {sms_error}")
            sms_response = {"success": False, "error": str(sms_error)}
        
        # Return a helpful error response
        error_response = SMSResponse(
            status="error",
            message=f"An error occurred while processing your SMS query: {str(e)}",
            textbelt_response=sms_response
        )
        
        return error_response

@router.get("/sms/quota", summary="Check SMS quota", 
           description="Check the remaining quota for the configured Textbelt API key.")
async def check_sms_quota():
    """Check the remaining quota for the Textbelt API key."""
    textbelt_client = get_textbelt_client()
    
    if not textbelt_client or not textbelt_client.is_available:
        return {
            "status": "error",
            "message": "Textbelt API key not available. SMS functionality is disabled."
        }
    
    try:
        quota_response = textbelt_client.check_quota()
        return {
            "status": "success",
            "quota": quota_response
        }
    except Exception as e:
        logging.error(f"Error checking SMS quota: {e}")
        return {
            "status": "error",
            "message": f"Error checking SMS quota: {str(e)}"
        }