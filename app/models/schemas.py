"""
API data schemas.

This module defines data schemas for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class HealthResponse(BaseModel):
    """Health check response schema."""
    api: str = Field(description="Status of the API")
    chroma: str = Field(description="Status of the ChromaDB connection")
    elasticsearch: Optional[str] = Field(None, description="Status of the Elasticsearch connection")
    ollama: str = Field(description="Status of the Ollama service")
    models: Dict[str, str] = Field(description="Status of the models")
    collection: Dict[str, Any] = Field(description="Status of the document collection")


class JobResponse(BaseModel):
    """Job status response schema."""
    job_id: str = Field(description="Unique identifier for the job")
    status: str = Field(description="Status of the job (queued, processing, completed, failed)")
    progress: int = Field(description="Progress percentage of the job")
    type: Optional[str] = Field(None, description="Type of job")
    error: Optional[str] = Field(None, description="Error message if the job failed")
    result: Optional[Dict[str, Any]] = Field(None, description="Result of the job if completed")


class ChunkInfo(BaseModel):
    """Information about a document chunk."""
    id: str = Field(description="Unique identifier for the chunk")
    text: str = Field(description="Text content of the chunk")
    filename: str = Field(description="Source filename")
    has_enrichment: bool = Field(description="Whether the chunk has semantic enrichment")
    enrichment: Optional[str] = Field("", description="Semantic enrichment if available")
    embedding_dimension: Optional[int] = Field(0, description="Dimension of the embedding vector")


class ChunkListResponse(BaseModel):
    """Response schema for listing chunks."""
    status: str = Field(description="Status of the request")
    total_in_db: int = Field(description="Total number of chunks in the database")
    total_matching: int = Field(description="Total number of chunks matching the filters")
    chunks_returned: int = Field(description="Number of chunks returned in this response")
    chunks: List[ChunkInfo] = Field(description="List of chunk information")
    message: Optional[str] = Field(None, description="Optional message")


class TermsListResponse(BaseModel):
    """Response schema for listing domain terms."""
    status: str = Field(description="Status of the request")
    term_count: int = Field(description="Number of domain terms")
    terms: List[str] = Field(description="List of domain terms")


class FileUploadResponse(BaseModel):
    """Response schema for file uploads."""
    status: str = Field(description="Status of the upload")
    message: str = Field(description="Status message")
    file_path: str = Field(description="Path to the saved file")
    job_id: Optional[str] = Field(None, description="Job ID if processing was started")
    processing_status: Optional[str] = Field(None, description="Status of processing if started")


class ChatMessage(BaseModel):
    """Schema for a chat message."""
    role: str = Field(description="Role of the message sender (user or assistant)")
    content: str = Field(description="Content of the message")


class ChatRequest(BaseModel):
    """Schema for a chat request with conversation history."""
    messages: List[ChatMessage] = Field(description="List of chat messages in the conversation")
    n_results: int = Field(3, description="Number of results to return")
    combine_chunks: bool = Field(True, description="Whether to combine chunks from the same document")
    web_search: Optional[bool] = Field(None, description="Whether to use web search (auto if None)")
    web_results_count: int = Field(3, description="Number of web search results to include")
    enhance_query: bool = Field(True, description="Whether to enhance the query for better retrieval")
    use_elasticsearch: Optional[bool] = Field(None, description="Whether to use Elasticsearch (auto if None)")
    hybrid_search: bool = Field(True, description="Whether to combine results from ChromaDB and Elasticsearch")
    apply_reranking: bool = Field(True, description="Whether to apply reranking to improve document relevance")
