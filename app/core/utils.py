"""
Utility functions for the API.

This module provides various utility functions used across the application.
"""

import re
import os
from typing import List, Dict, Any, Optional

from models.schemas import ChunkInfo

def clean_filename(filename: str) -> str:
    """
    Clean a filename to prevent path traversal and ensure it's safe.
    
    Args:
        filename: The original filename
        
    Returns:
        Cleaned filename
    """
    # Remove any path component and only keep the filename
    filename = os.path.basename(filename)
    
    # Replace any non-word characters with underscores
    return re.sub(r'[^\w\-\.]', '_', filename)

def extract_file_info(chunk_id: str) -> Dict[str, str]:
    """
    Extract file information from a chunk ID.
    
    Args:
        chunk_id: The chunk ID to parse
        
    Returns:
        Dictionary with filename and chunk number
    """
    if "#chunk-" in chunk_id:
        source_file = chunk_id.split("#chunk-")[0]
        chunk_num = chunk_id.split("#chunk-")[1]
        return {
            "filename": source_file,
            "chunk_num": chunk_num
        }
    else:
        return {
            "filename": chunk_id,
            "chunk_num": "0"
        }

def filter_chunks_by_filename(chunks: List[ChunkInfo], 
                             filename: Optional[str] = None, 
                             content: Optional[str] = None,
                             case_sensitive: bool = False) -> List[ChunkInfo]:
    """
    Filter ChunkInfo objects by filename or content.
    
    Args:
        chunks: List of ChunkInfo objects to filter
        filename: Optional filename filter (partial match)
        content: Optional content filter (partial match)
        case_sensitive: Whether to use case-sensitive matching
        
    Returns:
        Filtered list of ChunkInfo objects
    """
    if not filename and not content:
        return chunks
        
    filtered_chunks = []
    
    for chunk in chunks:
        # Get the filename to check
        chunk_filename = chunk.filename
        
        # Check filename filter if provided
        if filename:
            if case_sensitive:
                if filename not in chunk_filename:
                    continue
            else:
                if filename.lower() not in chunk_filename.lower():
                    continue
        
        # Check content filter if provided
        if content:
            chunk_text = chunk.text
            if case_sensitive:
                if content not in chunk_text:
                    continue
            else:
                if content.lower() not in chunk_text.lower():
                    continue
        
        # If we made it here, the chunk passed all filters
        filtered_chunks.append(chunk)
    
    return filtered_chunks