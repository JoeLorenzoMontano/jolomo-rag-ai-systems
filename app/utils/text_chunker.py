"""
Text chunking utilities.

This module provides functionality to split documents into manageable chunks
for embedding and retrieval.
"""

import re
from typing import List, Tuple, Generator


class TextChunker:
    """
    Handles document text chunking to create optimal pieces for embedding and retrieval.
    """
    
    def __init__(self, 
                max_chunk_size: int = 1000, 
                min_chunk_size: int = 200, 
                chunk_overlap: int = 100, 
                enable_chunking: bool = True):
        """
        Initialize the text chunker with chunking parameters.
        
        Args:
            max_chunk_size: Maximum size of a chunk in characters
            min_chunk_size: Minimum size of a chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            enable_chunking: Whether to enable chunking at all
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_chunking = enable_chunking
    
    def chunk_text(self, text: str, document_id: str) -> List[Tuple[str, str]]:
        """
        Splits text into chunks for better embedding and retrieval.
        Returns a list of (chunk_text, chunk_id) tuples.
        
        Args:
            text: The document text to chunk
            document_id: The identifier for the document (used for chunk IDs)
        
        Returns:
            List of tuples with (chunk_text, chunk_id)
        """
        # Skip empty documents
        if not text.strip():
            return []
        
        # If chunking is disabled or document is small, return as single chunk
        if not self.enable_chunking or len(text) < self.max_chunk_size:
            return [(text, document_id)]
        
        chunks = []
        
        # Split document into paragraphs based on double newlines
        # This preserves natural document structure
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        # Process each paragraph
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_length = len(para)
            
            # If this paragraph alone exceeds max chunk size, split it further
            if para_length > self.max_chunk_size:
                # If we have a current chunk, finalize it first
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunk_id = f"{document_id}#chunk-{chunk_index}"
                    chunks.append((chunk_text, chunk_id))
                    chunk_index += 1
                    current_chunk = []
                    current_length = 0
                
                # Split long paragraphs by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sentence_chunk = []
                sentence_length = 0
                
                for sentence in sentences:
                    if sentence_length + len(sentence) > self.max_chunk_size and sentence_length > self.min_chunk_size:
                        # Finalize this sentence chunk
                        chunk_text = " ".join(sentence_chunk)
                        chunk_id = f"{document_id}#chunk-{chunk_index}"
                        chunks.append((chunk_text, chunk_id))
                        chunk_index += 1
                        
                        # Start a new chunk with overlap
                        retain_sentences = []
                        retain_length = 0
                        
                        # Keep some sentences for context overlap
                        for prev_sent in reversed(sentence_chunk):
                            if retain_length + len(prev_sent) <= self.chunk_overlap:
                                retain_sentences.insert(0, prev_sent)
                                retain_length += len(prev_sent) + 1  # +1 for space
                            else:
                                break
                        
                        sentence_chunk = retain_sentences
                        sentence_length = retain_length
                    
                    sentence_chunk.append(sentence)
                    sentence_length += len(sentence) + 1  # +1 for space
                
                # Add the remaining sentences as a chunk
                if sentence_chunk:
                    chunk_text = " ".join(sentence_chunk)
                    chunk_id = f"{document_id}#chunk-{chunk_index}"
                    chunks.append((chunk_text, chunk_id))
                    chunk_index += 1
                
            # If adding this paragraph would exceed the limit, finalize the current chunk
            elif current_length + para_length > self.max_chunk_size and current_length > self.min_chunk_size:
                chunk_text = "\n\n".join(current_chunk)
                chunk_id = f"{document_id}#chunk-{chunk_index}"
                chunks.append((chunk_text, chunk_id))
                chunk_index += 1
                
                # For overlap, keep some content from the previous chunk
                overlap_paras = []
                overlap_length = 0
                
                # Find paragraphs to retain for overlap
                for prev_para in reversed(current_chunk):
                    if overlap_length + len(prev_para) <= self.chunk_overlap:
                        overlap_paras.insert(0, prev_para)
                        overlap_length += len(prev_para) + 2  # +2 for newlines
                    else:
                        break
                
                current_chunk = overlap_paras
                current_length = overlap_length
                
                # Add the current paragraph to the new chunk
                current_chunk.append(para)
                current_length += para_length
                
            # Otherwise add the paragraph to the current chunk
            else:
                current_chunk.append(para)
                current_length += para_length + 2  # +2 for the paragraph separator
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk_id = f"{document_id}#chunk-{chunk_index}"
            chunks.append((chunk_text, chunk_id))
        
        return chunks