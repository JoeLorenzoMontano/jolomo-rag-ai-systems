"""
PDF text extraction utility class.

This module provides a class for extracting and cleaning text from PDF files.
"""

import io
import os
import re
import PyPDF2

# Try to import pdfminer.six for better PDF extraction
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    from pdfminer.pdfparser import PDFSyntaxError
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False


class PDFExtractor:
    """Utility class for extracting and cleaning text from PDF files."""
    
    def __init__(self, temp_dir=None):
        """
        Initialize the PDF extractor.
        
        Args:
            temp_dir: Directory to use for temporary files. If None, the system temp directory is used.
        """
        self.temp_dir = temp_dir
    
    def extract_text(self, pdf_content, filename="document.pdf"):
        """
        Extract text from PDF content.
        
        Args:
            pdf_content: The binary content of the PDF file
            filename: The name of the PDF file (for logging purposes)
            
        Returns:
            The extracted text as a string
        """
        pdf_text = ""
        extraction_method = "unknown"
        
        try:
            # Try pdfminer.six first (usually better quality)
            if PDFMINER_AVAILABLE:
                try:
                    # Create a temporary file path
                    temp_file_path = self._get_temp_path(filename)
                    
                    # Save contents to a temporary file
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(pdf_content)
                        
                    try:
                        # Extract text using pdfminer
                        pdf_text = pdfminer_extract_text(temp_file_path)
                        if pdf_text:
                            extraction_method = "pdfminer"
                            print(f"PDF extracted with pdfminer.six: {len(pdf_text)} characters")
                    except (PDFSyntaxError, Exception) as e:
                        print(f"PDFMiner extraction failed, falling back to PyPDF2: {e}")
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                except Exception as e:
                    print(f"Error using pdfminer: {e}")
            
            # Fall back to PyPDF2 if pdfminer fails or isn't available
            if not pdf_text:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
                
                for page_num in range(len(pdf_reader.pages)):
                    page_text = pdf_reader.pages[page_num].extract_text()
                    # Clean up the text - this fixes the common issue with words being on separate lines
                    if page_text:
                        # Replace multiple newlines with a single one
                        page_text = re.sub(r'\n\s*\n', '\n\n', page_text)
                        # Fix words that got split across lines inappropriately (no period, comma, etc. before newline)
                        page_text = re.sub(r'(\w)\n(\w)', r'\1 \2', page_text)
                        # Fix cases where there might be a single letter followed by newline
                        page_text = re.sub(r'(\w)\n([a-z])\s', r'\1\2 ', page_text)
                    pdf_text += page_text + "\n\n"
                
                extraction_method = "pypdf2"
                print(f"PDF extracted with PyPDF2: {len(pdf_text)} characters")
            
            # Post-process and clean up text
            pdf_text = self._clean_text(pdf_text)
            
            print(f"PDF extraction complete using {extraction_method}")
            return pdf_text
            
        except Exception as e:
            print(f"PDF extraction failed: {e}")
            raise ValueError(f"Error extracting text from PDF: {str(e)}")
    
    def _get_temp_path(self, filename):
        """Generate a temporary file path."""
        base_name = os.path.basename(filename)
        temp_name = f"temp_{base_name}"
        
        if self.temp_dir and os.path.isdir(self.temp_dir):
            return os.path.join(self.temp_dir, temp_name)
        else:
            import tempfile
            return os.path.join(tempfile.gettempdir(), temp_name)
            
    def _clean_text(self, text):
        """
        Clean up extracted PDF text.
        
        Args:
            text: The raw extracted text
            
        Returns:
            The cleaned text
        """
        if not text:
            return ""
            
        # Fix common PDF extraction issues
        cleaned_text = text
        
        # Remove excessive whitespace
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        
        # Remove lines that just contain a single character (often artifacts)
        cleaned_text = re.sub(r'\n\s*[a-zA-Z]\s*\n', '\n', cleaned_text)
        
        # Fix cases where sentences continue on the next line
        cleaned_text = re.sub(r'([^.!?:])\n([a-z])', r'\1 \2', cleaned_text)
        
        # Replace multiple spaces with single space
        cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
        
        # Replace multiple newlines with double newline
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        # Remove unnecessary space before punctuation
        cleaned_text = re.sub(r' ([.,;:!?])', r'\1', cleaned_text)
        
        return cleaned_text.strip()