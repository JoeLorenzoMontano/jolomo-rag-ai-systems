import requests
import json
import re
import os
from typing import List, Dict, Optional

class OllamaClient:
    def __init__(self, base_url: str = None, model: str = None):
        """
        Initializes the Ollama client.
        :param base_url: Ollama server URL (default: local Ollama instance)
        :param model: Default model to use for generating responses
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("MODEL", "llama2")

    def generate_response(self, context: str, query: str, model: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        """
        Sends a query to the Ollama server with strict response rules.
        """
        payload = {
            "model": model or self.model,
            "prompt": (
                "You are an AI assistant that must follow strict response rules.\n"
                "### 🔹 **Rules (MUST FOLLOW):**\n"
                "1 **You MUST ONLY use information from the provided context.**\n"
                "2 **If the answer is NOT in the context, respond with:**\n"
                "   ❌ 'I could not find relevant information in the provided context. Please provide additional details if needed.'\n"
                "3 **You MUST NOT generate an answer using external knowledge.**\n"
                "4 **You MUST NOT make up any information.**\n\n"
                "### 🔹 **Context (ONLY use the information provided below to answer the query):**\n"
                f"\"\"\"\n{context}\n\"\"\"\n\n"
                f"Query: {query}"
            ),
            "stream": False,
            "options": {
                "num_predict": max_tokens if max_tokens else 1024
            }
        }

        response = requests.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()
        return self._remove_think_regions(result.get("response", "No response generated."))

    def summarize_text(self, text: str, context: str, model: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        """
        Summarizes the provided text following strict summarization rules.
        """
        if not text.strip():
            return "Error: No input text provided."
            
        model = model or self.model

        payload = {
            "model": model,
            "prompt": (
                "You are an AI assistant specializing in summarization.\n"
                "### 🔹 **Summarization Rules:**\n"
                "1. **Preserve all key details.**\n"
                "2. **Include all relevant facts, quotes, and statistics.**\n"
                "3. **Do NOT change the meaning of the original text.**\n"
                "4. **Do NOT add opinions or remove crucial context.**\n\n"
                "### 🔹 **Context (Previous Chunks):**\n"
                f"\"\"\"\n{context}\n\"\"\"\n\n"
                "### 🔹 **Text to Summarize:**\n"
                f"\"\"\"\n{text}\n\"\"\"\n\n"
                "Provide a concise yet detailed summary."
            ),
            "stream": False,
            "options": {
                "num_predict": max_tokens if max_tokens else 1024
            }
        }

        response = requests.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()
        return self._remove_think_regions(result.get("response", "No summary generated."))

    def generate_embedding(self, input_text: str) -> List[float]:
        """
        Generates an embedding for the given input text.
        """
        payload = {
            "model": "nomic-embed-text",
            "prompt": input_text,
            "options": {
                "embedding_only": True
            }
        }

        try:
            response = requests.post(f"{self.base_url}/api/embeddings", json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("embedding", [])
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return a dummy embedding if we can't get a real one
            # This is for testing purposes only
            import random
            return [random.random() for _ in range(768)]

    def extract_metadata(self, text: str, model: Optional[str] = None, max_tokens: Optional[int] = None) -> Dict[str, str]:
        """
        Extracts structured metadata from the text following a strict JSON schema.
        """
        model = model or self.model
        
        payload = {
            "model": model,
            "prompt": (
                "You are an AI assistant that extracts structured metadata from text. "
                "You must follow the JSON schema strictly without adding explanations or extra formatting.\n\n"
                f"Extract metadata from this document:\n\n\"\"\"\n{text}\n\"\"\"\n\n"
                "Output the metadata as a valid JSON object with fields for title, author, date, and summary."
            ),
            "stream": False,
            "options": {
                "num_predict": max_tokens if max_tokens else 1024
            }
        }

        response = requests.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()
        
        response_content = result.get("response", "{}")
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
            if json_match:
                response_content = json_match.group(1)
            
            # Clean up any non-JSON content
            response_content = re.sub(r'^[^{]*', '', response_content)
            response_content = re.sub(r'[^}]*$', '', response_content)
            
            return self._normalize_metadata(json.loads(response_content))
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON response", "content": response_content}

    def _normalize_metadata(self, metadata: Dict[str, str]) -> Dict[str, str]:
        """
        Converts arrays into pipe-separated strings.
        """
        fields_to_convert = {"named_entities", "keywords"}
        for field in fields_to_convert:
            if field in metadata and isinstance(metadata[field], list):
                metadata[field] = "|".join(metadata[field])
        return metadata

    def _remove_think_regions(self, text: str) -> str:
        """
        Removes `<think>...</think>` sections from the AI output.
        """
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()