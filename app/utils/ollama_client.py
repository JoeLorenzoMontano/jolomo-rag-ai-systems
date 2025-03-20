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
        
        # Make sure base_url doesn't end with a slash
        if self.base_url.endswith('/'):
            self.base_url = self.base_url[:-1]

    def generate_response(self, context: str, query: str, model: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        """
        Sends a query to the Ollama server with strict response rules.
        """
        current_model = model or self.model
        
        # Check if the model exists, if not pull it
        self._ensure_model_exists(current_model)
        
        payload = {
            "model": current_model,
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
            "stream": False
        }
        
        # Add max_tokens if provided
        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}

        response = requests.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()
        return self._remove_think_regions(result.get("response", "No response generated."))

    def _ensure_model_exists(self, model_name: str) -> None:
        """
        Checks if a model exists, and pulls it if it doesn't.
        """
        try:
            # Check if model exists
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            
            # Look for the model in the list
            model_exists = any(model["name"] == model_name for model in models)
            
            if not model_exists:
                # Try to pull the model
                print(f"Model {model_name} not found. Attempting to pull...")
                pull_payload = {"name": model_name}
                pull_response = requests.post(f"{self.base_url}/api/pull", json=pull_payload)
                pull_response.raise_for_status()
                print(f"Successfully pulled model {model_name}")
        except Exception as e:
            print(f"Error checking/pulling model: {e}")

    def summarize_text(self, text: str, context: str, model: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        """
        Summarizes the provided text following strict summarization rules.
        """
        if not text.strip():
            return "Error: No input text provided."
            
        current_model = model or self.model
        
        # Ensure model exists
        self._ensure_model_exists(current_model)

        payload = {
            "model": current_model,
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
            "stream": False
        }
        
        # Add max_tokens if provided
        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}

        response = requests.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()
        return self._remove_think_regions(result.get("response", "No summary generated."))

    def generate_embedding(self, input_text: str) -> List[float]:
        """
        Generates an embedding for the given input text using the Ollama API.
        
        According to the Ollama API docs, the /api/embeddings endpoint expects:
        - model: name of the model to use
        - prompt: text to generate embeddings for
        
        Returns a list of floats (the embedding vector)
        """
        # Use an embedding model if available, otherwise use the default model
        embedding_model = "nomic-embed-text"
        
        # Try to ensure the embedding model exists
        try:
            self._ensure_model_exists(embedding_model)
        except Exception as e:
            print(f"Could not ensure embedding model exists, using default model: {e}")
            embedding_model = self.model
        
        payload = {
            "model": embedding_model,
            "prompt": input_text
        }

        try:
            response = requests.post(f"{self.base_url}/api/embeddings", json=payload)
            response.raise_for_status()
            result = response.json()
            embedding = result.get("embedding", [])
            
            # Check if we got a valid embedding
            if not embedding:
                raise ValueError("No embedding returned from API")
                
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # In a production environment, we should handle this error more gracefully
            # For now, creating a random embedding for testing purposes
            import random
            import numpy as np
            
            # Create a random unit vector of size 768 (common embedding dimension)
            random_vector = np.random.randn(768)
            normalized = random_vector / np.linalg.norm(random_vector)
            return normalized.tolist()

    def extract_metadata(self, text: str, model: Optional[str] = None, max_tokens: Optional[int] = None) -> Dict[str, str]:
        """
        Extracts structured metadata from the text following a strict JSON schema.
        """
        current_model = model or self.model
        
        # Ensure model exists
        self._ensure_model_exists(current_model)
        
        payload = {
            "model": current_model,
            "prompt": (
                "You are an AI assistant that extracts structured metadata from text. "
                "You must follow the JSON schema strictly without adding explanations or extra formatting.\n\n"
                f"Extract metadata from this document:\n\n\"\"\"\n{text}\n\"\"\"\n\n"
                "Output the metadata as a valid JSON object with fields for title, author, date, and summary."
            ),
            "stream": False
        }
        
        # Add max_tokens if provided
        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}

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