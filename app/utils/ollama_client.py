import requests
import re
import os
from typing import List, Optional

class OllamaClient:
    def __init__(self, base_url: str = None, model: str = None, embedding_model: str = None):
        """
        Initializes the Ollama client.
        :param base_url: Ollama server URL (default: local Ollama instance)
        :param model: Default model to use for generating responses
        :param embedding_model: Model to use for embeddings (defaults to env var or all-minilm:l6-v2)
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("MODEL", "llama2")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "all-minilm:l6-v2")
        
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
                "   'I could not find relevant information in the provided context. Please provide additional details if needed.'\n"
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
        Checks if a model exists, and tries to use a default model if it doesn't.
        """
        try:
            # Check if model exists
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            models_data = response.json()
            # Handle different response formats
            models = []
            if "models" in models_data:
                models = models_data["models"]
            elif isinstance(models_data, list):
                models = models_data
                
            # Simplify model names for comparison
            available_models = [model["name"].split(":")[0] if ":" in model["name"] else model["name"] 
                               for model in models]
            model_name_simple = model_name.split(":")[0] if ":" in model_name else model_name
            
            # Check if our model exists (exact or base name match)
            if not (model_name in available_models or model_name_simple in available_models):
                print(f"Model {model_name} not available.")
                if models:
                    # Use first available model as fallback
                    self.model = models[0]["name"]
                    print(f"Using {self.model} as fallback.")
                else:
                    print(f"No models available. Will attempt to pull llama2 when needed.")
                    self.model = "llama2"
        except Exception as e:
            print(f"Error checking available models: {e}")
            print("Will use default model settings and let Ollama handle errors.")

    def generate_embedding(self, input_text: str) -> List[float]:
        """
        Generates an embedding for the given input text using the Ollama API.
        
        According to the Ollama API docs:
        POST /api/embed
        
        Returns a list of floats (the embedding vector).
        """
        # Use the dedicated embedding model
        model_name = self.embedding_model
        
        print(f"Generating embedding using Ollama embed API with model: {model_name}")
        
        # Format the request payload according to the API docs
        payload = {
            "model": model_name,
            "input": input_text
        }
        
        # Call the Ollama embed API
        response = requests.post(f"{self.base_url}/api/embed", json=payload)
        
        # Handle the response
        if response.status_code != 200:
            raise ValueError(f"Ollama embed API returned status code {response.status_code}: {response.text}")
            
        result = response.json()
        
        # Handle different response formats from Ollama
        # Some models return an "embedding" field with the vector directly
        if "embedding" in result:
            print("Successfully generated embedding from Ollama embed API (using 'embedding' field)")
            return result["embedding"]
        # Other models return an "embeddings" field with an array containing one or more vectors
        elif "embeddings" in result and len(result["embeddings"]) > 0:
            print("Successfully generated embedding from Ollama embed API (using 'embeddings' field)")
            return result["embeddings"][0]
        else:
            # For debugging, log the actual response format
            print(f"Unexpected response format from Ollama embed API: {result}")
            raise ValueError(f"Invalid response from Ollama embed API, missing both 'embedding' and 'embeddings' fields")

    def _remove_think_regions(self, text: str) -> str:
        """
        Removes `<think>...</think>` sections from the AI output.
        """
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()