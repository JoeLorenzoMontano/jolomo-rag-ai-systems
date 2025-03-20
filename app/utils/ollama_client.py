import requests
import json
import re
from typing import List, Dict, Optional

OPEN_WEBUI_API_KEY = "sk-0e21bac8a018452d92fad8127578d214"#os.getenv("OPEN_WEBUI_API_KEY")

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:3001", model: str = "deepseek-r1:8b"):
        """
        Initializes the Ollama client.
        :param base_url: Ollama server URL (default: local Ollama instance)
        :param model: Default model to use for generating responses
        """
        self.base_url = base_url
        self.model = model

    def generate_response(self, context: str, query: str, model: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        """
        Sends a query to the Ollama server with strict response rules.
        """
        payload = {
            "model": model or self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that must follow strict response rules.\n"
                        "### 🔹 **Rules (MUST FOLLOW):**\n"
                        "1 **You MUST ONLY use information from the provided context.**\n"
                        "2 **If the answer is NOT in the context, respond with:**\n"
                        "   ❌ 'I could not find relevant information in the provided context. Please provide additional details if needed.'\n"
                        "3 **You MUST NOT generate an answer using external knowledge.**\n"
                        "4 **You MUST NOT make up any information.**\n\n"
                        "### 🔹 **Context (ONLY use the information provided below to answer the query):**\n"
                        f"\"\"\"\n{context}\n\"\"\"\n\n"
                    ),
                },
                {"role": "user", "content": f"Query: {query}"},
            ],
            "stream": False,
            "max_tokens": max_tokens,
        }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPEN_WEBUI_API_KEY}"
        }

        response = requests.post(f"{self.base_url}/api/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        return self._remove_think_regions(result.get("choices", [{}])[0].get("message", {}).get("content", "No response generated."))

    def summarize_text(self, text: str, context: str, model: Optional[str] = "mistral:7b", max_tokens: Optional[int] = None) -> str:
        """
        Summarizes the provided text following strict summarization rules.
        """
        if not text.strip():
            return "Error: No input text provided."

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant specializing in summarization.\n"
                        "### 🔹 **Summarization Rules:**\n"
                        "1. **Preserve all key details.**\n"
                        "2. **Include all relevant facts, quotes, and statistics.**\n"
                        "3. **Do NOT change the meaning of the original text.**\n"
                        "4. **Do NOT add opinions or remove crucial context.**\n\n"
                        "### 🔹 **Context (Previous Chunks):**\n"
                        f"\"\"\"\n{context}\n\"\"\"\n\n"
                        "### 🔹 **Text to Summarize:**\n"
                        f"\"\"\"\n{text}\n\"\"\""
                    ),
                },
                {"role": "user", "content": "Provide a concise yet detailed summary."},
            ],
            "max_tokens": max_tokens,
        }

        response = requests.post(f"{self.base_url}/api/chat/completions", json=payload)
        response.raise_for_status()
        result = response.json()
        return self._remove_think_regions(result.get("choices", [{}])[0].get("message", {}).get("content", "No summary generated."))

    def generate_embedding(self, input_text: str) -> List[float]:
        """
        Generates an embedding for the given input text.
        """
        payload = {
            "model": "all-minilm:l6-v2",
            "input": [input_text],
        }

        response = requests.post(f"{self.base_url}/api/embeddings", json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("data", [{}])[0].get("embedding", [])

    def extract_metadata(self, text: str, model: str = "mistral:7b", max_tokens: Optional[int] = None) -> Dict[str, str]:
        """
        Extracts structured metadata from the text following a strict JSON schema.
        """
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that extracts structured metadata from text."
                        "You must follow the JSON schema strictly without adding explanations or extra formatting."
                    ),
                },
                {"role": "user", "content": f"Extract metadata from this document:\n\n\"\"\"\n{text}\n\"\"\""},
            ],
            "max_tokens": max_tokens,
        }

        response = requests.post(f"{self.base_url}/api/chat/completions", json=payload)
        response.raise_for_status()
        result = response.json()
        return self._normalize_metadata(json.loads(result.get("choices", [{}])[0].get("message", {}).get("content", "{}")))

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