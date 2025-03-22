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
    
    def enhance_query(self, original_query: str) -> str:
        """
        Enhances the original query to improve retrieval performance.
        
        This method expands the query, adds synonyms, removes contractions, 
        and normalizes possessives to increase the chance of matching relevant documents.
        
        Args:
            original_query: The user's original query text
            
        Returns:
            Enhanced query text optimized for retrieval
        """
        if not original_query.strip():
            return original_query
            
        try:
            # Use a condensed prompt to save tokens
            prompt = f"""Enhance this query for better document retrieval results:
            "{original_query}"
            
            Follow these rules:
            1. Expand acronyms and abbreviations
            2. Include alternative terms for key concepts
            3. Remove possessives (e.g., change "Joe's" to "Joe")
            4. Normalize contractions (e.g., change "don't" to "do not")
            5. Identify implied questions that aren't directly stated
            6. Keep the enhanced query concise (max 3 sentences)
            7. Maintain all important search terms from the original
            8. Format as a single, more effective search query

            Respond with ONLY the enhanced query text. No explanations.
            """
            
            # Use a fast response with the current model
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}  # Low temperature for deterministic results
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            enhanced_query = self._remove_think_regions(result.get("response", original_query))
            
            # If something went wrong or response is empty, return original
            if not enhanced_query or len(enhanced_query) < 3:
                return original_query
                
            # Remove quotes if the model included them
            enhanced_query = enhanced_query.strip('"\'')
            
            return enhanced_query
        except Exception as e:
            print(f"Error enhancing query: {e}")
            # Fall back to original query if enhancement fails
            return original_query
        
    def generate_semantic_enrichment(self, chunk_text: str, chunk_id: str, prev_chunk_text: str = None, next_chunk_text: str = None) -> str:
        """
        Generates a semantic enrichment for a document chunk to improve retrieval quality.
        This generates a contextual summary that captures the meaning of the chunk by considering
        surrounding context when available.
        
        Args:
            chunk_text: The original text chunk to enhance
            chunk_id: Identifier for the chunk (used for logging/debugging)
            prev_chunk_text: Text from the previous chunk (if available)
            next_chunk_text: Text from the next chunk (if available)
            
        Returns:
            Contextual summary that can be added to the original for better matching
        """
        # Check if the model exists, if not pull it
        current_model = self.model
        self._ensure_model_exists(current_model)
        
        # Set up context sections based on what's available
        context_parts = []
        if prev_chunk_text:
            # Include a shortened version of previous chunk for context
            prev_context = prev_chunk_text[:500] + "..." if len(prev_chunk_text) > 500 else prev_chunk_text
            context_parts.append(f"PREVIOUS CHUNK:\n{prev_context}")
            
        if next_chunk_text:
            # Include a shortened version of next chunk for context
            next_context = next_chunk_text[:500] + "..." if len(next_chunk_text) > 500 else next_chunk_text
            context_parts.append(f"NEXT CHUNK:\n{next_context}")
            
        # Join context sections if any exist
        context_section = "\n\n".join(context_parts)
        context_instruction = ""
        
        if context_section:
            context_instruction = f"""
            I'm providing surrounding chunks to help you understand the content's flow, but focus your summary on the MAIN CHUNK.

            {context_section}
            """
        
        # Improved prompt for more useful and focused summaries
        prompt = f"""You are an expert document analyzer that extracts key information for semantic search systems.

        Your task is to create a precise summary that will help match this content to relevant queries.

        {context_instruction}

        MAIN CHUNK:
        ```
        {chunk_text}
        ```

        Create a concise summary (2-3 sentences) that:
        1. Highlights the core subject matter and key points in the chunk
        2. Includes specific terminology, entities, and technical concepts present in the text
        3. Uses domain-specific vocabulary that would appear in related search queries
        4. Describes what problems or questions this content helps solve
        5. Avoids filler phrases like "this section describes" or "this document covers"
        
        Important guidelines:
        - Focus on factual content, not document structure
        - Start directly with the key information, not introductory phrases
        - Include specific technical terms, not general descriptions
        - Use natural language without bullet points or headings
        - Maintain technical accuracy
        """

        payload = {
            "model": current_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Use low temperature for consistent, focused output
                "num_predict": 300   # Limit to ~300 tokens for concise summary
            }
        }

        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            enrichment = result.get("response", "").strip()
            
            # Process the enrichment to clean it up
            processed_enrichment = self._process_contextual_summary(enrichment)
            
            print(f"Generated contextual summary for chunk {chunk_id} ({len(processed_enrichment)} chars)")
            return processed_enrichment
            
        except Exception as e:
            print(f"Error generating contextual summary for chunk {chunk_id}: {e}")
            return ""  # Return empty string on error
    
    def _process_contextual_summary(self, summary: str) -> str:
        """
        Processes the generated summary to ensure it's clean and useful for retrieval.
        """
        # Remove any code block syntax that might have been generated
        summary = summary.replace("```", "")
        
        # Remove any thinking indicators
        summary = self._remove_think_regions(summary)
        
        # Remove any common prefixes that might have been generated
        common_prefixes = [
            r'^SUMMARY:\s*',
            r'^KEY POINTS:\s*',
            r'^MAIN CONTENT:\s*',
            r'^OVERVIEW:\s*',
            r'^DESCRIPTION:\s*'
        ]
        for prefix_pattern in common_prefixes:
            summary = re.sub(prefix_pattern, '', summary, flags=re.IGNORECASE)
        
        # Remove any markdown or heading syntax
        summary = re.sub(r'^#+\s+', '', summary, re.MULTILINE)
        
        # Convert bullet points to regular text if present
        summary = re.sub(r'^\s*[-*•]\s+', '', summary, flags=re.MULTILINE)
        
        # Remove filler phrases that might have been generated
        filler_start_phrases = [
            r'^This section describes\s+',
            r'^This document covers\s+',
            r'^This part explains\s+',
            r'^This content details\s+'
        ]
        for phrase_pattern in filler_start_phrases:
            summary = re.sub(phrase_pattern, '', summary, flags=re.IGNORECASE)
        
        # If the summary still starts with "This section" or similar, replace it with something more useful
        summary = re.sub(r'^(This section|This document|This text|This content)\s+', '', summary, flags=re.IGNORECASE)
        
        # Add a prefix to make it clear this is a summary (but one that won't pollute term extraction)
        summary = "SEMANTIC CONTEXT: " + summary
                
        return summary.strip()