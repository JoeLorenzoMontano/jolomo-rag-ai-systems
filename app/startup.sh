#!/bin/bash
# Simple startup script for debugging

echo "Starting Document Processing API..."
echo "Environment variables:"
echo "OLLAMA_BASE_URL: ${OLLAMA_BASE_URL}"
echo "CHROMA_HOST: ${CHROMA_HOST}"
echo "CHROMA_PORT: ${CHROMA_PORT}"
echo "MODEL: ${MODEL}"

# Try to ping the services
echo "Network check:"
echo "Trying to reach Ollama..."
ping -c 1 ollama || echo "Cannot ping Ollama"
echo "Trying to reach ChromaDB..."
ping -c 1 chromadb || echo "Cannot ping ChromaDB"

# Start the actual application
echo "Starting the API..."
exec uvicorn document-processing-api:app --host 0.0.0.0 --port 8000 --reload