#!/bin/bash
# Startup script to ensure services are running properly

echo "Starting Document Processing API startup check..."

# Wait for ChromaDB to be ready
echo "Waiting for ChromaDB..."
for i in {1..30}; do
  if curl -s "http://${CHROMA_HOST:-chromadb}:${CHROMA_PORT:-8000}/api/v1/heartbeat" > /dev/null; then
    echo "ChromaDB is ready!"
    break
  fi
  
  if [ $i -eq 30 ]; then
    echo "ChromaDB did not start in time. Continuing anyway..."
  fi
  
  echo "Waiting for ChromaDB... (attempt $i/30)"
  sleep 1
done

# Wait for Ollama to be ready
echo "Waiting for Ollama..."
for i in {1..30}; do
  if curl -s "${OLLAMA_BASE_URL:-http://ollama:11434}/api/tags" > /dev/null; then
    echo "Ollama is ready!"
    break
  fi
  
  if [ $i -eq 30 ]; then
    echo "Ollama did not start in time. Continuing anyway..."
  fi
  
  echo "Waiting for Ollama... (attempt $i/30)"
  sleep 1
done

echo "Network check:"
echo "  - Ollama host: ${OLLAMA_BASE_URL:-http://ollama:11434}"
echo "  - ChromaDB host: ${CHROMA_HOST:-chromadb}:${CHROMA_PORT:-8000}"
echo "  - Ping test to Ollama:"
ping -c 1 ollama || echo "Cannot ping Ollama"
echo "  - Ping test to ChromaDB:"
ping -c 1 chromadb || echo "Cannot ping ChromaDB"

echo "All checks complete. Starting the API..."

# Start the actual application
exec uvicorn document-processing-api:app --host 0.0.0.0 --port 8000 --reload