#!/bin/bash
# Startup script with Ollama model check

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

# Wait for Ollama to be fully available
echo "Waiting for Ollama to be ready..."
for i in {1..30}; do
  if curl -s "${OLLAMA_BASE_URL}/api/tags" > /dev/null; then
    echo "Ollama is ready!"
    break
  fi
  echo "Ollama not ready yet, waiting..."
  sleep 2
  if [ $i -eq 30 ]; then
    echo "Timed out waiting for Ollama"
  fi
done

# Check available Ollama models
echo "Checking Ollama models..."
curl -s "${OLLAMA_BASE_URL}/api/tags"

# Define embedding model
EMBEDDING_MODEL=${EMBEDDING_MODEL:-"all-minilm:l6-v2"}

# Model download code commented out - models should be pre-installed
# This significantly reduces container startup time and build size

# # Make sure our main model is available
# echo "Ensuring main model (${MODEL}) is available..."
# curl -s -X POST "${OLLAMA_BASE_URL}/api/pull" -d "{\"name\":\"${MODEL}\"}"
# 
# # Make sure our embedding model is available
# echo "Ensuring embedding model (${EMBEDDING_MODEL}) is available..."
# curl -s -X POST "${OLLAMA_BASE_URL}/api/pull" -d "{\"name\":\"${EMBEDDING_MODEL}\"}"

echo "Skipping model downloads - using pre-installed models"

# Test embed endpoint with the embedding model
echo "Testing /api/embed endpoint with ${EMBEDDING_MODEL}:"
EMBED_TEST=$(curl -s -X POST "${OLLAMA_BASE_URL}/api/embed" -d "{\"model\":\"${EMBEDDING_MODEL}\",\"input\":\"test\"}")
if echo "$EMBED_TEST" | grep -q "embedding"; then
  echo "Embedding API working correctly! (using 'embedding' field)"
elif echo "$EMBED_TEST" | grep -q "embeddings"; then
  echo "Embedding API working correctly! (using 'embeddings' field)"
else
  echo "WARNING: Embedding API test failed. Response:"
  echo "$EMBED_TEST"
  # For better debugging, check the format of the response
  echo "Response fields: $(echo "$EMBED_TEST" | python3 -c "import sys, json; print(list(json.load(sys.stdin).keys()))")"
fi

# Test generate endpoint
echo "Testing /api/generate endpoint with ${MODEL}:"
curl -s -X POST "${OLLAMA_BASE_URL}/api/generate" -d "{\"model\":\"${MODEL}\",\"prompt\":\"test\"}" | head -c 100
echo

# Start the actual application
echo "Starting the API with extended timeouts..."
exec uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --timeout-keep-alive 300 \
  --log-level info \
  --workers 4