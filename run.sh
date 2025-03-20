#!/bin/bash
# Run script for Document Processing API

# Default settings
USE_GPU=false
GPU_DEVICE=0  
GPU_LAYERS=35
GPU_COUNT=1
GPU_MODE="shared"
MODEL="llama2"
EMBEDDING_MODEL="all-minilm:l6-v2"

# Document chunking settings
ENABLE_CHUNKING=true
MAX_CHUNK_SIZE=1000
MIN_CHUNK_SIZE=200
CHUNK_OVERLAP=100

# Usage help
show_help() {
    echo "Usage: ./run.sh [OPTIONS]"
    echo "Start the Document Processing API with or without GPU support"
    echo ""
    echo "Options:"
    echo "  --gpu               Enable GPU support (default: disabled)"
    echo "  --gpu-device N      Use GPU device N (default: 0)"
    echo "  --gpu-layers N      Offload N layers to GPU (default: 35)"
    echo "  --gpu-count N       Use N GPUs (default: 1)"
    echo "  --gpu-mode MODE     Set GPU mode to 'shared' or 'exclusive' (default: shared)"
    echo "  --model NAME        Model to use for responses (default: llama2)"
    echo "  --embedding-model NAME Model to use for embeddings (default: all-minilm:l6-v2)"
    echo ""
    echo "  # Document Chunking Options"
    echo "  --no-chunking       Disable document chunking (default: enabled)"
    echo "  --chunk-size N      Maximum size of document chunks (default: 1000 chars)"
    echo "  --min-chunk-size N  Minimum size of document chunks (default: 200 chars)"
    echo "  --chunk-overlap N   Overlap between chunks (default: 100 chars)"
    echo ""
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh --gpu --gpu-device 0 --gpu-layers 35"
    echo "  ./run.sh --model mistral --embedding-model nomic-embed-text"
    echo "  ./run.sh --chunk-size 1500 --chunk-overlap 200"
    echo "  ./run.sh --no-chunking"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            USE_GPU=true
            shift
            ;;
        --gpu-device)
            GPU_DEVICE="$2"
            shift 2
            ;;
        --gpu-layers)
            GPU_LAYERS="$2"
            shift 2
            ;;
        --gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        --gpu-mode)
            GPU_MODE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --embedding-model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        # Document chunking options
        --no-chunking)
            ENABLE_CHUNKING=false
            shift
            ;;
        --chunk-size)
            MAX_CHUNK_SIZE="$2"
            shift 2
            ;;
        --min-chunk-size)
            MIN_CHUNK_SIZE="$2"
            shift 2
            ;;
        --chunk-overlap)
            CHUNK_OVERLAP="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Set environment variables for GPU if enabled
if [ "$USE_GPU" = true ]; then
    echo "Enabling GPU support"
    echo "   Device: $GPU_DEVICE"
    echo "   Layers: $GPU_LAYERS"
    echo "   Count:  $GPU_COUNT"
    echo "   Mode:   $GPU_MODE"
    
    # Export GPU environment variables
    export OLLAMA_GPU_DEVICES="$GPU_DEVICE"
    export OLLAMA_GPU_LAYERS="$GPU_LAYERS"
    export OLLAMA_GPU_COUNT="$GPU_COUNT"
    export OLLAMA_GPU_MODE="$GPU_MODE"
else
    echo "Running with CPU only"
    # Ensure GPU variables are unset or empty
    export OLLAMA_GPU_DEVICES=""
    export OLLAMA_GPU_LAYERS=0
    export OLLAMA_GPU_COUNT=0
    export OLLAMA_GPU_MODE=""
fi

# Export model variables
export MODEL="$MODEL"
export EMBEDDING_MODEL="$EMBEDDING_MODEL"
echo "Using models:"
echo "   Response model: $MODEL"
echo "   Embedding model: $EMBEDDING_MODEL"

# Export chunking settings
export ENABLE_CHUNKING="$ENABLE_CHUNKING"
export MAX_CHUNK_SIZE="$MAX_CHUNK_SIZE"
export MIN_CHUNK_SIZE="$MIN_CHUNK_SIZE"
export CHUNK_OVERLAP="$CHUNK_OVERLAP"
echo "Document chunking settings:"
echo "   Chunking enabled: $ENABLE_CHUNKING"
echo "   Max chunk size: $MAX_CHUNK_SIZE chars"
echo "   Min chunk size: $MIN_CHUNK_SIZE chars"
echo "   Chunk overlap: $CHUNK_OVERLAP chars"

# Check if docker-compose is running
echo "Checking if services are already running..."
if docker-compose ps | grep -q "Up"; then
    echo "Services are already running. Restarting..."
    docker-compose down
fi

# Start the services
echo "Starting services..."
docker-compose up -d

# Wait for services to initialize
echo "Waiting for services to initialize..."
sleep 5

# Show logs
echo "Showing logs (press Ctrl+C to exit logs, services will continue running)"
docker-compose logs -f