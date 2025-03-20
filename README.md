# Document Processing API

A RAG (Retrieval-Augmented Generation) application that processes documents, stores their embeddings in ChromaDB, and uses Ollama to generate responses to user queries based on the most relevant documents.

## Components

- **FastAPI**: Web server for document processing and querying
- **ChromaDB**: Vector database for storing document embeddings
- **Ollama**: Local LLM service for generating embeddings and responses

## Setup and Usage

### Prerequisites

- Docker and Docker Compose

### Running the Application

#### Option 1: Using docker-compose directly

1. Clone this repository
2. Navigate to the project directory
3. Run:

```bash
docker-compose up -d
```

4. Wait for all services to start (this may take some time as Ollama downloads the necessary models)

#### Option 2: Using the run script

For convenience, a run script is provided that simplifies starting the application with or without GPU support:

```bash
# Run with CPU only
./run.sh

# Run with GPU support
./run.sh --gpu

# Run with custom GPU settings
./run.sh --gpu --gpu-device 0 --gpu-layers 35 --gpu-count 1
```

For all available options, run:

```bash
./run.sh --help
```

### Configuration Options

#### GPU Support

To enable GPU support for Ollama, you can either use the run script as shown above, or set the following environment variables before running docker-compose:

```bash
# Enable GPU support
export OLLAMA_GPU_DEVICES=0        # GPU devices to use (e.g., "0" or "0,1")
export OLLAMA_GPU_LAYERS=35        # Number of layers to offload to GPU (higher = more GPU utilization)
export OLLAMA_GPU_COUNT=1          # Number of GPUs to use
export OLLAMA_GPU_MODE="shared"    # GPU mode ("shared" or "exclusive")

# Then run docker-compose
docker-compose up -d
```

The GPU settings will be ignored if you don't set these variables, allowing the application to run on CPU only.

#### Models

The application uses two different models:
- `MODEL` (default: llama2) - Used for generating responses to queries
- `EMBEDDING_MODEL` (default: all-minilm:l6-v2) - Specialized model for generating embeddings

You can override these defaults by setting environment variables:

```bash
# Use specific models
export MODEL=mistral               # For generating responses
export EMBEDDING_MODEL=nomic-embed-text  # For generating embeddings

# Then run docker-compose
docker-compose up -d
```

### API Endpoints

- **POST /process**: Processes all documents in the `rag-documents` directory and stores their embeddings in ChromaDB
- **GET /query?query=YOUR_QUERY**: Returns a response based on the most relevant documents matching your query

### Example Usage

1. Once all services are up, process the documents:

```bash
curl -X POST http://localhost:8000/process
```

2. Query the system:

```bash
curl -X GET "http://localhost:8000/query?query=What%20are%20the%20tenant%20configuration%20settings?"
```

## Directory Structure

- `/app`: Contains the FastAPI application code
- `/rag-documents`: Contains markdown documents to be processed
- `/docker-compose.yml`: Docker Compose configuration for all services

## Troubleshooting

- If you encounter issues with the Ollama service, check the logs with `docker logs ollama-server`
- If you encounter issues with ChromaDB, check the logs with `docker logs chromadb`
- If you need to reset the system, you can run `docker-compose down -v` to remove all volumes and containers, then `docker-compose up -d` to start fresh