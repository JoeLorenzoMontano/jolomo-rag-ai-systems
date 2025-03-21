# Document Processing API with Web UI

A complete RAG (Retrieval-Augmented Generation) system that processes documents, stores their embeddings in ChromaDB, and generates AI responses based on the most relevant content. The system includes both an API backend and a web-based user interface.

## Architecture Overview

This application implements a full-stack Retrieval-Augmented Generation (RAG) pipeline with these key components:

- **API Backend** (FastAPI): Provides RESTful endpoints for document processing, embedding generation, vector storage, querying, and system health monitoring
- **Web Frontend** (Flask): Offers an intuitive interface for querying documents and visualizing results with source attribution
- **Vector Database** (ChromaDB): Persistent database that stores document embeddings and enables semantic similarity search
- **LLM Service** (Ollama): Local inference server that runs open-source LLMs without requiring cloud API access
  - Uses LLaMA2 (7B) for generating responses (configurable)
  - Uses all-minilm:l6-v2 for generating embeddings (configurable)
- **Document Chunking System**: Intelligently splits documents into semantic chunks with configurable parameters
- **Web Search Integration** (Optional): Supplements RAG results with internet-sourced information via Serper.dev API

## Setup and Usage

### Prerequisites

- **Docker Engine** (version 20.10.0 or higher)
- **Docker Compose** (version 2.0.0 or higher)
- At least 8GB of RAM available for Docker
- At least 10GB of free disk space
- Internet access for pulling container images and (optionally) web search functionality
- (Optional) NVIDIA GPU with compatible drivers for GPU acceleration

### System Requirements and Assumptions

This system makes the following assumptions:

1. **Document Format**: All documents to be processed must be in Markdown (.md) format and placed in the `rag-documents` directory
2. **Internet Access**: Initial setup requires internet access to pull Docker images and LLM models

### Quickstart Guide

1. Clone this repository
2. Navigate to the project directory
3. (Optional) Create a `.env` file for configuration:
   ```
   # For GPU support
   OLLAMA_GPU_DEVICES=all
   OLLAMA_GPU_COUNT=1
   OLLAMA_GPU_MODE=shared
   OLLAMA_GPU_LAYERS=32
   
   # For web search (optional)
   SERPER_API_KEY=your_serper_api_key_here
   ```
4. Start all services:
   ```bash
   docker-compose up -d
   ```
5. Wait for all services to initialize (this may take 5-10 minutes on first run as models are downloaded)
6. Access the web interface at http://localhost:5000
7. Process documents:
   - Go to http://localhost:5000/process and click "Process Documents", or
   - Use API directly: `curl -X POST http://localhost:8000/process`
8. Start querying your documents through the web UI or API

### Accessing the Services

- **Web UI**: http://localhost:5000
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **ChromaDB**: http://localhost:8001 (direct database access)
- **Ollama**: http://localhost:11434 (LLM server)

### Configuration Options

#### GPU Support

To enable GPU support for Ollama, set the following environment variables before running docker-compose:

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

#### Web Search API Integration

To enable the web search feature with Serper.dev API:

```bash
# Set the Serper API key
export SERPER_API_KEY=your_serper_api_key_here

# Then run docker-compose
docker-compose up -d
```

#### Web Search Integration

The application includes an optional web search feature that can augment the RAG results with information from the internet. This is particularly useful for queries that may require up-to-date information not available in your local document collection.

To use the web search feature:

1. Set the Serper API key as an environment variable:

```bash
# Add to your .env file or set directly
export SERPER_API_KEY=your_serper_api_key_here
```

2. Enable web search in your queries:

```bash
# Enable web search with 5 results
curl -X GET "http://localhost:8000/query?query=Your%20question&web_search=true&web_results_count=5"
```

3. In the UI, simply check the "Include Web Search Results" checkbox when making a query.

The web search results will be integrated into the context provided to the LLM, and will be displayed as additional sources in the UI.

Note: Web search requires an active internet connection and uses the Serper.dev API, which may have usage limits depending on your subscription.

#### Document Chunking

The application supports document chunking for improved retrieval performance. Chunking divides large documents into smaller pieces with some overlap, which helps make embeddings more focused and retrieval more accurate.

You can configure chunking with the following environment variables:

```bash
# Set chunking parameters
export ENABLE_CHUNKING=true     # Enable/disable chunking (default: true)
export MAX_CHUNK_SIZE=1000      # Maximum characters per chunk (default: 1000)
export MIN_CHUNK_SIZE=200       # Minimum size for a chunk (default: 200)
export CHUNK_OVERLAP=100        # Overlap between chunks (default: 100)

# Then run docker-compose
docker-compose up -d
```

You can also override chunking settings when processing documents through the API:

```bash
# Override chunking settings for a specific processing run
curl -X POST "http://localhost:8000/process?chunk_size=1500&min_size=300&overlap=150"

# Disable chunking for a specific run
curl -X POST "http://localhost:8000/process?enable_chunking=false"
```

### API Endpoints

- **POST /process**: Processes all documents in the `rag-documents` directory and stores their embeddings in ChromaDB
  - Optional query parameters:
    - `chunk_size`: Override max chunk size
    - `min_size`: Override min chunk size
    - `overlap`: Override chunk overlap
    - `enable_chunking`: Override chunking enabled setting (true/false)
  
- **GET /query?query=YOUR_QUERY**: Returns a response based on the most relevant documents matching your query
  - Optional query parameters:
    - `n_results`: Number of results to return (default: 3)
    - `combine_chunks`: Whether to combine chunks from the same document (default: true)
    - `web_search`: Whether to augment with web search results (default: false)
    - `web_results_count`: Number of web search results to include (default: 5)

- **GET /health**: Returns detailed health status of all services and components

- **POST /test-embedding?text=YOUR_TEXT**: Tests the embedding functionality with custom text (useful for debugging)

### Example Usage

1. Once all services are up, process the documents:

```bash
curl -X POST http://localhost:8000/process
```

2. Query the system:

```bash
curl -X GET "http://localhost:8000/query?query=What%20are%20the%20tenant%20configuration%20settings?"
```

## Project Structure

```
.
├── .env                            # Environment variables (create this file)
├── app/                            # API Backend (FastAPI)
│   ├── Dockerfile                  # Container definition for API
│   ├── document-processing-api.py  # Main FastAPI application
│   ├── requirements.txt            # Python dependencies
│   ├── startup.sh                  # API startup script
│   └── utils/                      # Utility modules
│       ├── ollama_client.py        # Ollama API client
│       └── web_search.py           # Web search integration
├── docker-compose.yml              # Container orchestration config
├── rag-documents/                  # Document collection to process
│   ├── *.md                        # Markdown documents
│   └── tenant/                     # Example subdirectory
├── ui/                             # Web Frontend (Flask)
│   ├── Dockerfile                  # Container definition for UI
│   ├── app.py                      # Main Flask application
│   ├── requirements.txt            # Python dependencies
│   ├── static/                     # Static assets
│   └── templates/                  # HTML templates
```

### Key Files

- **document-processing-api.py**: Core RAG implementation with document processing, embedding, and retrieval logic
- **docker-compose.yml**: Defines all services, networking, and volume configuration
- **ollama_client.py**: Handles communication with Ollama LLM service for embeddings and completions
- **web_search.py**: Implements the web search functionality using Serper.dev API
- **ui/app.py**: Flask application for the web interface

## Working with Documents

### Document Requirements

- Documents must be in Markdown (.md) format
- Place all documents in the `rag-documents` directory or its subdirectories 
- Documents will be recursively processed from all subdirectories
- File paths become document identifiers in the system

### Adding or Updating Documents

1. Add your markdown files to the `rag-documents` directory
2. Run the processing endpoint to index them:
   ```bash
   curl -X POST http://localhost:8000/process
   ```
   Or use the Process Documents page in the web UI

3. Verify documents were processed through the health endpoint:
   ```bash
   curl http://localhost:8000/health
   ```
   Check the `collection.document_count` field

### Document Chunking Strategy

The system splits documents into chunks using this strategy:

1. First splits by paragraph boundaries (double newlines)
2. For large paragraphs, further splits by sentence boundaries
3. Maintains overlap between chunks to preserve context
4. Respects minimum chunk size to avoid tiny fragments

This approach balances semantic coherence with vector retrieval efficiency.

## Troubleshooting

### Viewing Logs

```bash
# View logs from all services
docker-compose logs

# View logs from a specific service
docker-compose logs api
docker-compose logs ui
docker-compose logs ollama
docker-compose logs chromadb

# Follow logs in real-time
docker-compose logs -f
```

### Common Problems and Solutions

- **Startup Failures**: If services fail to start, check for port conflicts. The system requires ports 5000, 8000, 8001, and 11434.

- **Processing Hangs**: Document processing may time out with very large documents. Try disabling chunking or adjusting chunk parameters.

- **Out of Memory**: If you encounter OOM errors, increase Docker's memory allocation or reduce model layers.

- **System Reset**: To completely reset the system, remove all data and containers:
  ```bash
  docker-compose down -v
  docker-compose up -d
  ```

- **GPU Not Detected**: Check if your GPU is visible to Docker:
  ```bash
  docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
  ```

### Common Issues

#### Embedding API Response Format

The Ollama embedding API can return responses in different formats depending on the model:
- Some models return an `embedding` field with the vector directly
- Others (like all-minilm:l6-v2) return an `embeddings` field with an array of vectors

The application now handles both formats automatically. You can test embedding generation with:

```bash
# Test with the /test-embedding endpoint
curl -X POST "http://localhost:8000/test-embedding?text=This%20is%20a%20test"

# Or check the health endpoint for embedded status
curl http://localhost:8000/health
```

If you're experiencing embedding issues, you can also check Ollama's embed API directly:

```bash
# Direct test of the Ollama embedding API
curl -X POST http://localhost:11434/api/embed -d '{"model":"all-minilm:l6-v2","input":"test"}'
```

#### ChromaDB Connection Issues

If you encounter connection issues with ChromaDB, try:
1. Checking the ChromaDB logs: `docker logs chromadb`
2. Ensuring the correct ports are mapped in docker-compose.yml
3. Verifying there are no version conflicts with NumPy (should be <2.0.0)

#### GPU Support Compatibility

If GPU acceleration isn't working:
1. Verify your system has NVIDIA drivers installed
2. Check that the Docker NVIDIA runtime is installed
3. Ensure your GPU is compatible with the layers setting (try reducing OLLAMA_GPU_LAYERS)

#### Windows WSL2 GPU Support Considerations

If you're using Windows with Docker Desktop and WSL2, there are additional requirements for GPU support:

1. **NVIDIA GPU Driver for WSL**: You need to install the specific NVIDIA driver that supports WSL2 GPU passthrough
   - Download from [NVIDIA's WSL-compatible driver page](https://developer.nvidia.com/cuda/wsl)
   - Follow NVIDIA's installation instructions for WSL2 support

2. **WSL2 Configuration**: Ensure your WSL2 distribution is properly configured to access the GPU
   ```bash
   # Check if NVIDIA drivers are accessible from within WSL2
   nvidia-smi
   ```

3. **Potential Limitations**: Not all GPU features may be available through WSL2
   - Performance might be lower than on native Linux
   - Some CUDA features might not work as expected
   - Newer GPUs typically have better support

4. **Troubleshooting WSL2 GPU Issues**:
   - Ensure your Windows host has the latest Windows Updates
   - Update Docker Desktop to the latest version
   - Make sure WSL2 is set as the default WSL version: `wsl --set-default-version 2`
   - Check GPU is visible inside WSL2: `wsl --distribution <your-distro> --exec nvidia-smi`

## Advanced Usage

### Custom Prompting

The system uses a specific prompt template for querying, which you can view in the `ollama_client.py` file. The prompt includes instructions to:

1. Only use information from the provided context
2. Not generate answers from external knowledge
3. Admit when information is not available in the context

This helps ensure that responses are grounded in your document collection.

### Performance Tuning

- **RAM Usage**: Adjust the number of workers in startup.sh files if needed
- **Inference Speed**: GPU acceleration provides 5-10x speedup for compatible hardware
- **Vector DB Performance**: ChromaDB scales well to thousands of documents
- **Collection Size**: The system can handle medium-sized document collections (hundreds of files)

### LLM Response Formatting

The LLM response will attempt to:

1. Synthesize information from matched documents
2. Maintain consistent formatting with the source material
3. Include relevant details while avoiding hallucinations
4. Generate concise, focused answers to queries

## Limitations

- **Content Types**: Currently only processes text in Markdown format
- **Language Support**: Best performance with English language content
- **Token Limits**: There are context size limits (typically ~4000 tokens)
- **Response Quality**: Depends on the quality and relevance of the document collection
- **Inference Speed**: Local models are slower than cloud-based alternatives
- **No Authentication**: The system has no built-in security features and should not be exposed publicly
- **Web Search Limitations**: Web search relies on the Serper.dev API which may have rate limits or costs