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

### Prerequisites and Requirements

- **Docker Engine** and **Docker Compose**
- Internet access for pulling container images, LLM models, and (optionally) web search functionality
- (Optional) NVIDIA GPU with compatible drivers for GPU acceleration
- Documents must be in Markdown (.md) format and placed in the `rag-documents` directory

### Quickstart Guide

1. Clone this repository
2. Navigate to the project directory
3. (Optional) Create a `.env` file for configuration:
   ```
   # Ollama Settings
   MODEL=llama2:latest
   EMBEDDING_MODEL=all-minilm:l6-v2
   
   # Document chunking settings
   ENABLE_CHUNKING=true
   MAX_CHUNK_SIZE=1000
   MIN_CHUNK_SIZE=200
   CHUNK_OVERLAP=100
   
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
5. Wait for all services to initialize (this may take a few minutes on first run due to the models needing to be downloaded)
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
    - `web_search`: Whether to augment with web search results (null for auto-classification, true or false to explicitly set)
    - `web_results_count`: Number of web search results to include (default: 5)
    - `explain_classification`: Whether to include query classification explanation (default: false)

- **GET /health**: Returns detailed health status of all services and components

- **GET /terms**: Lists all domain-specific terms currently used by the query classifier

- **POST /refresh-terms**: Manually refreshes the domain-specific terms by extracting terminology from the current document collection

### Using Host Machine's Ollama

If you already have Ollama running on your host machine, you can use it instead of running Ollama in a container by using the dedicated docker-compose file:

1. Make sure Ollama is installed and running on your host machine
2. Create or edit your `.env` file with your LLM/embedding model settings
3. Start the services with: 

```bash
docker-compose -f docker-compose.host.yml up -d
```

This setup directs API requests to your host's Ollama installation and doesn't run a containerized Ollama. Benefits include:

- Uses your existing Ollama models (no need to re-download them)
- Leverages your host's GPU setup
- Easier management of Ollama models (pull, list, remove from host)
- Lower container resource usage

For macOS/Windows, the host Ollama will be accessed at `host.docker.internal:11434`.
For Linux, the special `host-gateway` setting enables access to the host machine.

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
- **query_classifier.py**: Intelligently determines when to use document retrieval vs. web search
- **document_processor.py**: Handles document chunking and preparation strategies
- **ui/app.py**: Flask application for the web interface

## Working with Documents

- Documents must be in Markdown (.md) format
- Place all documents in the `rag-documents` directory or its subdirectories
- Documents will be recursively processed from all subdirectories
- File paths become document identifiers in the system
- Process documents using the web UI or API endpoint

### Document Chunking Strategy

The system intelligently splits documents into chunks for optimal retrieval:

1. First splits by paragraph boundaries (double newlines)
2. For large paragraphs, further splits by sentence boundaries
3. Maintains overlap between chunks to preserve context
4. Respects minimum chunk size to avoid tiny fragments

This approach balances semantic coherence with efficient vector retrieval, leading to more relevant search results.

### Query Classification System

The system uses an intelligent classification mechanism to determine the optimal information source:

1. **Dynamic Domain Term Extraction**: Extracts important terminology from the document corpus
   - Analyzes document frequency and significance
   - Combines with predefined domain terms
   - Automatically updates when new documents are processed
   - Extracts both single terms and multi-word phrases

2. **Source Selection Logic**:
   - **Document Source**: Used when query contains domain-specific terminology or matches existing content well
   - **Web Search Source**: Used when query contains general knowledge questions outside document scope
   - **Hybrid Approach**: Used when confidence is moderate and both sources may contribute

3. **Classification Visualization**:
   - Toggle "Show Classification Details" to see how your query was classified
   - Displays matched domain terms and confidence scores
   - Shows a visual breakdown of the classification decision

The classifier automatically refreshes its domain term knowledge when documents are processed, ensuring it stays up-to-date with the content collection. You can also manually refresh terms or view the current term list using the dedicated API endpoints.

## Limitations

- **Content Types**: Currently only processes text in Markdown format
- **Language Support**: Best performance with English language content
- **Token Limits**: There are context size limits (typically ~4000 tokens)
- **Response Quality**: Depends on the quality and relevance of the document collection
- **Inference Speed**: Local models are slower than cloud-based alternatives
- **No Authentication**: The system has no built-in security features and should not be exposed publicly
- **Web Search Limitations**: Web search relies on the Serper.dev API which may have rate limits or costs
