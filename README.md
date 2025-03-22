# Document Processing API with Web UI

> **⚠️ IMPORTANT FOR REVIEWERS**: Running Ollama in a container with CPU-only will be noticeably slow. For better performance, install Ollama directly on your host machine and use the default `docker-compose up -d` command, which is already configured to use your host's Ollama installation. Host machine installation typically provides easier GPU access and significantly faster response times. See the ["Using Host Machine's Ollama"](#using-host-machines-ollama) section for details.

A complete RAG (Retrieval-Augmented Generation) system that processes documents, stores their embeddings in ChromaDB, and generates AI responses based on the most relevant content. The system includes both an API backend and a web-based user interface.

## Architecture Overview

This application implements a full-stack Retrieval-Augmented Generation (RAG) pipeline with these key components:

- **API Backend** (FastAPI): Provides RESTful endpoints for document processing, embedding generation, vector storage, querying, and system health monitoring
- **Web Frontend** (Flask): Offers both a simple chat interface with conversation memory and an advanced query interface with detailed source attribution
- **Vector Database** (ChromaDB): Persistent database that stores document embeddings and enables semantic similarity search
- **LLM Service** (Ollama): Local inference server that runs open-source LLMs without requiring cloud API access
  - Uses LLaMA2 (7B) for generating responses (configurable)
  - Uses all-minilm:l6-v2 for generating embeddings (configurable)
- **Document Chunking System**: Intelligently splits documents into semantic chunks with configurable parameters
- **Web Search Integration** (Optional): Supplements RAG results with internet-sourced information via Serper.dev API

## Setup and Usage

### Prerequisites and Requirements

- **Docker Engine** and **Docker Compose**
- **Ollama** installed on your host machine (recommended for better performance)
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
### API Documentation

For detailed information about all available API endpoints and their parameters, please visit the OpenAPI documentation at:

**API Documentation**: http://localhost:8000/docs

### Using Host Machine's Ollama

The default `docker-compose.yml` is already configured to use Ollama from your host machine:

1. Make sure Ollama is installed and running on your host machine
2. Create or edit your `.env` file with your LLM/embedding model settings
3. Start the services with the default command: 

```bash
docker-compose up -d
```

This setup directs API requests to your host's Ollama installation and doesn't run a containerized Ollama. Benefits include:

- Uses your existing Ollama models (no need to re-download them)
- Leverages your host's GPU setup
- Easier management of Ollama models (pull, list, remove from host)
- Lower container resource usage
- **Significantly faster inference speeds**

For macOS/Windows, the host Ollama will be accessed at `host.docker.internal:11434`.
For Linux, the special `host-gateway` setting enables access to the host machine.

If you want to run Ollama within Docker instead of using the host machine installation, use:

```bash
docker-compose -f docker-compose.docker.ollama.yml up -d
```

Note that running Ollama in a container may be significantly slower, especially on CPU-only machines.

## Working with Documents

- Place documents in Markdown (.md) format in the `rag-documents` directory
- Process documents using the Web UI or API endpoint
- Query your documents through the chat interface or advanced query page

## Key Features

- **Smart Document Chunking**: Automatically splits documents into semantic chunks for optimal retrieval
- **Query Classification**: Intelligently determines when to use documents vs. web search
- **Conversation Memory**: Maintains context for natural follow-up questions
- **Web Search Integration**: Optional integration with internet search for questions outside your document scope
