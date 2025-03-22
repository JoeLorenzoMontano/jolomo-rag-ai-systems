# Document Processing API with Web UI

> **⚠️ IMPORTANT FOR REVIEWERS**: 
>
> 1. **Docker Compose Options**:
>    - **Option A (Recommended)**: `docker-compose up -d` - Uses Ollama installed on your host machine
>    - **Option B (Slower)**: `docker-compose -f docker-compose.docker.ollama.yml up -d` - Runs Ollama in a container
>
>    **Why Option A is recommended**: Running Ollama in a container with CPU-only is noticeably slow, with responses taking 15-30+ seconds. With Ollama installed directly on your host machine, you'll get significantly faster responses (3-8 seconds) and easier GPU acceleration if available. See the ["Ollama Setup Options"](#ollama-setup-options) section for installation instructions.
>
> 2. **Pre-populated Database**: The ChromaDB database comes pre-populated with embeddings for all the documents in the `rag-documents` directory, so you can start querying immediately. If you wish to re-process the documents with different chunking settings, you can clear the database and re-process them through the System Info page in the UI or using the `/clear-db` and `/process` API endpoints.

A complete RAG (Retrieval-Augmented Generation) system that processes documents, stores their embeddings in ChromaDB, and generates AI responses based on the most relevant content. The system includes both an API backend and a web-based user interface.

## Setup and Usage

### Prerequisites and Requirements

**Required:**
- **Docker Engine** and **Docker Compose**
- Internet access (for pulling container images and models)

**Performance Options:**
- **Option 1 (Recommended):** Install **Ollama** on your host machine and use the default docker-compose file
- **Option 2:** Use the containerized Ollama setup with `docker-compose.docker.ollama.yml` (slower, but no host installation required)

**Optional:**
- NVIDIA GPU with compatible drivers (for faster inference)
- Serper.dev API key (for web search integration)

**Note:** Sample documents in Markdown format are already included in the `rag-documents` directory with pre-populated embeddings.

### Quickstart Guide

1. Clone this repository
2. Navigate to the project directory
3. (Optional) Create a `.env` file for configuration (or copy from `.env.sample`):
   ```
   # Ollama Model Settings
   MODEL=llama3:latest
   EMBEDDING_MODEL=all-minilm:l6-v2
   
   # Web Search Integration 
   SERPER_API_KEY=your_serper_api_key_here
   ```
4. Start all services:
   ```bash
   docker-compose up -d
   ```
5. Wait for all services to initialize (this may take a few minutes on first run due to the models needing to be downloaded)
6. Access the web interface at http://localhost:5000
7. Start querying immediately! ChromaDB comes pre-populated with embeddings for all documents in the `rag-documents` directory
8. (Optional) If you want to experiment with different chunking settings:
   - Navigate to System Info in the UI
   - Click "Clear Database" to remove all existing embeddings
   - Go to the Process Documents page to re-process the documents with your preferred settings

### Accessing the Services

- **Web UI**: http://localhost:5000
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **ChromaDB**: http://localhost:8001 (direct database access)
- **Ollama**: http://localhost:11434 (LLM server)
### API Documentation

For detailed information about all available API endpoints and their parameters, please visit the OpenAPI documentation at:

**API Documentation**: http://localhost:8000/docs

## Ollama Setup Options

### Option 1: Host Ollama Setup (Recommended)

For optimal performance, we recommend installing Ollama directly on your host machine:

1. Install Ollama by following the [official installation guide](https://ollama.com/download)
2. Start Ollama on your host (it should automatically run as a service)
3. Use the default docker-compose file, which connects to your host's Ollama:
   ```bash
   docker-compose up -d
   ```

**Benefits**:
- Significantly faster response times (3-8 seconds vs 15-30+ seconds)
- Automatic GPU acceleration if available on your system
- More stable performance for large language models

### Option 2: Containerized Ollama

If you prefer not to install software on your host, you can run Ollama in a container:

```bash
docker-compose -f docker-compose.docker.ollama.yml up -d
```

**Important**: This approach is significantly slower, especially on CPU-only machines, and is not recommended for reviewers who want to experience the system's optimal performance.

## Key Features

- **Smart Document Chunking**: Automatically splits documents into semantic chunks for optimal retrieval
- **Query Classification**: Intelligently determines when to use documents vs. web search
- **Conversation Memory**: Maintains context for natural follow-up questions
- **Web Search Integration**: Optional integration with internet search for questions outside your document scope
