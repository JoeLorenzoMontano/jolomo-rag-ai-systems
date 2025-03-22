# Document Processing API with Web UI

> **⚠️ IMPORTANT FOR REVIEWERS**: 
>
> 1. Running Ollama in a container with CPU-only will be noticeably slow. For better performance, install Ollama directly on your host machine and use the default `docker-compose up -d` command, which is already configured to use your host's Ollama installation. Host machine installation typically provides easier GPU access and significantly faster response times. See the ["Host Ollama Setup"](#host-ollama-setup) section for details.
>
> 2. The ChromaDB database comes pre-populated with embeddings for all the documents in the `rag-documents` directory, so you can start querying immediately. If you wish to re-process the documents with different chunking settings, you can clear the database and re-process them through the System Info page in the UI or using the `/clear-db` and `/process` API endpoints.

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

### Alternative: Containerized Ollama

If you prefer not to install Ollama on your host machine, you can use the containerized version instead:

```bash
docker-compose -f docker-compose.docker.ollama.yml up -d
```

**Note**: This alternative approach is typically significantly slower, especially on CPU-only machines, and is not recommended for the review process.
## Key Features

- **Smart Document Chunking**: Automatically splits documents into semantic chunks for optimal retrieval
- **Query Classification**: Intelligently determines when to use documents vs. web search
- **Conversation Memory**: Maintains context for natural follow-up questions
- **Web Search Integration**: Optional integration with internet search for questions outside your document scope

## Host Ollama Setup

For optimal performance, especially on machines with NVIDIA GPUs, it's recommended to install Ollama directly on your host machine:

1. Install Ollama on your host machine by following the [official installation guide](https://ollama.com/download)
2. Start Ollama on your host (it should automatically run as a service)
3. Use the default docker-compose file, which maps port 11434 from your host to the containers:
   ```bash
   docker-compose up -d
   ```

This configuration allows the containerized API to communicate with your host's Ollama instance, providing significantly faster inference, especially when using GPU acceleration.
