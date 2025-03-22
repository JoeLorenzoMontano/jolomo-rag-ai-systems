# Document Processing API with Web UI

> **⚠️ IMPORTANT FOR REVIEWERS**: Running Ollama in a container with CPU-only will be noticeably slow. For better performance, install Ollama directly on your host machine and use the default `docker-compose up -d` command, which is already configured to use your host's Ollama installation. Host machine installation typically provides easier GPU access and significantly faster response times. See the ["Using Host Machine's Ollama"](#using-host-machines-ollama) section for details.

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
3. (Optional) Create a `.env` file for configuration:
   ```
   # Ollama Settings
   MODEL=llama3:latest
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
7. Start querying immediately! ChromaDB comes pre-populated with embeddings for the documents in the `rag-documents` directory

### Accessing the Services

- **Web UI**: http://localhost:5000
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **ChromaDB**: http://localhost:8001 (direct database access)
- **Ollama**: http://localhost:11434 (LLM server)
### API Documentation

For detailed information about all available API endpoints and their parameters, please visit the OpenAPI documentation at:

**API Documentation**: http://localhost:8000/docs

### Using Containerized Ollama

If you don't want to install Ollama on your host machine, you can use the containerized version instead:

```bash
docker-compose -f docker-compose.docker.ollama.yml up -d
```

**Note**: This approach is typically significantly slower, especially on CPU-only machines, and is not recommended for the review process.
## Key Features

- **Smart Document Chunking**: Automatically splits documents into semantic chunks for optimal retrieval
- **Query Classification**: Intelligently determines when to use documents vs. web search
- **Conversation Memory**: Maintains context for natural follow-up questions
- **Web Search Integration**: Optional integration with internet search for questions outside your document scope
