# Document Processing API with Web UI

A complete RAG (Retrieval-Augmented Generation) system that processes documents, stores their embeddings in ChromaDB and indexes in Elasticsearch, and generates AI responses based on the most relevant content. The system combines multiple search techniques to deliver more accurate and relevant responses using a dual-database architecture with ChromaDB (vector embeddings) and Elasticsearch (text search) along with document reranking, providing a powerful and flexible solution that adapts to different query types. The system includes both an API backend and a web-based user interface.

**Key Features**:
- **Dual-Database Architecture**: Combines ChromaDB's vector search with Elasticsearch's text capabilities
- **Hybrid Search**: Intelligently merges vector similarity and text relevance with configurable weights
- **Smart Document Chunking**: Semantic chunking and enrichment for optimal retrieval
- **Document Reranking**: Improves result relevance using cross-encoder models via LangChain
- **Query Classification**: Automatically routes between document search, web search, or hybrid approaches
- **Conversation Memory**: Maintains context for natural follow-up questions in chat sessions
- **Web Search Integration**: Augments with internet search when appropriate for comprehensive answers

**Advanced RAG Capabilities**:
- **Query Preprocessing**: Enhances queries by expanding technical terms and acronyms
- **Fallback Strategies**: Uses BM25 text search when transformer models aren't available
- **Semantic Filtering**: Uses domain-specific terminology for better classification
- **Hybrid Result Merging**: Sophisticated algorithm for combining results from multiple sources
- **Adaptive Search Routing**: Selects the optimal search approach based on query characteristics

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
- **Elasticsearch**: http://localhost:9200 (text search engine)
- **Ollama**: http://localhost:11434 (LLM server)

## System Architecture

This system uses a dual-database approach to provide powerful, flexible document retrieval:

### Databases
- **ChromaDB**: Vector database that stores document embeddings for semantic similarity search
- **Elasticsearch**: Full-text search engine with BM25 algorithm for keyword and text-based search

### Search Methods
- **Vector Search**: Uses embeddings to find semantically similar content (better for conceptual queries)
- **Text Search**: Uses BM25 text matching for exact text matches and keyword search (better for specific terms)
- **Hybrid Search**: Combines both approaches with configurable weights to get the best of both worlds
- **Document Reranking**: Further improves result relevance using cross-encoder models

### Query Processing
1. **Preprocessing**: Enhances queries with context and synonyms
2. **Classification**: Determines optimal search approach based on query characteristics
3. **Retrieval**: Fetches relevant documents using the appropriate search methods
4. **Reranking**: Reorders retrieved documents based on relevance to the query
5. **Response Generation**: Uses LLM to create a natural language response based on retrieved context

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

## Future Plans

The following advanced RAG capabilities are planned for future development:

### Scalability & Performance Enhancements
- **Kafka Integration**: Implement Apache Kafka for asynchronous document processing and uploading pipelines
- **Distributed LLM Serving**: Replace Ollama with vLLM for distributed inference and optimized throughput
- **Multi-Level Caching**: Implement Redis-based caching for search results, embeddings, and API responses
- **Query Result Caching**: Store and reuse results for frequently asked questions to reduce latency
- **Vector Database Optimization**: Implement sharding and partitioning for improved ChromaDB/Elasticsearch performance

### Advanced RAG Techniques
- **Synthetic Data Augmentation**: Generate additional context-rich documents using AI to improve retrieval quality
- **Multi-vector Retrieval**: Store multiple embeddings per chunk (summary, entities, relationships) for improved semantic search
- **Query Optimization**: Implement query expansion and transformation for better retrieval
