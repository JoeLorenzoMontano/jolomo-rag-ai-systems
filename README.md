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

1. Clone this repository
2. Navigate to the project directory
3. Run:

```bash
docker-compose up -d
```

4. Wait for all services to start (this may take some time as Ollama downloads the necessary models)

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