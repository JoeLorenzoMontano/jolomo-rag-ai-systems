from fastapi import FastAPI, HTTPException, Depends
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import chromadb
import os
import requests
from utils.ollama_client import OllamaClient

# Initialize FastAPI app
app = FastAPI(title="Document Processing API", description="API for storing and retrieving documents with embeddings.", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize ChromaDB client
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

print(f"Connecting to ChromaDB at http://{CHROMA_HOST}:{CHROMA_PORT}")

# Connect to ChromaDB with retry logic
max_retries = 5
retry_delay = 2  # seconds

for attempt in range(max_retries):
    try:
        # Use the HttpClient which is the most reliable for server connections
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        
        # Verify connection with a heartbeat
        chroma_client.heartbeat()
        print(f"Successfully connected to ChromaDB on attempt {attempt + 1}")
        break
    except Exception as e:
        print(f"Connection attempt {attempt + 1} failed: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            import time
            time.sleep(retry_delay)
        else:
            print("All connection attempts failed. Raising exception.")
            raise RuntimeError(f"Could not connect to ChromaDB at http://{CHROMA_HOST}:{CHROMA_PORT} after {max_retries} attempts")

# Create or get collection with proper settings
try:
    db_collection = chroma_client.get_or_create_collection(
        name="documents",
        metadata={"description": "Main document collection for RAG processing"}
    )
    print(f"Collection 'documents' ready with {db_collection.count()} documents")
except Exception as e:
    print(f"Error creating collection: {e}")
    raise RuntimeError(f"Failed to create ChromaDB collection: {e}")

# Initialize Ollama Client
ollama_client = OllamaClient()

# Health check endpoints
@app.get("/", summary="Root endpoint", description="Returns a simple message indicating the API is running.")
async def root():
    return {"message": "Document Processing API is running"}

@app.get("/health", summary="Health check", description="Checks if all components are operational.")
async def health_check():
    health_status = {
        "api": "healthy",
        "chroma": "unknown",
        "ollama": "unknown"
    }
    
    # Check ChromaDB
    try:
        chroma_client.heartbeat()
        health_status["chroma"] = "healthy"
    except Exception as e:
        health_status["chroma"] = f"unhealthy: {str(e)}"
    
    # Check Ollama
    try:
        response = requests.get(f"{ollama_client.base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            health_status["ollama"] = "healthy"
        else:
            health_status["ollama"] = f"unhealthy: status code {response.status_code}"
    except Exception as e:
        health_status["ollama"] = f"unhealthy: {str(e)}"
    
    return health_status

# Folder to store raw documents
DOCS_FOLDER = "./data"
os.makedirs(DOCS_FOLDER, exist_ok=True)

@app.post("/process", summary="Process and store document embeddings", description="Processes documents and stores their embeddings in ChromaDB.")
async def process_documents():
    documents = []
    file_names = []
    
    # Function to process files recursively
    def process_directory(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isdir(file_path):
                # Recursively process subdirectories
                process_directory(file_path)
            elif os.path.isfile(file_path) and file_path.endswith('.md'):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Use relative path as identifier
                        rel_path = os.path.relpath(file_path, DOCS_FOLDER)
                        documents.append(content)
                        file_names.append(rel_path)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    
    # Process all files recursively
    process_directory(DOCS_FOLDER)
    
    if not documents:
        raise HTTPException(status_code=400, detail="No documents to process.")
    
    # Process documents in batches to avoid memory issues
    batch_size = 5
    successful = 0
    failed = 0
    failed_files = []
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_names = file_names[i:i+batch_size]
        
        try:
            # Generate embeddings for the current batch - use a safer approach to handle errors
            batch_embeddings = []
            valid_docs = []
            valid_names = []
            
            for j, doc in enumerate(batch_docs):
                try:
                    embedding = ollama_client.generate_embedding(doc)
                    batch_embeddings.append(embedding)
                    valid_docs.append(doc)
                    valid_names.append(batch_names[j])
                except Exception as e:
                    print(f"Error generating embedding for {batch_names[j]}: {e}")
                    failed += 1
                    failed_files.append(batch_names[j])
            
            # Update our batch to only include documents with valid embeddings
            batch_docs = valid_docs
            batch_names = valid_names
            
            # Skip to next batch if all embeddings failed
            if not batch_embeddings:
                continue
                
            # Add to ChromaDB
            ids = batch_names
            metadatas = [{"filename": name} for name in batch_names]
            
            db_collection.add(
                ids=ids,
                embeddings=batch_embeddings,
                metadatas=metadatas,
                documents=batch_docs
            )
            
            successful += len(batch_embeddings)
            print(f"Added batch: {len(batch_embeddings)} documents")
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            failed += len(batch_docs)
            failed_files.extend(batch_names)
    
    # Return detailed status
    if failed > 0:
        return {
            "message": "Documents processed with some errors",
            "total": len(documents),
            "successful": successful,
            "failed": failed,
            "failed_files": failed_files
        }
    else:
        return {
            "message": "All documents processed successfully",
            "total": len(documents),
            "successful": successful
        }

@app.get("/query", summary="Retrieve relevant documents", description="Query ChromaDB for the most relevant document based on input text.")
async def query_documents(query: str):
    try:
        # Check if ChromaDB has any documents at all
        doc_count = db_collection.count()
        if doc_count == 0:
            return {
                "query": query,
                "response": "No documents have been processed yet. Please use the /process endpoint first.",
                "sources": {"documents": [], "ids": [], "metadatas": []},
                "status": "error",
                "error": "Empty collection"
            }
            
        # Generate embedding for the query using Ollama
        query_embedding = ollama_client.generate_embedding(query)
        
        # Get the most relevant documents from ChromaDB
        results = db_collection.query(
            query_embeddings=[query_embedding], 
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        # Handle the case where no documents are found
        if not results["documents"] or len(results["documents"]) == 0:
            return {
                "query": query,
                "response": "No relevant documents found in the database.",
                "sources": {"documents": [], "ids": [], "metadatas": []},
                "status": "not_found"
            }
        
        # Safely get the best matching document(s)
        documents = results["documents"][0]
        best_match = ""
        
        # Handle different response formats from ChromaDB
        if isinstance(documents, list) and documents:
            best_match = documents[0]
        elif isinstance(documents, str):
            best_match = documents
        else:
            print(f"Unexpected result format: {type(documents)}")
            best_match = str(documents)
            
        # Generate a response based on the best match using Ollama
        response = ollama_client.generate_response(context=best_match, query=query)
        
        # Clean up the response for better frontend rendering
        cleaned_results = {
            "documents": results["documents"],
            "ids": results["ids"],
            "metadatas": results.get("metadatas", [{}] * len(results["ids"])),
            "distances": results.get("distances", [0] * len(results["ids"]))
        }
        
        return {
            "query": query, 
            "response": response, 
            "sources": cleaned_results,
            "status": "success"
        }
    except Exception as e:
        print(f"Error in query_documents: {e}")
        
        # Provide a more helpful error response
        error_response = {
            "query": query,
            "status": "error",
            "error": str(e)
        }
        
        # Add more context based on the type of error
        if "embed" in str(e).lower():
            error_response["response"] = "Error generating embeddings. The Ollama model may not support the embedding API."
            error_response["suggestion"] = "Check if the model supports embeddings or try a different model."
        elif "chroma" in str(e).lower():
            error_response["response"] = "Error connecting to the vector database."
            error_response["suggestion"] = "Verify ChromaDB is running and accessible."
        else:
            error_response["response"] = f"An error occurred while processing your query: {str(e)}"
            
        error_response["sources"] = {"documents": [], "ids": [], "metadatas": []}
        
        return error_response

# Custom OpenAPI documentation
@app.get("/openapi.json", include_in_schema=False)
def custom_openapi():
    return get_openapi(
        title="Document Processing API",
        version="1.0",
        description="API for processing and retrieving documents using embeddings.",
        routes=app.routes,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
