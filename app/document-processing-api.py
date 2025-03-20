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

# Try to connect to ChromaDB, with multiple fallback options
print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")

# Try approach 1: Modern client API
try:
    from chromadb.config import Settings
    
    chroma_settings = Settings(
        chroma_api_impl="rest",
        chroma_server_host=CHROMA_HOST,
        chroma_server_http_port=CHROMA_PORT
    )
    
    chroma_client = chromadb.Client(chroma_settings)
    chroma_client.heartbeat()
    print("Successfully connected to ChromaDB using modern REST API")
except Exception as e:
    print(f"Modern REST API connection failed: {e}")
    
    # Try approach 2: Direct HTTP client
    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        chroma_client.heartbeat()
        print("Successfully connected to ChromaDB using HTTP client")
    except Exception as e:
        print(f"HTTP client connection failed: {e}")
        
        # Try approach 3: Explicit URL
        try:
            chroma_client = chromadb.HttpClient(url=f"http://{CHROMA_HOST}:{CHROMA_PORT}")
            chroma_client.heartbeat()
            print("Successfully connected to ChromaDB using explicit URL")
        except Exception as e:
            print(f"Explicit URL connection failed: {e}")
            
            # Try approach 4: Default port
            try:
                chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=8000)
                chroma_client.heartbeat()
                print("Successfully connected to ChromaDB using default port 8000")
            except Exception as e:
                print(f"Default port connection failed: {e}")
                
                # Final fallback: In-memory client
                print("All connection attempts failed. Using in-memory client for testing purposes.")
                chroma_client = chromadb.EphemeralClient()

# Create or get collection
db_collection = chroma_client.get_or_create_collection(
    name="documents"
)

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
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_names = file_names[i:i+batch_size]
        
        # Generate embeddings for the current batch
        batch_embeddings = [ollama_client.generate_embedding(doc) for doc in batch_docs]
        
        # Add to ChromaDB
        ids = batch_names
        metadatas = [{"filename": name} for name in batch_names]
        db_collection.add(
            ids=ids,
            embeddings=batch_embeddings,
            metadatas=metadatas,
            documents=batch_docs
        )
    
    return {"message": "Documents processed and stored in ChromaDB", "total": len(documents)}

@app.get("/query", summary="Retrieve relevant documents", description="Query ChromaDB for the most relevant document based on input text.")
async def query_documents(query: str):
    try:
        query_embedding = ollama_client.generate_embedding(query)
        results = db_collection.query(query_embeddings=[query_embedding], n_results=3)
        
        # Handle various result formats from different ChromaDB versions
        if not results["documents"]:
            raise HTTPException(status_code=404, detail="No relevant documents found.")
        
        # Safely get the best matching document
        first_result = results["documents"][0]
        best_match = ""
        
        if isinstance(first_result, list) and first_result:
            best_match = first_result[0]
        elif isinstance(first_result, str):
            best_match = first_result
        else:
            print(f"Unexpected result format: {type(first_result)}")
            best_match = str(first_result)
            
        # Generate a response based on the best match
        response = ollama_client.generate_response(context=best_match, query=query)
        
        # Clean up the response for better frontend rendering
        cleaned_results = {
            "documents": results["documents"],
            "ids": results["ids"],
            "metadatas": results.get("metadatas", [{}] * len(results["ids"]))
        }
        
        return {
            "query": query, 
            "response": response, 
            "sources": cleaned_results,
            "status": "success"
        }
    except Exception as e:
        print(f"Error in query_documents: {e}")
        return {
            "query": query,
            "response": f"An error occurred while processing your query: {str(e)}",
            "sources": {"documents": [], "ids": [], "metadatas": []},
            "status": "error",
            "error": str(e)
        }

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
