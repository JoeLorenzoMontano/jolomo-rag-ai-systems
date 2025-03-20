from fastapi import FastAPI, HTTPException, Depends
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import chromadb
import os
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
chroma_client = chromadb.PersistentClient(path="./chroma_db")
db_collection = chroma_client.get_or_create_collection(name="documents")

# Initialize Ollama Client
ollama_client = OllamaClient()

# Folder to store raw documents
DOCS_FOLDER = "./data"
os.makedirs(DOCS_FOLDER, exist_ok=True)

@app.post("/process", summary="Process and store document embeddings", description="Processes documents and stores their embeddings in ChromaDB.")
async def process_documents():
    documents = []
    file_names = []
    
    for filename in os.listdir(DOCS_FOLDER):
        file_path = os.path.join(DOCS_FOLDER, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(content)
            file_names.append(filename)
    
    if not documents:
        raise HTTPException(status_code=400, detail="No documents to process.")
    
    embeddings = [ollama_client.generate_embedding(doc) for doc in documents]
    
    for i, doc in enumerate(documents):
        db_collection.add(ids=[file_names[i]], embeddings=[embeddings[i]], metadatas=[{"filename": file_names[i]}])
    
    return {"message": "Documents processed and stored in ChromaDB", "total": len(documents)}

@app.get("/query", summary="Retrieve relevant documents", description="Query ChromaDB for the most relevant document based on input text.")
async def query_documents(query: str):
    query_embedding = ollama_client.generate_embedding(query)
    results = db_collection.query(query_embeddings=[query_embedding], n_results=3)
    
    if not results["documents"]:
        raise HTTPException(status_code=404, detail="No relevant documents found.")
    
    best_match = results["documents"][0]
    response = ollama_client.generate_response(context=best_match, query=query)
    
    return {"query": query, "response": response, "sources": results}

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
