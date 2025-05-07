"""
Document Processing API.

This is the main entry point for the Document Processing API application.
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware

from core.config import log_config
from core.dependencies import get_all_services
from routers import health, documents, query, jobs, terms, sms

# Initialize FastAPI app
app = FastAPI(
    title="Document Processing API", 
    description="API for storing and retrieving documents with embeddings.", 
    version="1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(jobs.router)
app.include_router(terms.router)
app.include_router(sms.router)

@app.get("/", summary="Root endpoint", description="Returns a simple message indicating the API is running.")
async def root():
    """Root endpoint that returns a simple status message."""
    return {"message": "Document Processing API is running with SMS capabilities enabled"}

@app.get("/openapi.json", include_in_schema=False)
def custom_openapi():
    """Generate custom OpenAPI schema."""
    return get_openapi(
        title="Document Processing API",
        version="1.0",
        description="API for processing and retrieving documents using embeddings, with SMS capabilities.",
        routes=app.routes,
    )

# Initialize services on startup
@app.on_event("startup")
async def startup_event():
    """Initialize all services on application startup."""
    # Log configuration settings
    log_config()
    
    # Initialize all services
    _ = get_all_services()
    
    print("Document Processing API started successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)