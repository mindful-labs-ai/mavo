from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import sys
from fastapi.responses import RedirectResponse
from typing import Optional

# Add the current directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.controllers import api_router
from backend.controllers.docs import setup_docs_routes
import config

# Initialize FastAPI app
app = FastAPI(
    title="Mavo Voice Analysis Server",
    description="""
    A local server solution for voice analysis and diarization.
    
    This API provides endpoints for:
    * Uploading audio files
    * Transcribing speech to text
    * Identifying speakers (diarization)
    * Retrieving analysis results
    
    The server is designed to be faster and more accurate than cloud-based solutions.
    """,
    version="0.1.0",
    docs_url=None,  # Disable default docs to use custom docs
    redoc_url=None,  # Disable default redoc to use custom redoc
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Path to the public directory
public_dir = os.path.join(current_dir, "public")

# Mount the static files directory
app.mount("/public", StaticFiles(directory=public_dir, html=True), name="public")


# Add cache control middleware
@app.middleware("http")
async def add_cache_control_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# Add this route before the static files mount
@app.get("/mavo")
async def redirect_to_mavo(uuid: Optional[str] = None):
    """Redirect /mavo to /public/mavo.html with optional uuid parameter"""
    base_url = "/public/mavo.html"
    if uuid:
        return RedirectResponse(url=f"{base_url}?uuid={uuid}")
    return RedirectResponse(url=base_url)


# Include the API router with all controllers
app.include_router(api_router)

# Set up documentation routes
setup_docs_routes(app)

# Run the server if executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=config.DEBUG)
