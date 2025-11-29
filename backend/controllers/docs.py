from fastapi import APIRouter
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI
import sys
import os

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.logger import get_logger

# Get logger
logger = get_logger(__name__)

# Create API router for documentation
docs_router = APIRouter(tags=["Documentation"])

def get_openapi_schema(app: FastAPI):
    """
    Generate a custom OpenAPI schema for the application.
    
    Args:
        app: The FastAPI application instance
        
    Returns:
        dict: The OpenAPI schema
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Mavo Voice Analysis API",
        version="0.1.0",
        description="""
        A local server solution for voice analysis and diarization.
        
        This API provides endpoints for:
        * Uploading audio files
        * Transcribing speech to text
        * Identifying speakers (diarization)
        * Retrieving analysis results
        
        The server is designed to be faster and more accurate than cloud-based solutions.
        """,
        routes=app.routes,
    )
    
    # Add custom branding
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

def setup_docs_routes(app: FastAPI):
    """
    Set up the documentation routes for the application.
    
    Args:
        app: The FastAPI application instance
    """
    # Set custom OpenAPI schema generator
    app.openapi = lambda: get_openapi_schema(app)
    
    # Custom Swagger UI endpoint at /doc
    @docs_router.get("/doc", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        )

    # Custom ReDoc endpoint
    @docs_router.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - ReDoc",
            redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
        )
    
    # Include the docs router
    app.include_router(docs_router) 