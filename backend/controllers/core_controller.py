from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from typing import Dict, Any, List
import sys
import os

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from util.logger import get_logger
from util.monitor import MonitorThread

# Get logger
logger = get_logger(__name__)

# Create and start monitor thread
monitor_thread = MonitorThread()
monitor_thread.start()
logger.info("Started monitor thread")

# Create API router with tags for better organization in Swagger
router = APIRouter(tags=["Core"])


# Root endpoint - redirect to the HTML page
@router.get(
    "/", summary="Root Endpoint", description="Redirects to the HTML test page."
)
async def root():
    return RedirectResponse(url="/mavo")


# Root endpoint - redirect to the HTML page
@router.get(
    "/mavo", summary="Root Endpoint", description="Redirects to the HTML test page."
)
async def mavo_root():
    """
    Root endpoint that redirects to the HTML test page.

    Returns:
        RedirectResponse: Redirects to the HTML test page.
    """
    response = RedirectResponse(url="/public/mavo.html")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# API endpoint
@router.get(
    "/api/v1/",
    summary="API Information",
    description="Returns information about the API status and available endpoints.",
    response_description="API information object",
    status_code=status.HTTP_200_OK,
    response_model_exclude_none=True,
)
async def api_root() -> Dict[str, Any]:
    """
    Returns information about the API status and available endpoints.

    Returns:
        Dict[str, Any]: A dictionary containing API information.
    """

    return {
        "message": "Welcome to Mavo Voice Analysis Server",
        "status": "running",
        "version": "0.1.0",
        "dev_stage": 7,  # Current development stage
        "server_info": {
            "port": config.PORT,
            "host": config.HOST,
            "debug": config.DEBUG,
        },
    }
