from fastapi import APIRouter
from .upload_controller import router as upload_router
from .core_controller import router as core_router
from .maum_controller import router as maum_router
from .session_v2_controller import router as session_v2_router

# Create a main router that includes all controller routers
api_router = APIRouter()

# Include all controller routers
api_router.include_router(core_router)
api_router.include_router(upload_router)
api_router.include_router(maum_router)
api_router.include_router(session_v2_router)

# Export the routers
__all__ = [
    "api_router",
    "core_router",
    "upload_router",
    "maum_router",
    "session_v2_router",
]
