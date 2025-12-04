"""
TeamSync - AI-Powered Meeting Agent
Main entry point for the application
"""
import uvicorn
from src.config import settings


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
