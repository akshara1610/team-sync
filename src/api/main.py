"""
TeamSync FastAPI Application
Uses LangGraph + MCP orchestration
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from loguru import logger

from src.config import settings
from src.orchestrator import TeamSyncOrchestrator
from src.database.db import init_db


# Initialize FastAPI
app = FastAPI(
    title=f"{settings.APP_NAME} - LangGraph + MCP",
    version=settings.APP_VERSION,
    description="AI Meeting Agent with LangGraph workflow and MCP tool execution"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
orchestrator = TeamSyncOrchestrator()


# Request models
class MeetingStartRequest(BaseModel):
    room_name: str
    meeting_title: str
    access_token: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


# Endpoints

@app.on_event("startup")
async def startup():
    """Initialize database."""
    logger.info("Starting TeamSync API (LangGraph + MCP)")
    init_db()


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "architecture": "LangGraph + MCP"
    }


@app.post("/meetings/start")
async def start_meeting(request: MeetingStartRequest):
    """
    Start meeting processing with LangGraph + MCP workflow.

    Workflow:
    - LangGraph manages state transitions
    - MCP handles tool execution (Jira, Calendar)
    - Self-reflection loop for quality
    """
    try:
        result = await orchestrator.process_meeting(
            room_name=request.room_name,
            meeting_title=request.meeting_title,
            access_token=request.access_token
        )
        return result

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge/query")
async def query_knowledge(request: QueryRequest):
    """Query knowledge base via MCP."""
    try:
        result = await orchestrator.query_knowledge(request.query, request.top_k)
        return result

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/tools")
async def list_mcp_tools():
    """List all MCP tools."""
    return {"tools": orchestrator.mcp_server.list_tools()}


@app.get("/mcp/audit")
async def get_mcp_audit():
    """Get MCP audit log."""
    return {"audit_log": orchestrator.mcp_server.get_audit_log()}


@app.get("/workflow/info")
async def workflow_info():
    """Get LangGraph workflow information."""
    return {
        "workflow": "LangGraph State Machine",
        "nodes": ["listen", "summarize", "reflect", "improve", "execute_actions", "schedule_followup", "store_knowledge"],
        "mcp_tools": [t["name"] for t in orchestrator.mcp_server.list_tools()],
        "features": [
            "Stateful workflow execution",
            "Self-reflection loop",
            "MCP tool integration",
            "Automatic audit logging"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
