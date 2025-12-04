"""
FastAPI application with LangGraph-based orchestration.
Supports both traditional orchestrator and LangGraph orchestrator.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Literal
from loguru import logger

from src.config import settings
from src.orchestrator_langgraph import LangGraphMeetingOrchestrator
from src.database.db import init_db
from src.models.schemas import CalendarEvent


# Initialize FastAPI app
app = FastAPI(
    title=f"{settings.APP_NAME} - LangGraph Edition",
    version=settings.APP_VERSION,
    description="AI-Powered Meeting Agent with LangGraph multi-agent orchestration"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LangGraph orchestrator
orchestrator = LangGraphMeetingOrchestrator()


# Request/Response Models
class MeetingStartRequest(BaseModel):
    room_name: str
    meeting_title: str
    access_token: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class HealthResponse(BaseModel):
    status: str
    version: str
    orchestrator: str


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    logger.info("Starting TeamSync API with LangGraph...")
    init_db()
    logger.info("Database initialized")


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        orchestrator="LangGraph"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        orchestrator="LangGraph State Machine"
    )


@app.post("/meetings/start")
async def start_meeting(request: MeetingStartRequest):
    """
    Start processing a meeting with LangGraph workflow.

    This endpoint initiates the complete meeting lifecycle using LangGraph:
    1. Listen → Transcribe meeting
    2. Summarize → Generate initial summary
    3. Reflect → Validate and improve (with loop)
    4. Execute Actions → Create Jira tickets
    5. Schedule → Follow-up meetings
    6. Store → Add to knowledge base

    The workflow is managed as a state graph with automatic state transitions.
    """
    try:
        logger.info(f"Starting LangGraph workflow for: {request.meeting_title}")

        # Run LangGraph workflow
        result = await orchestrator.process_meeting(
            room_name=request.room_name,
            meeting_title=request.meeting_title,
            access_token=request.access_token
        )

        return result

    except Exception as e:
        logger.error(f"Error starting meeting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/meetings/{meeting_id}")
async def get_meeting(meeting_id: str):
    """Get meeting details by ID."""
    try:
        # Use the knowledge agent to retrieve
        context = orchestrator.knowledge_agent.get_meeting_context(meeting_id)

        if not context:
            raise HTTPException(status_code=404, detail="Meeting not found")

        return {
            "meeting_id": meeting_id,
            "context": context
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting meeting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge/query")
async def query_knowledge(request: QueryRequest):
    """
    Query the knowledge base using LangChain's RetrievalQA.

    Uses LangChain's full RAG pipeline:
    - ChromaDB for vector storage
    - OpenAI embeddings for semantic search
    - RetrievalQA chain for answer generation
    - Automatic source citation
    """
    try:
        result = await orchestrator.query_knowledge(
            query=request.query,
            top_k=request.top_k
        )

        return result

    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge/search")
async def similarity_search(
    query: str = Query(..., description="Search query"),
    k: int = Query(5, description="Number of results"),
    meeting_id: Optional[str] = Query(None, description="Filter by meeting ID")
):
    """
    Perform similarity search without answer generation.
    Returns raw documents with relevance scores.
    """
    try:
        filter_dict = {"meeting_id": meeting_id} if meeting_id else None

        results = orchestrator.knowledge_agent.similarity_search(
            query=query,
            k=k,
            filter_dict=filter_dict
        )

        return {
            "query": query,
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge/stats")
async def knowledge_stats():
    """Get knowledge base statistics."""
    try:
        stats = orchestrator.knowledge_agent.get_collection_stats()
        return stats

    except Exception as e:
        logger.error(f"Error getting knowledge stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/knowledge/meeting/{meeting_id}")
async def delete_meeting(meeting_id: str):
    """Delete a meeting from the knowledge base."""
    try:
        success = orchestrator.knowledge_agent.delete_meeting(meeting_id)

        if success:
            return {"message": f"Meeting {meeting_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Meeting not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting meeting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calendar/schedule")
async def schedule_event(event: CalendarEvent):
    """Schedule a new calendar event."""
    try:
        result = orchestrator.scheduler_agent.schedule_event(event)

        if result.get("status") == "failed":
            raise HTTPException(status_code=500, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/calendar/upcoming")
async def get_upcoming_events(max_results: int = 10):
    """Get upcoming calendar events."""
    try:
        events = orchestrator.scheduler_agent.get_upcoming_events(
            max_results=max_results
        )
        return {"events": events, "count": len(events)}

    except Exception as e:
        logger.error(f"Error getting upcoming events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jira/tickets")
async def get_jira_tickets(meeting_id: Optional[str] = None):
    """Get Jira tickets, optionally filtered by meeting."""
    try:
        tickets = orchestrator.action_agent.get_project_tickets(
            meeting_id=meeting_id
        )
        return {"tickets": tickets, "count": len(tickets)}

    except Exception as e:
        logger.error(f"Error getting Jira tickets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jira/tickets/{ticket_key}")
async def get_jira_ticket(ticket_key: str):
    """Get details of a specific Jira ticket."""
    try:
        ticket = orchestrator.action_agent.get_ticket_info(ticket_key)

        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")

        return ticket

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Jira ticket: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflow/visualize")
async def visualize_workflow():
    """
    Get information about the LangGraph workflow structure.
    """
    try:
        return {
            "workflow_type": "LangGraph State Machine",
            "nodes": [
                "listen",
                "summarize",
                "reflect",
                "improve",
                "execute_actions",
                "schedule_followup",
                "store_knowledge"
            ],
            "edges": [
                "listen → summarize",
                "summarize → reflect",
                "reflect → [improve OR execute_actions]",
                "improve → reflect (loop)",
                "execute_actions → schedule_followup",
                "schedule_followup → store_knowledge",
                "store_knowledge → END"
            ],
            "features": [
                "Stateful workflow execution",
                "Automatic state transitions",
                "Conditional branching",
                "Self-reflection loop (max 3 iterations)",
                "Error handling per node"
            ]
        }

    except Exception as e:
        logger.error(f"Error getting workflow info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main_langgraph:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
