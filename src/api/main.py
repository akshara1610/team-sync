from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from loguru import logger
from src.config import settings
from src.orchestrator import MeetingOrchestrator
from src.database.db import init_db
from src.models.schemas import RAGQuery, CalendarEvent


# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-Powered Meeting Agent for automated transcription, summarization, and task management"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
orchestrator = MeetingOrchestrator()


# Request/Response Models
class MeetingStartRequest(BaseModel):
    room_name: str
    meeting_title: str
    access_token: str
    auto_schedule_followup: bool = True


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class HealthResponse(BaseModel):
    status: str
    version: str


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    logger.info("Starting TeamSync API...")
    init_db()
    logger.info("Database initialized")


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION
    )


@app.post("/meetings/start")
async def start_meeting(
    request: MeetingStartRequest,
    background_tasks: BackgroundTasks
):
    """
    Start processing a meeting with full pipeline.

    This endpoint initiates the complete meeting lifecycle:
    1. Join meeting and transcribe
    2. Generate and validate summary
    3. Create Jira tickets
    4. Schedule follow-up
    5. Add to knowledge base
    """
    try:
        logger.info(f"Starting meeting: {request.meeting_title}")

        # Run pipeline in background
        result = await orchestrator.process_meeting_full_pipeline(
            room_name=request.room_name,
            meeting_title=request.meeting_title,
            access_token=request.access_token,
            auto_schedule_followup=request.auto_schedule_followup
        )

        return result

    except Exception as e:
        logger.error(f"Error starting meeting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/meetings/{meeting_id}")
async def get_meeting(meeting_id: str):
    """Get meeting details by ID."""
    try:
        result = orchestrator.get_meeting_summary(meeting_id)

        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting meeting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/meetings")
async def list_meetings(limit: int = 50):
    """List all meetings."""
    try:
        meetings = orchestrator.get_all_meetings(limit=limit)
        return {"meetings": meetings, "count": len(meetings)}

    except Exception as e:
        logger.error(f"Error listing meetings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge/query")
async def query_knowledge(request: QueryRequest):
    """
    Query the knowledge base about past meetings.

    Uses RAG (Retrieval-Augmented Generation) to find relevant information
    from historical meeting transcripts.
    """
    try:
        result = await orchestrator.query_knowledge_base(
            query=request.query,
            top_k=request.top_k
        )

        return result

    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
