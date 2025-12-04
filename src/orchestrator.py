from typing import Dict, Any
import uuid
from datetime import datetime
from loguru import logger
from src.agents.listener_agent import ListenerAgent
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.agents.reflection_agent import SelfReflectionAgent
from src.agents.action_agent import ActionAgent
from src.agents.scheduler_agent import SchedulerAgent
from src.models.schemas import MeetingRecord, MeetingStatus
from src.database.db import SessionLocal, Meeting


class MeetingOrchestrator:
    """
    Main orchestrator that coordinates all agents in the TeamSync system.
    Manages the complete meeting lifecycle from joining to post-processing.
    """

    def __init__(self):
        logger.info("Initializing Meeting Orchestrator...")

        # Initialize all agents
        self.listener_agent = ListenerAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.summarizer_agent = SummarizerAgent()
        self.reflection_agent = SelfReflectionAgent()
        self.action_agent = ActionAgent()
        self.scheduler_agent = SchedulerAgent()

        logger.info("All agents initialized successfully")

    async def process_meeting_full_pipeline(
        self,
        room_name: str,
        meeting_title: str,
        access_token: str,
        auto_schedule_followup: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the complete meeting pipeline.

        Pipeline stages:
        1. Join meeting and transcribe (Listener Agent)
        2. Generate summary (Summarizer Agent)
        3. Validate and improve (Self-Reflection Agent)
        4. Create Jira tickets (Action Agent)
        5. Schedule follow-up (Scheduler Agent)
        6. Store in knowledge base (Knowledge Agent)

        Args:
            room_name: LiveKit room name
            meeting_title: Title of the meeting
            access_token: LiveKit access token
            auto_schedule_followup: Whether to auto-schedule follow-up

        Returns:
            Complete pipeline results
        """
        meeting_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        logger.info(f"Starting full pipeline for meeting: {meeting_id}")

        # Create database record
        db = SessionLocal()
        meeting_record = Meeting(
            id=meeting_id,
            title=meeting_title,
            status=MeetingStatus.IN_PROGRESS,
            start_time=start_time,
            participants=[]
        )
        db.add(meeting_record)
        db.commit()

        try:
            # Stage 1: Join meeting and transcribe
            logger.info("Stage 1: Joining meeting and transcribing...")
            join_success = await self.listener_agent.join_meeting(
                room_name,
                meeting_id,
                access_token
            )

            if not join_success:
                raise Exception("Failed to join meeting")

            # Note: In real implementation, this would run continuously
            # For demo purposes, we'll simulate waiting for meeting to end
            # await asyncio.sleep(meeting_duration)

            # Get transcript
            transcript = self.listener_agent.get_transcript(meeting_title, start_time)
            transcript_path = f"data/transcripts/{meeting_id}.json"
            self.listener_agent.save_transcript(transcript_path)

            meeting_record.transcript_path = transcript_path
            meeting_record.participants = transcript.participants
            meeting_record.status = MeetingStatus.PROCESSING
            db.commit()

            # Stage 2: Generate summary
            logger.info("Stage 2: Generating meeting summary...")
            initial_summary = self.summarizer_agent.generate_summary(transcript)

            # Stage 3: Validate and improve with self-reflection
            logger.info("Stage 3: Validating summary with self-reflection...")
            reflection_feedback, final_summary = self.reflection_agent.validate_summary(
                initial_summary,
                transcript
            )

            # Save final summary
            summary_path = f"data/summaries/{meeting_id}.json"
            self.summarizer_agent.save_summary(final_summary, summary_path)
            meeting_record.summary_path = summary_path

            # Stage 4: Create Jira tickets
            logger.info("Stage 4: Creating Jira tickets for action items...")
            jira_tickets = []
            if final_summary.action_items:
                jira_tickets = self.action_agent.create_jira_tickets(
                    final_summary.action_items,
                    meeting_title,
                    meeting_id
                )

            # Stage 5: Schedule follow-up meeting
            logger.info("Stage 5: Scheduling follow-up meeting...")
            followup_event = None
            if auto_schedule_followup and (final_summary.action_items or final_summary.unresolved_questions):
                followup_event = self.scheduler_agent.schedule_follow_up_meeting(
                    final_summary,
                    days_ahead=7
                )

            # Stage 6: Store in knowledge base
            logger.info("Stage 6: Adding transcript to knowledge base...")
            self.knowledge_agent.add_transcript(transcript)

            # Update meeting record
            meeting_record.end_time = datetime.utcnow()
            meeting_record.status = MeetingStatus.COMPLETED
            db.commit()

            logger.info(f"Pipeline completed successfully for meeting: {meeting_id}")

            return {
                "meeting_id": meeting_id,
                "status": "success",
                "transcript": {
                    "path": transcript_path,
                    "segments_count": len(transcript.segments),
                    "participants": transcript.participants
                },
                "summary": {
                    "path": summary_path,
                    "executive_summary": final_summary.executive_summary,
                    "key_decisions_count": len(final_summary.key_decisions),
                    "action_items_count": len(final_summary.action_items),
                    "discussion_points_count": len(final_summary.discussion_points)
                },
                "reflection": {
                    "approved": reflection_feedback.approved,
                    "coherence_score": reflection_feedback.logical_coherence_score,
                    "issues_count": len(reflection_feedback.consistency_issues) + len(reflection_feedback.missing_action_items)
                },
                "jira_tickets": jira_tickets,
                "followup_meeting": followup_event,
                "processing_time_seconds": (datetime.utcnow() - start_time).total_seconds()
            }

        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            meeting_record.status = MeetingStatus.FAILED
            db.commit()

            return {
                "meeting_id": meeting_id,
                "status": "failed",
                "error": str(e)
            }

        finally:
            db.close()
            await self.listener_agent.leave_meeting()

    async def query_knowledge_base(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query the knowledge base for past meeting information.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            Query results
        """
        logger.info(f"Querying knowledge base: {query}")

        response = self.knowledge_agent.query(query, top_k)

        return {
            "query": response.query,
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": response.sources
        }

    def get_meeting_summary(self, meeting_id: str) -> Dict[str, Any]:
        """
        Get summary for a specific meeting.

        Args:
            meeting_id: Meeting identifier

        Returns:
            Meeting summary information
        """
        db = SessionLocal()
        try:
            meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()

            if not meeting:
                return {"error": "Meeting not found"}

            return {
                "meeting_id": meeting.id,
                "title": meeting.title,
                "status": meeting.status.value,
                "start_time": meeting.start_time.isoformat(),
                "end_time": meeting.end_time.isoformat() if meeting.end_time else None,
                "participants": meeting.participants,
                "transcript_path": meeting.transcript_path,
                "summary_path": meeting.summary_path
            }

        finally:
            db.close()

    def get_all_meetings(
        self,
        limit: int = 50
    ) -> list[Dict[str, Any]]:
        """
        Get all meetings from database.

        Args:
            limit: Maximum number of meetings to return

        Returns:
            List of meeting records
        """
        db = SessionLocal()
        try:
            meetings = db.query(Meeting).order_by(
                Meeting.created_at.desc()
            ).limit(limit).all()

            return [
                {
                    "meeting_id": m.id,
                    "title": m.title,
                    "status": m.status.value,
                    "start_time": m.start_time.isoformat(),
                    "participants": m.participants
                }
                for m in meetings
            ]

        finally:
            db.close()
