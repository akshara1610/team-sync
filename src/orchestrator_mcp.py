"""
MCP-Enabled Orchestrator
Uses Model Context Protocol for standardized tool access across all agents.
"""
from typing import Dict, Any
import uuid
from datetime import datetime
from loguru import logger

from src.agents.listener_agent import ListenerAgent
from src.agents.knowledge_agent_langchain import KnowledgeAgentLangChain
from src.agents.summarizer_agent import SummarizerAgent
from src.agents.reflection_agent import SelfReflectionAgent
from src.agents.action_agent_mcp import ActionAgentMCP
from src.agents.scheduler_agent import SchedulerAgent
from src.mcp.mcp_server import MCPServer, CalendarTool, KnowledgeBaseTool
from src.models.schemas import MeetingStatus
from src.database.db import SessionLocal, Meeting


class MCPMeetingOrchestrator:
    """
    Meeting Orchestrator that uses MCP for all tool interactions.

    Benefits:
    - Centralized tool registry
    - Audit logging of all tool calls
    - Context sharing across agents
    - Standardized interface for LLM function calling
    """

    def __init__(self):
        logger.info("Initializing MCP Meeting Orchestrator...")

        # Initialize MCP Server
        self.mcp_server = MCPServer()

        # Initialize agents
        self.listener_agent = ListenerAgent()
        self.knowledge_agent = KnowledgeAgentLangChain()
        self.summarizer_agent = SummarizerAgent()
        self.reflection_agent = SelfReflectionAgent()
        self.action_agent = ActionAgentMCP(self.mcp_server)  # MCP-enabled
        self.scheduler_agent = SchedulerAgent()

        # Register tools with MCP server
        self._register_tools()

        logger.info("MCP Orchestrator initialized")
        logger.info(f"Available tools: {[t['name'] for t in self.mcp_server.list_tools()]}")

    def _register_tools(self):
        """Register all tools with MCP server."""

        # Calendar tool
        calendar_tool = CalendarTool(self.scheduler_agent.service)
        self.mcp_server.register_tool(calendar_tool)

        # Knowledge tool
        knowledge_tool = KnowledgeBaseTool(self.knowledge_agent)
        self.mcp_server.register_tool(knowledge_tool)

        logger.info("[MCP] All tools registered")

    async def process_meeting_with_mcp(
        self,
        room_name: str,
        meeting_title: str,
        access_token: str,
        auto_schedule_followup: bool = True
    ) -> Dict[str, Any]:
        """
        Process meeting using MCP for all tool interactions.

        Pipeline:
        1. Transcribe meeting
        2. Generate and validate summary
        3. Create Jira tickets via MCP
        4. Schedule follow-up via MCP
        5. Store in knowledge base
        """
        meeting_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        logger.info(f"[MCP] Starting pipeline for meeting: {meeting_id}")

        # Set global context
        self.mcp_server.set_context("meeting_id", meeting_id)
        self.mcp_server.set_context("meeting_title", meeting_title)
        self.mcp_server.set_context("start_time", start_time.isoformat())

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
            # Stage 1: Join and transcribe
            logger.info("[MCP] Stage 1: Transcribing meeting...")
            join_success = await self.listener_agent.join_meeting(
                room_name,
                meeting_id,
                access_token
            )

            if not join_success:
                raise Exception("Failed to join meeting")

            transcript = self.listener_agent.get_transcript(meeting_title, start_time)
            transcript_path = f"data/transcripts/{meeting_id}.json"
            self.listener_agent.save_transcript(transcript_path)

            meeting_record.transcript_path = transcript_path
            meeting_record.participants = transcript.participants
            meeting_record.status = MeetingStatus.PROCESSING
            db.commit()

            # Stage 2: Generate summary
            logger.info("[MCP] Stage 2: Generating summary...")
            initial_summary = self.summarizer_agent.generate_summary(transcript)

            # Stage 3: Validate with self-reflection
            logger.info("[MCP] Stage 3: Self-reflection validation...")
            reflection_feedback, final_summary = self.reflection_agent.validate_summary(
                initial_summary,
                transcript
            )

            summary_path = f"data/summaries/{meeting_id}.json"
            self.summarizer_agent.save_summary(final_summary, summary_path)
            meeting_record.summary_path = summary_path

            # Stage 4: Create Jira tickets via MCP
            logger.info("[MCP] Stage 4: Creating Jira tickets via MCP...")
            jira_tickets = []
            if final_summary.action_items:
                jira_tickets = await self.action_agent.create_jira_tickets(
                    final_summary.action_items,
                    meeting_title,
                    meeting_id
                )

            # Stage 5: Schedule follow-up via MCP
            logger.info("[MCP] Stage 5: Scheduling follow-up via MCP...")
            followup_event = None
            if auto_schedule_followup and (final_summary.action_items or final_summary.unresolved_questions):
                # Use MCP calendar tool
                start_followup = datetime.utcnow()
                start_followup = start_followup.replace(
                    hour=start_followup.hour + (7 * 24)  # 7 days ahead
                )
                end_followup = start_followup.replace(hour=start_followup.hour + 1)

                followup_result = await self.mcp_server.execute_tool(
                    tool_name="calendar_create_event",
                    parameters={
                        "summary": f"Follow-up: {meeting_title}",
                        "description": f"Follow-up meeting for {meeting_title}",
                        "start_time": start_followup.isoformat(),
                        "end_time": end_followup.isoformat(),
                        "attendees": final_summary.participants
                    }
                )

                if followup_result.get("success"):
                    followup_event = followup_result

            # Stage 6: Store in knowledge base
            logger.info("[MCP] Stage 6: Storing in knowledge base...")
            self.knowledge_agent.add_transcript(transcript)

            # Update meeting record
            meeting_record.end_time = datetime.utcnow()
            meeting_record.status = MeetingStatus.COMPLETED
            db.commit()

            # Get MCP audit log
            audit_log = self.mcp_server.get_audit_log()

            logger.info(f"[MCP] Pipeline completed for meeting: {meeting_id}")

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
                    "action_items_count": len(final_summary.action_items),
                    "key_decisions_count": len(final_summary.key_decisions)
                },
                "reflection": {
                    "approved": reflection_feedback.approved,
                    "coherence_score": reflection_feedback.logical_coherence_score,
                    "issues_count": len(reflection_feedback.consistency_issues)
                },
                "jira_tickets": jira_tickets,
                "followup_meeting": followup_event,
                "mcp_stats": {
                    "tools_registered": len(self.mcp_server.list_tools()),
                    "tool_invocations": len(audit_log),
                    "audit_log": audit_log
                },
                "processing_time_seconds": (datetime.utcnow() - start_time).total_seconds()
            }

        except Exception as e:
            logger.error(f"[MCP] Pipeline error: {e}")
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

    async def query_via_mcp(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Query knowledge base via MCP."""
        logger.info(f"[MCP] Querying knowledge base: {query}")

        result = await self.mcp_server.execute_tool(
            tool_name="knowledge_query",
            parameters={
                "query": query,
                "top_k": top_k
            }
        )

        return result

    def list_available_tools(self) -> List[Dict[str, str]]:
        """List all tools available via MCP."""
        return self.mcp_server.list_tools()

    def get_tool_schemas_for_llm(self) -> List[Dict[str, Any]]:
        """
        Get tool schemas in OpenAI function calling format.
        Useful for allowing LLMs to call tools directly.
        """
        return self.mcp_server.get_all_schemas()

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get complete audit trail of all MCP tool calls."""
        return self.mcp_server.get_audit_log()
