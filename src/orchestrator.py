"""
Unified TeamSync Orchestrator
Combines LangGraph state machine workflow with MCP tool execution.

This is the main orchestrator - uses both LangGraph and MCP.
"""
from typing import Dict, Any, TypedDict, Annotated, Sequence
import operator
from datetime import datetime
import uuid
import asyncio
from loguru import logger

# LangGraph
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage

# Agents
from src.agents.listener_agent import ListenerAgent
from src.agents.knowledge_agent_langchain import KnowledgeAgentLangChain
from src.agents.summarizer_agent import SummarizerAgent
from src.agents.reflection_agent import SelfReflectionAgent
from src.agents.action_agent_mcp import ActionAgentMCP
from src.agents.scheduler_agent import SchedulerAgent

# MCP
from src.mcp.mcp_server import MCPServer, CalendarTool, KnowledgeBaseTool

# Database
from src.models.schemas import MeetingStatus
from src.database.db import SessionLocal, Meeting


class MeetingState(TypedDict):
    """LangGraph state for meeting workflow."""
    meeting_id: str
    meeting_title: str
    room_name: str
    start_time: datetime
    status: str
    transcript: Dict[str, Any]
    transcript_path: str
    initial_summary: Dict[str, Any]
    reflection_feedback: Dict[str, Any]
    final_summary: Dict[str, Any]
    summary_path: str
    jira_tickets: list
    followup_meeting: Dict[str, Any]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    validation_passed: bool
    reflection_iterations: int
    errors: list
    mcp_audit_log: list


def _run_async(coro):
    """Helper to run async functions from sync context."""
    try:
        loop = asyncio.get_running_loop()
        # There's already a running loop - this shouldn't happen in LangGraph nodes
        # Create a new task and wait for it
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return executor.submit(asyncio.run, coro).result()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(coro)


class TeamSyncOrchestrator:
    """
    Main TeamSync Orchestrator using LangGraph + MCP.

    Architecture:
    - LangGraph: State machine workflow coordination
    - MCP: Standardized tool execution (Jira, Calendar, Knowledge)
    - 6 Agents: Listener, Knowledge, Summarizer, Reflection, Action, Scheduler
    """

    def __init__(self):
        logger.info("=" * 60)
        logger.info("Initializing TeamSync Orchestrator")
        logger.info("Architecture: LangGraph + MCP")
        logger.info("=" * 60)

        # Initialize MCP Server
        self.mcp_server = MCPServer()

        # Initialize all agents
        self.listener_agent = ListenerAgent()
        self.knowledge_agent = KnowledgeAgentLangChain()
        self.summarizer_agent = SummarizerAgent()
        self.reflection_agent = SelfReflectionAgent()
        self.action_agent = ActionAgentMCP(self.mcp_server)
        self.scheduler_agent = SchedulerAgent()

        # Register MCP tools
        self._register_mcp_tools()

        # Build LangGraph workflow
        self.workflow = self._build_langgraph_workflow()
        self.app = self.workflow.compile()

        logger.info("✓ All agents initialized")
        logger.info(f"✓ MCP tools: {[t['name'] for t in self.mcp_server.list_tools()]}")
        logger.info(f"✓ LangGraph workflow compiled")
        logger.info("=" * 60)

    def _register_mcp_tools(self):
        """Register tools with MCP server."""
        calendar_tool = CalendarTool(self.scheduler_agent.service)
        self.mcp_server.register_tool(calendar_tool)

        knowledge_tool = KnowledgeBaseTool(self.knowledge_agent)
        self.mcp_server.register_tool(knowledge_tool)

    def _build_langgraph_workflow(self) -> StateGraph:
        """Build LangGraph state machine workflow."""
        workflow = StateGraph(MeetingState)

        # Add nodes
        workflow.add_node("listen", self._listen_node)
        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("reflect", self._reflect_node)
        workflow.add_node("improve", self._improve_node)
        workflow.add_node("execute_actions", self._action_node)
        workflow.add_node("schedule_followup", self._schedule_node)
        workflow.add_node("store_knowledge", self._store_node)

        # Define edges
        workflow.set_entry_point("listen")

        # Conditional: check if listen succeeded
        workflow.add_conditional_edges(
            "listen",
            lambda state: "summarize" if state.get("status") != "failed" else END,
            {"summarize": "summarize", END: END}
        )

        # Conditional: check if summarize succeeded
        workflow.add_conditional_edges(
            "summarize",
            lambda state: "reflect" if state.get("status") != "failed" else END,
            {"reflect": "reflect", END: END}
        )

        # Conditional: reflection passes or needs improvement
        workflow.add_conditional_edges(
            "reflect",
            self._should_improve,
            {"improve": "improve", "continue": "execute_actions"}
        )

        workflow.add_edge("improve", "reflect")
        workflow.add_edge("execute_actions", "schedule_followup")
        workflow.add_edge("schedule_followup", "store_knowledge")
        workflow.add_edge("store_knowledge", END)

        return workflow

    # LangGraph node implementations

    def _listen_node(self, state: MeetingState) -> Dict[str, Any]:
        """Node 1: Transcribe meeting."""
        logger.info(f"[LangGraph] Node: LISTEN - Meeting {state['meeting_id']}")

        try:
            transcript = self.listener_agent.get_transcript(
                state["meeting_title"],
                state["start_time"]
            )

            # Check if transcript has segments
            if not transcript.segments:
                error_msg = "No transcript segments captured. Meeting may not have started or no audio was detected."
                logger.error(error_msg)
                return {
                    "errors": [error_msg],
                    "status": "failed",
                    "transcript": {},
                    "transcript_path": ""
                }

            transcript_path = f"data/transcripts/{state['meeting_id']}.json"
            self.listener_agent.save_transcript(transcript_path)

            return {
                "transcript": transcript.model_dump(),
                "transcript_path": transcript_path,
                "messages": [HumanMessage(content=f"Transcribed {len(transcript.segments)} segments")],
                "status": "transcribed"
            }

        except Exception as e:
            logger.error(f"Listen node error: {e}")
            return {"errors": [f"Listen error: {str(e)}"], "status": "failed", "transcript": {}, "transcript_path": ""}

    def _summarize_node(self, state: MeetingState) -> Dict[str, Any]:
        """Node 2: Generate summary."""
        logger.info(f"[LangGraph] Node: SUMMARIZE")

        try:
            # Check if we have a valid transcript
            if not state.get("transcript") or state.get("status") == "failed":
                error_msg = "Cannot summarize: no valid transcript available"
                logger.error(error_msg)
                return {
                    "errors": state.get("errors", []) + [error_msg],
                    "status": "failed",
                    "initial_summary": {}
                }

            from src.models.schemas import TranscriptData
            transcript = TranscriptData(**state["transcript"])

            if not transcript.segments:
                error_msg = "Cannot summarize: transcript has no segments"
                logger.error(error_msg)
                return {
                    "errors": [error_msg],
                    "status": "failed",
                    "initial_summary": {}
                }

            summary = self.summarizer_agent.generate_summary(transcript)

            return {
                "initial_summary": summary.model_dump(),
                "messages": [HumanMessage(content=f"Generated summary with {len(summary.action_items)} actions")],
                "status": "summarized"
            }

        except Exception as e:
            logger.error(f"Summarize node error: {e}")
            return {"errors": [f"Summarize error: {str(e)}"], "status": "failed", "initial_summary": {}}

    def _reflect_node(self, state: MeetingState) -> Dict[str, Any]:
        """Node 3: Self-reflection validation."""
        logger.info(f"[LangGraph] Node: REFLECT - Iteration {state.get('reflection_iterations', 0) + 1}")

        try:
            from src.models.schemas import MeetingSummary, TranscriptData

            # Use final_summary if it exists (from previous improve cycle), otherwise use initial_summary
            summary_dict = state.get("final_summary") or state.get("initial_summary")

            if not summary_dict or summary_dict == {}:
                error_msg = "Cannot reflect: no summary available"
                logger.error(error_msg)
                return {
                    "errors": [error_msg],
                    "status": "failed",
                    "final_summary": state.get("initial_summary", {}),
                    "validation_passed": False
                }

            summary = MeetingSummary(**summary_dict)
            transcript = TranscriptData(**state["transcript"])

            feedback, improved_summary = self.reflection_agent.validate_summary(summary, transcript)
            iterations = state.get("reflection_iterations", 0) + 1

            return {
                "reflection_feedback": feedback.model_dump(),
                "final_summary": improved_summary.model_dump(),
                "validation_passed": feedback.approved,
                "reflection_iterations": iterations,
                "messages": [HumanMessage(
                    content=f"Reflection {iterations}: {'PASSED' if feedback.approved else 'NEEDS IMPROVEMENT'}"
                )]
            }

        except Exception as e:
            logger.error(f"Reflect node error: {e}")
            import traceback
            traceback.print_exc()
            # On error, preserve initial_summary as final_summary so workflow can continue
            return {
                "errors": [f"Reflect error: {str(e)}"],
                "status": "failed",
                "final_summary": state.get("initial_summary", {}),
                "validation_passed": True  # Force continue on error to avoid infinite loop
            }

    def _improve_node(self, state: MeetingState) -> Dict[str, Any]:
        """Node 4: Improvement step."""
        logger.info(f"[LangGraph] Node: IMPROVE")
        return {"messages": [HumanMessage(content="Summary improved")]}

    def _action_node(self, state: MeetingState) -> Dict[str, Any]:
        """Node 5: Create Jira tickets via MCP."""
        logger.info(f"[LangGraph] Node: ACTION (via MCP)")

        try:
            from src.models.schemas import MeetingSummary
            summary = MeetingSummary(**state["final_summary"])

            summary_path = f"data/summaries/{state['meeting_id']}.json"
            self.summarizer_agent.save_summary(summary, summary_path)

            # Create tickets via MCP
            jira_tickets = []
            if summary.action_items:
                jira_tickets = _run_async(self.action_agent.create_jira_tickets(
                    summary.action_items,
                    state["meeting_title"],
                    state["meeting_id"]
                ))

            return {
                "summary_path": summary_path,
                "jira_tickets": jira_tickets,
                "messages": [HumanMessage(content=f"Created {len(jira_tickets)} Jira tickets via MCP")]
            }

        except Exception as e:
            logger.error(f"Action node error: {e}")
            return {"jira_tickets": [], "errors": [f"Action error: {str(e)}"]}

    def _schedule_node(self, state: MeetingState) -> Dict[str, Any]:
        """Node 6: Schedule follow-up via MCP."""
        logger.info(f"[LangGraph] Node: SCHEDULE (via MCP)")

        try:
            from src.models.schemas import MeetingSummary
            summary = MeetingSummary(**state["final_summary"])

            followup = None
            if summary.action_items or summary.unresolved_questions:
                # Use MCP calendar tool - schedule for next week
                from datetime import timedelta
                start_time = datetime.utcnow() + timedelta(days=7)
                end_time = start_time + timedelta(hours=1)

                result = _run_async(self.mcp_server.execute_tool(
                    "calendar_create_event",
                    {
                        "summary": f"Follow-up: {state['meeting_title']}",
                        "description": f"Follow-up for {state['meeting_title']}",
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "attendees": summary.participants
                    }
                ))

                if result.get("success"):
                    followup = result

            return {
                "followup_meeting": followup or {},
                "messages": [HumanMessage(content="Follow-up scheduled via MCP" if followup else "No follow-up needed")]
            }

        except Exception as e:
            logger.error(f"Schedule node error: {e}")
            return {"followup_meeting": {}, "errors": [f"Schedule error: {str(e)}"]}

    def _store_node(self, state: MeetingState) -> Dict[str, Any]:
        """Node 7: Store in knowledge base."""
        logger.info(f"[LangGraph] Node: STORE")

        try:
            from src.models.schemas import TranscriptData
            transcript = TranscriptData(**state["transcript"])
            self.knowledge_agent.add_transcript(transcript)

            # Update database
            db = SessionLocal()
            meeting = db.query(Meeting).filter(Meeting.id == state["meeting_id"]).first()
            if meeting:
                meeting.transcript_path = state["transcript_path"]
                meeting.summary_path = state["summary_path"]
                meeting.status = MeetingStatus.COMPLETED
                meeting.end_time = datetime.utcnow()
                db.commit()
            db.close()

            # Get MCP audit log
            audit_log = self.mcp_server.get_audit_log()

            return {
                "mcp_audit_log": audit_log,
                "messages": [HumanMessage(content=f"Knowledge stored. MCP calls: {len(audit_log)}")]
            }

        except Exception as e:
            logger.error(f"Store node error: {e}")
            return {"errors": [f"Store error: {str(e)}"]}

    def _should_improve(self, state: MeetingState) -> str:
        """Conditional edge: improve or continue."""
        # If workflow has failed, don't loop - continue to end
        if state.get("status") == "failed":
            logger.error("Workflow failed, skipping improvement loop")
            return "continue"

        # If validation passed, continue
        if state.get("validation_passed", False):
            return "continue"

        # If max iterations reached, continue
        if state.get("reflection_iterations", 0) >= 3:
            logger.warning("Max reflection iterations reached")
            return "continue"

        return "improve"

    # Public API

    async def process_meeting(
        self,
        room_name: str,
        meeting_title: str,
        access_token: str
    ) -> Dict[str, Any]:
        """
        Process a meeting through LangGraph + MCP workflow.

        Returns complete results with MCP audit log.
        """
        meeting_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        logger.info(f"Starting LangGraph + MCP workflow for: {meeting_id}")

        # Create DB record
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
        db.close()

        # Join meeting
        await self.listener_agent.join_meeting(room_name, meeting_id, access_token)

        # Initialize state
        initial_state = MeetingState(
            meeting_id=meeting_id,
            meeting_title=meeting_title,
            room_name=room_name,
            start_time=start_time,
            status="initialized",
            transcript={},
            transcript_path="",
            initial_summary={},
            reflection_feedback={},
            final_summary={},
            summary_path="",
            jira_tickets=[],
            followup_meeting={},
            messages=[],
            validation_passed=False,
            reflection_iterations=0,
            errors=[],
            mcp_audit_log=[]
        )

        # Run LangGraph workflow
        try:
            final_state = self.app.invoke(initial_state)

            logger.info("✓ LangGraph + MCP workflow completed")

            return {
                "meeting_id": meeting_id,
                "status": "success",
                "transcript_path": final_state.get("transcript_path"),
                "summary_path": final_state.get("summary_path"),
                "jira_tickets": final_state.get("jira_tickets", []),
                "followup_meeting": final_state.get("followup_meeting", {}),
                "reflection_iterations": final_state.get("reflection_iterations", 0),
                "mcp_audit_log": final_state.get("mcp_audit_log", []),
                "workflow_messages": [msg.content for msg in final_state.get("messages", [])],
                "errors": final_state.get("errors", [])
            }

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return {"meeting_id": meeting_id, "status": "failed", "error": str(e)}

        finally:
            await self.listener_agent.leave_meeting()

    async def query_knowledge(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Query knowledge base via MCP."""
        result = await self.mcp_server.execute_tool(
            "knowledge_query",
            {"query": query, "top_k": top_k}
        )
        return result
