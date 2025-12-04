"""
LangGraph-based Meeting Orchestrator
Uses LangGraph for stateful multi-agent workflow coordination.
"""
from typing import Dict, Any, TypedDict, Annotated, Sequence
import operator
from datetime import datetime
import uuid
from loguru import logger

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage

from src.agents.listener_agent import ListenerAgent
from src.agents.knowledge_agent_langchain import KnowledgeAgentLangChain
from src.agents.summarizer_agent import SummarizerAgent
from src.agents.reflection_agent import SelfReflectionAgent
from src.agents.action_agent import ActionAgent
from src.agents.scheduler_agent import SchedulerAgent
from src.models.schemas import MeetingStatus, TranscriptData, MeetingSummary
from src.database.db import SessionLocal, Meeting


class MeetingState(TypedDict):
    """
    State object for LangGraph workflow.
    Tracks all data as it flows through the agent pipeline.
    """
    # Meeting metadata
    meeting_id: str
    meeting_title: str
    room_name: str
    start_time: datetime
    status: str

    # Agent outputs
    transcript: Dict[str, Any]
    transcript_path: str
    initial_summary: Dict[str, Any]
    reflection_feedback: Dict[str, Any]
    final_summary: Dict[str, Any]
    summary_path: str
    jira_tickets: list
    followup_meeting: Dict[str, Any]

    # Messages for agent communication
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Flags and counters
    validation_passed: bool
    reflection_iterations: int
    errors: list


class LangGraphMeetingOrchestrator:
    """
    LangGraph-based orchestrator that uses a state graph to coordinate agents.

    Benefits over manual orchestration:
    - Automatic state management
    - Built-in retries and error handling
    - Visual workflow representation
    - Easy to add conditional branches
    """

    def __init__(self):
        logger.info("Initializing LangGraph Meeting Orchestrator...")

        # Initialize all agents
        self.listener_agent = ListenerAgent()
        self.knowledge_agent = KnowledgeAgentLangChain()  # Using LangChain version
        self.summarizer_agent = SummarizerAgent()
        self.reflection_agent = SelfReflectionAgent()
        self.action_agent = ActionAgent()
        self.scheduler_agent = SchedulerAgent()

        # Build the workflow graph
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()

        logger.info("LangGraph orchestrator initialized with state graph")

    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow for meeting processing.

        Workflow:
        START → Listen → Summarize → Reflect → [Check] → Action → Schedule → Store → END
                                              ↓ (if failed)
                                            Improve → (back to Reflect)
        """
        workflow = StateGraph(MeetingState)

        # Add nodes (each represents an agent or processing step)
        workflow.add_node("listen", self._listen_node)
        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("reflect", self._reflect_node)
        workflow.add_node("improve", self._improve_node)
        workflow.add_node("execute_actions", self._action_node)
        workflow.add_node("schedule_followup", self._schedule_node)
        workflow.add_node("store_knowledge", self._store_node)

        # Define edges (workflow transitions)
        workflow.set_entry_point("listen")
        workflow.add_edge("listen", "summarize")
        workflow.add_edge("summarize", "reflect")

        # Conditional edge: reflection passes or needs improvement
        workflow.add_conditional_edges(
            "reflect",
            self._should_improve,
            {
                "improve": "improve",
                "continue": "execute_actions"
            }
        )

        workflow.add_edge("improve", "reflect")  # Loop back for re-validation
        workflow.add_edge("execute_actions", "schedule_followup")
        workflow.add_edge("schedule_followup", "store_knowledge")
        workflow.add_edge("store_knowledge", END)

        return workflow

    # Node implementations (each processes state and returns updated state)

    def _listen_node(self, state: MeetingState) -> Dict[str, Any]:
        """
        Node 1: Listen Agent - Transcribe meeting
        """
        logger.info(f"[LangGraph] Node: LISTEN - Meeting {state['meeting_id']}")

        try:
            # Get transcript from listener agent
            transcript = self.listener_agent.get_transcript(
                state["meeting_title"],
                state["start_time"]
            )

            # Save transcript
            transcript_path = f"data/transcripts/{state['meeting_id']}.json"
            self.listener_agent.save_transcript(transcript_path)

            return {
                "transcript": transcript.dict(),
                "transcript_path": transcript_path,
                "messages": [HumanMessage(content=f"Transcribed {len(transcript.segments)} segments")],
                "status": "transcribed"
            }

        except Exception as e:
            logger.error(f"Error in listen node: {e}")
            return {
                "errors": [f"Listen error: {str(e)}"],
                "status": "failed"
            }

    def _summarize_node(self, state: MeetingState) -> Dict[str, Any]:
        """
        Node 2: Summarizer Agent - Generate initial summary
        """
        logger.info(f"[LangGraph] Node: SUMMARIZE - Meeting {state['meeting_id']}")

        try:
            # Convert dict back to TranscriptData
            from src.models.schemas import TranscriptData
            transcript = TranscriptData(**state["transcript"])

            # Generate summary
            summary = self.summarizer_agent.generate_summary(transcript)

            return {
                "initial_summary": summary.dict(),
                "messages": [HumanMessage(content=f"Generated summary with {len(summary.action_items)} action items")],
                "status": "summarized"
            }

        except Exception as e:
            logger.error(f"Error in summarize node: {e}")
            return {
                "errors": [f"Summarize error: {str(e)}"],
                "status": "failed"
            }

    def _reflect_node(self, state: MeetingState) -> Dict[str, Any]:
        """
        Node 3: Self-Reflection Agent - Validate summary
        """
        logger.info(f"[LangGraph] Node: REFLECT - Meeting {state['meeting_id']}")

        try:
            from src.models.schemas import MeetingSummary, TranscriptData

            # Get current summary (either initial or improved)
            summary_dict = state.get("final_summary", state["initial_summary"])
            summary = MeetingSummary(**summary_dict)
            transcript = TranscriptData(**state["transcript"])

            # Validate
            feedback, improved_summary = self.reflection_agent.validate_summary(
                summary,
                transcript
            )

            # Increment reflection counter
            iterations = state.get("reflection_iterations", 0) + 1

            return {
                "reflection_feedback": feedback.dict(),
                "final_summary": improved_summary.dict(),
                "validation_passed": feedback.approved,
                "reflection_iterations": iterations,
                "messages": [HumanMessage(
                    content=f"Reflection iteration {iterations}: "
                            f"{'PASSED' if feedback.approved else 'NEEDS IMPROVEMENT'}"
                )],
                "status": "reflected"
            }

        except Exception as e:
            logger.error(f"Error in reflect node: {e}")
            return {
                "errors": [f"Reflect error: {str(e)}"],
                "status": "failed"
            }

    def _improve_node(self, state: MeetingState) -> Dict[str, Any]:
        """
        Node 4: Improvement step - Logged but improvement already done in reflect node
        """
        logger.info(f"[LangGraph] Node: IMPROVE - Iteration {state['reflection_iterations']}")

        # The actual improvement happens in reflect_node
        # This node just logs the improvement step

        return {
            "messages": [HumanMessage(content="Summary improved based on feedback")],
            "status": "improving"
        }

    def _action_node(self, state: MeetingState) -> Dict[str, Any]:
        """
        Node 5: Action Agent - Create Jira tickets
        """
        logger.info(f"[LangGraph] Node: ACTION - Meeting {state['meeting_id']}")

        try:
            from src.models.schemas import MeetingSummary
            summary = MeetingSummary(**state["final_summary"])

            # Save summary
            summary_path = f"data/summaries/{state['meeting_id']}.json"
            self.summarizer_agent.save_summary(summary, summary_path)

            # Create Jira tickets
            jira_tickets = []
            if summary.action_items:
                jira_tickets = self.action_agent.create_jira_tickets(
                    summary.action_items,
                    state["meeting_title"],
                    state["meeting_id"]
                )

            return {
                "summary_path": summary_path,
                "jira_tickets": jira_tickets,
                "messages": [HumanMessage(content=f"Created {len(jira_tickets)} Jira tickets")],
                "status": "actions_created"
            }

        except Exception as e:
            logger.error(f"Error in action node: {e}")
            return {
                "jira_tickets": [],
                "errors": [f"Action error: {str(e)}"],
                "status": "actions_failed"
            }

    def _schedule_node(self, state: MeetingState) -> Dict[str, Any]:
        """
        Node 6: Scheduler Agent - Schedule follow-up
        """
        logger.info(f"[LangGraph] Node: SCHEDULE - Meeting {state['meeting_id']}")

        try:
            from src.models.schemas import MeetingSummary
            summary = MeetingSummary(**state["final_summary"])

            # Schedule follow-up if needed
            followup = None
            if summary.action_items or summary.unresolved_questions:
                followup = self.scheduler_agent.schedule_follow_up_meeting(
                    summary,
                    days_ahead=7
                )

            return {
                "followup_meeting": followup or {},
                "messages": [HumanMessage(content="Follow-up meeting scheduled" if followup else "No follow-up needed")],
                "status": "scheduled"
            }

        except Exception as e:
            logger.error(f"Error in schedule node: {e}")
            return {
                "followup_meeting": {},
                "errors": [f"Schedule error: {str(e)}"],
                "status": "schedule_failed"
            }

    def _store_node(self, state: MeetingState) -> Dict[str, Any]:
        """
        Node 7: Knowledge Agent - Store in vector DB
        """
        logger.info(f"[LangGraph] Node: STORE - Meeting {state['meeting_id']}")

        try:
            from src.models.schemas import TranscriptData
            transcript = TranscriptData(**state["transcript"])

            # Store in ChromaDB via LangChain
            self.knowledge_agent.add_transcript(transcript)

            # Update database
            db = SessionLocal()
            meeting_record = db.query(Meeting).filter(
                Meeting.id == state["meeting_id"]
            ).first()

            if meeting_record:
                meeting_record.transcript_path = state["transcript_path"]
                meeting_record.summary_path = state["summary_path"]
                meeting_record.status = MeetingStatus.COMPLETED
                meeting_record.end_time = datetime.utcnow()
                db.commit()

            db.close()

            return {
                "messages": [HumanMessage(content="Knowledge stored successfully")],
                "status": "completed"
            }

        except Exception as e:
            logger.error(f"Error in store node: {e}")
            return {
                "errors": [f"Store error: {str(e)}"],
                "status": "store_failed"
            }

    def _should_improve(self, state: MeetingState) -> str:
        """
        Conditional edge: Determine if summary needs improvement.
        """
        # Check if validation passed
        if state.get("validation_passed", False):
            return "continue"

        # Check if we've exceeded max iterations
        max_iterations = 3
        if state.get("reflection_iterations", 0) >= max_iterations:
            logger.warning(f"Max reflection iterations ({max_iterations}) reached, proceeding anyway")
            return "continue"

        # Needs improvement
        return "improve"

    async def process_meeting(
        self,
        room_name: str,
        meeting_title: str,
        access_token: str
    ) -> Dict[str, Any]:
        """
        Process a meeting through the LangGraph workflow.

        Args:
            room_name: LiveKit room name
            meeting_title: Meeting title
            access_token: LiveKit access token

        Returns:
            Final state with all results
        """
        meeting_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        logger.info(f"[LangGraph] Starting workflow for meeting: {meeting_id}")

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
        db.close()

        # Join meeting first (outside of graph for async handling)
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
            errors=[]
        )

        # Run the workflow
        try:
            final_state = self.app.invoke(initial_state)

            logger.info(f"[LangGraph] Workflow completed for meeting: {meeting_id}")

            # Format response
            return {
                "meeting_id": meeting_id,
                "status": "success",
                "final_state": final_state,
                "transcript_path": final_state.get("transcript_path"),
                "summary_path": final_state.get("summary_path"),
                "jira_tickets": final_state.get("jira_tickets", []),
                "followup_meeting": final_state.get("followup_meeting", {}),
                "reflection_iterations": final_state.get("reflection_iterations", 0),
                "workflow_messages": [msg.content for msg in final_state.get("messages", [])],
                "errors": final_state.get("errors", [])
            }

        except Exception as e:
            logger.error(f"[LangGraph] Workflow failed: {e}")
            return {
                "meeting_id": meeting_id,
                "status": "failed",
                "error": str(e)
            }

        finally:
            await self.listener_agent.leave_meeting()

    async def query_knowledge(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query knowledge base using LangChain RAG.
        """
        response = self.knowledge_agent.query(query, top_k)
        return response.dict()

    def visualize_workflow(self, output_path: str = "workflow_graph.png"):
        """
        Visualize the LangGraph workflow.
        Requires graphviz to be installed.
        """
        try:
            from IPython.display import Image, display
            display(Image(self.app.get_graph().draw_png()))
            logger.info(f"Workflow visualization saved to: {output_path}")
        except Exception as e:
            logger.warning(f"Could not visualize workflow: {e}")
            logger.info("Install graphviz and pygraphviz to enable visualization")
