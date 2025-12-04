"""
Unit tests for TeamSync agents
Run with: pytest tests/ -v
"""
import pytest
from datetime import datetime
from src.models.schemas import (
    TranscriptData,
    SpeakerSegment,
    MeetingSummary,
    ActionItem
)


class TestSchemas:
    """Test data models and schemas."""

    def test_speaker_segment_creation(self):
        """Test creating a speaker segment."""
        segment = SpeakerSegment(
            speaker="Alice",
            start_time=0.0,
            end_time=5.0,
            text="Hello world"
        )
        assert segment.speaker == "Alice"
        assert segment.text == "Hello world"

    def test_transcript_data_creation(self):
        """Test creating transcript data."""
        transcript = TranscriptData(
            meeting_id="test-001",
            meeting_title="Test Meeting",
            start_time=datetime.utcnow(),
            segments=[],
            participants=["Alice", "Bob"]
        )
        assert transcript.meeting_id == "test-001"
        assert len(transcript.participants) == 2

    def test_action_item_creation(self):
        """Test creating action item."""
        action = ActionItem(
            title="Complete API documentation",
            description="Document all endpoints",
            assignee="Alice",
            priority="high",
            created_from_meeting="test-001"
        )
        assert action.title == "Complete API documentation"
        assert action.priority == "high"


class TestKnowledgeAgent:
    """Test Knowledge Agent with ChromaDB."""

    @pytest.fixture
    def sample_transcript(self):
        """Create sample transcript for testing."""
        return TranscriptData(
            meeting_id="test-meeting-001",
            meeting_title="API Review",
            start_time=datetime.utcnow(),
            segments=[
                SpeakerSegment(
                    speaker="Alice",
                    start_time=0.0,
                    end_time=5.0,
                    text="We need to implement rate limiting for the API."
                ),
                SpeakerSegment(
                    speaker="Bob",
                    start_time=5.5,
                    end_time=10.0,
                    text="I suggest using Redis for caching and rate limiting."
                ),
            ],
            participants=["Alice", "Bob"]
        )

    def test_add_transcript(self, sample_transcript):
        """Test adding transcript to ChromaDB."""
        from src.agents.knowledge_agent import KnowledgeAgent

        agent = KnowledgeAgent()
        success = agent.add_transcript(sample_transcript)
        assert success is True

    def test_query_knowledge(self):
        """Test querying the knowledge base."""
        from src.agents.knowledge_agent import KnowledgeAgent

        agent = KnowledgeAgent()
        response = agent.query("What was discussed about API?", top_k=3)

        assert response.query == "What was discussed about API?"
        assert response.confidence >= 0.0
        assert response.confidence <= 1.0
        assert isinstance(response.answer, str)

    def test_collection_stats(self):
        """Test getting collection statistics."""
        from src.agents.knowledge_agent import KnowledgeAgent

        agent = KnowledgeAgent()
        stats = agent.get_collection_stats()

        assert "total_segments" in stats
        assert "collection_name" in stats


class TestSummarizerAgent:
    """Test Summarizer Agent."""

    @pytest.fixture
    def sample_transcript(self):
        """Create sample transcript."""
        return TranscriptData(
            meeting_id="test-002",
            meeting_title="Sprint Planning",
            start_time=datetime.utcnow(),
            segments=[
                SpeakerSegment(
                    speaker="Alice",
                    start_time=0.0,
                    end_time=5.0,
                    text="Let's plan the next sprint. We have 10 tickets."
                ),
                SpeakerSegment(
                    speaker="Bob",
                    start_time=5.5,
                    end_time=10.0,
                    text="I'll take the authentication refactoring task."
                ),
            ],
            participants=["Alice", "Bob"]
        )

    def test_generate_summary(self, sample_transcript):
        """Test summary generation."""
        from src.agents.summarizer_agent import SummarizerAgent

        agent = SummarizerAgent()
        summary = agent.generate_summary(sample_transcript)

        assert isinstance(summary, MeetingSummary)
        assert summary.meeting_id == sample_transcript.meeting_id
        assert len(summary.executive_summary) > 0
        assert isinstance(summary.action_items, list)
        assert isinstance(summary.key_decisions, list)


class TestReflectionAgent:
    """Test Self-Reflection Agent."""

    @pytest.fixture
    def sample_summary(self):
        """Create sample summary."""
        return MeetingSummary(
            meeting_id="test-003",
            meeting_title="Test Meeting",
            date=datetime.utcnow(),
            participants=["Alice", "Bob"],
            duration_minutes=30,
            executive_summary="The team discussed API changes.",
            key_decisions=[],
            action_items=[
                ActionItem(
                    title="Update documentation",
                    description="Update API docs",
                    created_from_meeting="test-003"
                )
            ],
            discussion_points=["API design"],
            unresolved_questions=[],
            next_steps="Complete tasks"
        )

    @pytest.fixture
    def sample_transcript(self):
        """Create sample transcript."""
        return TranscriptData(
            meeting_id="test-003",
            meeting_title="Test Meeting",
            start_time=datetime.utcnow(),
            segments=[
                SpeakerSegment(
                    speaker="Alice",
                    start_time=0.0,
                    end_time=5.0,
                    text="We need to update the API documentation."
                )
            ],
            participants=["Alice", "Bob"]
        )

    def test_validate_summary(self, sample_summary, sample_transcript):
        """Test summary validation."""
        from src.agents.reflection_agent import SelfReflectionAgent

        agent = SelfReflectionAgent()
        feedback, improved_summary = agent.validate_summary(
            sample_summary,
            sample_transcript
        )

        assert feedback.logical_coherence_score >= 0.0
        assert feedback.logical_coherence_score <= 1.0
        assert isinstance(feedback.approved, bool)
        assert isinstance(improved_summary, MeetingSummary)


class TestActionAgent:
    """Test Action Agent."""

    @pytest.mark.skip(reason="Requires Jira credentials")
    def test_create_jira_ticket(self):
        """Test creating Jira ticket."""
        from src.agents.action_agent import ActionAgent

        agent = ActionAgent()
        action_items = [
            ActionItem(
                title="Test task",
                description="This is a test task",
                priority="medium",
                created_from_meeting="test-004"
            )
        ]

        tickets = agent.create_jira_tickets(
            action_items,
            "Test Meeting",
            "test-004"
        )

        assert len(tickets) == 1


class TestSchedulerAgent:
    """Test Scheduler Agent."""

    @pytest.mark.skip(reason="Requires Google Calendar credentials")
    def test_get_upcoming_events(self):
        """Test getting upcoming calendar events."""
        from src.agents.scheduler_agent import SchedulerAgent

        agent = SchedulerAgent()
        events = agent.get_upcoming_events(max_results=5)

        assert isinstance(events, list)


class TestOrchestrator:
    """Test Meeting Orchestrator."""

    def test_orchestrator_initialization(self):
        """Test orchestrator can be initialized."""
        from src.orchestrator import MeetingOrchestrator

        orchestrator = MeetingOrchestrator()
        assert orchestrator.listener_agent is not None
        assert orchestrator.knowledge_agent is not None
        assert orchestrator.summarizer_agent is not None

    def test_get_all_meetings(self):
        """Test retrieving all meetings."""
        from src.orchestrator import MeetingOrchestrator

        orchestrator = MeetingOrchestrator()
        meetings = orchestrator.get_all_meetings(limit=10)

        assert isinstance(meetings, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
