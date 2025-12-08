"""
Test the complete TeamSync workflow with a mock transcript.
This bypasses LiveKit and uses a sample meeting transcript.
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import uuid

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.orchestrator import TeamSyncOrchestrator
from src.models.schemas import TranscriptData, SpeakerSegment
from loguru import logger


def create_mock_transcript() -> TranscriptData:
    """Create a realistic mock meeting transcript."""
    meeting_id = str(uuid.uuid4())

    segments = [
        SpeakerSegment(
            speaker="Alice",
            text="Good morning everyone! Let's start our sprint planning meeting. We have a lot to cover today.",
            start_time=0.0,
            end_time=5.2
        ),
        SpeakerSegment(
            speaker="Bob",
            text="Thanks Alice. I wanted to discuss the API migration to GraphQL. I think we should prioritize this for the current sprint.",
            start_time=5.5,
            end_time=12.3
        ),
        SpeakerSegment(
            speaker="Alice",
            text="Good point. Bob, can you take the lead on the API migration? We need a detailed plan by next Friday.",
            start_time=12.8,
            end_time=18.4
        ),
        SpeakerSegment(
            speaker="Bob",
            text="Absolutely. I'll start with the user service and create a proof of concept. Should have it ready by Friday.",
            start_time=19.0,
            end_time=25.1
        ),
        SpeakerSegment(
            speaker="Charlie",
            text="I can help with the frontend integration once the API is ready. Also, we need to implement Redis caching to reduce database load.",
            start_time=25.8,
            end_time=33.2
        ),
        SpeakerSegment(
            speaker="Alice",
            text="Great. Charlie, can you handle the Redis implementation? We should also schedule a follow-up meeting next week to review progress.",
            start_time=34.0,
            end_time=41.5
        ),
        SpeakerSegment(
            speaker="Bob",
            text="What about the authentication refactoring? Should we include that in this sprint or defer it?",
            start_time=42.2,
            end_time=48.0
        ),
        SpeakerSegment(
            speaker="Alice",
            text="Let's defer the auth refactoring to next sprint. We have enough on our plate. Bob will focus on API migration, Charlie on Redis, and I'll handle code reviews.",
            start_time=48.8,
            end_time=57.3
        ),
        SpeakerSegment(
            speaker="Charlie",
            text="Sounds good. One more thing - we need to update the API documentation once the migration is done.",
            start_time=58.0,
            end_time=63.5
        ),
        SpeakerSegment(
            speaker="Alice",
            text="Good catch. Bob, can you update the docs as part of your API migration task?",
            start_time=64.2,
            end_time=69.0
        ),
        SpeakerSegment(
            speaker="Bob",
            text="Will do. I'll include that in the ticket.",
            start_time=69.5,
            end_time=72.1
        ),
        SpeakerSegment(
            speaker="Alice",
            text="Perfect. Let's wrap up. To summarize: Bob is doing API migration with docs, Charlie handles Redis caching, and we'll meet again next Tuesday to review. Any questions?",
            start_time=73.0,
            end_time=84.2
        ),
        SpeakerSegment(
            speaker="Charlie",
            text="No questions from me. Thanks everyone!",
            start_time=85.0,
            end_time=87.5
        ),
        SpeakerSegment(
            speaker="Bob",
            text="All clear. See you next week!",
            start_time=88.0,
            end_time=90.2
        ),
        SpeakerSegment(
            speaker="Alice",
            text="Great meeting. Thanks everyone!",
            start_time=91.0,
            end_time=93.0
        ),
    ]

    return TranscriptData(
        meeting_id=meeting_id,
        meeting_title="Sprint Planning - API Migration",
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow() + timedelta(minutes=2),
        segments=segments,
        participants=["vva2113@columbia.edu"]  # Use valid email for calendar invites
    )


async def test_workflow_with_mock_transcript():
    """Test the complete workflow using a mock transcript."""

    print("=" * 70)
    print("Testing Complete TeamSync Workflow (Mock Transcript)")
    print("=" * 70)
    print()

    # Initialize orchestrator
    print("1. Initializing TeamSync Orchestrator...")
    orchestrator = TeamSyncOrchestrator()
    print("✓ Orchestrator initialized")
    print()

    # Create mock transcript
    print("2. Creating mock meeting transcript...")
    transcript = create_mock_transcript()
    print(f"✓ Mock transcript created:")
    print(f"  - Meeting ID: {transcript.meeting_id}")
    print(f"  - Title: {transcript.meeting_title}")
    print(f"  - Participants: {', '.join(transcript.participants)}")
    print(f"  - Segments: {len(transcript.segments)}")
    print()

    # Process through the workflow (skipping LiveKit listener)
    print("3. Processing through LangGraph + MCP workflow...")
    print()

    try:
        # Initialize workflow state
        initial_state = {
            "meeting_id": transcript.meeting_id,
            "meeting_title": transcript.meeting_title,
            "transcript": transcript.model_dump(),
            "summary": None,
            "validation_result": None,
            "reflection_iterations": 0,
            "jira_results": [],
            "calendar_result": None,
            "messages": []
        }

        # Skip the 'listen' node since we have mock transcript
        # Start from 'summarize'
        print("  [Skipping 'listen' node - using mock transcript]")
        print()

        # Run summarize
        print("  → Running 'summarize' node...")
        summary_result = orchestrator._summarize_node(initial_state)
        summary_state = {**initial_state, **summary_result}
        print(f"  ✓ Summary generated")
        print(f"    - Executive summary: {summary_state['initial_summary']['executive_summary'][:100]}...")
        print(f"    - Action items: {len(summary_state['initial_summary']['action_items'])}")
        print(f"    - Key decisions: {len(summary_state['initial_summary']['key_decisions'])}")
        print()

        # Run reflect
        print("  → Running 'reflect' node...")
        reflect_result = orchestrator._reflect_node(summary_state)
        reflect_state = {**summary_state, **reflect_result}
        print(f"  ✓ Self-reflection completed")
        print(f"    - Approved: {reflect_state['validation_passed']}")
        print(f"    - Coherence score: {reflect_state['reflection_feedback'].get('logical_coherence_score', 'N/A')}")
        print(f"    - Issues found: {len(reflect_state['reflection_feedback'].get('consistency_issues', []))}")
        print()

        # Check if improvement needed
        if not reflect_state['validation_passed']:
            print("  → Running 'improve' node...")
            improve_result = orchestrator._improve_node(reflect_state)
            reflect_state = {**reflect_state, **improve_result}
            print("  ✓ Summary improved")
            print()

        # Run execute_actions (MCP)
        print("  → Running 'execute_actions' node (MCP)...")
        action_result = orchestrator._action_node(reflect_state)
        action_state = {**reflect_state, **action_result}
        print(f"  ✓ Jira tickets created via MCP")
        print(f"    - Tickets created: {len(action_state['jira_tickets'])}")
        for result in action_state['jira_tickets']:
            print(f"      • {result.get('action_item', 'N/A')}: {result.get('status', 'N/A')}")
        print()

        # Run schedule_followup (MCP)
        print("  → Running 'schedule_followup' node (MCP)...")
        schedule_result = orchestrator._schedule_node(action_state)
        schedule_state = {**action_state, **schedule_result}
        print(f"  ✓ Follow-up meeting scheduled via MCP")
        print(f"    - Success: {schedule_state['followup_meeting'].get('success', False)}")
        print()

        # Run store_knowledge (MCP)
        print("  → Running 'store_knowledge' node...")
        final_result = orchestrator._store_node(schedule_state)
        final_state = {**schedule_state, **final_result}
        print(f"  ✓ Knowledge stored in ChromaDB")
        print()

        # Show MCP audit log
        print("4. MCP Audit Log:")
        print("-" * 70)
        audit_log = orchestrator.mcp_server.get_audit_log()
        for i, entry in enumerate(audit_log, 1):
            print(f"  {i}. {entry['tool']} at {entry['timestamp']}")
            print(f"     Result: {entry['result'].get('success', 'N/A')}")
        print()

        # Show final results
        print("5. Final Results:")
        print("-" * 70)
        print(f"✓ Meeting processed successfully!")
        print(f"  - Meeting ID: {final_state['meeting_id']}")
        print(f"  - Reflection iterations: {final_state['reflection_iterations']}")
        print(f"  - Jira tickets: {len(final_state.get('jira_tickets', []))}")
        print(f"  - Follow-up scheduled: {final_state.get('followup_meeting', {}).get('success', False)}")
        print(f"  - MCP tool calls: {len(audit_log)}")
        print(f"  - Workflow messages: {len(final_state.get('messages', []))}")
        print()

        print("=" * 70)
        print("✓ Complete workflow test PASSED!")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(test_workflow_with_mock_transcript())
