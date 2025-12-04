"""
Test script for Summarizer and Self-Reflection Agents
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.agents.summarizer_agent import SummarizerAgent
from src.agents.reflection_agent import SelfReflectionAgent
from src.models.schemas import TranscriptData, SpeakerSegment
from loguru import logger


def main():
    """Test the Summarizer and Self-Reflection Agents."""
    logger.info("Testing Summarizer and Self-Reflection Agents...")

    # Create sample transcript
    sample_transcript = TranscriptData(
        meeting_id="test-meeting-002",
        meeting_title="Sprint Planning Meeting",
        start_time=datetime.utcnow(),
        segments=[
            SpeakerSegment(
                speaker="Alice",
                start_time=0.0,
                end_time=5.0,
                text="Let's review the backlog for next sprint. We have 15 tickets ready."
            ),
            SpeakerSegment(
                speaker="Bob",
                start_time=5.5,
                end_time=10.0,
                text="I'll take the authentication refactoring ticket. It's high priority."
            ),
            SpeakerSegment(
                speaker="Charlie",
                start_time=10.5,
                end_time=15.0,
                text="I can work on the API rate limiting feature. Should be done by Friday."
            ),
            SpeakerSegment(
                speaker="Alice",
                start_time=15.5,
                end_time=20.0,
                text="Great. Bob, can you also review Charlie's pull request before end of day?"
            ),
            SpeakerSegment(
                speaker="Bob",
                start_time=20.5,
                end_time=22.0,
                text="Sure, I'll review it this afternoon."
            ),
        ],
        participants=["Alice", "Bob", "Charlie"]
    )

    # Test Summarizer
    logger.info("\n1. Testing Summarizer Agent...")
    summarizer = SummarizerAgent()
    summary = summarizer.generate_summary(sample_transcript)

    logger.info(f"\n✓ Executive Summary:\n{summary.executive_summary}")
    logger.info(f"\n✓ Key Decisions: {len(summary.key_decisions)}")
    for decision in summary.key_decisions:
        logger.info(f"  - {decision.decision}")

    logger.info(f"\n✓ Action Items: {len(summary.action_items)}")
    for action in summary.action_items:
        assignee = f" ({action.assignee})" if action.assignee else ""
        logger.info(f"  - {action.title}{assignee}")

    # Test Self-Reflection
    logger.info("\n2. Testing Self-Reflection Agent...")
    reflection_agent = SelfReflectionAgent()
    feedback, improved_summary = reflection_agent.validate_summary(
        summary,
        sample_transcript
    )

    logger.info(f"\n✓ Reflection Results:")
    logger.info(f"  - Approved: {feedback.approved}")
    logger.info(f"  - Factually Consistent: {feedback.is_factually_consistent}")
    logger.info(f"  - Coherence Score: {feedback.logical_coherence_score:.2f}")
    logger.info(f"  - Issues Found: {len(feedback.consistency_issues) + len(feedback.missing_action_items)}")

    if feedback.consistency_issues:
        logger.info(f"\n  Consistency Issues:")
        for issue in feedback.consistency_issues:
            logger.info(f"    - {issue}")

    if feedback.missing_action_items:
        logger.info(f"\n  Missing Action Items:")
        for item in feedback.missing_action_items:
            logger.info(f"    - {item}")

    logger.info("\n✓ Test completed successfully!")


if __name__ == "__main__":
    main()
