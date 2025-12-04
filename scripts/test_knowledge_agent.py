"""
Test script for Knowledge Agent with ChromaDB
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.agents.knowledge_agent import KnowledgeAgent
from src.models.schemas import TranscriptData, SpeakerSegment
from loguru import logger


def main():
    """Test the Knowledge Agent."""
    logger.info("Testing Knowledge Agent with ChromaDB...")

    # Initialize agent
    agent = KnowledgeAgent()

    # Create sample transcript
    sample_transcript = TranscriptData(
        meeting_id="test-meeting-001",
        meeting_title="API Architecture Review",
        start_time=datetime.utcnow(),
        segments=[
            SpeakerSegment(
                speaker="Alice",
                start_time=0.0,
                end_time=5.0,
                text="We need to discuss the API migration strategy for the new microservices architecture."
            ),
            SpeakerSegment(
                speaker="Bob",
                start_time=5.5,
                end_time=10.0,
                text="I suggest we use GraphQL instead of REST for better performance."
            ),
            SpeakerSegment(
                speaker="Charlie",
                start_time=10.5,
                end_time=15.0,
                text="Agreed. Let's also implement caching with Redis to reduce database load."
            ),
        ],
        participants=["Alice", "Bob", "Charlie"]
    )

    # Add to knowledge base
    logger.info("Adding sample transcript to ChromaDB...")
    success = agent.add_transcript(sample_transcript)

    if success:
        logger.info("✓ Transcript added successfully!")

        # Test query
        logger.info("\nQuerying: 'What was discussed about API?'")
        response = agent.query("What was discussed about API?", top_k=3)

        logger.info(f"\nAnswer: {response.answer}")
        logger.info(f"Confidence: {response.confidence:.2f}")
        logger.info(f"\nSources ({len(response.sources)}):")
        for i, source in enumerate(response.sources, 1):
            logger.info(f"{i}. {source['speaker']}: {source['text'][:100]}...")

        # Get stats
        stats = agent.get_collection_stats()
        logger.info(f"\nCollection Stats: {stats}")

    else:
        logger.error("✗ Failed to add transcript")


if __name__ == "__main__":
    main()
