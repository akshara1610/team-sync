"""
Test script for LangChain-based Knowledge Agent with RetrievalQA
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.agents.knowledge_agent_langchain import KnowledgeAgentLangChain
from src.models.schemas import TranscriptData, SpeakerSegment
from loguru import logger


def main():
    """Test the LangChain-based Knowledge Agent."""
    logger.info("=" * 60)
    logger.info("Testing LangChain Knowledge Agent with RetrievalQA")
    logger.info("=" * 60)

    # Initialize agent
    logger.info("\n1. Initializing LangChain Knowledge Agent...")
    agent = KnowledgeAgentLangChain()
    logger.info("✓ Agent initialized with:")
    logger.info("  - OpenAI embeddings (text-embedding-3-small)")
    logger.info("  - ChatOpenAI LLM")
    logger.info("  - ChromaDB vector store")
    logger.info("  - RetrievalQA chain")

    # Create sample transcripts
    logger.info("\n2. Creating sample meeting transcripts...")

    transcript1 = TranscriptData(
        meeting_id="test-meeting-langchain-001",
        meeting_title="API Migration Strategy Discussion",
        start_time=datetime.utcnow(),
        segments=[
            SpeakerSegment(
                speaker="Alice",
                start_time=0.0,
                end_time=5.0,
                text="We need to migrate from REST to GraphQL for our new microservices architecture. "
                     "This will give us better performance and flexibility."
            ),
            SpeakerSegment(
                speaker="Bob",
                start_time=5.5,
                end_time=12.0,
                text="I agree. GraphQL will reduce over-fetching. We should also implement Redis "
                     "caching to reduce database load. I propose we start with the user service first."
            ),
            SpeakerSegment(
                speaker="Charlie",
                start_time=12.5,
                end_time=18.0,
                text="Good idea. Let's create a proof of concept by next Friday. I'll handle the "
                     "Redis setup, Bob can work on the GraphQL schema."
            ),
            SpeakerSegment(
                speaker="Alice",
                start_time=18.5,
                end_time=22.0,
                text="Perfect. We also need to update our documentation. The new API docs should "
                     "be ready before we release to staging."
            ),
        ],
        participants=["Alice", "Bob", "Charlie"]
    )

    transcript2 = TranscriptData(
        meeting_id="test-meeting-langchain-002",
        meeting_title="Database Optimization Sprint Planning",
        start_time=datetime.utcnow(),
        segments=[
            SpeakerSegment(
                speaker="Bob",
                start_time=0.0,
                end_time=6.0,
                text="Our PostgreSQL queries are getting slow. We need to add proper indexes "
                     "on the users and orders tables. I've identified 5 slow queries."
            ),
            SpeakerSegment(
                speaker="Alice",
                start_time=6.5,
                end_time=12.0,
                text="Let's also consider adding connection pooling. We're opening too many "
                     "database connections. I'll look into pgBouncer."
            ),
            SpeakerSegment(
                speaker="Charlie",
                start_time=12.5,
                end_time=16.0,
                text="We should run EXPLAIN ANALYZE on all those slow queries first. "
                     "Let's document the query plans before and after optimization."
            ),
        ],
        participants=["Alice", "Bob", "Charlie"]
    )

    # Add transcripts to knowledge base
    logger.info("\n3. Adding transcripts to ChromaDB via LangChain...")
    success1 = agent.add_transcript(transcript1)
    success2 = agent.add_transcript(transcript2)

    if success1 and success2:
        logger.info("✓ Both transcripts added successfully!")
    else:
        logger.error("✗ Failed to add transcripts")
        return

    # Get collection stats
    logger.info("\n4. Collection Statistics:")
    stats = agent.get_collection_stats()
    for key, value in stats.items():
        logger.info(f"  - {key}: {value}")

    # Test RetrievalQA chain
    logger.info("\n5. Testing RetrievalQA Chain...")
    logger.info("=" * 60)

    queries = [
        "What did we decide about the API migration?",
        "What database optimizations were discussed?",
        "Who is responsible for Redis setup?",
        "What is the deadline for the proof of concept?"
    ]

    for i, query in enumerate(queries, 1):
        logger.info(f"\nQuery {i}: {query}")
        logger.info("-" * 60)

        response = agent.query(query, top_k=3)

        logger.info(f"Answer: {response.answer}")
        logger.info(f"Confidence: {response.confidence:.2f}")
        logger.info(f"\nSources ({len(response.sources)}):")

        for j, source in enumerate(response.sources, 1):
            logger.info(f"\n  Source {j}:")
            logger.info(f"    Meeting: {source['meeting_title']}")
            logger.info(f"    Speaker: {source['speaker']}")
            logger.info(f"    Text: {source['text'][:150]}...")

    # Test similarity search
    logger.info("\n" + "=" * 60)
    logger.info("6. Testing Similarity Search (without answer generation)...")
    logger.info("=" * 60)

    search_query = "Redis caching"
    logger.info(f"\nSearching for: {search_query}")

    results = agent.similarity_search(search_query, k=3)

    logger.info(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        logger.info(f"\n  Result {i}:")
        logger.info(f"    Text: {result['text']}")
        logger.info(f"    Meeting: {result['metadata']['meeting_title']}")
        logger.info(f"    Speaker: {result['metadata']['speaker']}")
        logger.info(f"    Relevance: {result['relevance_score']:.3f}")

    # Test meeting context retrieval
    logger.info("\n" + "=" * 60)
    logger.info("7. Testing Full Meeting Context Retrieval...")
    logger.info("=" * 60)

    context = agent.get_meeting_context("test-meeting-langchain-001")
    logger.info(f"\n{context}")

    # Test conversational chain
    logger.info("\n" + "=" * 60)
    logger.info("8. Creating Conversational Chain (with memory)...")
    logger.info("=" * 60)

    conversational_chain = agent.create_conversational_chain()
    logger.info("✓ Conversational chain created!")
    logger.info("  This chain maintains conversation history for follow-up questions")

    logger.info("\n" + "=" * 60)
    logger.info("✓ All tests completed successfully!")
    logger.info("=" * 60)

    logger.info("\nLangChain Integration Features Demonstrated:")
    logger.info("  ✓ OpenAI embeddings for semantic search")
    logger.info("  ✓ ChromaDB vector store via LangChain")
    logger.info("  ✓ RetrievalQA chain for Q&A")
    logger.info("  ✓ Automatic source citation")
    logger.info("  ✓ Similarity search with scores")
    logger.info("  ✓ Conversational chain with memory")


if __name__ == "__main__":
    main()
