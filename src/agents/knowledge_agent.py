from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from loguru import logger
from src.config import settings
from src.models.schemas import RAGQuery, RAGResponse, TranscriptData


class KnowledgeAgent:
    """
    Knowledge Agent (RAG): Maintains a ChromaDB vector database of historical
    meeting transcripts embedded using sentence-transformers. Responds to queries
    during meetings by retrieving relevant context from past discussions.
    """

    def __init__(self):
        logger.info("Initializing Knowledge Agent with ChromaDB...")

        # Initialize ChromaDB client
        self.chroma_client = chromadb.Client(ChromaSettings(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            anonymized_telemetry=False
        ))

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize embedding model
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

        # Initialize LLM for answer generation
        self.llm = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3,
            model_name=settings.SUMMARIZER_MODEL
        )

        logger.info("Knowledge Agent initialized successfully")

    def add_transcript(self, transcript: TranscriptData) -> bool:
        """
        Add a meeting transcript to the vector database.

        Args:
            transcript: Complete transcript data

        Returns:
            bool: Success status
        """
        try:
            documents = []
            metadatas = []
            ids = []

            # Process each segment and create embeddings
            for idx, segment in enumerate(transcript.segments):
                doc_text = f"{segment.speaker}: {segment.text}"
                documents.append(doc_text)

                metadata = {
                    "meeting_id": transcript.meeting_id,
                    "meeting_title": transcript.meeting_title,
                    "speaker": segment.speaker,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "timestamp": transcript.start_time.isoformat()
                }
                metadatas.append(metadata)

                # Create unique ID
                doc_id = f"{transcript.meeting_id}_seg_{idx}"
                ids.append(doc_id)

            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(documents)} segments from meeting {transcript.meeting_id} to ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Error adding transcript to ChromaDB: {e}")
            return False

    def query(
        self,
        query: str,
        top_k: int = None,
        meeting_id_filter: Optional[str] = None
    ) -> RAGResponse:
        """
        Query the knowledge base using semantic search.

        Args:
            query: Natural language query
            top_k: Number of results to retrieve
            meeting_id_filter: Optional filter by meeting ID

        Returns:
            RAG response with answer and sources
        """
        try:
            if top_k is None:
                top_k = settings.KNOWLEDGE_AGENT_TOP_K

            # Build where clause for filtering
            where_clause = None
            if meeting_id_filter:
                where_clause = {"meeting_id": meeting_id_filter}

            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause
            )

            # Extract results
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []

            if not documents:
                return RAGResponse(
                    query=query,
                    answer="No relevant information found in past meetings.",
                    sources=[],
                    confidence=0.0
                )

            # Build context from retrieved documents
            context = self._build_context(documents, metadatas)

            # Generate answer using LLM
            answer = self._generate_answer(query, context)

            # Build sources
            sources = []
            for doc, meta, dist in zip(documents, metadatas, distances):
                sources.append({
                    "meeting_id": meta.get("meeting_id"),
                    "meeting_title": meta.get("meeting_title"),
                    "speaker": meta.get("speaker"),
                    "text": doc,
                    "relevance_score": 1 - dist  # Convert distance to similarity
                })

            # Calculate confidence based on relevance scores
            avg_relevance = sum(1 - d for d in distances) / len(distances)

            return RAGResponse(
                query=query,
                answer=answer,
                sources=sources,
                confidence=avg_relevance
            )

        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return RAGResponse(
                query=query,
                answer=f"Error retrieving information: {str(e)}",
                sources=[],
                confidence=0.0
            )

    def _build_context(self, documents: List[str], metadatas: List[Dict]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        for doc, meta in zip(documents, metadatas):
            context_parts.append(
                f"Meeting: {meta.get('meeting_title', 'Unknown')} "
                f"(ID: {meta.get('meeting_id')})\n"
                f"Speaker: {meta.get('speaker')}\n"
                f"Content: {doc}\n"
            )
        return "\n---\n".join(context_parts)

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM with retrieved context."""
        prompt_template = """You are a helpful assistant that answers questions about past meetings.
Use the following context from previous meetings to answer the question.
If you cannot find the answer in the context, say so.

Context from past meetings:
{context}

Question: {question}

Answer: Provide a clear, concise answer based on the context. Include specific references to meetings and speakers when relevant."""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        try:
            formatted_prompt = prompt.format(context=context, question=query)
            answer = self.llm(formatted_prompt)
            return answer.strip()

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Unable to generate answer at this time."

    def search_meetings(
        self,
        keyword: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for meetings containing specific keywords.

        Args:
            keyword: Keyword to search for
            top_k: Number of results

        Returns:
            List of matching segments
        """
        try:
            results = self.collection.query(
                query_texts=[keyword],
                n_results=top_k
            )

            segments = []
            if results['documents']:
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    segments.append({
                        "meeting_id": meta.get("meeting_id"),
                        "meeting_title": meta.get("meeting_title"),
                        "speaker": meta.get("speaker"),
                        "text": doc,
                        "timestamp": meta.get("timestamp")
                    })

            return segments

        except Exception as e:
            logger.error(f"Error searching meetings: {e}")
            return []

    def get_meeting_history(self, meeting_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all segments from a specific meeting.

        Args:
            meeting_id: Meeting identifier

        Returns:
            List of all segments from the meeting
        """
        try:
            results = self.collection.get(
                where={"meeting_id": meeting_id}
            )

            segments = []
            if results['documents']:
                for doc, meta in zip(results['documents'], results['metadatas']):
                    segments.append({
                        "speaker": meta.get("speaker"),
                        "text": doc,
                        "start_time": meta.get("start_time"),
                        "end_time": meta.get("end_time")
                    })

            # Sort by start_time
            segments.sort(key=lambda x: x.get("start_time", 0))
            return segments

        except Exception as e:
            logger.error(f"Error retrieving meeting history: {e}")
            return []

    def clear_collection(self):
        """Clear all data from the collection (use with caution)."""
        try:
            self.chroma_client.delete_collection(settings.CHROMA_COLLECTION_NAME)
            self.collection = self.chroma_client.create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "total_segments": count,
                "collection_name": settings.CHROMA_COLLECTION_NAME,
                "persist_directory": settings.CHROMA_PERSIST_DIR
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
