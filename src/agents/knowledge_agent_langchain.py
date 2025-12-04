"""
Enhanced Knowledge Agent using LangChain's RetrievalQA and ChromaDB integration.
This replaces the manual RAG implementation with LangChain's built-in components.
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import chromadb
from chromadb.config import Settings as ChromaSettings

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import settings
from src.models.schemas import RAGQuery, RAGResponse, TranscriptData


class KnowledgeAgentLangChain:
    """
    Knowledge Agent using full LangChain stack for RAG.

    Uses:
    - langchain-chroma for vector store integration
    - langchain-openai for embeddings and LLM
    - RetrievalQA chain for question answering
    """

    def __init__(self):
        logger.info("Initializing Knowledge Agent with LangChain + ChromaDB...")

        # Initialize OpenAI embeddings (replaces sentence-transformers)
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model="text-embedding-3-small"  # Latest embedding model
        )

        # Initialize ChatOpenAI for LLM
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.SUMMARIZER_MODEL,
            temperature=0.3
        )

        # Initialize Chroma vector store with LangChain
        self.vector_store = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )

        # Create custom prompt template for RAG
        self.qa_prompt_template = """You are a helpful AI assistant that answers questions about past meetings.
Use the following context from previous meetings to answer the question accurately.
If the answer cannot be found in the context, clearly state that you don't have that information.

Context from past meetings:
{context}

Question: {question}

Instructions:
- Provide a clear, concise answer based only on the context
- Include specific references to meetings, speakers, and decisions when relevant
- If you're not certain, express appropriate uncertainty
- Use bullet points for multiple items

Answer:"""

        self.qa_prompt = PromptTemplate(
            template=self.qa_prompt_template,
            input_variables=["context", "question"]
        )

        # Create RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff" puts all docs in prompt, "map_reduce" for larger contexts
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": settings.KNOWLEDGE_AGENT_TOP_K}
            ),
            chain_type_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True
        )

        logger.info("Knowledge Agent initialized with LangChain successfully")

    def add_transcript(self, transcript: TranscriptData) -> bool:
        """
        Add a meeting transcript to the vector database using LangChain Documents.

        Args:
            transcript: Complete transcript data

        Returns:
            bool: Success status
        """
        try:
            documents = []

            # Convert each segment to a LangChain Document
            for idx, segment in enumerate(transcript.segments):
                # Create document text
                doc_text = f"{segment.speaker}: {segment.text}"

                # Create metadata
                metadata = {
                    "meeting_id": transcript.meeting_id,
                    "meeting_title": transcript.meeting_title,
                    "speaker": segment.speaker,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "timestamp": transcript.start_time.isoformat(),
                    "segment_index": idx
                }

                # Create LangChain Document
                doc = Document(
                    page_content=doc_text,
                    metadata=metadata
                )
                documents.append(doc)

            # Add documents to vector store
            self.vector_store.add_documents(documents)

            logger.info(
                f"Added {len(documents)} segments from meeting {transcript.meeting_id} "
                f"to ChromaDB via LangChain"
            )
            return True

        except Exception as e:
            logger.error(f"Error adding transcript to vector store: {e}")
            return False

    def query(
        self,
        query: str,
        top_k: int = None,
        meeting_id_filter: Optional[str] = None
    ) -> RAGResponse:
        """
        Query the knowledge base using LangChain's RetrievalQA chain.

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

            # Update retriever with custom k if different from default
            if top_k != settings.KNOWLEDGE_AGENT_TOP_K:
                self.qa_chain.retriever.search_kwargs["k"] = top_k

            # Add filter if meeting_id specified
            if meeting_id_filter:
                self.qa_chain.retriever.search_kwargs["filter"] = {
                    "meeting_id": meeting_id_filter
                }

            # Run the RetrievalQA chain
            result = self.qa_chain({"query": query})

            answer = result["result"]
            source_documents = result.get("source_documents", [])

            # Build sources from retrieved documents
            sources = []
            relevance_scores = []

            for doc in source_documents:
                metadata = doc.metadata
                sources.append({
                    "meeting_id": metadata.get("meeting_id"),
                    "meeting_title": metadata.get("meeting_title"),
                    "speaker": metadata.get("speaker"),
                    "text": doc.page_content,
                    "timestamp": metadata.get("timestamp")
                })
                # Approximate relevance (RetrievalQA doesn't expose scores by default)
                relevance_scores.append(0.8)  # Placeholder

            # Calculate confidence
            confidence = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

            # Reset filters
            if meeting_id_filter:
                self.qa_chain.retriever.search_kwargs.pop("filter", None)

            return RAGResponse(
                query=query,
                answer=answer,
                sources=sources,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return RAGResponse(
                query=query,
                answer=f"Error retrieving information: {str(e)}",
                sources=[],
                confidence=0.0
            )

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search without answer generation.

        Args:
            query: Search query
            k: Number of results
            filter_dict: Optional metadata filters

        Returns:
            List of similar documents with metadata
        """
        try:
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                filter=filter_dict
            )

            documents = []
            for doc, score in results:
                documents.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(1 / (1 + score))  # Convert distance to similarity
                })

            return documents

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

    def get_meeting_context(self, meeting_id: str) -> str:
        """
        Get full context for a specific meeting using LangChain.

        Args:
            meeting_id: Meeting identifier

        Returns:
            Formatted meeting context
        """
        try:
            # Search for all documents from this meeting
            docs = self.vector_store.similarity_search(
                "",  # Empty query
                k=100,  # Get many results
                filter={"meeting_id": meeting_id}
            )

            if not docs:
                return f"No context found for meeting: {meeting_id}"

            # Sort by segment index
            docs.sort(key=lambda x: x.metadata.get("segment_index", 0))

            # Build formatted context
            context_parts = [
                f"Meeting: {docs[0].metadata.get('meeting_title', 'Unknown')}",
                f"Meeting ID: {meeting_id}",
                f"Timestamp: {docs[0].metadata.get('timestamp', 'Unknown')}",
                "",
                "Transcript:"
            ]

            for doc in docs:
                context_parts.append(doc.page_content)

            return "\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error retrieving meeting context: {e}")
            return ""

    def delete_meeting(self, meeting_id: str) -> bool:
        """
        Delete all documents for a specific meeting.

        Args:
            meeting_id: Meeting identifier

        Returns:
            Success status
        """
        try:
            # Get all document IDs for this meeting
            docs = self.vector_store.get(
                where={"meeting_id": meeting_id}
            )

            if docs and "ids" in docs:
                self.vector_store.delete(ids=docs["ids"])
                logger.info(f"Deleted {len(docs['ids'])} documents for meeting {meeting_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error deleting meeting: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            # Get collection from the underlying ChromaDB client
            collection = self.vector_store._collection
            count = collection.count()

            return {
                "total_documents": count,
                "collection_name": settings.CHROMA_COLLECTION_NAME,
                "persist_directory": settings.CHROMA_PERSIST_DIR,
                "embedding_model": "text-embedding-3-small (OpenAI)"
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def create_conversational_chain(self):
        """
        Create a conversational retrieval chain with memory.
        Useful for interactive Q&A sessions.
        """
        from langchain.chains import ConversationalRetrievalChain
        from langchain.memory import ConversationBufferMemory

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": settings.KNOWLEDGE_AGENT_TOP_K}
            ),
            memory=memory,
            return_source_documents=True
        )

        return conversational_chain
