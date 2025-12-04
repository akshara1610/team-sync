"""
Enhanced Knowledge Agent that stores both transcripts AND summaries in ChromaDB
This allows RAG queries over both raw conversations and structured summaries
"""
from typing import List, Dict, Any
from loguru import logger
from src.agents.knowledge_agent import KnowledgeAgent
from src.models.schemas import MeetingSummary


class EnhancedKnowledgeAgent(KnowledgeAgent):
    """
    Extended Knowledge Agent that also stores meeting summaries in ChromaDB.
    This enables RAG queries over:
    1. Raw transcript segments (speaker-level detail)
    2. Structured summaries (high-level decisions and actions)
    """

    def add_summary(self, summary: MeetingSummary) -> bool:
        """
        Add a meeting summary to the vector database.
        Stores executive summary, decisions, and action items as searchable documents.

        Args:
            summary: Complete meeting summary

        Returns:
            bool: Success status
        """
        try:
            documents = []
            metadatas = []
            ids = []

            # Add executive summary
            documents.append(summary.executive_summary)
            metadatas.append({
                "meeting_id": summary.meeting_id,
                "meeting_title": summary.meeting_title,
                "document_type": "executive_summary",
                "timestamp": summary.date.isoformat()
            })
            ids.append(f"{summary.meeting_id}_summary")

            # Add each key decision
            for idx, decision in enumerate(summary.key_decisions):
                doc_text = f"Decision: {decision.decision}\nContext: {decision.context}"
                documents.append(doc_text)
                metadatas.append({
                    "meeting_id": summary.meeting_id,
                    "meeting_title": summary.meeting_title,
                    "document_type": "decision",
                    "participants": ",".join(decision.participants),
                    "timestamp": summary.date.isoformat()
                })
                ids.append(f"{summary.meeting_id}_decision_{idx}")

            # Add each action item
            for idx, action in enumerate(summary.action_items):
                doc_text = f"Action: {action.title}\nDescription: {action.description}"
                if action.assignee:
                    doc_text += f"\nAssignee: {action.assignee}"

                documents.append(doc_text)
                metadatas.append({
                    "meeting_id": summary.meeting_id,
                    "meeting_title": summary.meeting_title,
                    "document_type": "action_item",
                    "assignee": action.assignee or "unassigned",
                    "priority": action.priority,
                    "timestamp": summary.date.isoformat()
                })
                ids.append(f"{summary.meeting_id}_action_{idx}")

            # Add discussion points as a single document
            if summary.discussion_points:
                discussion_text = "Discussion Points:\n" + "\n".join(
                    f"- {point}" for point in summary.discussion_points
                )
                documents.append(discussion_text)
                metadatas.append({
                    "meeting_id": summary.meeting_id,
                    "meeting_title": summary.meeting_title,
                    "document_type": "discussion_points",
                    "timestamp": summary.date.isoformat()
                })
                ids.append(f"{summary.meeting_id}_discussion")

            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(
                f"Added summary with {len(documents)} components from meeting "
                f"{summary.meeting_id} to ChromaDB"
            )
            return True

        except Exception as e:
            logger.error(f"Error adding summary to ChromaDB: {e}")
            return False

    def query_decisions(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query specifically for decisions made in past meetings.

        Args:
            query: Natural language query
            top_k: Number of results

        Returns:
            List of decision documents
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"document_type": "decision"}
            )

            decisions = []
            if results['documents']:
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    decisions.append({
                        "meeting_id": meta.get("meeting_id"),
                        "meeting_title": meta.get("meeting_title"),
                        "decision_text": doc,
                        "participants": meta.get("participants", "").split(","),
                        "timestamp": meta.get("timestamp")
                    })

            return decisions

        except Exception as e:
            logger.error(f"Error querying decisions: {e}")
            return []

    def query_action_items(
        self,
        query: str = None,
        assignee: str = None,
        priority: str = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query for action items with optional filters.

        Args:
            query: Natural language query (optional)
            assignee: Filter by assignee (optional)
            priority: Filter by priority (optional)
            top_k: Number of results

        Returns:
            List of action item documents
        """
        try:
            # Build where clause
            where_clause = {"document_type": "action_item"}
            if assignee:
                where_clause["assignee"] = assignee
            if priority:
                where_clause["priority"] = priority

            # Query
            if query:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where=where_clause
                )
            else:
                # If no query, just filter
                results = self.collection.get(
                    where=where_clause,
                    limit=top_k
                )

            actions = []
            documents = results.get('documents', results.get('documents', []))
            metadatas = results.get('metadatas', results.get('metadatas', []))

            # Handle both query and get responses
            if query and documents:
                docs = documents[0]
                metas = metadatas[0]
            else:
                docs = documents
                metas = metadatas

            for doc, meta in zip(docs, metas):
                actions.append({
                    "meeting_id": meta.get("meeting_id"),
                    "meeting_title": meta.get("meeting_title"),
                    "action_text": doc,
                    "assignee": meta.get("assignee"),
                    "priority": meta.get("priority"),
                    "timestamp": meta.get("timestamp")
                })

            return actions

        except Exception as e:
            logger.error(f"Error querying action items: {e}")
            return []

    def get_meeting_summary_from_db(self, meeting_id: str) -> Dict[str, Any]:
        """
        Retrieve all summary components for a specific meeting from ChromaDB.

        Args:
            meeting_id: Meeting identifier

        Returns:
            Dictionary with all summary components
        """
        try:
            results = self.collection.get(
                where={"meeting_id": meeting_id}
            )

            summary_components = {
                "executive_summary": None,
                "decisions": [],
                "action_items": [],
                "discussion_points": None,
                "transcript_segments": []
            }

            if results['documents']:
                for doc, meta in zip(results['documents'], results['metadatas']):
                    doc_type = meta.get("document_type")

                    if doc_type == "executive_summary":
                        summary_components["executive_summary"] = doc
                    elif doc_type == "decision":
                        summary_components["decisions"].append({
                            "text": doc,
                            "participants": meta.get("participants", "").split(",")
                        })
                    elif doc_type == "action_item":
                        summary_components["action_items"].append({
                            "text": doc,
                            "assignee": meta.get("assignee"),
                            "priority": meta.get("priority")
                        })
                    elif doc_type == "discussion_points":
                        summary_components["discussion_points"] = doc
                    else:
                        # Transcript segment
                        summary_components["transcript_segments"].append({
                            "speaker": meta.get("speaker"),
                            "text": doc
                        })

            return summary_components

        except Exception as e:
            logger.error(f"Error retrieving meeting summary: {e}")
            return {}
