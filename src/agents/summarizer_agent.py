from typing import List
import json
import time
import re
from datetime import datetime
from loguru import logger
import google.generativeai as genai
from src.config import settings
from src.models.schemas import (
    TranscriptData,
    MeetingSummary,
    ActionItem,
    KeyDecision
)


class SummarizerAgent:
    """
    Summarizer Agent: Processes complete transcripts to generate Minutes of Meeting (MoM),
    extract key decisions, identify action items with assignees, and highlight unresolved questions.
    """

    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(settings.SUMMARIZER_MODEL)
        logger.info(f"Summarizer Agent initialized with Gemini model: {settings.SUMMARIZER_MODEL}")

    def generate_summary(self, transcript: TranscriptData) -> MeetingSummary:
        """
        Generate comprehensive meeting summary from transcript.

        Args:
            transcript: Complete transcript data

        Returns:
            Meeting summary with action items and decisions
        """
        try:
            logger.info(f"Generating summary for meeting: {transcript.meeting_id}")

            # Build transcript text
            transcript_text = self._build_transcript_text(transcript)

            # Generate executive summary
            executive_summary = self._generate_executive_summary(transcript_text)

            # Extract key decisions
            key_decisions = self._extract_key_decisions(transcript_text, transcript)

            # Extract action items
            action_items = self._extract_action_items(transcript_text, transcript)

            # Extract discussion points
            discussion_points = self._extract_discussion_points(transcript_text)

            # Extract unresolved questions
            unresolved_questions = self._extract_unresolved_questions(transcript_text)

            # Generate next steps
            next_steps = self._generate_next_steps(action_items, key_decisions)

            # Calculate duration
            duration_minutes = 0
            if transcript.end_time:
                duration_minutes = int(
                    (transcript.end_time - transcript.start_time).total_seconds() / 60
                )

            summary = MeetingSummary(
                meeting_id=transcript.meeting_id,
                meeting_title=transcript.meeting_title,
                date=transcript.start_time,
                participants=transcript.participants,
                duration_minutes=duration_minutes,
                executive_summary=executive_summary,
                key_decisions=key_decisions,
                action_items=action_items,
                discussion_points=discussion_points,
                unresolved_questions=unresolved_questions,
                next_steps=next_steps
            )

            logger.info(f"Summary generated successfully with {len(action_items)} action items")
            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise

    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Helper method to call Gemini with rate limit handling."""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                    # Rate limit hit - extract wait time or use default
                    wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
                    match = re.search(r'seconds:\s*(\d+)', error_str)
                    if match:
                        wait_time = int(match.group(1)) + 5  # Add buffer
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    raise e
        raise Exception(f"Max retries ({max_retries}) exceeded for LLM call")

    def _build_transcript_text(self, transcript: TranscriptData) -> str:
        """Build formatted transcript text from segments."""
        lines = []
        for segment in transcript.segments:
            timestamp = f"[{segment.start_time:.2f}s]"
            lines.append(f"{timestamp} {segment.speaker}: {segment.text}")
        return "\n".join(lines)

    def _generate_executive_summary(self, transcript_text: str) -> str:
        """Generate executive summary using LLM."""
        prompt = f"""You are an expert meeting summarizer. Create a concise executive summary (2-3 paragraphs) of the following meeting transcript.
Focus on the main topics discussed, key outcomes, and overall purpose.

Transcript:
{transcript_text}

Executive Summary:"""

        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return "Unable to generate executive summary."

    def _extract_key_decisions(
        self,
        transcript_text: str,
        transcript: TranscriptData
    ) -> List[KeyDecision]:
        """Extract key decisions from transcript."""
        prompt = f"""You are an expert at extracting key decisions from meetings. Analyze the following meeting transcript and extract ALL key decisions that were made.
For each decision, provide:
1. The decision that was made
2. The context/reasoning behind it
3. The timestamp (in seconds) when it was discussed
4. Participants involved

Format your response as a JSON array of objects with keys: decision, context, timestamp, participants.

Transcript:
{transcript_text}

Key Decisions (JSON array):"""

        try:
            content = self._call_llm(prompt)

            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            decisions_data = json.loads(content)

            decisions = []
            for item in decisions_data:
                decision = KeyDecision(
                    decision=item.get("decision", ""),
                    context=item.get("context", ""),
                    timestamp=float(item.get("timestamp", 0)),
                    participants=item.get("participants", [])
                )
                decisions.append(decision)

            return decisions

        except Exception as e:
            logger.error(f"Error extracting key decisions: {e}")
            return []

    def _extract_action_items(
        self,
        transcript_text: str,
        transcript: TranscriptData
    ) -> List[ActionItem]:
        """Extract action items with assignees from transcript."""
        prompt = f"""You are an expert at extracting action items from meetings. Analyze the following meeting transcript and extract ALL action items (tasks, todos, assignments).
For each action item, identify:
1. The task title (concise)
2. Detailed description
3. Who is assigned (if mentioned)
4. Priority (high/medium/low)
5. Due date if mentioned

Format your response as a JSON array of objects with keys: title, description, assignee, priority, due_date.

Transcript:
{transcript_text}

Action Items (JSON array):"""

        try:
            content = self._call_llm(prompt)

            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            items_data = json.loads(content)

            action_items = []
            for item in items_data:
                action = ActionItem(
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    assignee=item.get("assignee"),
                    priority=item.get("priority", "medium"),
                    due_date=None,  # Can be parsed if provided
                    created_from_meeting=transcript.meeting_id
                )
                action_items.append(action)

            return action_items

        except Exception as e:
            logger.error(f"Error extracting action items: {e}")
            return []

    def _extract_discussion_points(self, transcript_text: str) -> List[str]:
        """Extract main discussion points."""
        prompt = f"""You are an expert at identifying key discussion points. Analyze the following meeting transcript and extract the main discussion points/topics.
Provide a bulleted list of 5-8 key discussion points.

Transcript:
{transcript_text}

Discussion Points (as a simple list, one per line):"""

        try:
            content = self._call_llm(prompt)

            # Parse bullet points
            points = [
                line.strip().lstrip("-•* ").strip()
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            return points[:8]  # Limit to 8 points

        except Exception as e:
            logger.error(f"Error extracting discussion points: {e}")
            return []

    def _extract_unresolved_questions(self, transcript_text: str) -> List[str]:
        """Extract unresolved questions and open issues."""
        prompt = f"""You are an expert at identifying unresolved questions. Analyze the following meeting transcript and identify any unresolved questions,
open issues, or topics that need further discussion.

Transcript:
{transcript_text}

Unresolved Questions (as a simple list, one per line):"""

        try:
            content = self._call_llm(prompt)

            questions = [
                line.strip().lstrip("-•* ").strip()
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            return questions

        except Exception as e:
            logger.error(f"Error extracting unresolved questions: {e}")
            return []

    def _generate_next_steps(
        self,
        action_items: List[ActionItem],
        key_decisions: List[KeyDecision]
    ) -> str:
        """Generate next steps summary."""
        next_steps_parts = []

        if action_items:
            next_steps_parts.append(
                f"**Action Items:** {len(action_items)} tasks have been identified and assigned."
            )

        if key_decisions:
            next_steps_parts.append(
                f"**Decisions:** {len(key_decisions)} key decisions were made and should be communicated to stakeholders."
            )

        if not next_steps_parts:
            return "No specific action items or decisions recorded."

        return " ".join(next_steps_parts)

    def save_summary(self, summary: MeetingSummary, output_path: str):
        """Save summary to JSON file."""
        try:
            summary_dict = summary.model_dump(mode='json')
            with open(output_path, 'w') as f:
                json.dump(summary_dict, f, indent=2, default=str)
            logger.info(f"Summary saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving summary: {e}")
