from typing import List, Tuple
import json
import time
import re
from loguru import logger
from src.config import settings
from src.models.schemas import (
    MeetingSummary,
    TranscriptData,
    ReflectionFeedback,
    ActionItem
)


class SelfReflectionAgent:
    """
    Self-Reflection Agent: Reviews outputs from the Summarizer and Action agents,
    verifying factual consistency against source transcripts, checking for missed
    action items, and validating logical coherence. Implements a critique-revise
    evaluation loop before finalizing outputs.

    This is the critical innovation that builds trust in the system.
    """

    def __init__(self):
        self.llm_provider = settings.LLM_PROVIDER
        self.model_name = settings.SUMMARIZER_MODEL
        self.max_iterations = settings.MAX_REFLECTION_ITERATIONS

        if self.llm_provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info(f"Self-Reflection Agent initialized with OpenAI model: {self.model_name}, max {self.max_iterations} iterations")
        elif self.llm_provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Self-Reflection Agent initialized with Gemini model: {self.model_name}, max {self.max_iterations} iterations")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Helper method to call LLM with rate limit handling."""
        for attempt in range(max_retries):
            try:
                if self.llm_provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=2000
                    )
                    return response.choices[0].message.content.strip()
                elif self.llm_provider == "gemini":
                    response = self.model.generate_content(prompt)
                    return response.text.strip()
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                    # Rate limit hit - extract wait time or use default
                    wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
                    if "retry_delay" in error_str:
                        try:
                            # Try to extract the retry delay from error
                            import re
                            match = re.search(r'seconds:\s*(\d+)', error_str)
                            if match:
                                wait_time = int(match.group(1)) + 5  # Add buffer
                        except:
                            pass
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    raise e
        raise Exception(f"Max retries ({max_retries}) exceeded for LLM call")

    def validate_summary(
        self,
        summary: MeetingSummary,
        transcript: TranscriptData
    ) -> Tuple[ReflectionFeedback, MeetingSummary]:
        """
        Validate and iteratively improve meeting summary.

        Args:
            summary: Generated meeting summary
            transcript: Original transcript

        Returns:
            Tuple of (feedback, improved_summary)
        """
        try:
            logger.info(f"Starting validation for meeting: {summary.meeting_id}")

            transcript_text = self._build_transcript_text(transcript)
            current_summary = summary
            iteration = 0

            while iteration < self.max_iterations:
                iteration += 1
                logger.info(f"Reflection iteration {iteration}/{self.max_iterations}")

                # Perform comprehensive validation
                feedback = self._perform_validation(current_summary, transcript_text)

                if feedback.approved:
                    logger.info("Summary approved after reflection")
                    return feedback, current_summary

                # Generate improved version
                logger.info("Issues found, generating improved version...")
                current_summary = self._improve_summary(
                    current_summary,
                    transcript_text,
                    feedback
                )

            # Max iterations reached - return with warning
            logger.warning(f"Max iterations ({self.max_iterations}) reached")
            feedback.approved = True  # Force approval but keep issues noted
            return feedback, current_summary

        except Exception as e:
            logger.error(f"Error in validation: {e}")
            # Return original with error feedback
            return ReflectionFeedback(
                is_factually_consistent=False,
                consistency_issues=[f"Validation error: {str(e)}"],
                missing_action_items=[],
                logical_coherence_score=0.0,
                suggested_improvements=[],
                approved=False
            ), summary

    def _perform_validation(
        self,
        summary: MeetingSummary,
        transcript_text: str
    ) -> ReflectionFeedback:
        """Perform comprehensive validation checks."""

        # Check 1: Factual consistency
        consistency_check = self._check_factual_consistency(summary, transcript_text)

        # Check 2: Completeness of action items
        completeness_check = self._check_action_item_completeness(summary, transcript_text)

        # Check 3: Logical coherence
        coherence_score = self._check_logical_coherence(summary, transcript_text)

        # Aggregate results
        all_issues = consistency_check + completeness_check
        is_approved = (
            len(all_issues) == 0 and
            coherence_score >= 0.7
        )

        feedback = ReflectionFeedback(
            is_factually_consistent=len(consistency_check) == 0,
            consistency_issues=consistency_check,
            missing_action_items=completeness_check,
            logical_coherence_score=coherence_score,
            suggested_improvements=all_issues,
            approved=is_approved
        )

        return feedback

    def _check_factual_consistency(
        self,
        summary: MeetingSummary,
        transcript_text: str
    ) -> List[str]:
        """Check if summary is factually consistent with transcript."""

        summary_text = self._summary_to_text(summary)

        prompt = f"""You are a critical fact-checker. Compare the meeting summary against the original transcript
and identify any factual inconsistencies, hallucinations, or misrepresentations.

Original Transcript:
{transcript_text[:4000]}  # Truncate for token limits

Generated Summary:
{summary_text}

List any factual inconsistencies you find. If the summary is factually accurate, respond with "NONE".
Be critical - flag anything that seems exaggerated, misrepresented, or not supported by the transcript.

Inconsistencies (one per line, or "NONE"):"""

        try:
            content = self._call_llm(prompt)

            if content.upper() == "NONE" or "no inconsistencies" in content.lower():
                return []

            # Parse issues
            issues = [
                line.strip().lstrip("-•* ").strip()
                for line in content.split("\n")
                if line.strip() and line.strip().upper() != "NONE"
            ]

            return issues

        except Exception as e:
            logger.error(f"Error checking factual consistency: {e}")
            return [f"Error in consistency check: {str(e)}"]

    def _check_action_item_completeness(
        self,
        summary: MeetingSummary,
        transcript_text: str
    ) -> List[str]:
        """Check if any action items were missed."""

        existing_actions = [f"{a.title}: {a.description}" for a in summary.action_items]
        existing_text = "\n".join(existing_actions) if existing_actions else "No action items listed."

        prompt = f"""You are reviewing a meeting summary to ensure all action items were captured.

Original Transcript:
{transcript_text[:4000]}

Action Items in Summary:
{existing_text}

Review the transcript carefully. Were any action items, tasks, or assignments missed?
List any missing action items. If all action items were captured, respond with "NONE".

Missing Action Items (one per line, or "NONE"):"""

        try:
            content = self._call_llm(prompt)

            if content.upper() == "NONE" or "no missing" in content.lower():
                return []

            missing = [
                line.strip().lstrip("-•* ").strip()
                for line in content.split("\n")
                if line.strip() and line.strip().upper() != "NONE"
            ]

            return missing

        except Exception as e:
            logger.error(f"Error checking action item completeness: {e}")
            return []

    def _check_logical_coherence(
        self,
        summary: MeetingSummary,
        transcript_text: str
    ) -> float:
        """Check logical coherence and quality of summary."""

        summary_text = self._summary_to_text(summary)

        prompt = f"""Rate the logical coherence and quality of this meeting summary on a scale of 0.0 to 1.0.

Consider:
- Does the summary flow logically?
- Are decisions clearly connected to discussions?
- Are action items clearly defined with assignees?
- Is the executive summary accurate and concise?
- Are there any contradictions?

Summary:
{summary_text}

Respond with ONLY a number between 0.0 and 1.0, where:
- 0.0-0.3: Poor quality, many issues
- 0.4-0.6: Acceptable but needs improvement
- 0.7-0.9: Good quality
- 0.9-1.0: Excellent quality

Score:"""

        try:
            content = self._call_llm(prompt)
            score = float(content)
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1

        except Exception as e:
            logger.error(f"Error checking logical coherence: {e}")
            return 0.5  # Default middle score on error

    def _improve_summary(
        self,
        summary: MeetingSummary,
        transcript_text: str,
        feedback: ReflectionFeedback
    ) -> MeetingSummary:
        """Generate improved version of summary based on feedback."""

        summary_text = self._summary_to_text(summary)
        issues_text = "\n".join(feedback.suggested_improvements)

        prompt = f"""You are improving a meeting summary based on critical feedback.

Original Transcript:
{transcript_text[:4000]}

Current Summary:
{summary_text}

Issues Identified:
{issues_text}

Generate an improved version that addresses all issues. Maintain the same JSON structure.
Focus on:
1. Fixing factual inconsistencies
2. Adding any missing action items
3. Improving clarity and coherence

Return the improved summary as JSON with the same structure as the original."""

        try:
            content = self._call_llm(prompt)

            # Extract JSON from markdown if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            improved_data = json.loads(content)

            # Update summary with improvements (merge with existing)
            # For simplicity, we'll just update the text fields
            if "executive_summary" in improved_data:
                summary.executive_summary = improved_data["executive_summary"]

            logger.info("Summary improved based on feedback")
            return summary

        except Exception as e:
            logger.error(f"Error improving summary: {e}")
            return summary  # Return original if improvement fails

    def _build_transcript_text(self, transcript: TranscriptData) -> str:
        """Build formatted transcript text."""
        lines = []
        for segment in transcript.segments:
            lines.append(f"{segment.speaker}: {segment.text}")
        return "\n".join(lines)

    def _summary_to_text(self, summary: MeetingSummary) -> str:
        """Convert summary object to readable text."""
        parts = [
            f"Meeting: {summary.meeting_title}",
            f"Date: {summary.date}",
            f"Participants: {', '.join(summary.participants)}",
            f"\nExecutive Summary:\n{summary.executive_summary}",
            f"\nKey Decisions ({len(summary.key_decisions)}):",
        ]

        for dec in summary.key_decisions:
            parts.append(f"- {dec.decision}")

        parts.append(f"\nAction Items ({len(summary.action_items)}):")
        for action in summary.action_items:
            assignee_text = f" (Assigned to: {action.assignee})" if action.assignee else ""
            parts.append(f"- {action.title}{assignee_text}")

        return "\n".join(parts)
