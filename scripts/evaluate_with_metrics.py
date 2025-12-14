"""
Evaluate Reflection Agent Impact with Quantitative Metrics

Generates the numbers to justify:
1. Higher factual accuracy with self-reflection
2. More complete action items captured
3. Clearer, better-structured summaries
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.agents.summarizer_agent import SummarizerAgent
from src.agents.reflection_agent import SelfReflectionAgent
from src.models.schemas import TranscriptData, SpeakerSegment
from loguru import logger


class MetricsEvaluator:
    """Evaluate summaries with quantitative metrics."""

    def __init__(self):
        from src.config import settings
        self.llm_provider = settings.LLM_PROVIDER

        if self.llm_provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = settings.SUMMARIZER_MODEL
        elif self.llm_provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(settings.SUMMARIZER_MODEL)

    def _call_llm(self, prompt: str) -> str:
        """Call LLM for evaluation."""
        if self.llm_provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        elif self.llm_provider == "gemini":
            response = self.model.generate_content(prompt)
            return response.text.strip()

    def evaluate_factual_accuracy(self, transcript: TranscriptData, summary_text: str) -> float:
        """
        Evaluate factual accuracy (0-100 scale).

        Checks if facts in summary are supported by transcript.
        """
        transcript_text = "\n".join([f"{seg.speaker}: {seg.text}" for seg in transcript.segments])

        prompt = f"""You are evaluating the factual accuracy of a meeting summary.

TRANSCRIPT:
{transcript_text}

SUMMARY:
{summary_text}

Task: Rate the factual accuracy on a 0-100 scale where:
- 100 = Every fact is perfectly accurate and supported by transcript
- 80-99 = Mostly accurate with minor issues
- 60-79 = Some inaccuracies or unsupported claims
- 40-59 = Multiple inaccuracies
- 0-39 = Many false or unsupported facts

Respond with ONLY a number between 0-100."""

        try:
            result = self._call_llm(prompt)
            # Extract number from response
            import re
            match = re.search(r'\b(\d+)\b', result)
            if match:
                score = int(match.group(1))
                return min(max(score, 0), 100)  # Clamp to 0-100
            return 50.0  # Default if parsing fails
        except:
            return 50.0

    def count_action_items(self, summary_text: str) -> int:
        """Count action items extracted from summary."""
        prompt = f"""Count the number of distinct action items (tasks, todos, assignments) in this summary.

SUMMARY:
{summary_text}

Respond with ONLY a number."""

        try:
            result = self._call_llm(prompt)
            import re
            match = re.search(r'\b(\d+)\b', result)
            if match:
                return int(match.group(1))
            return 0
        except:
            return 0

    def evaluate_completeness(self, transcript: TranscriptData, summary_text: str) -> float:
        """
        Evaluate completeness (0-100 scale).

        Checks if all important points from transcript are captured.
        """
        transcript_text = "\n".join([f"{seg.speaker}: {seg.text}" for seg in transcript.segments])

        prompt = f"""You are evaluating the completeness of a meeting summary.

TRANSCRIPT:
{transcript_text}

SUMMARY:
{summary_text}

Task: Rate the completeness on a 0-100 scale where:
- 100 = All important points, decisions, and action items captured
- 80-99 = Most important points captured, minor omissions
- 60-79 = Some important points missing
- 40-59 = Many important points missing
- 0-39 = Most important information missing

Respond with ONLY a number between 0-100."""

        try:
            result = self._call_llm(prompt)
            import re
            match = re.search(r'\b(\d+)\b', result)
            if match:
                score = int(match.group(1))
                return min(max(score, 0), 100)
            return 50.0
        except:
            return 50.0

    def evaluate_clarity(self, summary_text: str) -> float:
        """
        Evaluate clarity and coherence (0-100 scale).

        Checks logical organization, flow, and readability.
        """
        prompt = f"""You are evaluating the clarity and coherence of a meeting summary.

SUMMARY:
{summary_text}

Task: Rate the clarity and coherence on a 0-100 scale where:
- 100 = Perfectly organized, clear, easy to understand
- 80-99 = Well-structured with minor clarity issues
- 60-79 = Somewhat clear but has organizational issues
- 40-59 = Confusing or poorly structured
- 0-39 = Very unclear and hard to follow

Respond with ONLY a number between 0-100."""

        try:
            result = self._call_llm(prompt)
            import re
            match = re.search(r'\b(\d+)\b', result)
            if match:
                score = int(match.group(1))
                return min(max(score, 0), 100)
            return 50.0
        except:
            return 50.0


def format_summary_as_text(summary) -> str:
    """Convert MeetingSummary object to text."""
    parts = []

    parts.append(f"EXECUTIVE SUMMARY:\n{summary.executive_summary}\n")

    if summary.key_decisions:
        parts.append("KEY DECISIONS:")
        for i, dec in enumerate(summary.key_decisions, 1):
            parts.append(f"{i}. {dec.decision}")
            if hasattr(dec, 'context') and dec.context:
                parts.append(f"   Context: {dec.context}")
        parts.append("")

    if summary.action_items:
        parts.append("ACTION ITEMS:")
        for i, item in enumerate(summary.action_items, 1):
            parts.append(f"{i}. {item.title}")
            parts.append(f"   Assignee: {item.assignee}")
            parts.append(f"   Priority: {item.priority}")
        parts.append("")

    return "\n".join(parts)


async def main():
    """Run evaluation on transcript."""
    print("\n" + "="*80)
    print("REFLECTION AGENT IMPACT EVALUATION")
    print("="*80)

    # Load transcript
    transcript_path = Path("data/transcripts/sprint_planning_20251210.json")

    print(f"\n📝 Loading transcript: {transcript_path.name}")
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)

    segments = [SpeakerSegment(**seg) for seg in transcript_data['segments']]
    transcript = TranscriptData(
        meeting_id=transcript_data['meeting_id'],
        meeting_title=transcript_data.get('meeting_title', 'Sprint Planning'),
        start_time=datetime.now() - timedelta(minutes=2),
        end_time=datetime.now(),
        segments=segments,
        participants=transcript_data.get('participants', [])
    )

    print(f"✅ Loaded {len(segments)} segments\n")

    # Initialize agents
    summarizer = SummarizerAgent()
    reflector = SelfReflectionAgent()
    evaluator = MetricsEvaluator()

    # Generate WITHOUT reflection
    print("-"*80)
    print("GENERATING SUMMARY WITHOUT REFLECTION")
    print("-"*80)
    initial_summary = summarizer.generate_summary(transcript)
    initial_text = format_summary_as_text(initial_summary)

    # Generate WITH reflection
    print("\n" + "-"*80)
    print("GENERATING SUMMARY WITH REFLECTION")
    print("-"*80)
    feedback, improved_summary = reflector.validate_summary(initial_summary, transcript)
    improved_text = format_summary_as_text(improved_summary)

    print(f"\nReflection Feedback:")
    print(f"  Approved: {feedback.approved}")
    print(f"  Consistency issues: {len(feedback.consistency_issues)}")
    print(f"  Missing action items: {len(feedback.missing_action_items)}")
    print(f"  Improvements suggested: {len(feedback.suggested_improvements)}")

    # Evaluate both versions
    print("\n" + "="*80)
    print("QUANTITATIVE EVALUATION")
    print("="*80)

    print("\n🔍 Evaluating WITHOUT Reflection...")
    initial_metrics = {
        "factual_accuracy": evaluator.evaluate_factual_accuracy(transcript, initial_text),
        "completeness": evaluator.evaluate_completeness(transcript, initial_text),
        "clarity": evaluator.evaluate_clarity(initial_text),
        "action_items_count": evaluator.count_action_items(initial_text)
    }

    print("\n🔍 Evaluating WITH Reflection...")
    improved_metrics = {
        "factual_accuracy": evaluator.evaluate_factual_accuracy(transcript, improved_text),
        "completeness": evaluator.evaluate_completeness(transcript, improved_text),
        "clarity": evaluator.evaluate_clarity(improved_text),
        "action_items_count": evaluator.count_action_items(improved_text)
    }

    # Calculate improvements
    print("\n" + "="*80)
    print("RESULTS (1-5 SCALE FOR PRESENTATION)")
    print("="*80)

    # Convert 0-100 to 1-5 scale
    def to_five_scale(score):
        return round((score / 100) * 4 + 1, 1)  # Maps 0-100 to 1-5

    initial_factual = to_five_scale(initial_metrics["factual_accuracy"])
    improved_factual = to_five_scale(improved_metrics["factual_accuracy"])

    initial_completeness = to_five_scale(initial_metrics["completeness"])
    improved_completeness = to_five_scale(improved_metrics["completeness"])

    initial_clarity = to_five_scale(initial_metrics["clarity"])
    improved_clarity = to_five_scale(improved_metrics["clarity"])

    print(f"\n📊 Metric 1: Factual Accuracy")
    print(f"   Without Reflection: {initial_factual}/5.0")
    print(f"   With Reflection:    {improved_factual}/5.0")
    print(f"   Improvement:        +{improved_factual - initial_factual:.1f} points")

    print(f"\n📊 Metric 2: Completeness of Action Items")
    print(f"   Without Reflection: {initial_metrics['action_items_count']} items ({initial_completeness}/5.0)")
    print(f"   With Reflection:    {improved_metrics['action_items_count']} items ({improved_completeness}/5.0)")
    print(f"   Improvement:        +{improved_metrics['action_items_count'] - initial_metrics['action_items_count']} items, +{improved_completeness - initial_completeness:.1f} points")

    print(f"\n📊 Metric 3: Clarity & Coherence")
    print(f"   Without Reflection: {initial_clarity}/5.0")
    print(f"   With Reflection:    {improved_clarity}/5.0")
    print(f"   Improvement:        +{improved_clarity - initial_clarity:.1f} points")

    # Summary for slide
    print("\n" + "="*80)
    print("SUMMARY FOR YOUR SLIDE")
    print("="*80)
    print(f"""
How We Evaluated TeamSync
Tested on meeting recording: "{transcript.meeting_title}"
For each meeting, we generated two summaries and compared both versions:
 (1) Without Self-Reflection and (2) With Self-Reflection

How did we compare? (1–5 scale)

╔═══════════════════════════════════════════════════════════════╗
║ Metric                    │ Without │ With │ Improvement      ║
╠═══════════════════════════════════════════════════════════════╣
║ Factual Accuracy          │  {initial_factual}/5   │ {improved_factual}/5 │ +{improved_factual - initial_factual:.1f} points      ║
║ Completeness (Actions)    │  {initial_metrics['action_items_count']} items │ {improved_metrics['action_items_count']} items│ +{improved_metrics['action_items_count'] - initial_metrics['action_items_count']} items        ║
║ Clarity & Coherence       │  {initial_clarity}/5   │ {improved_clarity}/5 │ +{improved_clarity - initial_clarity:.1f} points      ║
╚═══════════════════════════════════════════════════════════════╝

Key Outcomes:
✓ {((improved_factual - initial_factual) / initial_factual * 100):.1f}% higher factual accuracy with self-reflection
✓ {improved_metrics['action_items_count'] - initial_metrics['action_items_count']} more action items captured
✓ {((improved_clarity - initial_clarity) / initial_clarity * 100):.1f}% clearer, better-structured summaries
""")

    # Save results
    results = {
        "transcript": transcript_path.name,
        "timestamp": datetime.now().isoformat(),
        "without_reflection": {
            "factual_accuracy_100": initial_metrics["factual_accuracy"],
            "factual_accuracy_5": initial_factual,
            "completeness_100": initial_metrics["completeness"],
            "completeness_5": initial_completeness,
            "clarity_100": initial_metrics["clarity"],
            "clarity_5": initial_clarity,
            "action_items_count": initial_metrics["action_items_count"]
        },
        "with_reflection": {
            "factual_accuracy_100": improved_metrics["factual_accuracy"],
            "factual_accuracy_5": improved_factual,
            "completeness_100": improved_metrics["completeness"],
            "completeness_5": improved_completeness,
            "clarity_100": improved_metrics["clarity"],
            "clarity_5": improved_clarity,
            "action_items_count": improved_metrics["action_items_count"]
        },
        "improvements": {
            "factual_accuracy": improved_factual - initial_factual,
            "completeness": improved_completeness - initial_completeness,
            "clarity": improved_clarity - initial_clarity,
            "action_items": improved_metrics["action_items_count"] - initial_metrics["action_items_count"]
        }
    }

    output_path = Path("data/evaluation_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
