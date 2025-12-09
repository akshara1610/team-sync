"""
Email utility to send Minutes of Meeting to participants
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from pathlib import Path
from typing import List
from loguru import logger
from src.config import settings
from src.models.schemas import MeetingSummary


class EmailSender:
    """Send Minutes of Meeting via email to participants."""

    def __init__(self):
        # Configure these in your .env file
        self.smtp_server = "smtp.gmail.com"  # For Gmail
        self.smtp_port = 587
        # Use dedicated TeamSync email account
        self.sender_email = getattr(settings, 'TEAMSYNC_EMAIL', 'teamsync990@gmail.com')
        # For Gmail, use App Password: https://support.google.com/accounts/answer/185833
        self.sender_password = getattr(settings, 'EMAIL_PASSWORD', None)  # Read from .env

    def send_meeting_minutes(
        self,
        summary: MeetingSummary,
        recipients: List[str],
        summary_file_path: str = None
    ) -> bool:
        """
        Send Minutes of Meeting email to all participants.

        Args:
            summary: Meeting summary object
            recipients: List of participant emails
            summary_file_path: Optional path to JSON summary file to attach

        Returns:
            Success status
        """
        try:
            # Create email
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"Meeting Minutes: {summary.meeting_title}"

            # Create HTML email body
            html_body = self._create_html_body(summary)
            msg.attach(MIMEText(html_body, 'html'))

            # Attach summary JSON file if provided
            if summary_file_path and Path(summary_file_path).exists():
                with open(summary_file_path, 'rb') as f:
                    attachment = MIMEApplication(f.read(), _subtype='json')
                    attachment.add_header(
                        'Content-Disposition',
                        'attachment',
                        filename=f'{summary.meeting_id}_summary.json'
                    )
                    msg.attach(attachment)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            logger.info(f"Meeting minutes sent to {len(recipients)} participants")
            return True

        except Exception as e:
            logger.error(f"Failed to send meeting minutes: {e}")
            return False

    def _create_html_body(self, summary: MeetingSummary) -> str:
        """Create formatted HTML email body."""

        # Format action items
        action_items_html = ""
        if summary.action_items:
            action_items_html = "<h3>📋 Action Items</h3><ul>"
            for item in summary.action_items:
                assignee = f" - <strong>{item.assignee}</strong>" if item.assignee else ""
                priority_badge = f'<span style="background-color: {"#ff4444" if item.priority == "high" else "#ffaa00" if item.priority == "medium" else "#44ff44"}; padding: 2px 6px; border-radius: 3px; color: white; font-size: 10px;">{item.priority.upper()}</span>'
                action_items_html += f"<li>{item.title}{assignee} {priority_badge}<br><small>{item.description}</small></li>"
            action_items_html += "</ul>"

        # Format key decisions
        decisions_html = ""
        if summary.key_decisions:
            decisions_html = "<h3>🎯 Key Decisions</h3><ul>"
            for decision in summary.key_decisions:
                decisions_html += f"<li><strong>{decision.decision}</strong><br><small>{decision.context}</small></li>"
            decisions_html += "</ul>"

        # Format discussion points
        discussion_html = ""
        if summary.discussion_points:
            discussion_html = "<h3>💬 Discussion Points</h3><ul>"
            for point in summary.discussion_points:
                discussion_html += f"<li>{point}</li>"
            discussion_html += "</ul>"

        # Format unresolved questions
        questions_html = ""
        if summary.unresolved_questions:
            questions_html = "<h3>❓ Unresolved Questions</h3><ul>"
            for question in summary.unresolved_questions:
                questions_html += f"<li>{question}</li>"
            questions_html += "</ul>"

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h3 {{ color: #34495e; margin-top: 20px; }}
                ul {{ list-style-type: none; padding-left: 0; }}
                li {{ padding: 8px 0; border-bottom: 1px solid #ecf0f1; }}
                .header {{ background-color: #3498db; color: white; padding: 20px; border-radius: 5px; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 2px solid #ecf0f1; color: #7f8c8d; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>📝 Minutes of Meeting</h2>
                <p><strong>{summary.meeting_title}</strong></p>
                <p>Date: {summary.date.strftime('%B %d, %Y at %H:%M')}</p>
                <p>Duration: {summary.duration_minutes} minutes</p>
                <p>Participants: {', '.join(summary.participants)}</p>
            </div>

            <div class="summary">
                <h3>📊 Executive Summary</h3>
                <p>{summary.executive_summary}</p>
            </div>

            {action_items_html}

            {decisions_html}

            {discussion_html}

            {questions_html}

            <div class="footer">
                <p>🤖 <em>This summary was automatically generated by TeamSync AI</em></p>
                <p>Meeting ID: {summary.meeting_id}</p>
            </div>
        </body>
        </html>
        """

        return html


# Simple text-only version (no SMTP required, just prints)
def print_meeting_minutes(summary: MeetingSummary):
    """Print meeting minutes to console (for testing)."""
    print("\n" + "="*80)
    print(f"📝 MINUTES OF MEETING")
    print("="*80)
    print(f"\n📌 {summary.meeting_title}")
    print(f"📅 Date: {summary.date.strftime('%B %d, %Y at %H:%M')}")
    print(f"⏱️  Duration: {summary.duration_minutes} minutes")
    print(f"👥 Participants: {', '.join(summary.participants)}")

    print(f"\n📊 EXECUTIVE SUMMARY")
    print("-"*80)
    print(summary.executive_summary)

    if summary.action_items:
        print(f"\n📋 ACTION ITEMS")
        print("-"*80)
        for i, item in enumerate(summary.action_items, 1):
            assignee = f" → {item.assignee}" if item.assignee else ""
            print(f"{i}. [{item.priority.upper()}] {item.title}{assignee}")
            print(f"   {item.description}")

    if summary.key_decisions:
        print(f"\n🎯 KEY DECISIONS")
        print("-"*80)
        for i, decision in enumerate(summary.key_decisions, 1):
            print(f"{i}. {decision.decision}")
            print(f"   Context: {decision.context}")

    if summary.discussion_points:
        print(f"\n💬 DISCUSSION POINTS")
        print("-"*80)
        for point in summary.discussion_points:
            print(f"  • {point}")

    if summary.unresolved_questions:
        print(f"\n❓ UNRESOLVED QUESTIONS")
        print("-"*80)
        for question in summary.unresolved_questions:
            print(f"  • {question}")

    print("\n" + "="*80)
    print(f"🤖 Generated by TeamSync AI | Meeting ID: {summary.meeting_id}")
    print("="*80 + "\n")
