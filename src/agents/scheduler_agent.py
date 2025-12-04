from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from loguru import logger
from src.config import settings
from src.models.schemas import CalendarEvent, ActionItem, MeetingSummary


class SchedulerAgent:
    """
    Scheduler Agent: Analyzes action item dependencies and meeting outcomes to
    propose follow-up meetings. Integrates with Google Calendar API to schedule
    events and send invitations to participants.
    """

    SCOPES = ['https://www.googleapis.com/auth/calendar']

    def __init__(self):
        logger.info("Initializing Scheduler Agent...")
        self.service = self._authenticate()
        logger.info("Scheduler Agent initialized successfully")

    def _authenticate(self):
        """Authenticate with Google Calendar API."""
        creds = None

        # The file token.json stores the user's access and refresh tokens
        if os.path.exists(settings.GOOGLE_TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(
                settings.GOOGLE_TOKEN_FILE,
                self.SCOPES
            )

        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    settings.GOOGLE_CREDENTIALS_FILE,
                    self.SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open(settings.GOOGLE_TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())

        return build('calendar', 'v3', credentials=creds)

    def schedule_event(self, event: CalendarEvent) -> Dict[str, Any]:
        """
        Schedule a new calendar event.

        Args:
            event: Calendar event to schedule

        Returns:
            Created event information
        """
        try:
            # Build event body
            event_body = {
                'summary': event.summary,
                'description': event.description,
                'start': {
                    'dateTime': event.start_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': event.end_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'attendees': [{'email': email} for email in event.attendees],
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 24 * 60},
                        {'method': 'popup', 'minutes': 30},
                    ],
                },
            }

            if event.location:
                event_body['location'] = event.location

            # Create event
            created_event = self.service.events().insert(
                calendarId='primary',
                body=event_body,
                sendUpdates='all'  # Send invitations to all attendees
            ).execute()

            logger.info(f"Created calendar event: {created_event['id']}")

            return {
                'id': created_event['id'],
                'htmlLink': created_event['htmlLink'],
                'summary': created_event['summary'],
                'start': created_event['start']['dateTime'],
                'status': 'created'
            }

        except HttpError as e:
            logger.error(f"Error creating calendar event: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def schedule_follow_up_meeting(
        self,
        summary: MeetingSummary,
        days_ahead: int = 7,
        duration_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Schedule a follow-up meeting based on meeting summary.

        Args:
            summary: Meeting summary with action items
            days_ahead: Number of days in the future to schedule
            duration_minutes: Duration of follow-up meeting

        Returns:
            Created event information
        """
        try:
            # Generate follow-up meeting details
            follow_up_title = f"Follow-up: {summary.meeting_title}"

            # Build description with action items
            description_parts = [
                f"This is a follow-up meeting for: {summary.meeting_title}",
                f"Original meeting date: {summary.date.strftime('%Y-%m-%d')}",
                "",
                "**Action Items to Review:**"
            ]

            for action in summary.action_items[:5]:  # Limit to 5 items
                assignee_text = f" ({action.assignee})" if action.assignee else ""
                description_parts.append(f"• {action.title}{assignee_text}")

            if len(summary.action_items) > 5:
                description_parts.append(f"... and {len(summary.action_items) - 5} more items")

            if summary.unresolved_questions:
                description_parts.extend([
                    "",
                    "**Unresolved Questions:**"
                ])
                for question in summary.unresolved_questions[:3]:
                    description_parts.append(f"• {question}")

            description = "\n".join(description_parts)

            # Calculate timing
            start_time = datetime.utcnow() + timedelta(days=days_ahead)
            end_time = start_time + timedelta(minutes=duration_minutes)

            # Create event
            event = CalendarEvent(
                summary=follow_up_title,
                description=description,
                start_time=start_time,
                end_time=end_time,
                attendees=summary.participants
            )

            result = self.schedule_event(event)
            logger.info(f"Scheduled follow-up meeting {days_ahead} days ahead")

            return result

        except Exception as e:
            logger.error(f"Error scheduling follow-up meeting: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def suggest_meeting_time(
        self,
        attendee_emails: List[str],
        duration_minutes: int = 60,
        days_to_check: int = 7
    ) -> Optional[datetime]:
        """
        Suggest an optimal meeting time based on attendee availability.

        Args:
            attendee_emails: List of attendee emails
            duration_minutes: Required meeting duration
            days_to_check: Number of days to check for availability

        Returns:
            Suggested meeting start time
        """
        try:
            # Get current time
            now = datetime.utcnow()

            # Build time range to check
            time_min = now.isoformat() + 'Z'
            time_max = (now + timedelta(days=days_to_check)).isoformat() + 'Z'

            # Query free/busy information
            body = {
                "timeMin": time_min,
                "timeMax": time_max,
                "items": [{"id": email} for email in attendee_emails]
            }

            freebusy_query = self.service.freebusy().query(body=body).execute()

            # Find common free slots
            suggested_time = self._find_common_free_slot(
                freebusy_query,
                duration_minutes,
                now,
                days_to_check
            )

            return suggested_time

        except HttpError as e:
            logger.error(f"Error checking availability: {e}")
            return None

    def _find_common_free_slot(
        self,
        freebusy_data: Dict,
        duration_minutes: int,
        start_date: datetime,
        days_to_check: int
    ) -> Optional[datetime]:
        """Find a common free slot for all attendees."""

        # Simple heuristic: find first slot during business hours (9 AM - 5 PM)
        # This is a simplified version; production would need more sophisticated logic

        for day_offset in range(days_to_check):
            check_date = start_date + timedelta(days=day_offset)

            # Check business hours (9 AM - 5 PM)
            for hour in range(9, 17):
                potential_start = check_date.replace(
                    hour=hour,
                    minute=0,
                    second=0,
                    microsecond=0
                )

                potential_end = potential_start + timedelta(minutes=duration_minutes)

                # Check if this slot is free for all attendees
                is_free = self._is_slot_free(
                    freebusy_data,
                    potential_start,
                    potential_end
                )

                if is_free:
                    return potential_start

        return None

    def _is_slot_free(
        self,
        freebusy_data: Dict,
        start_time: datetime,
        end_time: datetime
    ) -> bool:
        """Check if a time slot is free for all attendees."""

        calendars = freebusy_data.get('calendars', {})

        for calendar_id, calendar_data in calendars.items():
            busy_periods = calendar_data.get('busy', [])

            for busy_period in busy_periods:
                busy_start = datetime.fromisoformat(
                    busy_period['start'].replace('Z', '+00:00')
                )
                busy_end = datetime.fromisoformat(
                    busy_period['end'].replace('Z', '+00:00')
                )

                # Check for overlap
                if (start_time < busy_end and end_time > busy_start):
                    return False

        return True

    def update_event(
        self,
        event_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update an existing calendar event.

        Args:
            event_id: Event ID to update
            updates: Dictionary of fields to update

        Returns:
            Updated event information
        """
        try:
            # Get existing event
            event = self.service.events().get(
                calendarId='primary',
                eventId=event_id
            ).execute()

            # Apply updates
            for key, value in updates.items():
                event[key] = value

            # Update event
            updated_event = self.service.events().update(
                calendarId='primary',
                eventId=event_id,
                body=event,
                sendUpdates='all'
            ).execute()

            logger.info(f"Updated calendar event: {event_id}")

            return {
                'id': updated_event['id'],
                'htmlLink': updated_event['htmlLink'],
                'status': 'updated'
            }

        except HttpError as e:
            logger.error(f"Error updating event: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def cancel_event(self, event_id: str) -> bool:
        """
        Cancel a calendar event.

        Args:
            event_id: Event ID to cancel

        Returns:
            Success status
        """
        try:
            self.service.events().delete(
                calendarId='primary',
                eventId=event_id,
                sendUpdates='all'
            ).execute()

            logger.info(f"Cancelled calendar event: {event_id}")
            return True

        except HttpError as e:
            logger.error(f"Error cancelling event: {e}")
            return False

    def get_upcoming_events(
        self,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get upcoming calendar events.

        Args:
            max_results: Maximum number of events to return

        Returns:
            List of upcoming events
        """
        try:
            now = datetime.utcnow().isoformat() + 'Z'

            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=now,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            events = events_result.get('items', [])

            return [
                {
                    'id': event['id'],
                    'summary': event.get('summary', 'No title'),
                    'start': event['start'].get('dateTime', event['start'].get('date')),
                    'htmlLink': event['htmlLink']
                }
                for event in events
            ]

        except HttpError as e:
            logger.error(f"Error getting upcoming events: {e}")
            return []
