from typing import List, Dict, Any
from jira import JIRA
from loguru import logger
from src.config import settings
from src.models.schemas import ActionItem, JiraTicket


class ActionAgent:
    """
    Action Agent: Translates extracted action items into concrete workflow tasks.
    Creates Jira tickets with proper fields (summary, description, assignee, due date)
    and sends notifications to relevant team members via MCP tool integrations.
    """

    def __init__(self):
        try:
            logger.info("Initializing Action Agent with Jira integration...")

            # Initialize Jira client
            self.jira_client = JIRA(
                server=settings.JIRA_URL,
                basic_auth=(settings.JIRA_EMAIL, settings.JIRA_API_TOKEN)
            )

            self.project_key = settings.JIRA_PROJECT_KEY

            # Verify connection
            self.jira_client.myself()
            logger.info(f"Successfully connected to Jira: {settings.JIRA_URL}")

        except Exception as e:
            logger.error(f"Failed to initialize Jira client: {e}")
            raise

    def create_jira_tickets(
        self,
        action_items: List[ActionItem],
        meeting_title: str,
        meeting_id: str
    ) -> List[Dict[str, Any]]:
        """
        Create Jira tickets for all action items.

        Args:
            action_items: List of action items from meeting
            meeting_title: Title of the meeting
            meeting_id: Meeting identifier

        Returns:
            List of created ticket information
        """
        created_tickets = []

        for action_item in action_items:
            try:
                ticket_info = self._create_single_ticket(
                    action_item,
                    meeting_title,
                    meeting_id
                )
                created_tickets.append(ticket_info)
                logger.info(f"Created Jira ticket: {ticket_info['key']}")

            except Exception as e:
                logger.error(f"Failed to create ticket for '{action_item.title}': {e}")
                created_tickets.append({
                    "action_item": action_item.title,
                    "status": "failed",
                    "error": str(e)
                })

        logger.info(f"Created {len([t for t in created_tickets if t.get('status') != 'failed'])} out of {len(action_items)} tickets")
        return created_tickets

    def _create_single_ticket(
        self,
        action_item: ActionItem,
        meeting_title: str,
        meeting_id: str
    ) -> Dict[str, Any]:
        """Create a single Jira ticket."""

        # Build ticket description
        description = self._build_ticket_description(
            action_item,
            meeting_title,
            meeting_id
        )

        # Map priority
        priority_map = {
            "high": "High",
            "medium": "Medium",
            "low": "Low"
        }
        jira_priority = priority_map.get(action_item.priority.lower(), "Medium")

        # Build issue fields
        issue_fields = {
            "project": {"key": self.project_key},
            "summary": action_item.title,
            "description": description,
            "issuetype": {"name": "Task"},
            "priority": {"name": jira_priority},
            "labels": ["teamsync", f"meeting-{meeting_id}"]
        }

        # Add assignee if specified and valid
        if action_item.assignee:
            try:
                # Try to find user by email or username
                assignee_id = self._find_jira_user(action_item.assignee)
                if assignee_id:
                    issue_fields["assignee"] = {"accountId": assignee_id}
            except Exception as e:
                logger.warning(f"Could not assign to {action_item.assignee}: {e}")

        # Add due date if specified
        if action_item.due_date:
            issue_fields["duedate"] = action_item.due_date.strftime("%Y-%m-%d")

        # Create the issue
        new_issue = self.jira_client.create_issue(fields=issue_fields)

        return {
            "action_item": action_item.title,
            "key": new_issue.key,
            "url": f"{settings.JIRA_URL}/browse/{new_issue.key}",
            "status": "created",
            "assignee": action_item.assignee
        }

    def _build_ticket_description(
        self,
        action_item: ActionItem,
        meeting_title: str,
        meeting_id: str
    ) -> str:
        """Build formatted ticket description."""
        description_parts = [
            f"*Action item from meeting:* {meeting_title}",
            f"*Meeting ID:* {meeting_id}",
            "",
            "*Description:*",
            action_item.description,
            "",
            f"*Priority:* {action_item.priority.capitalize()}",
        ]

        if action_item.assignee:
            description_parts.append(f"*Assigned to:* {action_item.assignee}")

        if action_item.due_date:
            description_parts.append(f"*Due date:* {action_item.due_date.strftime('%Y-%m-%d')}")

        description_parts.extend([
            "",
            "---",
            "_This ticket was automatically created by TeamSync AI Meeting Agent_"
        ])

        return "\n".join(description_parts)

    def _find_jira_user(self, identifier: str) -> str:
        """
        Find Jira user by email or username.

        Args:
            identifier: Email or username

        Returns:
            Account ID of the user
        """
        try:
            # Search for users matching the identifier
            users = self.jira_client.search_users(query=identifier)

            if users:
                return users[0].accountId

            # Try searching by email
            users = self.jira_client.search_users(query=identifier, maxResults=1)
            if users:
                return users[0].accountId

            logger.warning(f"No Jira user found for: {identifier}")
            return None

        except Exception as e:
            logger.error(f"Error finding Jira user {identifier}: {e}")
            return None

    def update_ticket_status(
        self,
        ticket_key: str,
        status: str
    ) -> bool:
        """
        Update ticket status (e.g., Done, In Progress).

        Args:
            ticket_key: Jira ticket key
            status: New status

        Returns:
            Success status
        """
        try:
            issue = self.jira_client.issue(ticket_key)
            transitions = self.jira_client.transitions(issue)

            # Find matching transition
            for transition in transitions:
                if transition['name'].lower() == status.lower():
                    self.jira_client.transition_issue(issue, transition['id'])
                    logger.info(f"Updated {ticket_key} status to: {status}")
                    return True

            logger.warning(f"No transition found for status: {status}")
            return False

        except Exception as e:
            logger.error(f"Error updating ticket status: {e}")
            return False

    def add_comment(
        self,
        ticket_key: str,
        comment: str
    ) -> bool:
        """
        Add comment to a Jira ticket.

        Args:
            ticket_key: Jira ticket key
            comment: Comment text

        Returns:
            Success status
        """
        try:
            self.jira_client.add_comment(ticket_key, comment)
            logger.info(f"Added comment to {ticket_key}")
            return True

        except Exception as e:
            logger.error(f"Error adding comment: {e}")
            return False

    def get_ticket_info(self, ticket_key: str) -> Dict[str, Any]:
        """
        Get ticket information.

        Args:
            ticket_key: Jira ticket key

        Returns:
            Ticket information dictionary
        """
        try:
            issue = self.jira_client.issue(ticket_key)

            return {
                "key": issue.key,
                "summary": issue.fields.summary,
                "description": issue.fields.description,
                "status": issue.fields.status.name,
                "assignee": issue.fields.assignee.displayName if issue.fields.assignee else None,
                "priority": issue.fields.priority.name if issue.fields.priority else None,
                "url": f"{settings.JIRA_URL}/browse/{issue.key}"
            }

        except Exception as e:
            logger.error(f"Error getting ticket info: {e}")
            return {}

    def link_tickets(
        self,
        inward_ticket: str,
        outward_ticket: str,
        link_type: str = "Relates"
    ) -> bool:
        """
        Link two Jira tickets.

        Args:
            inward_ticket: Inward ticket key
            outward_ticket: Outward ticket key
            link_type: Type of link (e.g., "Relates", "Blocks")

        Returns:
            Success status
        """
        try:
            self.jira_client.create_issue_link(
                type=link_type,
                inwardIssue=inward_ticket,
                outwardIssue=outward_ticket
            )
            logger.info(f"Linked {inward_ticket} to {outward_ticket}")
            return True

        except Exception as e:
            logger.error(f"Error linking tickets: {e}")
            return False

    def get_project_tickets(
        self,
        meeting_id: str = None,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get tickets for the project, optionally filtered by meeting.

        Args:
            meeting_id: Optional meeting ID to filter by
            max_results: Maximum number of results

        Returns:
            List of ticket information
        """
        try:
            jql = f"project = {self.project_key} AND labels = teamsync"

            if meeting_id:
                jql += f" AND labels = meeting-{meeting_id}"

            jql += " ORDER BY created DESC"

            issues = self.jira_client.search_issues(jql, maxResults=max_results)

            tickets = []
            for issue in issues:
                tickets.append({
                    "key": issue.key,
                    "summary": issue.fields.summary,
                    "status": issue.fields.status.name,
                    "assignee": issue.fields.assignee.displayName if issue.fields.assignee else None,
                    "url": f"{settings.JIRA_URL}/browse/{issue.key}"
                })

            return tickets

        except Exception as e:
            logger.error(f"Error getting project tickets: {e}")
            return []
