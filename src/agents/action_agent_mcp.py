"""
MCP-Enabled Action Agent
Uses Model Context Protocol for standardized tool execution.
"""
from typing import List, Dict, Any
from loguru import logger
from jira import JIRA

from src.config import settings
from src.models.schemas import ActionItem
from src.mcp.mcp_server import MCPServer, JiraTool


class ActionAgentMCP:
    """
    Action Agent that uses MCP for tool execution.

    Benefits of MCP:
    - Standardized interface for all tools
    - Audit logging of all actions
    - Context sharing across tool invocations
    - Easy to add new tools
    - Compatible with LLM function calling
    """

    def __init__(self, mcp_server: MCPServer):
        logger.info("Initializing MCP-Enabled Action Agent...")

        self.mcp_server = mcp_server

        # Initialize Jira client
        self.jira_client = JIRA(
            server=settings.JIRA_URL,
            basic_auth=(settings.JIRA_EMAIL, settings.JIRA_API_TOKEN)
        )

        # Register Jira tool with MCP server
        # Participants will be provided via MCP context at runtime
        jira_tool = JiraTool(self.jira_client)
        self.mcp_server.register_tool(jira_tool)

        logger.info("Action Agent initialized with MCP")

    async def create_jira_tickets(
        self,
        action_items: List[ActionItem],
        meeting_title: str,
        meeting_id: str
    ) -> List[Dict[str, Any]]:
        """
        Create Jira tickets using MCP tool execution.

        Args:
            action_items: List of action items
            meeting_title: Meeting title
            meeting_id: Meeting identifier

        Returns:
            List of ticket creation results
        """
        # Set context for all tool calls
        self.mcp_server.set_context("meeting_id", meeting_id)
        self.mcp_server.set_context("meeting_title", meeting_title)

        created_tickets = []

        for action_item in action_items:
            # Skip action items that are actually meetings (should go to Calendar instead)
            if self._is_meeting_action(action_item):
                logger.info(f"[MCP] Skipping meeting action item (should be in Calendar): {action_item.title}")
                continue

            try:
                logger.info(f"[MCP] Creating ticket for: {action_item.title}")

                # Build description
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
                priority = priority_map.get(action_item.priority.lower(), "Medium")

                # Execute via MCP
                result = await self.mcp_server.execute_tool(
                    tool_name="jira_create_ticket",
                    parameters={
                        "summary": action_item.title,
                        "description": description,
                        "assignee": action_item.assignee,
                        "priority": priority,
                        "project_key": settings.JIRA_PROJECT_KEY
                    },
                    context={
                        "action_item_id": action_item.title,
                        "source": "teamsync_agent"
                    }
                )

                if result.get("success"):
                    # Handle multiple tickets (when assignee is "Everyone" or "All team members")
                    if "tickets" in result:
                        for ticket in result["tickets"]:
                            if ticket.get("success"):
                                created_tickets.append({
                                    "action_item": action_item.title,
                                    "status": "created",
                                    "ticket_key": ticket["ticket_key"],
                                    "ticket_url": ticket["ticket_url"],
                                    "assignee": ticket.get("assignee", action_item.assignee)
                                })
                            else:
                                created_tickets.append({
                                    "action_item": action_item.title,
                                    "status": "failed",
                                    "error": ticket.get("error"),
                                    "assignee": ticket.get("assignee", action_item.assignee)
                                })
                    else:
                        # Single ticket
                        created_tickets.append({
                            "action_item": action_item.title,
                            "status": "created",
                            "ticket_key": result["ticket_key"],
                            "ticket_url": result["ticket_url"],
                            "assignee": result.get("assignee", action_item.assignee)
                        })
                else:
                    created_tickets.append({
                        "action_item": action_item.title,
                        "status": "failed",
                        "error": result.get("error")
                    })

            except Exception as e:
                logger.error(f"Error creating ticket: {e}")
                created_tickets.append({
                    "action_item": action_item.title,
                    "status": "failed",
                    "error": str(e)
                })

        logger.info(f"[MCP] Created {len([t for t in created_tickets if t.get('status') == 'created'])} tickets")

        return created_tickets

    def _is_meeting_action(self, action_item: ActionItem) -> bool:
        """
        Determine if an action item is actually a meeting (should go to Calendar).

        Returns:
            True if this is a meeting action, False if it's a task action
        """
        meeting_keywords = [
            "attend meeting",
            "join meeting",
            "meeting on",
            "schedule meeting",
            "mandatory meeting",
            "attend session",
            "participate in meeting"
        ]

        title_lower = action_item.title.lower()
        desc_lower = action_item.description.lower()

        # Check if title or description contains meeting keywords
        for keyword in meeting_keywords:
            if keyword in title_lower or keyword in desc_lower:
                return True

        return False

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
            "_This ticket was automatically created by TeamSync AI via MCP_"
        ])

        return "\n".join(description_parts)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get schemas for all registered tools.
        Useful for LLM function calling.
        """
        return self.mcp_server.get_all_schemas()

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get audit log of all MCP tool executions."""
        return self.mcp_server.get_audit_log()
