"""
Model Context Protocol (MCP) Server for TeamSync
Provides standardized tool access for AI agents via Anthropic's MCP.

MCP enables agents to:
1. Call external tools (Jira, Google Calendar) through a standard interface
2. Share context across tool invocations
3. Have controlled, auditable access to resources
"""
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from loguru import logger

# Note: This is a conceptual implementation
# The actual MCP package from Anthropic would be used in production
# For now, we create a compatible interface


class MCPTool:
    """Base class for MCP tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        raise NotImplementedError


class JiraTool(MCPTool):
    """MCP tool for Jira integration."""

    def __init__(self, jira_client, participants: Optional[List[str]] = None):
        super().__init__(
            name="jira_create_ticket",
            description="Create a Jira ticket with title, description, assignee, and priority"
        )
        self.jira_client = jira_client
        self.participants = participants or []

        # Name to email mapping
        self.name_mapping = {
            "vrinda": "vva2113@columbia.edu",
            "akshara": "ap4613@columbia.edu",
            "sachi": "sk5476@columbia.edu",
        }

    def _resolve_assignee(self, assignee_str: str) -> List[str]:
        """
        Resolve assignee string to list of email addresses.

        Handles:
        - "All team members" -> all participants
        - First name (e.g., "Vrinda") -> mapped email
        - Direct email -> single email

        Returns:
            List of email addresses
        """
        if not assignee_str:
            return []

        assignee_lower = assignee_str.lower().strip()

        # Handle "All team members" or similar
        if "all" in assignee_lower or "everyone" in assignee_lower or "team" in assignee_lower:
            logger.info(f"[MCP] Assignee '{assignee_str}' maps to all participants: {self.participants}")
            return self.participants

        # Check name mapping (case-insensitive)
        if assignee_lower in self.name_mapping:
            email = self.name_mapping[assignee_lower]
            logger.info(f"[MCP] Mapped name '{assignee_str}' to {email}")
            return [email]

        # Check if it's already an email
        if "@" in assignee_str:
            return [assignee_str]

        # Try to find partial match in participants
        for participant in self.participants:
            # Extract first name from email (before @)
            participant_name = participant.split('@')[0].lower()
            if assignee_lower in participant_name or participant_name in assignee_lower:
                logger.info(f"[MCP] Matched '{assignee_str}' to participant {participant}")
                return [participant]

        # If no match, return as-is (will be handled by JIRA search)
        logger.warning(f"[MCP] Could not map assignee '{assignee_str}' to participant")
        return [assignee_str]

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Jira ticket via MCP.

        Parameters:
            - summary (str): Ticket title
            - description (str): Ticket description
            - assignee (str, optional): Assignee email/username/name or "All team members"
            - priority (str): Priority level (High/Medium/Low)
            - project_key (str): Jira project key
        """
        try:
            # Get participants from context if available
            context = parameters.get("_context", {})
            if context.get("participants"):
                self.participants = context["participants"]
                logger.info(f"[MCP] Using participants from context: {self.participants}")

            assignee_str = parameters.get("assignee", "")
            assignee_emails = self._resolve_assignee(assignee_str)

            # If multiple assignees (e.g., "All team members"), create multiple tickets
            if len(assignee_emails) > 1:
                logger.info(f"[MCP] Creating {len(assignee_emails)} tickets (one per assignee)")
                created_tickets = []

                for assignee_email in assignee_emails:
                    result = await self._create_single_ticket(
                        parameters=parameters,
                        assignee_email=assignee_email
                    )
                    created_tickets.append(result)

                # Return consolidated result
                successful = [t for t in created_tickets if t.get("success")]
                failed = [t for t in created_tickets if not t.get("success")]

                return {
                    "success": len(successful) > 0,
                    "tickets": created_tickets,
                    "message": f"Created {len(successful)}/{len(assignee_emails)} tickets",
                    "summary": {
                        "successful": len(successful),
                        "failed": len(failed)
                    }
                }

            # Single assignee or no assignee
            assignee_email = assignee_emails[0] if assignee_emails else None
            return await self._create_single_ticket(parameters, assignee_email)

        except Exception as e:
            logger.error(f"[MCP] Jira tool error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _create_single_ticket(
        self,
        parameters: Dict[str, Any],
        assignee_email: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a single JIRA ticket with optional assignee."""
        try:
            logger.info(f"[MCP] Creating Jira ticket: {parameters.get('summary')}")

            issue_fields = {
                "project": {"key": parameters["project_key"]},
                "summary": parameters["summary"],
                "description": parameters["description"],
                "issuetype": {"name": "Task"},
                "priority": {"name": parameters.get("priority", "Medium")}
            }

            # Add assignee if provided
            if assignee_email:
                # Find user by email in JIRA
                users = self.jira_client.search_users(query=assignee_email)
                if users:
                    issue_fields["assignee"] = {"accountId": users[0].accountId}
                    logger.info(f"[MCP] Assigned to {assignee_email} (accountId: {users[0].accountId})")
                else:
                    logger.warning(f"[MCP] User {assignee_email} not found in JIRA, creating unassigned ticket")

            new_issue = self.jira_client.create_issue(fields=issue_fields)

            return {
                "success": True,
                "ticket_key": new_issue.key,
                "ticket_url": f"{self.jira_client._options['server']}/browse/{new_issue.key}",
                "message": f"Created ticket {new_issue.key}",
                "assignee": assignee_email
            }

        except Exception as e:
            logger.error(f"[MCP] Jira tool error for assignee {assignee_email}: {e}")
            return {
                "success": False,
                "error": str(e),
                "assignee": assignee_email
            }


class CalendarTool(MCPTool):
    """MCP tool for Google Calendar integration."""

    def __init__(self, calendar_service):
        super().__init__(
            name="calendar_create_event",
            description="Create a Google Calendar event with title, time, and attendees"
        )
        self.calendar_service = calendar_service

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create calendar event via MCP.

        Parameters:
            - summary (str): Event title
            - description (str): Event description
            - start_time (str): ISO format datetime
            - end_time (str): ISO format datetime
            - attendees (list): List of email addresses
        """
        try:
            logger.info(f"[MCP] Creating calendar event: {parameters.get('summary')}")

            event_body = {
                'summary': parameters['summary'],
                'description': parameters.get('description', ''),
                'start': {
                    'dateTime': parameters['start_time'],
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': parameters['end_time'],
                    'timeZone': 'UTC',
                },
                'attendees': [{'email': email} for email in parameters.get('attendees', []) if email and email.strip()],
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 24 * 60},
                        {'method': 'popup', 'minutes': 30},
                    ],
                },
            }

            created_event = self.calendar_service.events().insert(
                calendarId='primary',
                body=event_body,
                sendUpdates='all'
            ).execute()

            return {
                "success": True,
                "event_id": created_event['id'],
                "event_url": created_event['htmlLink'],
                "message": f"Created event: {created_event['summary']}"
            }

        except Exception as e:
            logger.error(f"[MCP] Calendar tool error: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class KnowledgeBaseTool(MCPTool):
    """MCP tool for querying the knowledge base."""

    def __init__(self, knowledge_agent):
        super().__init__(
            name="knowledge_query",
            description="Query the meeting knowledge base using RAG"
        )
        self.knowledge_agent = knowledge_agent

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query knowledge base via MCP.

        Parameters:
            - query (str): Natural language query
            - top_k (int): Number of results to return
        """
        try:
            logger.info(f"[MCP] Querying knowledge base: {parameters.get('query')}")

            response = self.knowledge_agent.query(
                query=parameters['query'],
                top_k=parameters.get('top_k', 5)
            )

            return {
                "success": True,
                "answer": response.answer,
                "sources": response.sources,
                "confidence": response.confidence
            }

        except Exception as e:
            logger.error(f"[MCP] Knowledge tool error: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class MCPServer:
    """
    MCP Server that manages tool registry and execution.
    Provides a standardized interface for agents to interact with tools.
    """

    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        self.context: Dict[str, Any] = {}
        self.audit_log: List[Dict[str, Any]] = []
        logger.info("Initializing MCP Server...")

    def register_tool(self, tool: MCPTool):
        """Register a tool with the MCP server."""
        self.tools[tool.name] = tool
        logger.info(f"[MCP] Registered tool: {tool.name}")

    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self.tools.values()
        ]

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool via MCP with context management.

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            context: Optional context to merge with server context

        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}"
            }

        # Merge context
        if context:
            self.context.update(context)

        # Add context to parameters if needed
        parameters["_context"] = self.context

        # Log the invocation
        invocation = {
            "timestamp": datetime.utcnow().isoformat(),
            "tool": tool_name,
            "parameters": {k: v for k, v in parameters.items() if k != "_context"}
        }

        # Execute tool
        tool = self.tools[tool_name]
        result = await tool.execute(parameters)

        # Add to audit log
        invocation["result"] = result
        self.audit_log.append(invocation)

        logger.info(f"[MCP] Executed {tool_name}: {result.get('success', False)}")

        return result

    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get the schema for a tool (for LLM function calling)."""
        if tool_name not in self.tools:
            return {}

        tool = self.tools[tool_name]

        # This would be auto-generated from tool annotations in production
        schemas = {
            "jira_create_ticket": {
                "name": "jira_create_ticket",
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "Ticket title"},
                        "description": {"type": "string", "description": "Ticket description"},
                        "assignee": {"type": "string", "description": "Assignee email"},
                        "priority": {"type": "string", "enum": ["High", "Medium", "Low"]},
                        "project_key": {"type": "string", "description": "Jira project key"}
                    },
                    "required": ["summary", "description", "project_key"]
                }
            },
            "calendar_create_event": {
                "name": "calendar_create_event",
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string", "description": "Event title"},
                        "description": {"type": "string", "description": "Event description"},
                        "start_time": {"type": "string", "description": "ISO datetime"},
                        "end_time": {"type": "string", "description": "ISO datetime"},
                        "attendees": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["summary", "start_time", "end_time"]
                }
            },
            "knowledge_query": {
                "name": "knowledge_query",
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language query"},
                        "top_k": {"type": "integer", "description": "Number of results"}
                    },
                    "required": ["query"]
                }
            }
        }

        return schemas.get(tool_name, {})

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools (for LLM function calling)."""
        return [self.get_tool_schema(name) for name in self.tools.keys()]

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get the audit log of all tool invocations."""
        return self.audit_log

    def clear_context(self):
        """Clear the shared context."""
        self.context = {}
        logger.info("[MCP] Context cleared")

    def set_context(self, key: str, value: Any):
        """Set a context value."""
        self.context[key] = value

    def get_context(self, key: str) -> Any:
        """Get a context value."""
        return self.context.get(key)
