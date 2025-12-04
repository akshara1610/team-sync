# Model Context Protocol (MCP) Integration Guide

## What is MCP?

**Model Context Protocol (MCP)** is Anthropic's standard for connecting LLMs to external tools and data sources. It provides:

- ✅ **Standardized tool interfaces** - Consistent API for all external tools
- ✅ **Context management** - Share state across tool invocations
- ✅ **Audit logging** - Track all tool executions
- ✅ **Security** - Controlled access to resources
- ✅ **LLM integration** - Easy function calling for AI agents

## MCP in TeamSync

### Where MCP is Now Used

TeamSync now uses MCP for:

1. **Jira ticket creation** - via `JiraTool`
2. **Calendar event scheduling** - via `CalendarTool`
3. **Knowledge base queries** - via `KnowledgeBaseTool`

### Architecture

```
┌─────────────────────────────────────────────────┐
│              MCP Server                          │
│  (Central tool registry & execution layer)      │
├─────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ JiraTool │  │CalendarTool│ │KnowledgeTool│  │
│  └────┬─────┘  └─────┬──────┘ └────┬─────┘     │
└───────┼──────────────┼─────────────┼───────────┘
        │              │             │
        ▼              ▼             ▼
   ┌─────────┐   ┌──────────┐  ┌──────────┐
   │  Jira   │   │  Google  │  │ ChromaDB │
   │   API   │   │ Calendar │  │    RAG   │
   └─────────┘   └──────────┘  └──────────┘
```

## Files Created

### 1. **MCP Server** ([src/mcp/mcp_server.py](src/mcp/mcp_server.py))

Core MCP implementation with:
- Tool registration system
- Context management
- Audit logging
- Schema generation for LLM function calling

### 2. **MCP-Enabled Action Agent** ([src/agents/action_agent_mcp.py](src/agents/action_agent_mcp.py))

Action agent that uses MCP instead of direct API calls.

### 3. **MCP Orchestrator** ([src/orchestrator_mcp.py](src/orchestrator_mcp.py))

Orchestrator that coordinates all agents via MCP.

## Usage Examples

### Example 1: Basic MCP Server Setup

```python
from src.mcp.mcp_server import MCPServer, JiraTool, CalendarTool

# Initialize MCP server
mcp_server = MCPServer()

# Register tools
jira_tool = JiraTool(jira_client)
mcp_server.register_tool(jira_tool)

calendar_tool = CalendarTool(calendar_service)
mcp_server.register_tool(calendar_tool)

# List available tools
tools = mcp_server.list_tools()
print(tools)
# [
#   {"name": "jira_create_ticket", "description": "..."},
#   {"name": "calendar_create_event", "description": "..."}
# ]
```

### Example 2: Execute Tool via MCP

```python
# Create Jira ticket via MCP
result = await mcp_server.execute_tool(
    tool_name="jira_create_ticket",
    parameters={
        "summary": "Fix authentication bug",
        "description": "Users unable to log in",
        "assignee": "alice@example.com",
        "priority": "High",
        "project_key": "PROJ"
    },
    context={
        "meeting_id": "meeting-123",
        "source": "teamsync"
    }
)

if result["success"]:
    print(f"Created ticket: {result['ticket_key']}")
    print(f"URL: {result['ticket_url']}")
```

### Example 3: Create Calendar Event via MCP

```python
# Schedule meeting via MCP
result = await mcp_server.execute_tool(
    tool_name="calendar_create_event",
    parameters={
        "summary": "Sprint Planning",
        "description": "Plan next sprint tasks",
        "start_time": "2024-01-15T10:00:00Z",
        "end_time": "2024-01-15T11:00:00Z",
        "attendees": ["alice@example.com", "bob@example.com"]
    }
)

if result["success"]:
    print(f"Created event: {result['event_id']}")
```

### Example 4: Query Knowledge Base via MCP

```python
# Query via MCP
result = await mcp_server.execute_tool(
    tool_name="knowledge_query",
    parameters={
        "query": "What was discussed about API migration?",
        "top_k": 5
    }
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
```

### Example 5: Using MCP Orchestrator

```python
from src.orchestrator_mcp import MCPMeetingOrchestrator

# Initialize orchestrator with MCP
orchestrator = MCPMeetingOrchestrator()

# Process meeting (all tools via MCP)
result = await orchestrator.process_meeting_with_mcp(
    room_name="standup",
    meeting_title="Daily Standup",
    access_token="token"
)

# Check MCP stats
print(f"Tools used: {result['mcp_stats']['tools_registered']}")
print(f"Tool calls: {result['mcp_stats']['tool_invocations']}")
print(f"Audit log: {result['mcp_stats']['audit_log']}")
```

## MCP Server Features

### 1. Tool Registration

```python
class MyCustomTool(MCPTool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="Does something useful"
        )

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        # Tool implementation
        return {"success": True, "result": "..."}

# Register with MCP
mcp_server.register_tool(MyCustomTool())
```

### 2. Context Management

```python
# Set context (available to all tools)
mcp_server.set_context("meeting_id", "meeting-123")
mcp_server.set_context("user_id", "user-456")

# Get context
meeting_id = mcp_server.get_context("meeting_id")

# Context is automatically passed to tool executions
result = await mcp_server.execute_tool(
    "some_tool",
    parameters={...},
    context={"additional": "data"}  # Merged with server context
)
```

### 3. Audit Logging

```python
# Get audit log
audit_log = mcp_server.get_audit_log()

# Each entry contains:
for entry in audit_log:
    print(f"Time: {entry['timestamp']}")
    print(f"Tool: {entry['tool']}")
    print(f"Parameters: {entry['parameters']}")
    print(f"Result: {entry['result']}")
```

### 4. LLM Function Calling

```python
# Get tool schemas for OpenAI function calling
schemas = mcp_server.get_all_schemas()

# Use with OpenAI
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[...],
    functions=schemas  # MCP tool schemas
)

# If function call is requested
if response["choices"][0]["message"].get("function_call"):
    func_name = response["choices"][0]["message"]["function_call"]["name"]
    func_args = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])

    # Execute via MCP
    result = await mcp_server.execute_tool(func_name, func_args)
```

## Benefits of MCP Integration

### 1. Standardization

**Before (Direct API calls):**
```python
# Different interface for each tool
jira_client.create_issue(fields={...})
calendar_service.events().insert(body={...}).execute()
knowledge_agent.query(query)
```

**After (MCP):**
```python
# Consistent interface for all tools
await mcp_server.execute_tool("jira_create_ticket", {...})
await mcp_server.execute_tool("calendar_create_event", {...})
await mcp_server.execute_tool("knowledge_query", {...})
```

### 2. Audit Trail

Every tool execution is logged:
```python
{
  "timestamp": "2024-01-15T10:30:00Z",
  "tool": "jira_create_ticket",
  "parameters": {"summary": "...", "priority": "High"},
  "result": {"success": True, "ticket_key": "PROJ-123"}
}
```

### 3. Context Sharing

```python
# Set once, available everywhere
mcp_server.set_context("meeting_id", "meeting-123")

# All tools can access context
# No need to pass meeting_id to every function
```

### 4. Easy LLM Integration

```python
# LLMs can discover and call tools automatically
schemas = mcp_server.get_all_schemas()

# GPT-4 can now call any registered tool
# without hardcoding tool names
```

## Comparison: Traditional vs MCP

| Feature | Traditional | MCP |
|---------|------------|-----|
| **Tool Interface** | Different for each | Standardized |
| **Audit Log** | Manual | Automatic |
| **Context Sharing** | Pass manually | Server-managed |
| **LLM Integration** | Custom code | Built-in schemas |
| **Error Handling** | Per-tool | Centralized |
| **Testing** | Mock each API | Mock MCP server |

## Tool Schemas

MCP automatically generates OpenAI-compatible function calling schemas:

```json
{
  "name": "jira_create_ticket",
  "description": "Create a Jira ticket with title, description, assignee, and priority",
  "parameters": {
    "type": "object",
    "properties": {
      "summary": {
        "type": "string",
        "description": "Ticket title"
      },
      "description": {
        "type": "string",
        "description": "Ticket description"
      },
      "assignee": {
        "type": "string",
        "description": "Assignee email"
      },
      "priority": {
        "type": "string",
        "enum": ["High", "Medium", "Low"]
      },
      "project_key": {
        "type": "string",
        "description": "Jira project key"
      }
    },
    "required": ["summary", "description", "project_key"]
  }
}
```

## Testing MCP

### Test Individual Tool

```python
from src.mcp.mcp_server import MCPServer, JiraTool

mcp_server = MCPServer()
jira_tool = JiraTool(jira_client)
mcp_server.register_tool(jira_tool)

# Test execution
result = await mcp_server.execute_tool(
    "jira_create_ticket",
    {"summary": "Test", "description": "Test", "project_key": "TEST"}
)

assert result["success"] == True
```

### Test MCP Orchestrator

```python
from src.orchestrator_mcp import MCPMeetingOrchestrator

orchestrator = MCPMeetingOrchestrator()

# Check registered tools
tools = orchestrator.list_available_tools()
assert len(tools) >= 3  # Jira, Calendar, Knowledge

# Process meeting
result = await orchestrator.process_meeting_with_mcp(...)
assert result["status"] == "success"

# Check audit log
audit = orchestrator.get_audit_trail()
assert len(audit) > 0
```

## Adding New MCP Tools

### Step 1: Create Tool Class

```python
from src.mcp.mcp_server import MCPTool

class SlackTool(MCPTool):
    def __init__(self, slack_client):
        super().__init__(
            name="slack_send_message",
            description="Send a message to a Slack channel"
        )
        self.slack_client = slack_client

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = self.slack_client.chat_postMessage(
                channel=parameters["channel"],
                text=parameters["message"]
            )

            return {
                "success": True,
                "message_ts": result["ts"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

### Step 2: Register Tool

```python
# In orchestrator or agent
slack_tool = SlackTool(slack_client)
mcp_server.register_tool(slack_tool)
```

### Step 3: Use Tool

```python
result = await mcp_server.execute_tool(
    "slack_send_message",
    {
        "channel": "#general",
        "message": "Meeting summary attached"
    }
)
```

## MCP vs Direct API Calls

### Direct API Call (Original)

```python
# action_agent.py
issue = self.jira_client.create_issue(fields={
    "project": {"key": "PROJ"},
    "summary": action_item.title,
    "description": action_item.description,
    "issuetype": {"name": "Task"}
})
```

**Problems:**
- ❌ No audit trail
- ❌ No context sharing
- ❌ Hard to test
- ❌ Not LLM-friendly

### MCP Approach (New)

```python
# action_agent_mcp.py
result = await mcp_server.execute_tool(
    "jira_create_ticket",
    {
        "summary": action_item.title,
        "description": action_item.description,
        "project_key": "PROJ"
    }
)
```

**Benefits:**
- ✅ Automatic audit logging
- ✅ Context automatically included
- ✅ Easy to mock for testing
- ✅ LLM can discover and call

## API Integration

You can expose MCP functionality via API:

```python
# In FastAPI
@app.get("/mcp/tools")
async def list_tools():
    """List all available MCP tools."""
    return orchestrator.list_available_tools()

@app.get("/mcp/schemas")
async def get_schemas():
    """Get tool schemas for LLM function calling."""
    return orchestrator.get_tool_schemas_for_llm()

@app.get("/mcp/audit")
async def get_audit():
    """Get MCP audit trail."""
    return orchestrator.get_audit_trail()

@app.post("/mcp/execute")
async def execute_tool(tool_name: str, parameters: Dict):
    """Execute an MCP tool."""
    result = await orchestrator.mcp_server.execute_tool(
        tool_name,
        parameters
    )
    return result
```

## Summary

### What MCP Provides

1. ✅ **Standardized tool interface** - One API for all tools
2. ✅ **Automatic audit logging** - Track all executions
3. ✅ **Context management** - Share state across tools
4. ✅ **LLM integration** - Function calling schemas
5. ✅ **Security** - Controlled access layer
6. ✅ **Testing** - Easy to mock MCP server

### MCP Integration Status

- ✅ **MCP Server implemented** - [src/mcp/mcp_server.py](src/mcp/mcp_server.py)
- ✅ **JiraTool** - Ticket creation via MCP
- ✅ **CalendarTool** - Event scheduling via MCP
- ✅ **KnowledgeBaseTool** - RAG queries via MCP
- ✅ **MCP Action Agent** - [src/agents/action_agent_mcp.py](src/agents/action_agent_mcp.py)
- ✅ **MCP Orchestrator** - [src/orchestrator_mcp.py](src/orchestrator_mcp.py)

### Files Summary

| File | Purpose |
|------|---------|
| `src/mcp/mcp_server.py` | Core MCP implementation |
| `src/agents/action_agent_mcp.py` | MCP-enabled action agent |
| `src/orchestrator_mcp.py` | MCP-enabled orchestrator |

**TeamSync now has full Model Context Protocol integration! 🎉**
