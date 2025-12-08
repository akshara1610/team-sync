# TeamSync - Setup and Run Guide

## Architecture Overview

**TeamSync uses:**
- ✅ **LangGraph** - State machine workflow coordination
- ✅ **MCP** - Model Context Protocol for tool execution
- ✅ **LangChain** - RetrievalQA for knowledge base
- ✅ **ChromaDB** - Vector database for RAG
- ✅ **6 AI Agents** - Listener, Knowledge, Summarizer, Reflection, Action, Scheduler

## Prerequisites

- Python 3.9+
- PostgreSQL 15+
- Redis 7+
- API Keys (see below)

---

## Step 1: Clone and Setup Virtual Environment

```bash
cd "/Users/aksharapramod/Documents/Columbia Documents/Course Work/Sem3/LLM Based Gen AI/team-sync"

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key packages:**
- `langchain`, `langchain-openai`, `langchain-chroma` - LangChain ecosystem
- `langgraph` - State machine workflows
- `chromadb` - Vector database
- `fastapi` - API framework
- `livekit` - Meeting audio
- `jira`, `google-api-python-client` - External integrations

---

## Step 3: Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use any editor
```

### Required API Keys

#### 1. OpenAI API Key
```env
OPENAI_API_KEY=sk-...
```
Get from: https://platform.openai.com/api-keys

#### 2. HuggingFace Token
```env
HF_TOKEN=hf_...
```
- Get from: https://huggingface.co/settings/tokens
- **Important:** Accept terms at https://huggingface.co/pyannote/speaker-diarization

#### 3. Jira Configuration
```env
JIRA_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=...
JIRA_PROJECT_KEY=PROJ
```
Generate token: https://id.atlassian.com/manage-profile/security/api-tokens

#### 4. Google Calendar
- Follow: https://developers.google.com/calendar/api/quickstart/python
- Download `credentials.json` to project root

#### 5. Database (Local Development)
```env
DATABASE_URL=postgresql://teamsync:password@localhost:5432/teamsync
REDIS_URL=redis://localhost:6379/0
```

#### 6. LiveKit (Optional for testing)
```env
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
```

---

## Step 4: Start Database Services

### Option A: Docker (Recommended)

```bash
# Start PostgreSQL and Redis only
docker-compose up -d postgres redis
```

### Option B: Local PostgreSQL

```bash
# Create database
createdb teamsync

# Or with psql
psql postgres
CREATE DATABASE teamsync;
\q
```

---

## Step 5: Initialize Database

```bash
python scripts/setup_db.py
```

Expected output:
```
Initializing database...
Database initialized successfully!
```

---

## Step 6: Test Individual Components

### Test 1: LangChain Knowledge Agent

```bash
python scripts/test_langchain_rag.py
```

Expected output:
```
✓ Agent initialized with LangChain
✓ Transcripts added to ChromaDB
✓ RetrievalQA queries working
✓ Similarity search functional
```

### Test 2: Summarizer + Reflection

```bash
python scripts/test_summarizer.py
```

Expected output:
```
✓ Executive Summary generated
✓ Action Items: 2
✓ Reflection approved: True
✓ Coherence Score: 0.89
```

---

## Step 7: Start the API Server

```bash
python main.py
```

Expected output:
```
====================================================================
Initializing TeamSync Orchestrator
Architecture: LangGraph + MCP
====================================================================
✓ All agents initialized
✓ MCP tools: ['jira_create_ticket', 'calendar_create_event', 'knowledge_query']
✓ LangGraph workflow compiled
====================================================================
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Step 8: Verify API is Running

```bash
# In another terminal
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "architecture": "LangGraph + MCP"
}
```

### Check Workflow Info

```bash
curl http://localhost:8000/workflow/info
```

Response:
```json
{
  "workflow": "LangGraph State Machine",
  "nodes": ["listen", "summarize", "reflect", "improve", "execute_actions", "schedule_followup", "store_knowledge"],
  "mcp_tools": ["jira_create_ticket", "calendar_create_event", "knowledge_query"],
  "features": [
    "Stateful workflow execution",
    "Self-reflection loop",
    "MCP tool integration",
    "Automatic audit logging"
  ]
}
```

### Check MCP Tools

```bash
curl http://localhost:8000/mcp/tools
```

---

## Step 9: Test the Complete Workflow

### Option 1: Via API Docs (Easiest)

1. Open browser: http://localhost:8000/docs
2. Click on `POST /meetings/start`
3. Click "Try it out"
4. Enter test data:
```json
{
  "room_name": "test-meeting",
  "meeting_title": "Test Meeting",
  "access_token": "your_livekit_token"
}
```
5. Click "Execute"

### Option 2: Via cURL

```bash
curl -X POST http://localhost:8000/meetings/start \
  -H "Content-Type: application/json" \
  -d '{
    "room_name": "test",
    "meeting_title": "Daily Standup",
    "access_token": "livekit_token"
  }'
```

### Expected Response:

```json
{
  "meeting_id": "uuid-here",
  "status": "success",
  "transcript_path": "data/transcripts/uuid.json",
  "summary_path": "data/summaries/uuid.json",
  "jira_tickets": [
    {
      "action_item": "Update API docs",
      "status": "created",
      "ticket_key": "PROJ-123",
      "ticket_url": "https://..."
    }
  ],
  "followup_meeting": {
    "success": true,
    "event_id": "...",
    "event_url": "https://..."
  },
  "reflection_iterations": 2,
  "mcp_audit_log": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "tool": "jira_create_ticket",
      "result": {"success": true}
    }
  ],
  "workflow_messages": [
    "Transcribed 15 segments",
    "Generated summary with 3 actions",
    "Reflection 1: NEEDS IMPROVEMENT",
    "Reflection 2: PASSED",
    "Created 3 Jira tickets via MCP",
    "Follow-up scheduled via MCP",
    "Knowledge stored. MCP calls: 5"
  ]
}
```

---

## Step 10: Query Knowledge Base

```bash
curl -X POST http://localhost:8000/knowledge/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was discussed about the API?",
    "top_k": 5
  }'
```

Response:
```json
{
  "success": true,
  "answer": "The team discussed migrating from REST to GraphQL...",
  "sources": [
    {
      "meeting_id": "...",
      "meeting_title": "API Review",
      "speaker": "Alice",
      "text": "..."
    }
  ],
  "confidence": 0.87
}
```

---

## Workflow Execution Steps

When you call `/meetings/start`, here's what happens:

```
1. [LangGraph Node: listen]
   → Listener Agent transcribes meeting
   → Saves to data/transcripts/

2. [LangGraph Node: summarize]
   → Summarizer Agent generates MoM
   → Extracts action items and decisions

3. [LangGraph Node: reflect]
   → Self-Reflection Agent validates
   → Checks factual consistency
   → Scores coherence

4. [Conditional Edge]
   → If validation_passed: continue
   → If needs improvement: go to improve (max 3 iterations)

5. [LangGraph Node: execute_actions]
   → Action Agent creates Jira tickets
   → Uses MCP: execute_tool("jira_create_ticket", ...)
   → MCP logs audit trail

6. [LangGraph Node: schedule_followup]
   → Scheduler Agent creates calendar event
   → Uses MCP: execute_tool("calendar_create_event", ...)
   → Auto-invites participants

7. [LangGraph Node: store_knowledge]
   → Knowledge Agent stores in ChromaDB
   → LangChain RetrievalQA enabled
   → Updates database

8. [END]
   → Returns complete results
   → MCP audit log included
```

---

## Troubleshooting

### Issue: "Database connection failed"

```bash
# Check PostgreSQL is running
pg_isready

# If not running:
brew services start postgresql  # Mac
sudo systemctl start postgresql  # Linux
```

### Issue: "HuggingFace authentication failed"

- Accept terms: https://huggingface.co/pyannote/speaker-diarization
- Verify token in `.env`

### Issue: "Jira API authentication failed"

```bash
# Test Jira connection
curl -u your-email@example.com:your-api-token \
  https://your-domain.atlassian.net/rest/api/3/myself
```

### Issue: "Google Calendar credentials not found"

- Download `credentials.json` from Google Cloud Console
- Place in project root

### Issue: "LangChain import errors"

```bash
pip install langchain-openai langchain-chroma --upgrade
```

### Issue: "Port 8000 already in use"

```bash
# Change port in .env
API_PORT=8001

# Or kill process
lsof -ti:8000 | xargs kill
```

---

## Testing Checklist

- [ ] Database initialized
- [ ] LangChain RAG test passed
- [ ] Summarizer test passed
- [ ] API health check returns 200
- [ ] Workflow info shows all nodes
- [ ] MCP tools list shows 3 tools
- [ ] Complete meeting workflow executes
- [ ] Jira ticket created
- [ ] Calendar event scheduled
- [ ] Knowledge base queryable
- [ ] MCP audit log captured

---

## Next Steps

1. ✅ **Process a real meeting** - Use LiveKit for actual audio
2. ✅ **Query knowledge base** - Ask questions about past meetings
3. ✅ **Review Jira tickets** - Check auto-created tasks
4. ✅ **Check calendar** - Verify follow-up meeting scheduled
5. ✅ **Review audit log** - See all MCP tool calls

---

## Architecture Summary

**Files in Use:**
- `src/orchestrator.py` - Main orchestrator (LangGraph + MCP)
- `src/api/main.py` - FastAPI application
- `src/agents/knowledge_agent_langchain.py` - LangChain RAG
- `src/agents/action_agent_mcp.py` - MCP-enabled actions
- `src/mcp/mcp_server.py` - MCP server implementation

**Technologies:**
- LangGraph: State machine workflow
- MCP: Standardized tool execution
- LangChain: RetrievalQA for knowledge
- ChromaDB: Vector database
- FastAPI: REST API

**All working together for production-ready meeting intelligence! 🚀**
