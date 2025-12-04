# TeamSync - Quick Start Guide

This guide will help you get TeamSync up and running in 15 minutes.

## Prerequisites

Before starting, ensure you have:
- Python 3.9+ installed
- PostgreSQL 15+ running
- Redis 7+ running
- Required API keys (see below)

## Step 1: Clone and Setup

```bash
# Clone repository
git clone https://github.com/your-org/team-sync.git
cd team-sync

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env
```

Edit `.env` and add your API keys:

### Required API Keys

1. **OpenAI API Key**
   ```
   OPENAI_API_KEY=sk-...
   ```
   Get from: https://platform.openai.com/api-keys

2. **HuggingFace Token**
   ```
   HF_TOKEN=hf_...
   ```
   - Get from: https://huggingface.co/settings/tokens
   - Accept pyannote terms: https://huggingface.co/pyannote/speaker-diarization

3. **Jira Configuration**
   ```
   JIRA_URL=https://your-domain.atlassian.net
   JIRA_EMAIL=your-email@example.com
   JIRA_API_TOKEN=...
   JIRA_PROJECT_KEY=PROJ
   ```
   Generate token: https://id.atlassian.com/manage-profile/security/api-tokens

4. **Google Calendar**
   - Follow: https://developers.google.com/calendar/api/quickstart/python
   - Download `credentials.json` to project root

5. **LiveKit** (Optional for testing)
   ```
   LIVEKIT_URL=ws://localhost:7880
   LIVEKIT_API_KEY=devkey
   LIVEKIT_API_SECRET=secret
   ```

### Database Configuration

```env
DATABASE_URL=postgresql://teamsync:password@localhost:5432/teamsync
REDIS_URL=redis://localhost:6379/0
```

## Step 3: Setup Database

### Option A: Using PostgreSQL locally

```bash
# Create database
createdb teamsync

# Initialize tables
python scripts/setup_db.py
```

### Option B: Using Docker

```bash
# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Initialize database
python scripts/setup_db.py
```

## Step 4: Test Individual Agents

### Test Knowledge Agent (ChromaDB + RAG)

```bash
python scripts/test_knowledge_agent.py
```

Expected output:
```
✓ Transcript added successfully!
✓ Answer: In the meeting, the team discussed migrating to...
✓ Confidence: 0.87
```

### Test Summarizer + Self-Reflection

```bash
python scripts/test_summarizer.py
```

Expected output:
```
✓ Executive Summary: The team discussed sprint planning...
✓ Action Items: 3
✓ Reflection approved: True
✓ Coherence Score: 0.89
```

## Step 5: Start the API Server

```bash
python main.py
```

The server will start at: http://localhost:8000

API Documentation: http://localhost:8000/docs

## Step 6: Test the API

### Health Check

```bash
curl http://localhost:8000/health
```

### Query Knowledge Base

```bash
curl -X POST http://localhost:8000/knowledge/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was discussed about API migration?",
    "top_k": 5
  }'
```

### List Meetings

```bash
curl http://localhost:8000/meetings
```

### Get Jira Tickets

```bash
curl http://localhost:8000/jira/tickets
```

## Step 7: Process a Meeting (Full Pipeline)

To process a complete meeting, you need:
1. A LiveKit room running
2. An access token for the room

```bash
curl -X POST http://localhost:8000/meetings/start \
  -H "Content-Type: application/json" \
  -d '{
    "room_name": "test-meeting",
    "meeting_title": "Team Standup",
    "access_token": "your_livekit_token",
    "auto_schedule_followup": true
  }'
```

## Docker Deployment (Alternative)

If you prefer Docker:

```bash
# Configure .env first
cp .env.example .env
# Edit .env with your API keys

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f api

# Access API
curl http://localhost:8000/health
```

## Troubleshooting

### Issue: "Database connection failed"

**Solution**: Ensure PostgreSQL is running and credentials in `.env` are correct.

```bash
# Test connection
psql postgresql://teamsync:password@localhost:5432/teamsync
```

### Issue: "HuggingFace authentication failed"

**Solution**: Accept model terms at https://huggingface.co/pyannote/speaker-diarization

### Issue: "Jira API authentication failed"

**Solution**: Verify your API token is valid and has correct permissions.

### Issue: "Google Calendar credentials not found"

**Solution**: Download `credentials.json` from Google Cloud Console.

## Next Steps

1. **Read the full documentation**: [README.md](README.md)
2. **Explore API endpoints**: http://localhost:8000/docs
3. **Customize agents**: Modify files in `src/agents/`
4. **Add custom prompts**: Edit `src/agents/summarizer_agent.py`

## Getting Help

- Check [README.md](README.md) for detailed documentation
- Review example scripts in `scripts/`
- Contact:
  - vva2113@columbia.edu
  - sk5476@columbia.edu
  - ap4613@columbia.edu

---

**You're all set! 🚀**
