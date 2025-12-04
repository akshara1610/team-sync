# TeamSync - Quick Reference Card

## 🚀 Quick Start

### Installation
```bash
cd team-sync
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
python scripts/setup_db.py
```

### Start API
```bash
# Traditional orchestrator
python main.py

# LangGraph orchestrator (recommended)
python src/api/main_langgraph.py
```

## 📁 File Structure

```
team-sync/
├── src/
│   ├── agents/                        # AI Agents
│   │   ├── listener_agent.py          # Transcription
│   │   ├── knowledge_agent.py         # RAG (original)
│   │   ├── knowledge_agent_langchain.py # RAG (LangChain) ⭐
│   │   ├── summarizer_agent.py        # MoM generation
│   │   ├── reflection_agent.py        # Self-validation ⭐
│   │   ├── action_agent.py            # Jira integration
│   │   └── scheduler_agent.py         # Google Calendar
│   ├── api/
│   │   ├── main.py                    # Original API
│   │   └── main_langgraph.py          # LangGraph API ⭐
│   ├── orchestrator.py                # Traditional orchestrator
│   └── orchestrator_langgraph.py      # LangGraph orchestrator ⭐
├── scripts/
│   ├── setup_db.py                    # Initialize database
│   ├── test_knowledge_agent.py        # Test RAG
│   ├── test_summarizer.py             # Test summarization
│   └── test_langchain_rag.py          # Test LangChain ⭐
└── docs/
    ├── README.md                      # Main documentation
    ├── QUICKSTART.md                  # 15-min setup guide
    ├── ARCHITECTURE.md                # Technical details
    └── LANGCHAIN_LANGGRAPH_GUIDE.md   # LangChain guide ⭐
```

## 🔧 Choose Your Stack

### Option 1: Traditional (Original)
- Manual orchestration
- sentence-transformers embeddings
- Manual RAG implementation

**Use:** `src/orchestrator.py` + `src/api/main.py`

### Option 2: LangChain + LangGraph (Recommended) ⭐
- LangChain RetrievalQA
- OpenAI embeddings
- LangGraph state machine

**Use:** `src/orchestrator_langgraph.py` + `src/api/main_langgraph.py`

## 🧪 Testing

```bash
# Test ChromaDB integration
python scripts/test_knowledge_agent.py

# Test summarization + reflection
python scripts/test_summarizer.py

# Test LangChain RAG
python scripts/test_langchain_rag.py

# Full test suite
pytest tests/ -v
```

## 🌐 API Endpoints

### Base URL
```
http://localhost:8000
```

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/meetings/start` | POST | Process meeting |
| `/meetings/{id}` | GET | Get meeting details |
| `/knowledge/query` | POST | RAG query |
| `/knowledge/stats` | GET | Knowledge base stats |
| `/jira/tickets` | GET | List Jira tickets |
| `/calendar/upcoming` | GET | Upcoming events |
| `/workflow/visualize` | GET | LangGraph workflow (new) |

### Example Requests

```bash
# Health check
curl http://localhost:8000/health

# Query knowledge base
curl -X POST http://localhost:8000/knowledge/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was discussed?", "top_k": 5}'

# Start meeting
curl -X POST http://localhost:8000/meetings/start \
  -H "Content-Type: application/json" \
  -d '{
    "room_name": "standup",
    "meeting_title": "Daily Standup",
    "access_token": "token"
  }'
```

## 📊 Agent Pipeline

```
Meeting → Listener → Knowledge → Summarizer → Reflection → Action → Scheduler
                        ↑                         ↓
                        └─────── (loop if needed) ─┘
```

### Traditional Flow
```python
transcript = listener.transcribe()
summary = summarizer.generate(transcript)
feedback = reflection.validate(summary)
if not feedback.approved:
    summary = summarizer.improve(summary)
tickets = action.create_tickets(summary)
```

### LangGraph Flow
```python
workflow = StateGraph(MeetingState)
workflow.add_node("listen", listen_node)
workflow.add_conditional_edges("reflect", should_improve)
final_state = workflow.compile().invoke(initial_state)
```

## 🔑 Key Components

### 1. Listener Agent
- **Input:** LiveKit audio stream
- **Output:** Transcript with speaker labels
- **Tech:** Whisper, pyannote.audio

### 2. Knowledge Agent
- **Input:** Query string
- **Output:** Answer + sources
- **Tech:** ChromaDB, LangChain RetrievalQA

### 3. Summarizer Agent
- **Input:** Transcript
- **Output:** MoM with action items
- **Tech:** GPT-4

### 4. Self-Reflection Agent ⭐
- **Input:** Summary + transcript
- **Output:** Validation feedback
- **Tech:** GPT-4 (critique mode)

### 5. Action Agent
- **Input:** Action items
- **Output:** Jira tickets
- **Tech:** Jira API

### 6. Scheduler Agent
- **Input:** Meeting summary
- **Output:** Calendar events
- **Tech:** Google Calendar API

## 💡 Usage Examples

### Python

```python
# Traditional
from src.orchestrator import MeetingOrchestrator
orchestrator = MeetingOrchestrator()
result = await orchestrator.process_meeting_full_pipeline(...)

# LangGraph
from src.orchestrator_langgraph import LangGraphMeetingOrchestrator
orchestrator = LangGraphMeetingOrchestrator()
result = await orchestrator.process_meeting(...)

# LangChain RAG
from src.agents.knowledge_agent_langchain import KnowledgeAgentLangChain
agent = KnowledgeAgentLangChain()
response = agent.query("What was discussed?")
```

### cURL

```bash
# Query
curl -X POST http://localhost:8000/knowledge/query \
  -d '{"query": "API migration", "top_k": 5}'

# Workflow info
curl http://localhost:8000/workflow/visualize
```

## 🔐 Environment Variables

### Required
```env
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
JIRA_URL=https://your-domain.atlassian.net
JIRA_EMAIL=you@example.com
JIRA_API_TOKEN=...
```

### Optional
```env
ANTHROPIC_API_KEY=...
LIVEKIT_URL=ws://localhost:7880
DATABASE_URL=postgresql://...
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `pip install -r requirements.txt` |
| Database errors | `python scripts/setup_db.py` |
| Jira auth failed | Check API token and email |
| HuggingFace auth | Accept pyannote terms |
| Port in use | Change `API_PORT` in .env |

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Main documentation |
| `QUICKSTART.md` | 15-minute setup |
| `ARCHITECTURE.md` | Technical details |
| `LANGCHAIN_LANGGRAPH_GUIDE.md` | LangChain/Graph guide |
| `PROJECT_SUMMARY.md` | Implementation summary |
| `QUICK_REFERENCE.md` | This file |

## 🎯 Key Features

- ✅ Real-time meeting transcription
- ✅ Speaker diarization
- ✅ RAG-based knowledge queries
- ✅ Automatic MoM generation
- ✅ Self-reflection validation ⭐
- ✅ Jira ticket creation
- ✅ Calendar scheduling
- ✅ LangChain integration ⭐
- ✅ LangGraph orchestration ⭐

## 🔗 Quick Links

- **API Docs:** http://localhost:8000/docs
- **Health:** http://localhost:8000/health
- **Workflow:** http://localhost:8000/workflow/visualize

## 📞 Support

- vva2113@columbia.edu
- sk5476@columbia.edu
- ap4613@columbia.edu

---

**TeamSync - AI-Powered Meeting Intelligence with LangChain & LangGraph 🚀**
