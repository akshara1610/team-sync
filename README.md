# TeamSync: AI-Powered Smart Meeting Agent

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TeamSync** is an autonomous AI-powered meeting agent that manages the complete meeting lifecycle - from joining and transcribing live meetings to generating summaries, creating tasks, and scheduling follow-ups. The system uses a novel **self-reflective multi-agent architecture** to ensure factual consistency and completeness in meeting intelligence.

## Authors

- **Vrinda Ahuja** - Columbia University ([vva2113@columbia.edu](mailto:vva2113@columbia.edu))
- **Sachi Kaushik** - Columbia University ([sk5476@columbia.edu](mailto:sk5476@columbia.edu))
- **Akshara Pramod** - Columbia University ([ap4613@columbia.edu](mailto:ap4613@columbia.edu))

---

## Overview

TeamSync addresses the universal bottleneck of post-meeting work through a sophisticated multi-agent AI system that:

- ✅ **Joins and transcribes** live meetings in real-time using LiveKit and Whisper
- ✅ **Answers questions** during meetings using RAG from past discussions
- ✅ **Generates comprehensive** Minutes of Meeting (MoM) with key decisions
- ✅ **Extracts action items** with assignees and creates Jira tickets automatically
- ✅ **Self-validates** outputs using a Self-Reflection Agent for trustworthiness
- ✅ **Schedules follow-ups** in Google Calendar based on meeting outcomes

### Core Innovation: Self-Reflection Agent

Unlike traditional "fire-and-forget" AI tools, TeamSync includes a **Self-Reflection Agent** that critiques and improves outputs before human review, creating a self-correcting evaluation loop that ensures:

- Factual consistency with source transcripts
- Completeness of action items
- Logical coherence of summaries
- Elimination of hallucinations

---

## System Architecture

TeamSync uses a distributed multi-agent topology with six specialized agents:

```
┌─────────────────────────────────────────────────────────────┐
│                    Meeting Orchestrator                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Listener   │ │  Knowledge  │ │ Summarizer  │
│   Agent     │ │   Agent     │ │   Agent     │
│ (LiveKit +  │ │ (RAG with   │ │ (GPT-4 +    │
│  Whisper)   │ │  ChromaDB)  │ │  Prompts)   │
└─────────────┘ └─────────────┘ └─────────────┘
       │               │               │
       └───────────────┼───────────────┘
                       ▼
              ┌─────────────────┐
              │ Self-Reflection │
              │     Agent       │
              │  (Validation)   │
              └────────┬────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Action    │ │  Scheduler  │ │   Database  │
│   Agent     │ │   Agent     │ │  (Postgres) │
│  (Jira)     │ │  (GCal)     │ └─────────────┘
└─────────────┘ └─────────────┘
```

### Agent Descriptions

1. **Listener Agent**: Captures real-time audio via LiveKit, transcribes with Whisper, and performs speaker diarization
2. **Knowledge Agent**: Maintains ChromaDB vector database with sentence-transformers embeddings for RAG queries
3. **Summarizer Agent**: Generates MoM, extracts decisions and action items using LLMs
4. **Self-Reflection Agent**: Validates outputs through factual consistency checks and iterative improvement
5. **Action Agent**: Creates Jira tickets with proper fields and assignees
6. **Scheduler Agent**: Analyzes dependencies and schedules follow-up meetings in Google Calendar

---

## Installation

### Prerequisites

- Python 3.9 or higher
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional)

### Clone Repository

```bash
git clone https://github.com/your-org/team-sync.git
cd team-sync
```

### Option 1: Local Setup

1. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. **Initialize database**:
```bash
# Start PostgreSQL and Redis
# Then run migrations
python -c "from src.database.db import init_db; init_db()"
```

5. **Start the application**:
```bash
python main.py
```

### Option 2: Docker Setup

```bash
# Configure .env file first
cp .env.example .env

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f api
```

---

## Configuration

### Required API Keys

1. **OpenAI API Key** - For Whisper transcription and GPT-4 summarization
   - Get from: https://platform.openai.com/api-keys

2. **Anthropic API Key** - For Claude model (optional)
   - Get from: https://console.anthropic.com/

3. **HuggingFace Token** - For pyannote.audio speaker diarization
   - Get from: https://huggingface.co/settings/tokens
   - Accept terms: https://huggingface.co/pyannote/speaker-diarization

4. **Jira API Token**
   - Generate from: https://id.atlassian.com/manage-profile/security/api-tokens

5. **Google Calendar Credentials**
   - Follow: https://developers.google.com/calendar/api/quickstart/python
   - Download `credentials.json` to project root

6. **LiveKit Credentials**
   - Deploy LiveKit: https://docs.livekit.io/deploy/
   - Or use LiveKit Cloud: https://cloud.livekit.io/

### Environment Variables

See [.env.example](.env.example) for all configuration options.

---

## Usage

### Starting the API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### API Endpoints

#### 1. Start Meeting Processing

```bash
POST /meetings/start
```

**Request Body**:
```json
{
  "room_name": "team-standup",
  "meeting_title": "Daily Standup - Jan 15",
  "access_token": "livekit_access_token",
  "auto_schedule_followup": true
}
```

**Response**:
```json
{
  "meeting_id": "uuid",
  "status": "success",
  "transcript": {
    "path": "data/transcripts/uuid.json",
    "segments_count": 145,
    "participants": ["Alice", "Bob", "Charlie"]
  },
  "summary": {
    "executive_summary": "...",
    "key_decisions_count": 3,
    "action_items_count": 7
  },
  "reflection": {
    "approved": true,
    "coherence_score": 0.89
  },
  "jira_tickets": [...],
  "followup_meeting": {...}
}
```

#### 2. Query Knowledge Base

```bash
POST /knowledge/query
```

**Request Body**:
```json
{
  "query": "What did we decide about the API migration?",
  "top_k": 5
}
```

**Response**:
```json
{
  "query": "What did we decide about the API migration?",
  "answer": "In the meeting on Jan 10, the team decided to...",
  "confidence": 0.87,
  "sources": [
    {
      "meeting_id": "...",
      "meeting_title": "Architecture Review",
      "speaker": "Alice",
      "text": "..."
    }
  ]
}
```

#### 3. Get Meeting Details

```bash
GET /meetings/{meeting_id}
```

#### 4. List All Meetings

```bash
GET /meetings?limit=50
```

#### 5. Schedule Calendar Event

```bash
POST /calendar/schedule
```

#### 6. Get Jira Tickets

```bash
GET /jira/tickets?meeting_id={meeting_id}
```

---

## Project Structure

```
team-sync/
├── src/
│   ├── agents/
│   │   ├── listener_agent.py          # LiveKit + Whisper transcription
│   │   ├── knowledge_agent.py         # RAG with ChromaDB
│   │   ├── summarizer_agent.py        # MoM generation
│   │   ├── reflection_agent.py        # Self-validation
│   │   ├── action_agent.py            # Jira integration
│   │   └── scheduler_agent.py         # Google Calendar
│   ├── api/
│   │   └── main.py                    # FastAPI application
│   ├── database/
│   │   └── db.py                      # SQLAlchemy models
│   ├── models/
│   │   └── schemas.py                 # Pydantic schemas
│   ├── orchestrator.py                # Main pipeline coordinator
│   └── config.py                      # Configuration management
├── data/
│   ├── transcripts/                   # Meeting transcripts
│   ├── summaries/                     # Generated summaries
│   └── chroma_db/                     # ChromaDB persistence
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment template
├── docker-compose.yml                 # Docker setup
├── Dockerfile                         # Container definition
└── README.md                          # This file
```

---

## Technical Pipeline

### Stage 1: Real-Time Capture
- LiveKit establishes WebRTC connection
- WhisperX transcribes audio in real-time
- Pyannote.audio performs speaker diarization
- Transcripts stored as vector embeddings

### Stage 2: Intelligent Retrieval
- Query embeddings via sentence-transformers
- Similarity search in ChromaDB
- LangChain orchestrates RAG for contextual answers

### Stage 3: Self-Reflective Summarization
- Summarizer Agent generates initial MoM
- Self-Reflection Agent validates:
  - Factual consistency
  - Completeness of action items
  - Logical coherence
- Iterative revision loop (max 3 iterations)

### Stage 4: Autonomous Execution
- Action Agent creates Jira tickets via API
- Scheduler Agent proposes follow-up meetings
- Google Calendar integration for scheduling

---

## Development

### Running Tests

```bash
pytest tests/ -v --cov=src
```

### Code Quality

```bash
# Format code
black src/

# Lint
pylint src/

# Type checking
mypy src/
```

---

## Research Questions

This project explores:

> **Can adding a self-reflection mechanism within a multi-agent LLM system significantly improve the factual consistency and completeness of meeting summaries and action items compared to traditional, non-reflective methods?**

### Evaluation Metrics

1. **Transcription Accuracy**: Word Error Rate (WER)
2. **RAG Performance**: Precision, Recall, F1
3. **Summary Quality**: ROUGE scores
4. **Action Item Extraction**: Precision, Recall, F1
5. **Self-Reflection Impact**: A/B comparison with/without reflection

---

## Limitations & Future Work

### Current Limitations
- Requires stable internet for API calls
- Limited to English language transcription
- Speaker diarization accuracy depends on audio quality
- Jira/Calendar require manual OAuth setup

### Future Enhancements
- [ ] Multi-language support
- [ ] Real-time in-meeting Q&A interface
- [ ] Integration with Slack/Teams
- [ ] Custom fine-tuned models for domain-specific terminology
- [ ] Enhanced speaker identification with voice profiles
- [ ] Automated sentiment analysis

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{teamsync2024,
  title={TeamSync: A Self-Reflective Multi-Agent System for Meeting Intelligence},
  author={Ahuja, Vrinda and Kaushik, Sachi and Pramod, Akshara},
  year={2024},
  institution={Columbia University}
}
```

---

## Acknowledgments

- **ReAct Framework**: Yao et al. (2023)
- **RAG**: Lewis et al. (2020)
- **LangChain**: Chase et al. (2022)
- **Model Context Protocol**: Anthropic (2024)
- **WhisperX**: Bain et al. (2023)
- **pyannote.audio**: Bredin et al. (2020)

---

## Contact

For questions or support, please contact:

- Vrinda Ahuja: [vva2113@columbia.edu](mailto:vva2113@columbia.edu)
- Sachi Kaushik: [sk5476@columbia.edu](mailto:sk5476@columbia.edu)
- Akshara Pramod: [ap4613@columbia.edu](mailto:ap4613@columbia.edu)

---

**Built with ❤️ at Columbia University**
