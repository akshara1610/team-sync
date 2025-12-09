# TeamSync: AI-Powered Smart Meeting Agent

**TeamSync** is an autonomous AI-powered meeting agent that manages the complete meeting lifecycle - from joining and transcribing live meetings to generating summaries, creating tasks, and scheduling follow-ups. The system uses a novel **self-reflective multi-agent architecture** to ensure factual consistency and completeness in meeting intelligence.

## Authors

- **Vrinda Ahuja** - Columbia University ([vva2113@columbia.edu](mailto:vva2113@columbia.edu))
- **Sachi Kaushik** - Columbia University ([sk5476@columbia.edu](mailto:sk5476@columbia.edu))
- **Akshara Pramod** - Columbia University ([ap4613@columbia.edu](mailto:ap4613@columbia.edu))

---

## Overview

TeamSync addresses the universal bottleneck of post-meeting work through a sophisticated multi-agent AI system that:

- **Joins and transcribes** live meetings in real-time using Google Meet Adapter and Whisper
- **Generates comprehensive** Minutes of Meeting (MoM) with key decisions
- **Extracts action items** with assignees and creates Jira tickets automatically
- **Self-validates** outputs using a Self-Reflection Agent for trustworthiness
- **Schedules follow-ups** in Google Calendar based on meeting outcomes

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
│ (GMeet   +  │ │ (RAG with   │ │ (GPT-4 +    │
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

1. **Listener Agent**: Captures real-time audio via Google Meet Adapter, transcribes with OpenAI Whisper Model, and performs speaker diarization via Pyannote
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
docker-compose up postgres redis
```

5. **Start the application**:
```bash
python main.py
```

6. **Test the application**:
```bash
python scripts/demo_meeting_bot.py
```


---

## Configuration

### Required API Keys

1. **OpenAI API Key** - For Whisper transcription and GPT-4 summarization
   - Get from: https://platform.openai.com/api-keys

2. **HuggingFace Token** - For pyannote.audio speaker diarization
   - Get from: https://huggingface.co/settings/tokens
   - Accept terms: https://huggingface.co/pyannote/speaker-diarization

3. **Jira API Token**
   - Generate from: https://id.atlassian.com/manage-profile/security/api-tokens

4. **Google Calendar Credentials**
   - Follow: https://developers.google.com/calendar/api/quickstart/python
   - Download `credentials.json` to project root


---

## Technical Pipeline

### Stage 1: Real-Time Capture
- Google Meet Adapter and blackhole-2ch capture the audio meeting
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


## Limitations & Future Work

### Current Limitations
- Requires stable internet for API calls
- Limited to English language transcription
- Speaker diarization accuracy depends on audio quality

### Future Enhancements
- [ ] Multi-language support
- [ ] Real-time in-meeting Q&A interface
- [ ] Integration with Slack/Teams
- [ ] Custom fine-tuned models for domain-specific terminology
- [ ] Enhanced speaker identification with voice profiles

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
