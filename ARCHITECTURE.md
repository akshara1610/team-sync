# TeamSync Architecture Documentation

## Overview

TeamSync is built as a multi-agent system where specialized AI agents collaborate through a central orchestrator to manage the complete meeting lifecycle. This document provides technical details about the system architecture.

## System Components

### 1. Agent Layer

#### Listener Agent (`src/agents/listener_agent.py`)
- **Purpose**: Real-time audio capture and transcription
- **Technologies**:
  - LiveKit (WebRTC)
  - Faster-Whisper (OpenAI Whisper)
  - pyannote.audio (Speaker diarization)
- **Input**: LiveKit audio stream
- **Output**: Structured transcript with speaker labels and timestamps
- **Key Methods**:
  - `join_meeting()`: Connect to LiveKit room
  - `_process_audio_stream()`: Process audio in real-time
  - `perform_diarization()`: Identify speakers
  - `get_transcript()`: Export transcript data

#### Knowledge Agent (`src/agents/knowledge_agent.py`)
- **Purpose**: RAG-based knowledge management
- **Technologies**:
  - ChromaDB (Vector database)
  - sentence-transformers (Embeddings)
  - LangChain (RAG orchestration)
- **Input**: Meeting transcripts, user queries
- **Output**: Contextual answers with citations
- **Key Methods**:
  - `add_transcript()`: Store meeting in vector DB
  - `query()`: Semantic search and answer generation
  - `search_meetings()`: Keyword-based search
  - `get_collection_stats()`: Database statistics

#### Summarizer Agent (`src/agents/summarizer_agent.py`)
- **Purpose**: Generate comprehensive meeting summaries
- **Technologies**:
  - OpenAI GPT-4
  - Custom prompt engineering
- **Input**: Meeting transcript
- **Output**: MoM with decisions, action items, discussion points
- **Key Methods**:
  - `generate_summary()`: Main summary generation
  - `_extract_key_decisions()`: Identify decisions
  - `_extract_action_items()`: Find tasks with assignees
  - `_extract_unresolved_questions()`: Track open issues

#### Self-Reflection Agent (`src/agents/reflection_agent.py`)
- **Purpose**: Validate and improve outputs (NOVEL CONTRIBUTION)
- **Technologies**:
  - OpenAI GPT-4 (for critique)
  - Multi-pass validation
- **Input**: Summary + original transcript
- **Output**: Validation feedback + improved summary
- **Key Methods**:
  - `validate_summary()`: Main validation loop
  - `_check_factual_consistency()`: Verify accuracy
  - `_check_action_item_completeness()`: Find missing items
  - `_check_logical_coherence()`: Score quality
  - `_improve_summary()`: Generate revised version

#### Action Agent (`src/agents/action_agent.py`)
- **Purpose**: Execute workflow automation
- **Technologies**:
  - Jira Python API
  - Model Context Protocol (MCP)
- **Input**: Action items from summary
- **Output**: Created Jira tickets
- **Key Methods**:
  - `create_jira_tickets()`: Batch ticket creation
  - `update_ticket_status()`: Modify ticket state
  - `link_tickets()`: Create dependencies
  - `get_project_tickets()`: Query existing tickets

#### Scheduler Agent (`src/agents/scheduler_agent.py`)
- **Purpose**: Calendar integration and meeting scheduling
- **Technologies**:
  - Google Calendar API
  - OAuth 2.0 authentication
- **Input**: Meeting summary, attendee availability
- **Output**: Scheduled calendar events
- **Key Methods**:
  - `schedule_event()`: Create calendar event
  - `schedule_follow_up_meeting()`: Auto-schedule based on MoM
  - `suggest_meeting_time()`: Find optimal slot
  - `get_upcoming_events()`: Query calendar

### 2. Orchestration Layer

#### Meeting Orchestrator (`src/orchestrator.py`)
- **Purpose**: Coordinate all agents in the pipeline
- **Key Workflow**:
  1. Join meeting → Transcribe (Listener)
  2. Generate summary (Summarizer)
  3. Validate & improve (Self-Reflection)
  4. Create tickets (Action)
  5. Schedule follow-up (Scheduler)
  6. Store knowledge (Knowledge)
- **Key Methods**:
  - `process_meeting_full_pipeline()`: Execute complete workflow
  - `query_knowledge_base()`: Delegate to Knowledge Agent
  - `get_meeting_summary()`: Retrieve from database

### 3. API Layer

#### FastAPI Application (`src/api/main.py`)
- **Purpose**: RESTful API for system interaction
- **Endpoints**:
  - `POST /meetings/start`: Trigger meeting processing
  - `GET /meetings/{id}`: Get meeting details
  - `POST /knowledge/query`: Query knowledge base
  - `POST /calendar/schedule`: Create calendar event
  - `GET /jira/tickets`: List tickets
- **Features**:
  - CORS middleware
  - Background task processing
  - Error handling
  - API documentation (Swagger/OpenAPI)

### 4. Data Layer

#### Database Models (`src/database/db.py`)
- **Purpose**: Persistent storage
- **Technology**: SQLAlchemy + PostgreSQL
- **Models**:
  - `Meeting`: Store meeting metadata
  - Tracks: ID, title, status, participants, timestamps

#### Schemas (`src/models/schemas.py`)
- **Purpose**: Data validation and serialization
- **Technology**: Pydantic
- **Key Schemas**:
  - `TranscriptData`: Meeting transcript structure
  - `MeetingSummary`: Complete summary with nested objects
  - `ActionItem`: Task with assignee and metadata
  - `RAGQuery/Response`: Knowledge base interaction
  - `ReflectionFeedback`: Validation results

### 5. Configuration

#### Settings (`src/config.py`)
- **Purpose**: Centralized configuration management
- **Technology**: Pydantic Settings
- **Configuration Sources**:
  - Environment variables
  - `.env` file
  - Default values
- **Categories**:
  - Database URLs
  - API keys
  - Agent parameters
  - Server settings

## Data Flow

### Complete Meeting Pipeline

```
1. Meeting Start
   └─> LiveKit Connection (Listener Agent)
       └─> Audio Stream
           └─> Whisper Transcription
               └─> Speaker Diarization
                   └─> Structured Transcript JSON

2. Transcript Processing
   └─> Transcript → Summarizer Agent
       └─> GPT-4 Prompts
           ├─> Executive Summary
           ├─> Key Decisions
           ├─> Action Items
           ├─> Discussion Points
           └─> Unresolved Questions

3. Self-Reflection Loop
   └─> Initial Summary → Self-Reflection Agent
       ├─> Factual Consistency Check
       ├─> Completeness Verification
       ├─> Coherence Scoring
       └─> [IF ISSUES] → Improve → Validate Again
           └─> [MAX 3 ITERATIONS]

4. Action Execution
   └─> Validated Action Items → Action Agent
       └─> Create Jira Tickets
           ├─> Set assignees
           ├─> Add descriptions
           ├─> Set priorities
           └─> Link to meeting

5. Follow-up Scheduling
   └─> Meeting Outcomes → Scheduler Agent
       ├─> Analyze action items
       ├─> Check attendee availability
       └─> Schedule follow-up meeting

6. Knowledge Storage
   └─> Transcript → Knowledge Agent
       └─> Embed segments (sentence-transformers)
           └─> Store in ChromaDB
               └─> Available for RAG queries
```

## Technology Stack

### Core Framework
- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### AI/ML
- **OpenAI GPT-4**: Summarization and reasoning
- **Whisper/Faster-Whisper**: Speech-to-text
- **sentence-transformers**: Text embeddings
- **LangChain**: RAG orchestration

### Storage
- **PostgreSQL**: Relational database
- **ChromaDB**: Vector database
- **Redis**: Caching and queues

### Integrations
- **LiveKit**: Real-time audio/video
- **Jira API**: Task management
- **Google Calendar API**: Scheduling
- **pyannote.audio**: Speaker diarization

### Deployment
- **Docker**: Containerization
- **docker-compose**: Multi-container orchestration

## Key Design Decisions

### 1. Multi-Agent Architecture
**Rationale**: Separation of concerns allows each agent to specialize, making the system modular, testable, and maintainable.

### 2. Self-Reflection Loop
**Rationale**: Single-pass LLM outputs are unreliable. Iterative validation significantly improves factual consistency.

### 3. ChromaDB over FAISS
**Rationale**:
- Persistent storage out-of-the-box
- Better metadata filtering
- Active development and updates
- Easier deployment

### 4. Orchestrator Pattern
**Rationale**: Central coordination simplifies workflow management and provides a single point of control for the pipeline.

### 5. FastAPI for REST API
**Rationale**:
- Modern async support
- Automatic API documentation
- Type safety with Pydantic
- High performance

## Scalability Considerations

### Current Limitations
- Single-threaded meeting processing
- In-memory audio buffering
- Synchronous database operations

### Future Improvements
1. **Horizontal Scaling**
   - Message queue (RabbitMQ/Kafka) for distributed processing
   - Multiple worker instances
   - Load balancing

2. **Async Processing**
   - Convert database operations to async
   - Stream processing for audio
   - Concurrent agent execution

3. **Caching**
   - Redis caching for RAG results
   - Embedding cache for frequent queries
   - API response caching

4. **Monitoring**
   - Prometheus metrics
   - Distributed tracing (OpenTelemetry)
   - Error tracking (Sentry)

## Security Considerations

1. **API Keys**: Stored in environment variables, never committed
2. **Database**: Connection strings encrypted
3. **OAuth**: Google Calendar uses OAuth 2.0 flow
4. **API Access**: Should add authentication middleware in production
5. **Rate Limiting**: Recommended for production deployment

## Testing Strategy

1. **Unit Tests**: Individual agent functionality (`tests/test_agents.py`)
2. **Integration Tests**: Multi-agent workflows
3. **End-to-End Tests**: Complete pipeline with mock data
4. **Manual Testing**: Scripts in `scripts/` directory

## Deployment Architecture

### Development
```
Local Machine
├── Python virtual environment
├── PostgreSQL (local)
├── Redis (local)
└── LiveKit (Docker or cloud)
```

### Production (Recommended)
```
Cloud Platform (GCP/AWS)
├── Kubernetes Cluster
│   ├── API Pods (FastAPI)
│   ├── Worker Pods (Agent processing)
│   └── LiveKit Pods
├── Cloud SQL (PostgreSQL)
├── Redis (Managed)
├── ChromaDB (Persistent volume)
└── Load Balancer
```

## Performance Metrics

### Expected Performance
- Transcription: Real-time (~1x meeting duration)
- Summarization: 30-60 seconds per 30-minute meeting
- Self-Reflection: 60-120 seconds (3 iterations max)
- Action Creation: <5 seconds
- RAG Query: <2 seconds

### Optimization Opportunities
1. Batch embedding generation
2. Parallel agent execution where possible
3. Cache frequent queries
4. Use lighter models for non-critical tasks

---

For implementation details, see individual agent files in `src/agents/`.
