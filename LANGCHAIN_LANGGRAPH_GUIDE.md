# LangChain & LangGraph Integration Guide

This guide explains how TeamSync uses LangChain and LangGraph for advanced RAG and multi-agent orchestration.

## Overview

TeamSync now has **two orchestration modes**:

1. **Traditional Orchestrator** ([src/orchestrator.py](src/orchestrator.py)) - Manual Python coordination
2. **LangGraph Orchestrator** ([src/orchestrator_langgraph.py](src/orchestrator_langgraph.py)) - State machine-based workflow ⭐ **Recommended**

## What's New

### ✅ LangChain Integration

The new [knowledge_agent_langchain.py](src/agents/knowledge_agent_langchain.py) properly uses LangChain's full stack:

#### Before (Manual Implementation):
```python
# Manual prompt formatting
prompt = f"Context: {context}\nQuestion: {query}"
answer = openai_client.complete(prompt)
```

#### After (LangChain):
```python
# LangChain's RetrievalQA chain
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=Chroma().as_retriever(),
    return_source_documents=True
)

result = qa_chain({"query": query})
```

**Benefits:**
- ✅ Automatic prompt management
- ✅ Built-in source citation
- ✅ Support for conversational memory
- ✅ Easy to swap LLMs
- ✅ Community-maintained prompts

### ✅ LangGraph Integration

The new [orchestrator_langgraph.py](src/orchestrator_langgraph.py) uses LangGraph for stateful workflow:

#### Traditional Approach:
```python
# Manual sequential execution
transcript = await listener.transcribe()
summary = summarizer.generate(transcript)
feedback = reflector.validate(summary)
if not feedback.approved:
    summary = summarizer.improve(summary, feedback)
tickets = action_agent.create_tickets(summary)
```

#### LangGraph Approach:
```python
# State machine with automatic transitions
workflow = StateGraph(MeetingState)
workflow.add_node("listen", listen_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("reflect", reflect_node)
workflow.add_conditional_edges("reflect", should_improve)
app = workflow.compile()

# Execute workflow
final_state = app.invoke(initial_state)
```

**Benefits:**
- ✅ Automatic state management
- ✅ Visual workflow representation
- ✅ Easy conditional branching
- ✅ Built-in error handling per node
- ✅ Workflow introspection

## Architecture Comparison

### Traditional vs LangGraph Orchestration

```
Traditional:                  LangGraph:
┌─────────────┐              ┌─────────────┐
│ Orchestrator│              │ StateGraph  │
│   (Manual)  │              │  (Auto)     │
└──────┬──────┘              └──────┬──────┘
       │                            │
       ├─→ Agent 1                  ├─→ Node 1 (listen)
       │   (await)                  │   (state → state)
       ├─→ Agent 2                  ├─→ Node 2 (summarize)
       │   (await)                  │   (state → state)
       ├─→ if/else                  ├─→ Conditional Edge
       │   (manual)                 │   (automatic)
       └─→ Agent 3                  └─→ Node 3 (action)
```

## Files Created

### 1. LangChain Knowledge Agent
**File:** [src/agents/knowledge_agent_langchain.py](src/agents/knowledge_agent_langchain.py)

**Features:**
- Uses `langchain-chroma` for vector store
- Uses `langchain-openai` for embeddings and LLM
- Implements `RetrievalQA` chain
- Supports conversational memory
- Automatic source citation

**Usage:**
```python
from src.agents.knowledge_agent_langchain import KnowledgeAgentLangChain

agent = KnowledgeAgentLangChain()

# Add transcript
agent.add_transcript(transcript)

# Query with RetrievalQA
response = agent.query("What was discussed about API?")
print(response.answer)
print(response.sources)

# Similarity search
results = agent.similarity_search("Redis caching", k=5)

# Conversational chain with memory
conv_chain = agent.create_conversational_chain()
```

### 2. LangGraph Orchestrator
**File:** [src/orchestrator_langgraph.py](src/orchestrator_langgraph.py)

**Features:**
- State machine workflow
- 7 nodes (listen, summarize, reflect, improve, action, schedule, store)
- Conditional branching (reflection loop)
- Per-node error handling
- Workflow visualization

**Workflow Structure:**
```
START
  ↓
listen (transcribe)
  ↓
summarize (generate MoM)
  ↓
reflect (validate)
  ↓
[CONDITIONAL]
  ├─ if validation_passed → execute_actions
  └─ if needs_improvement → improve → (back to reflect)
      └─ max 3 iterations
  ↓
execute_actions (Jira tickets)
  ↓
schedule_followup (Calendar)
  ↓
store_knowledge (ChromaDB)
  ↓
END
```

**Usage:**
```python
from src.orchestrator_langgraph import LangGraphMeetingOrchestrator

orchestrator = LangGraphMeetingOrchestrator()

# Process meeting (runs entire workflow)
result = await orchestrator.process_meeting(
    room_name="standup",
    meeting_title="Daily Standup",
    access_token="livekit_token"
)

# Query knowledge
response = await orchestrator.query_knowledge("What was discussed?")
```

### 3. LangGraph API
**File:** [src/api/main_langgraph.py](src/api/main_langgraph.py)

Enhanced API with LangGraph support:
- All original endpoints
- New `/workflow/visualize` endpoint
- Uses LangGraph orchestrator

**Start Server:**
```bash
python -m uvicorn src.api.main_langgraph:app --reload
```

### 4. Test Script
**File:** [scripts/test_langchain_rag.py](scripts/test_langchain_rag.py)

Comprehensive test suite for LangChain features.

**Run Tests:**
```bash
python scripts/test_langchain_rag.py
```

## Updated Requirements

New dependencies added to [requirements.txt](requirements.txt):
```
langchain-openai==0.0.5      # OpenAI integration
langchain-chroma==0.1.0      # ChromaDB integration
```

## Usage Guide

### Option 1: Use LangGraph Orchestrator (Recommended)

```python
# In your code
from src.orchestrator_langgraph import LangGraphMeetingOrchestrator

orchestrator = LangGraphMeetingOrchestrator()

# Process meeting with state machine
result = await orchestrator.process_meeting(
    room_name="team-meeting",
    meeting_title="Sprint Planning",
    access_token="token"
)

# Check workflow execution
print(f"Reflection iterations: {result['reflection_iterations']}")
print(f"Workflow messages: {result['workflow_messages']}")
print(f"Final status: {result['status']}")
```

### Option 2: Use LangChain Knowledge Agent Directly

```python
from src.agents.knowledge_agent_langchain import KnowledgeAgentLangChain

agent = KnowledgeAgentLangChain()

# Add documents
agent.add_transcript(transcript)

# Query with full RAG pipeline
response = agent.query(
    "What decisions were made about the database?",
    top_k=5
)

print(f"Answer: {response.answer}")
for source in response.sources:
    print(f"  - {source['meeting_title']}: {source['text']}")
```

### Option 3: Traditional Orchestrator

The original [src/orchestrator.py](src/orchestrator.py) still works:

```python
from src.orchestrator import MeetingOrchestrator

orchestrator = MeetingOrchestrator()
result = await orchestrator.process_meeting_full_pipeline(...)
```

## API Endpoints

### Using LangGraph API

Start the server:
```bash
# LangGraph version
python src/api/main_langgraph.py

# Or use uvicorn
uvicorn src.api.main_langgraph:app --reload
```

### New Endpoints

#### 1. Workflow Visualization
```bash
GET /workflow/visualize
```

Returns workflow structure and features.

#### 2. Enhanced Knowledge Query
```bash
POST /knowledge/query
{
  "query": "What was discussed about API migration?",
  "top_k": 5
}
```

Uses LangChain's RetrievalQA chain.

#### 3. Similarity Search
```bash
POST /knowledge/search?query=Redis&k=5&meeting_id=optional
```

Direct vector similarity search.

#### 4. Delete Meeting
```bash
DELETE /knowledge/meeting/{meeting_id}
```

Remove meeting from knowledge base.

## LangChain Features Used

### 1. Embeddings
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
```

### 2. Vector Store
```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="meetings",
    embedding_function=embeddings,
    persist_directory="./data/chroma_db"
)
```

### 3. LLM
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model_name="gpt-4-turbo-preview",
    temperature=0.3
)
```

### 4. Prompt Template
```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Context: {context}\nQuestion: {question}\nAnswer:",
    input_variables=["context", "question"]
)
```

### 5. RetrievalQA Chain
```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)
```

### 6. Conversational Chain
```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory
)
```

## LangGraph Features Used

### 1. State Definition
```python
from typing import TypedDict

class MeetingState(TypedDict):
    meeting_id: str
    transcript: Dict
    summary: Dict
    validation_passed: bool
    # ... more fields
```

### 2. Graph Construction
```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(MeetingState)

# Add nodes
workflow.add_node("listen", listen_function)
workflow.add_node("reflect", reflect_function)

# Add edges
workflow.add_edge("listen", "reflect")

# Conditional edges
workflow.add_conditional_edges(
    "reflect",
    should_continue_func,
    {
        "continue": "next_node",
        "retry": "previous_node"
    }
)
```

### 3. Workflow Execution
```python
# Compile graph
app = workflow.compile()

# Run workflow
final_state = app.invoke(initial_state)
```

## Testing

### Test LangChain RAG
```bash
python scripts/test_langchain_rag.py
```

Expected output:
- ✓ Agent initialization
- ✓ Transcript addition
- ✓ RetrievalQA queries with answers
- ✓ Similarity search results
- ✓ Meeting context retrieval
- ✓ Conversational chain creation

### Test LangGraph Workflow
```bash
# Start API
python src/api/main_langgraph.py

# In another terminal
curl -X POST http://localhost:8000/meetings/start \
  -H "Content-Type: application/json" \
  -d '{
    "room_name": "test",
    "meeting_title": "Test Meeting",
    "access_token": "token"
  }'
```

## Benefits Summary

### LangChain Benefits
1. **Standardization**: Use community-maintained components
2. **Flexibility**: Easy to swap models and prompts
3. **Reliability**: Battle-tested implementations
4. **Features**: Memory, agents, tools out-of-the-box
5. **Documentation**: Extensive examples and guides

### LangGraph Benefits
1. **Clarity**: Visual workflow representation
2. **Maintainability**: Easy to modify workflow
3. **Debugging**: Per-node error handling
4. **State Management**: Automatic state tracking
5. **Scalability**: Easy to add new nodes/branches

## Migration from Traditional

To migrate existing code:

### Replace Knowledge Agent
```python
# OLD
from src.agents.knowledge_agent import KnowledgeAgent
agent = KnowledgeAgent()

# NEW
from src.agents.knowledge_agent_langchain import KnowledgeAgentLangChain
agent = KnowledgeAgentLangChain()
```

### Replace Orchestrator
```python
# OLD
from src.orchestrator import MeetingOrchestrator
orchestrator = MeetingOrchestrator()

# NEW
from src.orchestrator_langgraph import LangGraphMeetingOrchestrator
orchestrator = LangGraphMeetingOrchestrator()
```

### API stays the same
Both implementations expose the same interface, so API code doesn't change!

## Troubleshooting

### Issue: LangChain import errors
```bash
pip install langchain-openai langchain-chroma
```

### Issue: LangGraph not found
```bash
pip install langgraph
```

### Issue: "No module named 'langchain_core'"
```bash
pip install langchain --upgrade
```

## Next Steps

1. **Test the integration**:
   ```bash
   python scripts/test_langchain_rag.py
   ```

2. **Start LangGraph API**:
   ```bash
   python src/api/main_langgraph.py
   ```

3. **Visualize workflow**:
   ```bash
   curl http://localhost:8000/workflow/visualize
   ```

4. **Process a meeting**:
   Use the API to run the full LangGraph workflow

## Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ChromaDB Integration](https://python.langchain.com/docs/integrations/vectorstores/chroma)
- [RetrievalQA Guide](https://python.langchain.com/docs/use_cases/question_answering/)

---

**TeamSync now fully integrates LangChain and LangGraph for production-ready RAG and multi-agent orchestration! 🚀**
