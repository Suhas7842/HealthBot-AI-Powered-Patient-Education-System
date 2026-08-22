# Getting Started with HealthBot

**A Step-by-Step Guide for Understanding the Complete Project**

This guide walks you through the HealthBot codebase systematically, explaining what each component does and how they work together.

---

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Project Overview](#project-overview)
3. [Understanding the Codebase - Step by Step](#understanding-the-codebase---step-by-step)
4. [Running the Application](#running-the-application)
5. [Testing](#testing)
6. [Key Concepts](#key-concepts)

---

## Quick Start

### Prerequisites

- Python 3.10+
- API Keys: OpenAI/Groq/Gemini (at least one)
- Git

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd HealthBot-AI-Powered-Patient-Education-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Run the Application

```bash
# FastAPI server
python api.py

# Or Streamlit UI
streamlit run app.py
```

---

## Project Overview

**What is HealthBot?**

HealthBot is a **GenAI medical education assistant** that uses:
- **LangGraph ReAct Agent** for intelligent tool orchestration
- **Hybrid Retrieval** (BM25 + Semantic Search) over 716 curated PubMed articles
- **4 Custom Tools**: Medical RAG, PubMed API, Medical Calculator, Web Search
- **Research Mode** for complex multi-source evidence synthesis

**Key Innovation**: It's not a ChatGPT wrapper - it's an orchestration layer over YOUR infrastructure. The LLM decides WHICH tools to call, not generates answers from training data.

---

## Understanding the Codebase - Step by Step

Follow this order to understand the complete workflow:

### Level 1: Configuration & Setup

**Start here to understand project configuration:**

#### 1. `.env.example` and `healthbot/config.py`
- **What**: Configuration management
- **Why first**: Understand how the app configures LLM providers, API keys, settings
- **Key concepts**: Environment variables, provider selection (Gemini/OpenAI/Groq)

```python
# healthbot/config.py
class Settings(BaseSettings):
    LLM_PROVIDER: str = "groq"  # or "gemini", "openai"
    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str
```

**Read**: `healthbot/config.py` (103 lines)

---

### Level 2: Data & Knowledge Base

**Understand where medical knowledge comes from:**

#### 2. Knowledge Base Structure
- **Location**: `data/pubmed_qa/pubmed_qa.jsonl`
- **What**: 716 curated medical articles from PubMed
- **Format**: JSONL with question, context, long_answer

#### 3. Vector Database Setup: `healthbot/embeddings.py`
- **What**: Creates Chroma vector database from PubMed articles
- **Why**: Enables semantic search over medical knowledge
- **Key function**: `create_medical_knowledge_base()`

**Read**: `healthbot/embeddings.py` (137 lines)

---

### Level 3: Retrieval System (Phase 1-3)

**How the system retrieves medical information:**

#### 4. Retrieval Architecture: `healthbot/retrieval/`

**Start here**: `healthbot/retrieval/retriever.py`
- **What**: Hybrid retrieval (BM25 + Semantic Search + RRF)
- **Why**: Combines keyword matching (BM25) with semantic understanding (embeddings)
- **Key class**: `HybridRetriever`

```python
# Retrieval flow
Query -> [BM25 search] + [Semantic search] -> RRF fusion -> Rerank (optional) -> Top-k docs
```

**Read order**:
1. `healthbot/retrieval/retriever.py` (381 lines) - Main retrieval logic
2. `healthbot/retrieval/reranker.py` (81 lines) - Optional cross-encoder reranking
3. `healthbot/tools.py` (231 lines) - Tool wrappers around retrieval

**Key takeaway**: The retrieval system is YOUR curated medical knowledge base, not ChatGPT's general knowledge.

---

### Level 4: Custom Tools (Phase 4)

**The tools that the agent orchestrates:**

#### 5. Tool Definitions: `healthbot/agent_tools.py`

This is **critical** - it defines the 4 tools available to the LLM agent:

```python
@tool
def medical_rag_search(query: str, k: int = 5):
    """Search 716 PubMed articles in local knowledge base"""
    
@tool
def medical_calculator(calculation_type, weight_kg, height_m, ...):
    """BMI, dosage, creatinine clearance calculations"""
    
@tool  
def pubmed_api_search(query: str, max_results: int = 5):
    """Live search of 35M+ PubMed research papers"""
    
@tool
def web_search(query: str, max_results: int = 3):
    """General web search for current health news"""
```

**Why important**: These are YOUR tools, not LLM's built-in capabilities. The LLM is a ROUTER that decides which tool to call.

**Read**: 
1. `healthbot/agent_tools.py` (265 lines) - Tool definitions
2. `healthbot/tools/medical_calculator.py` (118 lines) - Calculator implementation
3. `healthbot/tools/pubmed_api.py` (175 lines) - PubMed API client

**Key takeaway**: Agent doesn't generate answers - it orchestrates YOUR tools.

---

### Level 5: Agent Prompts & Routing

**How the agent decides what to do:**

#### 6. Agent System Prompts: `healthbot/prompts_agent.py`

Two prompts defined here:

```python
AGENT_SYSTEM_PROMPT = """
You are a medical education research assistant with access to multiple tools.

YOUR APPROACH:
1. UNDERSTAND the user's question
2. PLAN which tool(s) to use
3. CALL the appropriate tool(s)
4. SYNTHESIZE the results
5. RESPOND with clear, cited information
"""

RESEARCH_AGENT_PROMPT_TEMPLATE = """
You are conducting multi-step medical research.

1. DECOMPOSE the question into sub-questions
2. RESEARCH each sub-question
3. COMPARE and CONTRAST evidence
4. SYNTHESIZE into coherent response
5. CITE all sources
"""
```

**Why two prompts?**
- **Standard**: For simple queries ("What is diabetes?")
- **Research**: For complex queries ("What does recent research say about diabetes risk factors?")

**Read**: `healthbot/prompts_agent.py` (189 lines)

#### 7. Query Classification: `healthbot/routing.py`

Determines if query is "normal" or "research" style:

```python
research_patterns = [
    r"\brecent (research|studies|evidence)\b",
    r"\bcompare (recent )?studies\b",
    r"\bmodifiable risk factors?\b",
]
```

**Read**: `healthbot/routing.py` (290 lines)

**Key takeaway**: System adapts behavior based on query complexity.

---

### Level 6: State Management

**How conversation context is tracked:**

#### 8. State Definition: `healthbot/state.py`

```python
class PatientState(TypedDict):
    # Core input
    topic: str                      # User's query
    patient_level: str              # Education level
    
    # Agent conversation
    messages: list[BaseMessage]     # LangChain message history
    
    # Output
    summary: str | None             # Agent's response
    
    # Query classification
    query_type: str | None          # "normal" or "research"
    
    # Tool tracking
    tools_called: list[str]         # Which tools agent used
    tool_results: list[dict]        # Results from tools
```

**Read**: `healthbot/state.py` (41 lines)

**Key takeaway**: State tracks the entire conversation flow through the agent workflow.

---

### Level 7: Agent Workflow (THE CORE)

**This is where everything comes together:**

#### 9. Agent Graph: `healthbot/agent_graph.py`

**THE MOST IMPORTANT FILE** - This orchestrates the entire agent workflow.

```
Workflow:
1. safety_check -> Emergency detection
2. agent_node -> LLM with tool calling
   - Detect if research query
   - Choose appropriate prompt
   - Call create_react_agent (LangGraph)
   - Agent decides which tools to call
   - Agent can call multiple tools sequentially
   - Agent synthesizes results
3. END -> Return response
```

**Key function**: `agent_node(state: PatientState)`

```python
def agent_node(state):
    # 1. Detect research vs normal
    is_research = classifier.is_research_query(topic)
    
    # 2. Choose prompt
    if is_research:
        system_prompt = get_research_prompt(topic)
    else:
        system_prompt = get_agent_prompt()
    
    # 3. Create ReAct agent (LangGraph handles tool calling loop)
    agent_executor = create_react_agent(llm, tools)
    
    # 4. Invoke agent
    result = agent_executor.invoke(agent_input)
    
    # 5. Return response with tool tracking
    return {
        "summary": final_response,
        "tools_called": tools_called,
        "query_type": "research" or "normal"
    }
```

**Read**: `healthbot/agent_graph.py` (260 lines)

**Key takeaway**: This is the orchestration layer. LangGraph's `create_react_agent` handles the tool-calling loop automatically.

---

### Level 8: Safety & Medical Disclaimers

**Critical for medical applications:**

#### 10. Safety Checks: `healthbot/safety.py`

```python
def check_emergency(query: str) -> bool:
    """Detect emergency keywords"""
    emergency_keywords = [
        "chest pain", "heart attack", "stroke",
        "severe bleeding", "suicide", "overdose"
    ]
    return any(keyword in query.lower() for keyword in emergency_keywords)
```

**Read**: `healthbot/safety.py` (80 lines)

**Key takeaway**: Emergency detection happens BEFORE agent processing. System never diagnoses or prescribes.

---

### Level 9: API & UI

**How users interact with the system:**

#### 11. FastAPI Backend: `api.py`

```python
@app.post("/query")
async def query_agent(request: QueryRequest):
    # Run through agent workflow
    result = run_agent_query(
        query=request.query,
        patient_level=request.patient_level
    )
    return result
```

**Read**: `api.py` (135 lines)

#### 12. Streamlit UI: `app.py`

User-friendly chat interface with:
- Query input
- Patient level selection
- Tool tracking display
- Citation display

**Read**: `app.py` (267 lines)

---

### Level 10: Evaluation System (Optional - Advanced)

**For validating agent performance:**

#### 13. Evaluation Framework: `healthbot/evaluation/`

Three-file architecture for free-tier LLM evaluation:

```
agent_cache.py     -> Persistent JSONL cache (avoid repeat LLM calls)
agent_executor.py  -> Rate-limited execution with retries
agent_eval.py      -> Offline evaluation metrics
```

**Why separate execution from evaluation?**
- First run: 5 LLM calls (cache traces)
- Second run: 0 LLM calls (evaluate cached traces)
- Change metrics without new LLM calls

**Read**:
1. `healthbot/evaluation/agent_cache.py` (220 lines)
2. `healthbot/evaluation/agent_executor.py` (346 lines)
3. `healthbot/evaluation/agent_eval.py` (527 lines)

**Documentation**: See `docs/EVALUATION_SYSTEM.md` for complete details.

---

## Running the Application

### Option 1: FastAPI (Recommended for API access)

```bash
python api.py
```

Visit: http://localhost:8000/docs for API documentation

### Option 2: Streamlit (Recommended for UI)

```bash
streamlit run app.py
```

Visit: http://localhost:8501

### Option 3: Direct Python

```python
from healthbot.agent_graph import run_agent_query

result = run_agent_query(
    query="What is Type 2 diabetes?",
    patient_level="beginner"
)

print(result["summary"])
print(result["tools_called"])
print(result["query_type"])  # "normal" or "research"
```

---

## Testing

### Run All Tests

```bash
pytest
```

### Run Specific Test Suites

```bash
# Agent tests
pytest tests/test_agent_graph.py -v

# Research mode tests
pytest tests/test_research_mode.py -v

# Retrieval tests  
pytest tests/test_retrieval.py -v

# Evaluation tests (no LLM required)
pytest tests/test_eval_free_tier.py -v

# Calculator tests
pytest tests/test_medical_calculator.py -v
```

### Run Examples

```bash
# Research mode demo
python examples/demo_research_mode.py

# Agent evaluation (requires LLM access)
python run_agent_evaluation.py --mode mock  # 0 LLM calls
python run_smoke_test.py                     # 5 LLM calls (first run)
```

---

## Key Concepts

### 1. Why is this NOT a ChatGPT wrapper?

**Bad approach**: Send medical question to ChatGPT, return answer
- Problem: No source control, hallucinations, outdated info

**HealthBot approach**: 
- LLM is a **router/orchestrator**, not answer generator
- Tools are YOUR infrastructure (curated medical DB, PubMed API, validated formulas)
- LLM decides WHICH tool to call based on query analysis
- Final answer synthesized from YOUR tools' results

### 2. Normal vs Research Mode

**Normal Mode**: Simple questions
```
"What is diabetes?"
-> medical_rag_search -> Answer
```

**Research Mode**: Complex questions
```
"What does recent research say about diabetes risk factors?"
-> medical_rag_search (established knowledge)
-> pubmed_api_search (recent literature)  
-> Synthesize both sources -> Answer
```

### 3. Tool Selection

Agent decides which tool(s) to call:

| Query Type | Tools Called |
|------------|--------------|
| Definition question | `medical_rag_search` |
| BMI calculation | `medical_calculator` |
| Recent research | `pubmed_api_search` |
| Multi-part question | `medical_rag_search` + others |

### 4. ReAct Agent (Reason + Act)

LangGraph's `create_react_agent` implements:
```
Loop:
1. Think: What do I need to know?
2. Act: Call appropriate tool
3. Observe: Read tool result
4. Think: Do I need more information?
5. If yes, repeat. If no, synthesize answer.
```

This is automatic - you don't implement the loop manually.

---

## Next Steps

After understanding the codebase:

1. **Experiment with queries**: Try normal vs research queries
2. **Add custom tools**: Follow pattern in `agent_tools.py`
3. **Modify prompts**: Adjust guidance in `prompts_agent.py`
4. **Extend knowledge base**: Add more medical articles to `data/`
5. **Deploy**: See deployment guides for production setup

---

## Additional Documentation

- **Evaluation System**: `docs/EVALUATION_SYSTEM.md`
- **Research Mode Implementation**: `docs/RESEARCH_MODE_IMPLEMENTATION.md`
- **Free-Tier Evaluation**: `docs/IMPLEMENTATION_NOTES.md`
- **Main README**: `README.md`

---

## Common Questions

### Q: Why LangGraph instead of plain LangChain?

**A**: LangGraph provides:
- State management for multi-step workflows
- Conditional routing (safety check before agent)
- Tool-calling loop automation via `create_react_agent`

### Q: Why separate cache from evaluation?

**A**: Free-tier LLM APIs have rate limits. Separating execution (expensive) from evaluation (cheap) means:
- First run: Cache N traces
- Iterate on metrics: 0 additional LLM calls

### Q: Can I use this with my own medical data?

**A**: Yes! Replace `data/pubmed_qa/pubmed_qa.jsonl` with your articles and regenerate embeddings with `python healthbot/embeddings.py`.

### Q: Is this production-ready?

**A**: It's a research/portfolio project. For production:
- Add authentication
- Implement proper logging
- Add monitoring/observability  
- Deploy with proper security
- Add compliance measures (HIPAA if applicable)

---

## File Reading Order Summary

**Essential path** (understand 80% of the system):

1. `healthbot/config.py` - Configuration
2. `healthbot/retrieval/retriever.py` - How knowledge is retrieved
3. `healthbot/agent_tools.py` - What tools agent has
4. `healthbot/prompts_agent.py` - How agent is guided
5. `healthbot/routing.py` - Query classification
6. `healthbot/state.py` - What state is tracked
7. **`healthbot/agent_graph.py`** - How everything orchestrates (MOST IMPORTANT)
8. `api.py` or `app.py` - How users interact

**Advanced path** (evaluation, research mode, tools):

9. `healthbot/evaluation/` - Agent evaluation system
10. `healthbot/tools/` - Tool implementations
11. `tests/` - Test patterns and validation

**Total essential reading**: ~1500 lines to understand core system

---

## Getting Help

- Check existing tests for usage examples
- Read tool docstrings for parameter details
- Run with `--help` for CLI options
- See error logs in console output

---

**Ready to dive in? Start with** `healthbot/config.py` **and follow the reading order above!**
