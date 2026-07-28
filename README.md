# 🏥 HealthBot - AI-Powered Patient Education System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.19-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-teal.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, extensible **Retrieval-Augmented Generation (RAG)** system for medical education, powered by LangGraph, Pinecone, and Google Gemini. Provides accurate, evidence-based health information through interactive chat, quiz generation, and comprehensive evaluation framework.

---

## ✨ Key Features

### 🎯 Core Capabilities
- **🔍 Hybrid RAG Retrieval** - Combines semantic search (Pinecone) + keyword matching (BM25) with reciprocal rank fusion
- **📚 Real Medical Data** - 716 PubMed articles embedded as 2,578 chunks across 10 common conditions
- **🤖 LangGraph Orchestration** - 13-node stateful workflow with conditional routing
- **✅ Structured Outputs** - Type-safe Pydantic models (no string parsing)
- **🛡️ Medical Safety** - Emergency detection with 23 critical keywords
- **📊 Comprehensive Evaluation** - RAGAS framework integrated + 50-case test suite

### 🚀 Deployment Options
- **💬 Streamlit UI** - Interactive chat interface with metrics dashboard
- **🔌 FastAPI Backend** - RESTful API with auto-generated docs (5 endpoints)
- **🖥️ CLI Interface** - Terminal-based interaction
- **☁️ Cloud-Ready** - Docker production configuration with Pinecone vector DB and Gemini LLM

### 📈 Measured Performance
- **Retrieval Success**: 100% (verified on 10-case medical test suite)
- **Average Latency**: 418ms per query (hybrid semantic + BM25 retrieval)
- **Architecture**: Stateless containers (120 MB) enabling horizontal scaling
- **Observability**: Tracks node latencies, token usage, retrieval scores, confidence metrics
- **Data**: 2,578 medical document embeddings in cloud vector database (Pinecone)
- **Cost**: Free tier usage (Google Gemini 1,500 req/day, Pinecone 100K vectors)

---

## 📊 System Architecture

```
User → [Streamlit/FastAPI] → LangGraph Workflow
                                     ↓
                              [Safety Check]
                                     ↓
                              [Tool Selection]
                                     ↓
                        ┌────────────┴────────────┐
                        ↓                         ↓
                  [RAG Pipeline]            [Tavily Search]
               (Semantic + BM25)              (Fallback)
                        ↓
                  [RRF Fusion]
                        ↓
                  [LLM Generation]
                  (Structured Output)
                        ↓
               [Response + Sources]
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed diagrams.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Google Gemini API key (free tier: 1,500 requests/day)
- Pinecone API key (free tier: 100K vectors)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/healthbot.git
cd healthbot

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example config.env
# Edit config.env and add:
#   GOOGLE_API_KEY=your_gemini_key
#   PINECONE_API_KEY=your_pinecone_key
```

### Knowledge Base Setup

**Option 1: Use Pre-Loaded Pinecone (Recommended)**
- The system connects to existing Pinecone index with 2,578 embeddings
- No local setup required - just add API keys to config.env

**Option 2: Build Locally (Optional)**
```bash
# Fetch PubMed articles (~30-45 minutes)
python -m healthbot.data.loader

# Upload to Pinecone (~5-10 minutes)
python -m healthbot.retrieval.pinecone_store
```

### Run HealthBot

**Option 1: Streamlit UI** (Recommended)
```bash
streamlit run app.py
```
Open http://localhost:8501

**Option 2: FastAPI Backend**
```bash
uvicorn api:app --reload
```
API docs: http://localhost:8000/docs

**Option 3: CLI**
```bash
python -m healthbot.graph
```

---

## 💻 Usage Examples

### Streamlit UI
1. Ask a health question in the chat
2. View structured summary with sources
3. Generate quiz to test comprehension
4. Monitor system metrics in sidebar

### API (curl)
```bash
# Chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are symptoms of Type 2 diabetes?"}'

# Quiz generation
curl -X POST http://localhost:8000/quiz \
  -H "Content-Type: application/json" \
  -d '{"summary": "Diabetes is a metabolic disorder..."}'

# System metrics
curl http://localhost:8000/metrics
```

### Python SDK
```python
from healthbot.graph import healthbot_app

result = healthbot_app.invoke({"topic": "What causes hypertension?", "messages": []})

print(result["summary"])
```

---

## 🧪 Evaluation

### Test Suite
The project includes a comprehensive evaluation framework:
- **50 medical test cases** with ground truth answers
- **10 conditions covered**: diabetes, hypertension, asthma, heart disease, arthritis, depression, migraine, COPD, obesity, thyroid
- **Verified Results**: 100% retrieval success, 418ms avg latency (see [EVALUATION_REPORT.md](EVALUATION_REPORT.md))

### Run Evaluation
```bash
# Run retrieval performance evaluation
python -m healthbot.evaluation.simple_eval

# For advanced LLM-based evaluation (requires additional setup)
python -m healthbot.evaluation.ragas_eval
```

### Latest Results (10-Case Sample)
- **Success Rate**: 100% (10/10 queries retrieved relevant documents)
- **Avg Latency**: 0.418s (418ms)
- **Min/Max**: 0.280s - 1.452s (first query includes cold start)
- **Method Distribution**: 54% semantic, 34% BM25, 12% hybrid
- **Conditions Tested**: Heart disease, COVID-19, stroke, asthma, obesity, hypertension, depression

**Metrics Tracked:**
- Retrieval latency and success rate
- RRF quality scores
- Method distribution (semantic vs BM25 vs hybrid)
- Per-condition performance breakdown
- Token usage and cost estimates

---

## 📁 Project Structure

```
healthbot/
├── healthbot/
│   ├── config.py              # Pydantic settings
│   ├── state.py               # PatientState TypedDict
│   ├── schemas.py             # Pydantic models
│   ├── logger.py              # Logging + decorators
│   ├── safety.py              # Emergency detection
│   ├── prompts.py             # LLM prompts
│   ├── models.py              # LLM wrapper (retry logic)
│   ├── tools.py               # RAG + Tavily integration
│   ├── nodes.py               # 13 LangGraph nodes
│   ├── graph.py               # Workflow orchestration
│   ├── data/                  # PubMed loader, chunking
│   ├── retrieval/             # Embeddings, vector store, retriever
│   └── evaluation/            # RAGAS, metrics, test suite
├── app.py                     # Streamlit UI
├── api.py                     # FastAPI backend
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md        # System design
│   ├── IMPLEMENTATION_GUIDE.md
│   └── DAY*.md               # Development logs
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🛠️ Technology Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Orchestration** | LangGraph | Stateful workflow with 13 nodes |
| **LLM** | Google Gemini Flash | Text generation with structured outputs |
| **Vector DB** | Pinecone | Cloud-native semantic search (2,578 vectors) |
| **Embeddings** | HuggingFace Transformers | 384-dim sentence embeddings |
| **Keyword Search** | BM25Okapi (rank-bm25) | Lexical matching for hybrid RAG |
| **Fusion** | Reciprocal Rank Fusion | Combines semantic + BM25 results |
| **Web Search** | Tavily API | Fallback for rare conditions |
| **Data Source** | PubMed (716 articles) | Medical literature via Biopython |
| **API Framework** | FastAPI | RESTful backend (5 endpoints) |
| **UI Framework** | Streamlit | Interactive chat interface |
| **Evaluation** | RAGAS | RAG quality metrics framework |
| **Config** | Pydantic Settings | Type-safe configuration |

---

## 📊 System Specifications

| Component | Details |
|-----------|---------|
| **Medical Knowledge Base** | |
| PubMed Articles | 716 articles |
| Vector Embeddings | 2,578 chunks (384-dim) |
| Conditions Covered | 10 common health conditions |
| **LangGraph Workflow** | |
| Total Nodes | 13 nodes with conditional routing |
| State Fields Tracked | 14 (messages, retrieval, metrics, safety) |
| **Hybrid RAG Pipeline** | |
| Semantic Search | Pinecone vector similarity |
| Keyword Search | BM25Okapi algorithm |
| Fusion Method | Reciprocal Rank Fusion |
| **Evaluation Framework** | |
| Test Cases | 50 medical questions with ground truth |
| RAGAS Metrics | Faithfulness, relevancy, recall, precision |
| Observability | Per-node latency, token usage, confidence scores |
| **Deployment** | |
| Container Size | 120 MB (stateless) |
| Cloud Services | Pinecone + Gemini (free tier) |
| Scaling | Horizontal (1-1000 instances) |

---

## 🔒 Medical Safety

### Emergency Detection
Automatically detects 23 critical keywords:
- Chest pain, difficulty breathing, stroke symptoms
- Severe bleeding, unconscious, suicidal thoughts
- Triggers immediate emergency response

### Disclaimers
All responses include medical disclaimers:
> ⚠️ This information is for educational purposes only. Always consult a qualified healthcare professional for medical advice.

### Scope
- ✅ Educational information
- ✅ General health concepts
- ❌ No diagnosis
- ❌ No prescriptions
- ❌ No emergency treatment

---

## 🧰 Development

### Run Tests
```bash
pytest tests/
```

### Check Types
```bash
mypy healthbot/
```

### Format Code
```bash
black healthbot/ api.py app.py
```

### Generate Graph Visualization
```python
from healthbot.graph import visualize_graph

visualize_graph("docs/workflow_graph.png")
```

---

## 📈 Evaluation Framework

### Test Suite Coverage
- **Total Cases**: 50 medical questions with ground truth answers
- **Conditions**: 10 (diabetes, hypertension, asthma, heart disease, arthritis, depression, migraine, COPD, obesity, thyroid)
- **Cases per Condition**: 5 carefully curated questions
- **Latest Evaluation**: 100% success rate, 418ms avg latency (10-case sample)

### Evaluation Metrics
The system supports multiple evaluation approaches:

**Simple Retrieval Evaluation** (Recommended):
```bash
python -m healthbot.evaluation.simple_eval
```
Measures: retrieval success rate, latency, RRF scores, method distribution

**RAGAS Integration** (Advanced - requires additional dependencies):
```bash
python -m healthbot.evaluation.ragas_eval
```
Measures: faithfulness, answer relevancy, context recall, context precision

See [EVALUATION_REPORT.md](EVALUATION_REPORT.md) for detailed results and analysis.

---

## 🚀 Deployment

### Docker Production Deployment

```bash
# Build optimized production container (120 MB)
docker build -f Dockerfile.production -t healthbot:prod .

# Run single instance
docker run -p 8000:8000 --env-file config.env healthbot:prod

# Run with load balancing (3 replicas)
docker-compose -f docker-compose.production.yml up --scale api=3
```

### Cloud Deployment
- **Railway**: Deploy directly from GitHub (see DEPLOY.md)
- **AWS ECS**: Stateless containers with auto-scaling
- **Google Cloud Run**: Serverless container deployment

See [DEPLOY.md](DEPLOY.md) for complete deployment guide.

---

## 📖 Documentation

- **[Architecture](docs/ARCHITECTURE.md)** - System design and technical diagrams
- **[Deployment Guide](DEPLOY.md)** - Production deployment instructions
- **[Code Audit Summary](CODE_AUDIT_SUMMARY.md)** - Implementation verification report

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Follow CLAUDE.md coding principles
4. Add tests for new features
5. Submit a pull request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PubMed/NCBI** - Medical literature data
- **LangChain/LangGraph** - Workflow orchestration framework
- **Pinecone** - Cloud vector database
- **Google** - Gemini LLM with structured outputs
- **RAGAS** - RAG evaluation framework
- **Tavily** - Real-time web search API

---

## 📬 Contact

**Author**: Suhas
- Email: rsuhaskumar3@gmail.com
- GitHub: [github.com/Suhas7842](https://github.com/Suhas7842)

---

## ⚠️ Disclaimer

HealthBot is an educational tool only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions regarding medical conditions.

---

**Built with ❤️ using LangGraph, Pinecone, and Google Gemini | v2.0.0**
