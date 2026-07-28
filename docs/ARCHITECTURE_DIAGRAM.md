# HealthBot System Architecture

## High-Level System Architecture

```mermaid
graph TB
    subgraph "User Interfaces"
        UI1[Streamlit UI]
        UI2[FastAPI REST]
        UI3[CLI Interface]
    end

    subgraph "LangGraph Orchestration"
        START([START])
        COLLECT[Collect Topic]
        SAFETY[Check Safety]
        RETRIEVE[Retrieve Knowledge]
        GENERATE[Generate Summary]
        PRESENT[Present Summary]
        QUIZ[Generate Quiz]
        EVAL[Evaluate Answer]
        CONTINUE{Continue?}
        EMERGENCY[Emergency Exit]
        END([END])
        
        START --> COLLECT
        COLLECT --> SAFETY
        SAFETY -->|Emergency| EMERGENCY
        SAFETY -->|Normal| RETRIEVE
        RETRIEVE --> GENERATE
        GENERATE --> PRESENT
        PRESENT --> QUIZ
        QUIZ --> EVAL
        EVAL --> CONTINUE
        CONTINUE -->|Yes| COLLECT
        CONTINUE -->|No| END
        EMERGENCY --> END
    end

    subgraph "Hybrid RAG Pipeline"
        TOOLS[Tool Selector]
        SEM[Semantic Search]
        BM25[BM25 Keyword]
        RRF[Reciprocal Rank Fusion]
        
        TOOLS --> SEM
        TOOLS --> BM25
        SEM --> RRF
        BM25 --> RRF
    end

    subgraph "Data Layer"
        PINECONE[(Pinecone Vector DB<br/>2,578 embeddings)]
        BM25IDX[BM25 Index<br/>In-Memory]
        PARQUET[(PubMed Articles<br/>716 articles)]
    end

    subgraph "External Services"
        GEMINI[Google Gemini LLM]
        TAVILY[Tavily Web Search]
    end

    UI1 --> START
    UI2 --> START
    UI3 --> START
    
    RETRIEVE --> TOOLS
    RRF --> GENERATE
    
    SEM --> PINECONE
    BM25 --> BM25IDX
    BM25IDX --> PARQUET
    
    GENERATE --> GEMINI
    TOOLS --> TAVILY

    style START fill:#90EE90
    style END fill:#FFB6C1
    style SAFETY fill:#FFD700
    style EMERGENCY fill:#FF6B6B
    style RETRIEVE fill:#87CEEB
    style GENERATE fill:#DDA0DD
    style RRF fill:#FFE4B5
```

---

## Detailed Component Architecture

```mermaid
graph LR
    subgraph "Frontend Layer"
        A1[Streamlit app.py]
        A2[FastAPI api.py]
        A3[CLI graph.py]
    end

    subgraph "Orchestration Layer"
        B1[graph.py<br/>StateGraph]
        B2[nodes.py<br/>13 Nodes]
        B3[state.py<br/>PatientState]
    end

    subgraph "Business Logic"
        C1[safety.py<br/>Emergency Detection]
        C2[tools.py<br/>Tool Selector]
        C3[models.py<br/>LLM Wrapper]
        C4[schemas.py<br/>Pydantic Models]
    end

    subgraph "Retrieval System"
        D1[retriever.py<br/>Hybrid Retriever]
        D2[pinecone_store.py<br/>Vector Store]
        D3[embeddings.py<br/>Sentence Transformers]
    end

    subgraph "Data Pipeline"
        E1[loader.py<br/>PubMed Fetcher]
        E2[processor.py<br/>Document Processor]
        E3[chunker.py<br/>Text Chunker]
    end

    subgraph "Evaluation"
        F1[simple_eval.py<br/>Performance]
        F2[test_suite.py<br/>50 Test Cases]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2
    B2 --> B3
    B2 --> C1
    B2 --> C2
    B2 --> C3
    C2 --> D1
    C3 --> C4
    D1 --> D2
    D2 --> D3
    E1 --> E2
    E2 --> E3
    E3 --> D2
    F1 --> D1
    F1 --> F2

    style B1 fill:#87CEEB
    style D1 fill:#FFE4B5
    style E2 fill:#90EE90
    style F1 fill:#DDA0DD
```

---

## Data Flow: Query to Response

```mermaid
sequenceDiagram
    participant User
    participant LangGraph
    participant Safety
    participant Retriever
    participant Pinecone
    participant BM25
    participant RRF
    participant Gemini
    participant User as Response

    User->>LangGraph: Ask medical question
    LangGraph->>Safety: Check for emergency keywords
    
    alt Emergency Detected
        Safety->>User: Immediate emergency message
    else Normal Query
        Safety->>Retriever: Proceed with retrieval
        
        par Parallel Search
            Retriever->>Pinecone: Semantic search (k=10)
            Pinecone-->>Retriever: Top 10 semantic results
        and
            Retriever->>BM25: Keyword search (k=10)
            BM25-->>Retriever: Top 10 BM25 results
        end
        
        Retriever->>RRF: Fuse results (Reciprocal Rank Fusion)
        RRF-->>Retriever: Top 5 combined results
        
        Retriever->>Gemini: Generate answer from context
        Gemini-->>LangGraph: Structured summary
        
        LangGraph->>Gemini: Generate quiz question
        Gemini-->>LangGraph: Quiz with options
        
        LangGraph->>User: Present summary + quiz
        User->>LangGraph: Submit answer
        
        LangGraph->>Gemini: Evaluate answer
        Gemini-->>User: Grade + explanation
    end
```

---

## Deployment Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        CLIENT[Web Browser/<br/>Terminal]
    end

    subgraph "Application Layer"
        LB[Load Balancer/<br/>Nginx]
        APP1[HealthBot Instance 1<br/>Docker Container]
        APP2[HealthBot Instance 2<br/>Docker Container]
        APP3[HealthBot Instance 3<br/>Docker Container]
    end

    subgraph "Cloud Services"
        PINE[Pinecone<br/>Vector Database<br/>2,578 vectors]
        GEM[Google Gemini<br/>LLM API<br/>Flash Model]
        TAV[Tavily<br/>Web Search API]
    end

    subgraph "Monitoring"
        LOG[Logging<br/>healthbot.logger]
        METRICS[Metrics<br/>Latency, Tokens]
    end

    CLIENT --> LB
    LB --> APP1
    LB --> APP2
    LB --> APP3
    
    APP1 --> PINE
    APP2 --> PINE
    APP3 --> PINE
    
    APP1 --> GEM
    APP2 --> GEM
    APP3 --> GEM
    
    APP1 --> TAV
    APP2 --> TAV
    APP3 --> TAV
    
    APP1 --> LOG
    APP2 --> LOG
    APP3 --> LOG
    
    LOG --> METRICS

    style APP1 fill:#87CEEB
    style APP2 fill:#87CEEB
    style APP3 fill:#87CEEB
    style PINE fill:#FFE4B5
    style GEM fill:#DDA0DD
    style METRICS fill:#90EE90
```

---

## State Management

```mermaid
stateDiagram-v2
    [*] --> CollectTopic: User starts
    
    CollectTopic --> CheckSafety: Topic received
    
    CheckSafety --> EmergencyExit: Emergency detected
    CheckSafety --> Retrieve: Normal query
    
    Retrieve --> GenerateSummary: Documents retrieved
    GenerateSummary --> PresentSummary: Summary generated
    PresentSummary --> GenerateQuiz: Summary presented
    GenerateQuiz --> PresentQuiz: Quiz generated
    PresentQuiz --> CollectAnswer: Quiz presented
    CollectAnswer --> Evaluate: Answer received
    Evaluate --> PresentGrade: Evaluation complete
    PresentGrade --> AskContinue: Grade presented
    
    AskContinue --> CollectTopic: User continues
    AskContinue --> [*]: User ends
    
    EmergencyExit --> [*]: Emergency message shown
    
    note right of CheckSafety
        23 emergency keywords
        "chest pain", "stroke",
        "suicidal thoughts", etc.
    end note
    
    note right of Retrieve
        Hybrid RAG:
        - Semantic (Pinecone)
        - BM25 (In-memory)
        - RRF Fusion
    end note
    
    note right of GenerateSummary
        Structured output:
        - Pydantic models
        - Type-safe responses
    end note
```

---

## Hybrid RAG Architecture (Detailed)

```mermaid
graph TD
    QUERY[User Query:<br/>"What are diabetes symptoms?"]
    
    subgraph "Embedding Pipeline"
        EMB1[sentence-transformers/<br/>all-MiniLM-L6-v2]
        VEC[384-dim Vector]
    end
    
    subgraph "Semantic Search Path"
        PINE[(Pinecone Index<br/>2,578 vectors)]
        SEM_RES[Top 10 Semantic<br/>Cosine Similarity]
    end
    
    subgraph "Keyword Search Path"
        TOK[Tokenize Query<br/>["diabetes", "symptoms"]]
        BM25_IDX[BM25Okapi Index<br/>2,578 documents]
        BM25_RES[Top 10 BM25<br/>Term Frequency]
    end
    
    subgraph "Fusion Layer"
        RRF_CALC[RRF Score = Σ 1/(60 + rank)]
        MERGE[Deduplicate & Rerank]
        TOP5[Top 5 Combined Results]
    end
    
    subgraph "Context Preparation"
        FORMAT[Format Sources<br/>[Source 1] Title<br/>PMID | Condition<br/>Text...]
        CTX[RAG Context String]
    end
    
    QUERY --> EMB1
    QUERY --> TOK
    
    EMB1 --> VEC
    VEC --> PINE
    PINE --> SEM_RES
    
    TOK --> BM25_IDX
    BM25_IDX --> BM25_RES
    
    SEM_RES --> RRF_CALC
    BM25_RES --> RRF_CALC
    
    RRF_CALC --> MERGE
    MERGE --> TOP5
    TOP5 --> FORMAT
    FORMAT --> CTX
    
    CTX --> LLM[Google Gemini<br/>Generate Answer]

    style QUERY fill:#90EE90
    style PINE fill:#FFE4B5
    style RRF_CALC fill:#FFD700
    style LLM fill:#DDA0DD
```

---

## File Organization

```
healthbot/
├── 📁 Core Orchestration
│   ├── graph.py          # LangGraph workflow (13 nodes)
│   ├── nodes.py          # Node implementations
│   ├── state.py          # PatientState TypedDict
│   └── schemas.py        # Pydantic models
│
├── 📁 Business Logic
│   ├── safety.py         # Emergency detection
│   ├── tools.py          # Tool selector (RAG/Tavily)
│   ├── models.py         # LLM wrapper with retry
│   ├── prompts.py        # System prompts
│   └── logger.py         # Logging & decorators
│
├── 📁 Retrieval System
│   ├── retrieval/
│   │   ├── retriever.py      # Hybrid retriever (main)
│   │   ├── pinecone_store.py # Pinecone client
│   │   ├── embeddings.py     # Embedding manager
│   │   └── README.md         # Retrieval docs
│
├── 📁 Data Pipeline
│   ├── data/
│   │   ├── loader.py         # PubMed fetcher
│   │   ├── processor.py      # Document processor
│   │   └── chunker.py        # Text chunker
│
├── 📁 Evaluation
│   ├── evaluation/
│   │   ├── simple_eval.py    # Performance evaluator
│   │   ├── test_suite.py     # 50 test cases
│   │   └── ragas_eval.py     # RAGAS integration
│
└── 📁 Configuration
    └── config.py         # Pydantic settings
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Orchestration** | LangGraph 0.2.19 | Stateful workflow with conditional routing |
| **LLM** | Google Gemini Flash | Text generation with structured outputs |
| **Vector DB** | Pinecone | Cloud-native semantic search (2,578 vectors) |
| **Embeddings** | sentence-transformers | 384-dim embeddings (all-MiniLM-L6-v2) |
| **Keyword Search** | rank-bm25 (BM25Okapi) | Lexical matching for medical terms |
| **Fusion** | RRF (Reciprocal Rank Fusion) | Combines semantic + BM25 results |
| **Web Search** | Tavily API | Fallback for rare conditions |
| **Data Source** | PubMed (Biopython) | 716 medical articles via Entrez API |
| **API Framework** | FastAPI 0.111.0 | RESTful backend (5 endpoints) |
| **UI Framework** | Streamlit 1.35.0 | Interactive chat interface |
| **Config** | Pydantic Settings | Type-safe environment config |
| **Logging** | Python logging | Structured logs with decorators |

---

## Performance Characteristics

### Measured Latencies (10-case evaluation)
- **Average**: 418ms
- **Min**: 280ms
- **Max**: 1,452ms (cold start with model loading)
- **Typical**: 300-350ms (after warm-up)

### Breakdown (Estimated)
```
Total: ~400ms
├─ Embedding: ~50ms (CPU, sentence-transformers)
├─ Pinecone Query: ~200ms (network + search)
├─ BM25 Search: ~20ms (in-memory)
├─ RRF Fusion: ~10ms (computation)
└─ Context Formatting: ~10ms
```

### Scalability
- **Stateless Containers**: Each instance is independent (no shared state)
- **Horizontal Scaling**: 1-1000 instances (limited only by cloud quotas)
- **Concurrency**: 10+ queries/sec per instance (limited by Pinecone throughput)
- **Cost**: Free tier supports ~1,500 queries/day (Gemini limit)

---

## Security & Safety

### Medical Safety
- **Emergency Detection**: 23 critical keywords → immediate exit
- **Disclaimers**: Shown on every response
- **Scope Limitations**: Educational only, no diagnosis/prescriptions

### Data Privacy
- **No PII Storage**: Queries not persisted
- **Stateless Design**: No conversation history stored server-side
- **Cloud Services**: Pinecone (embeddings only), Gemini (ephemeral)

### API Security
- **Rate Limiting**: Configurable per endpoint
- **CORS**: Restricted origins in production
- **API Keys**: Environment variables, never committed

---

**Generated**: 2026-07-29  
**Version**: 1.0.0  
**For**: Interview presentations, system understanding, onboarding
