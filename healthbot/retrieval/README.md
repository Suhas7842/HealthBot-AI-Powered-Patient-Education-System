# Hybrid RAG Retrieval System

This module implements a **Hybrid Retrieval-Augmented Generation (RAG)** pipeline that combines multiple search strategies for improved medical information retrieval.

---

## Architecture

```
Query
  ↓
┌─────────────────────────────┐
│   Hybrid Retriever          │
│                             │
│  ┌──────────┐  ┌─────────┐ │
│  │ Semantic │  │  BM25   │ │
│  │  Search  │  │ Keyword │ │
│  └──────────┘  └─────────┘ │
│       ↓             ↓       │
│  ┌─────────────────────┐   │
│  │ Reciprocal Rank     │   │
│  │ Fusion (RRF)        │   │
│  └─────────────────────┘   │
└─────────────────────────────┘
  ↓
Top-K Results
```

---

## Components

### 1. Semantic Search (Pinecone)

**What**: Vector similarity search using dense embeddings

**How**: 
- Query → 384-dim embedding vector (sentence-transformers)
- Cosine similarity against 2,578 medical document embeddings
- Returns documents semantically similar to query meaning

**Why**:
- Captures conceptual similarity ("diabetes causes" matches "Type 2 etiology")
- Robust to paraphrasing and synonyms
- Understands context beyond exact keyword matches

**Implementation**: [`pinecone_store.py`](pinecone_store.py)

---

### 2. Keyword Search (BM25)

**What**: Term frequency-based ranking using BM25Okapi algorithm

**How**:
- Query → tokenized keywords
- BM25 scoring over document corpus
- Returns documents with matching terms, weighted by frequency and rarity

**Why**:
- Precise matching for medical terminology (drug names, conditions)
- Handles rare terms that embeddings might miss
- Fast, deterministic, no neural network required

**Implementation**: [`retriever.py`](retriever.py) → `keyword_search()`

---

### 3. Reciprocal Rank Fusion (RRF)

**What**: Ranking fusion algorithm that combines multiple ranked lists

**Formula**: For each document, sum RRF scores from all methods:
```
score(doc) = Σ (1 / (k + rank_i))
```
Where:
- `k = 60` (constant, standard RRF parameter)
- `rank_i` = position in method i's result list

**How**:
- Semantic results: [doc_A (rank 1), doc_B (rank 2), ...]
- BM25 results: [doc_C (rank 1), doc_A (rank 3), ...]
- RRF combines both lists with balanced weighting

**Why**:
- No need to normalize different score scales (cosine vs BM25)
- Equal weighting prevents one method from dominating
- Documents appearing in both lists get boosted (high confidence)
- Proven effective in information retrieval research

**Implementation**: [`retriever.py`](retriever.py) → `reciprocal_rank_fusion()`

---

## Usage

### Basic Retrieval

```python
from healthbot.retrieval.retriever import HybridRetriever

retriever = HybridRetriever()

# Retrieve top 5 documents
results = retriever.retrieve("What are symptoms of Type 2 diabetes?", k=5)

for doc in results:
    print(f"RRF Score: {doc['rrf_score']:.4f}")
    print(
        f"Methods: {doc['methods']}"
    )  # ['semantic'], ['bm25'], or ['bm25', 'semantic']
    print(f"Text: {doc['text']}")
```

### Format Context for LLM

```python
# Format retrieved documents as context string
context = retriever.format_context(results)

# Use in prompt
prompt = f"""
Context:
{context}

Question: {user_query}
"""
```

---

## Performance Characteristics

### Semantic Search
- **Strength**: Conceptual understanding, synonym matching
- **Weakness**: May miss exact medical terminology
- **Latency**: ~200-500ms (network call to Pinecone)

### BM25 Keyword Search
- **Strength**: Exact term matching, fast
- **Weakness**: No semantic understanding
- **Latency**: ~50-100ms (local in-memory index)

### Hybrid (RRF)
- **Strength**: Best of both worlds - precision + recall
- **Latency**: Sum of both methods + fusion (~300-600ms)
- **Improvement**: 10-20% better retrieval quality vs single method (typical RAG benchmarks)

---

## Data Pipeline

```
PubMed Articles (716)
  ↓
[Document Processor]
  ↓
Chunks (2,578)
  ↓
┌─────────────────┬──────────────────┐
│                 │                  │
[Embeddings]      [BM25 Index]
(384-dim)         (Tokenized)
│                 │
[Pinecone]        [In-Memory]
(Cloud)           (Local)
```

See [`processor.py`](../data/processor.py) for data pipeline details.

---

## Files

| File | Purpose |
|------|---------|
| `embeddings.py` | Sentence-transformer embedding model wrapper |
| `pinecone_store.py` | Pinecone vector database client (semantic search) |
| `retriever.py` | Hybrid retriever combining semantic + BM25 + RRF |
| `vector_store.py` | (Legacy) Local ChromaDB store - not used in production |

---

## Configuration

Set in [`config.py`](../config.py):

```python
PINECONE_API_KEY: str  # Cloud vector database
PINECONE_INDEX_NAME: str = "medical-knowledge"
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
```

---

## Testing

Test retrieval pipeline:

```bash
# Test hybrid retriever
python -m healthbot.retrieval.retriever

# Test Pinecone connection
python -m healthbot.retrieval.pinecone_store
```

---

## Why Hybrid RAG?

### Problem with Semantic-Only Search
- Neural embeddings can miss exact medical terms
- Example: "What is metformin?" might not match if "metformin" embedding is poor

### Problem with Keyword-Only Search
- No understanding of meaning
- Example: "diabetes causes" vs "Type 2 diabetes etiology" won't match

### Solution: Hybrid
Combines strengths of both methods:
- Semantic for conceptual matching
- BM25 for precise terminology
- RRF for balanced fusion

**Result**: More robust retrieval across diverse medical queries.

---

## References

- **BM25**: Robertson & Zaragoza (2009) - "The Probabilistic Relevance Framework: BM25 and Beyond"
- **RRF**: Cormack et al. (2009) - "Reciprocal Rank Fusion outperforms Condorcet"
- **Sentence-BERT**: Reimers & Gurevych (2019) - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

---

**Built with**: Pinecone (vector DB) + rank-bm25 (keyword search) + sentence-transformers (embeddings)
