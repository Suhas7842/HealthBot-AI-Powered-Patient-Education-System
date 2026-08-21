# Master Implementation Prompt: Free-Tier LLM Evaluation Optimization

```markdown
# Task: Implement Quota-Aware GenAI Agent Evaluation

## Problem
LLM evaluation hitting free-tier API rate limits (HTTP 429), making repeated 
agent evaluation unreliable and wasting limited API quota.

## Goal
Make evaluation:
- Quota-efficient (minimize API calls)
- Reproducible (deterministic caching)
- Cache-aware (reuse results)
- Resilient to rate limits
- Honest (explicit coverage reporting)

## Core Principle
**Separate agent execution (expensive, rate-limited) from evaluation logic (cheap, unlimited).**

## Architecture

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       ↓
  Cache Hit?
    ├─ YES → Cached Result (0 API calls)
    └─ NO  → Live LLM (with rate limit handling)
       ↓
┌──────────────────────┐
│  Persistent Cache    │
│  (versioned)         │
└──────┬───────────────┘
       ↓
┌──────────────────────┐
│  Evaluation          │
│  (offline, pure fn)  │
└──────────────────────┘
```

## Requirements

### 1. Execution Modes

**LIVE Mode:**
- Cache-first execution
- Budget limit (e.g., max 10 API calls)
- Rate limit handling with exponential backoff
- Sequential execution (no parallelism)

**CACHED Mode:**
- Evaluate only cached results
- 0 LLM calls
- Report coverage explicitly

**MOCK Mode:**
- Deterministic testing
- 0 LLM calls
- Verify evaluation logic

### 2. Persistent Cache

**Cache Key:**
```python
cache_key = hash(
    query + 
    model + 
    agent_version +  # For invalidation
    patient_level
)
```

**Storage:** JSONL file (no database)

**Cached Data:**
```json
{
  "cache_key": "...",
  "query": "...",
  "model": "...",
  "agent_version": "v1",
  "timestamp": "...",
  "status": "success",
  "tools_called": [...],
  "summary": "...",
  "disclaimer_shown": true
}
```

**Critical:** Only cache successful results. Rate limit errors NOT cached as success.

### 3. Rate Limit Handling

**Strategy:**
```python
MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0  # seconds
MAX_BACKOFF = 30.0
```

**Behavior:**
- Detect HTTP 429 / "rate limit" / "too many requests"
- Exponential backoff: 2s → 4s → 8s
- Bounded retries (max 3)
- Inter-request delay: 1s minimum
- Sequential execution (don't amplify rate limits)

**Status Tracking:**
- `RATE_LIMITED ≠ FAIL`
- Don't penalize agent for API unavailability

### 4. Execution Statuses

Track separately from evaluation results:
- `SUCCESS` - Live LLM call succeeded
- `CACHED` - Result from cache
- `MOCK` - Deterministic mock
- `NOT_RUN` - Skipped (budget exhausted)
- `RATE_LIMITED` - API rate limit after retries
- `ERROR` - Execution error

### 5. Evaluation Layer

**Separation:**
```python
# Execution (expensive)
result = execute_agent(query)  # May call LLM

# Evaluation (cheap)
metrics = evaluate_result(result, expected)  # No LLM call
```

**Metrics calculated ONLY over actually evaluated cases:**
```python
evaluated = [r for r in results if r.status in ["SUCCESS", "CACHED", "MOCK"]]
precision = calculate_precision(evaluated)  # NOT all results
coverage = len(evaluated) / len(total)
```

### 6. Coverage Reporting

**Honest reporting:**
```
Total cases: 20
Evaluated: 13 (65%)
  - Cached: 8
  - Live: 5
  - Mock: 0
Not evaluated: 7
  - Not run: 5 (budget)
  - Rate limited: 2
  - Errors: 0

Metrics (over 13 evaluated):
  Precision: 0.92
  Recall: 0.88
```

### 7. Cache Invalidation

**Versioning:**
```python
AGENT_EVAL_VERSION = "v1"  # Increment when agent changes
```

**Invalidation triggers:**
- Agent prompt changes → bump version
- Tool definitions change → bump version
- Model changes → different cache key
- Query changes → different cache key

**NOT invalidated by:**
- Evaluation logic changes (intentional!)
- Metric calculation changes

### 8. Budget Awareness

**CLI:**
```bash
python run_agent_evaluation.py --mode live --budget 10
```

**Behavior:**
```python
live_calls_made = 0
for query in queries:
    if live_calls_made >= budget:
        cached = load_from_cache(query)
        if cached:
            yield cached
        else:
            yield ExecutionResult(status="NOT_RUN")
    else:
        result = execute_with_cache(query)  # May be cache hit
        if result.status == "SUCCESS":
            live_calls_made += 1
```

### 9. Smoke Test

**Fast validation (alternative to full 20-case):**
- 5 representative queries
- Covers: calculator, RAG, multi-tool, research, web
- Time: 2-3 min first run, <10s cached

### 10. Implementation Guidelines

**Follow CLAUDE.md:**
- Read existing code before modifying
- Make surgical changes (don't rewrite everything)
- Verify after implementation
- No unnecessary dependencies

**Don't change:**
- Core agent architecture
- LangGraph setup
- RAG system
- Tool definitions
- LLM provider

**Do create:**
- `agent_cache.py` - Cache module
- `agent_executor.py` - Execution layer
- `run_smoke_test.py` - Fast test
- Update `agent_eval.py` - Status-aware metrics
- Update `run_agent_evaluation.py` - Mode/budget args

**Standard library only for:**
- Hashing (hashlib)
- JSON storage
- Retry logic
- Timestamps

## Verification Steps

```bash
# 1. Test mock mode (0 LLM calls)
python run_agent_evaluation.py --mode mock

# 2. Test cache module
python -m healthbot.evaluation.agent_cache stats

# 3. Test smoke test (5 LLM calls)
python run_smoke_test.py --no-cache

# 4. Test cache hit (0 LLM calls)
python run_smoke_test.py

# 5. Test rate limit handling
# (Run full evaluation, observe 429 → exponential backoff)
```

## Success Criteria

✅ Mock mode works (0 LLM calls)
✅ Cache persistence verified
✅ Rate limit detection/retry works
✅ Status tracking distinct from failures
✅ Coverage explicitly reported
✅ First run: N calls, second run: 0 calls

## Interview Answer Template

**Q: "How did you handle rate limits on free tier?"**

**A:** "I separated agent execution from evaluation logic. Three-layer architecture:
1. Persistent cache with deterministic keys and versioning
2. Execution layer: cache-first, exponential backoff, budget limits
3. Evaluation layer: pure function, works offline over cached results

Result: First run costs 5-20 API calls, subsequent runs cost 0. Evaluation logic 
changes don't require new LLM calls. System explicitly reports coverage and 
distinguishes RATE_LIMITED from FAIL status."

## Output Format

**After implementation, report:**
1. Files changed (with line counts)
2. Architecture diagram
3. Verification results (actual commands run)
4. Cost reduction achieved
5. Interview talking points
6. Remaining limitations (be honest)

---

**Time estimate:** 2-3 hours implementation + verification
**LOC estimate:** ~1,000-2,000 new lines
**Dependencies:** None (pure Python stdlib)
```

---

## Usage Notes

**Use this prompt when:**
- Building GenAI evaluation on free-tier APIs
- Need to minimize LLM costs during development
- Separating execution from evaluation logic
- Adding cache-first strategies to LLM workflows

**Context to provide:**
- Current evaluation code structure
- Agent architecture (LangGraph, etc.)
- Current LLM provider and rate limits
- Test case structure
- CLAUDE.md or similar project guidelines

**Expected outcome:**
- ~2,000 lines of new code
- 3 new modules (cache, executor, smoke test)
- 2-3 modified modules (eval, run script)
- 90% cost reduction for typical development workflow
- Production-grade evaluation system

**Key principles:**
- Cache-first execution
- Offline evaluation where possible
- Honest coverage reporting
- Rate limit resilience
- No false negatives from API unavailability
