# HealthBot Agent Evaluation System

## Overview

Free-tier-aware evaluation system for GenAI agent validation. Separates agent execution (expensive, rate-limited) from evaluation logic (cheap, unlimited).

**Key Principle:** Use expensive live LLM calls only to validate real model behavior. Use deterministic tests and cached outputs for everything else.

## Architecture

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       ↓
┌──────────────────────┐
│  Execution Layer     │
│  (agent_executor.py) │
└──────┬───────────────┘
       ↓
   Cache Hit?
    ├─ YES → Cached Result (0 API calls)
    ├─ NO  → Live LLM call (with rate limit handling)
    └─ MOCK → Deterministic mock
       ↓
┌──────────────────────┐
│  Persistent Cache    │
│  (agent_cache.py)    │
└──────┬───────────────┘
       ↓
┌──────────────────────┐
│  Evaluation Layer    │
│  (agent_eval.py)     │
└──────┬───────────────┘
       ↓
    Metrics
```

## Execution Modes

### 1. LIVE Mode (Default)

Cache-first execution with live LLM budget:

```bash
python run_agent_evaluation.py --mode live --budget 10
```

**Behavior:**
- Check cache first
- If cached → return immediately (0 API calls)
- If not cached → call live LLM (up to budget limit)
- Save successful results to cache
- Handle rate limits with exponential backoff

**Use when:** First run, or after agent/prompt changes

### 2. CACHED Mode

Evaluate only cached results (0 LLM calls):

```bash
python run_agent_evaluation.py --mode cached
```

**Behavior:**
- Load results from cache
- Mark uncached queries as NOT_RUN
- Evaluate cached results offline

**Use when:** Testing evaluation logic changes, generating reports

### 3. MOCK Mode

Deterministic testing (0 LLM calls):

```bash
python run_agent_evaluation.py --mode mock
```

**Behavior:**
- Generate mock outputs matching expected tools
- Test evaluation logic without API dependency
- Verify metrics calculations

**Use when:** Unit testing, CI/CD, development

## Cache System

### Cache Key Design

```python
cache_key = hash(
    query +
    model +
    agent_version +
    patient_level
)
```

Cache invalidates automatically when:
- Agent prompt changes (increment `AGENT_EVAL_VERSION` in `agent_cache.py`)
- Model changes
- Query changes

### Cache Location

```
evaluation_cache/
└── agent_results.jsonl  # Persistent JSONL cache
```

**Added to `.gitignore`** - never commit cached results

### Cache Management

```bash
# View cache statistics
python -m healthbot.evaluation.agent_cache stats

# Clear cache
python -m healthbot.evaluation.agent_cache clear
```

### Cache Entry Format

```json
{
  "cache_key": "e36e37db70cb6bbe",
  "query": "What is diabetes?",
  "model": "openai/gpt-oss-120b",
  "agent_version": "v1",
  "patient_level": "beginner",
  "timestamp": "2026-08-22T01:15:30",
  "status": "success",
  "tools_called": ["medical_rag_search"],
  "summary": "...",
  "disclaimer_shown": true,
  "tool_call_trace": ["Called tool: medical_rag_search"]
}
```

**Important:** Only `status: "success"` entries are reused. Rate limit errors and failures are NOT cached as successful results.

## Rate Limit Handling

### Strategy

```
LLM Call
  ↓
429 Too Many Requests?
  ├─ NO → Success
  └─ YES
      ↓
  Exponential Backoff
      ↓
  Retry (max 3 times)
      ↓
  Still rate limited?
      ↓
  Mark as RATE_LIMITED
  Continue evaluation
```

### Configuration

```python
MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0  # seconds
MAX_BACKOFF = 30.0  # seconds
INTER_REQUEST_DELAY = 1.0  # Minimum delay between requests
```

### Behavior

- **Bounded retries:** Max 3 attempts (not infinite loop)
- **Exponential backoff:** 2s → 4s → 8s
- **Sequential execution:** No parallel requests (avoids amplifying rate limits)
- **Status tracking:** RATE_LIMITED ≠ FAIL (don't penalize agent for API limits)

## Evaluation Metrics

Metrics calculated **only over actually evaluated cases**:

### Statuses

- `SUCCESS` - Live LLM call succeeded
- `CACHED` - Result from cache
- `MOCK` - Deterministic mock
- `NOT_RUN` - Skipped (budget exhausted)
- `RATE_LIMITED` - API rate limit after retries
- `ERROR` - Execution error

### Coverage

```
Coverage = evaluated_cases / total_cases

evaluated_cases = SUCCESS + CACHED + MOCK
```

**NOT_RUN and RATE_LIMITED do NOT count as failures.**

### Tool Selection Metrics

```
Precision = correct_tools / actual_tools
Recall = required_tools_called / required_tools
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

## Smoke Test

Fast validation with 5 representative queries:

```bash
python run_smoke_test.py
```

**Test Cases:**
1. Calculator (single tool)
2. RAG (single tool)
3. Multi-tool (calculator + RAG)
4. Research (PubMed)
5. Web search

**Time:** 2-3 minutes first run, <10s cached

**Use when:** Quick validation after changes

## Workflows

### First Evaluation Run

```bash
# Full 20-case evaluation with live LLM
python run_agent_evaluation.py --mode live --budget 20
```

**Expected:**
- 20 live LLM calls
- ~10-15 minutes (with rate limit retries)
- Results cached

### Subsequent Runs

```bash
# Cached evaluation (0 LLM calls)
python run_agent_evaluation.py --mode cached
```

**Expected:**
- 0 LLM calls
- <10 seconds
- Same metrics (evaluation logic rerun over cached outputs)

### After Changing Evaluation Logic

```bash
# 1. Increment AGENT_EVAL_VERSION in agent_cache.py
# 2. Re-evaluate cached results
python run_agent_evaluation.py --mode cached
```

**Expected:**
- 0 LLM calls (old cache still valid if agent didn't change)
- New metrics from new evaluation logic

### After Changing Agent Prompt/Tools

```bash
# 1. Increment AGENT_EVAL_VERSION in agent_cache.py
# 2. Run live evaluation with budget
python run_agent_evaluation.py --mode live --budget 10
```

**Expected:**
- Cache miss (version changed)
- Up to 10 new live LLM calls
- Old cache entries ignored

### Development Workflow

```bash
# 1. Unit tests (no LLM)
pytest tests/

# 2. Mock evaluation (no LLM)
python run_agent_evaluation.py --mode mock

# 3. Smoke test (5 LLM calls)
python run_smoke_test.py

# 4. Full evaluation (budget-aware)
python run_agent_evaluation.py --mode live --budget 5
```

## Free-Tier Strategy

### API Quota Management

**Groq Free Tier:** ~30 requests/minute

**Strategy:**
1. **Cache-first:** Reuse results whenever possible
2. **Budget limits:** Cap live calls per run
3. **Sequential execution:** Don't amplify rate limits
4. **Smoke test:** Quick 5-query validation instead of full 20
5. **Mock mode:** Test evaluation logic without API

### Cost Analysis

| Action | LLM Calls | Time | When to Use |
|--------|-----------|------|-------------|
| Mock evaluation | 0 | 10s | CI/CD, unit tests |
| Cached evaluation | 0 | 10s | After eval logic changes |
| Smoke test | 5 | 2-3 min | Quick validation |
| Full evaluation (first) | 20 | 10-15 min | Initial validation |
| Full evaluation (cached) | 0 | 10s | Repeat runs |

## Files

### Core Modules

- **`healthbot/evaluation/agent_cache.py`** - Persistent cache with versioning
- **`healthbot/evaluation/agent_executor.py`** - Execution layer (cache + rate limits)
- **`healthbot/evaluation/agent_eval.py`** - Offline evaluation logic

### Scripts

- **`run_agent_evaluation.py`** - Full 20-case evaluation
- **`run_smoke_test.py`** - Fast 5-query smoke test
- **`verify_agent_behavior.py`** - Original 4-query verification (deprecated, use smoke test)

### Configuration

- **`evaluation_cache/`** - Persistent cache directory (gitignored)
- **`AGENT_EVAL_VERSION`** - Cache version in `agent_cache.py`

## Verification Steps

### 1. Test Mock Mode (0 LLM calls)

```bash
python run_agent_evaluation.py --mode mock
```

**Expected output:**
```
Mode: MOCK
Total test cases: 20
Mock mode - testing evaluation logic (0 LLM calls)
...
Execution Coverage: 20/20 (100.0%)
Execution Status:
  Mock results: 20
Tool Selection Accuracy (over 20 evaluated):
  Exact Match: ...
```

### 2. Test Cache Module

```bash
python -c "from healthbot.evaluation.agent_cache import get_cache_key, AGENT_EVAL_VERSION; print(f'Version: {AGENT_EVAL_VERSION}')"
```

### 3. Test Cache Hit/Miss

```bash
# First run: cache MISS (calls LLM)
python run_smoke_test.py --no-cache

# Second run: cache HIT (0 LLM calls)
python run_smoke_test.py

# Clear cache
python -m healthbot.evaluation.agent_cache clear

# Third run: cache MISS again
python run_smoke_test.py
```

### 4. Test Rate Limit Handling

```bash
# Run full evaluation without cache
# Will hit rate limits and retry
python run_agent_evaluation.py --mode live --no-cache --budget 20
```

**Expected behavior:**
- HTTP 429 errors logged
- Exponential backoff (2s, 4s, 8s...)
- Bounded retries (max 3)
- Graceful degradation (NOT_RUN for budget exhaustion)

## Interview Explanation

> **"How did you handle LLM API rate limits during evaluation given that you were using a free tier?"**

**Answer:**

"I built a quota-aware evaluation system that separates agent execution from evaluation logic. The key insight was that most evaluation runs don't need to call the LLM - they're testing evaluation logic, not agent behavior.

The architecture has three layers:
1. **Persistent cache** - Deterministic cache key based on query + model + agent version
2. **Execution layer** - Cache-first with live budget limits and exponential backoff for rate limits
3. **Evaluation layer** - Pure function that works offline over cached results

This means:
- **First run:** 20 LLM calls with rate limit handling
- **Subsequent runs:** 0 LLM calls (evaluate cached outputs)
- **Evaluation logic changes:** 0 LLM calls (recompute metrics over same outputs)
- **Agent prompt changes:** Cache invalidates via versioning, new LLM calls needed

For free-tier development, I use:
- **Mock mode** for unit testing (deterministic outputs, 0 cost)
- **Smoke test** for quick validation (5 queries instead of 20)
- **Budget limits** to cap API usage per run

The system tracks execution status separately from evaluation results - a rate-limited API call is NOT a failed test. This prevents false negatives when the agent works but the API is temporarily unavailable.

The result: Maximum evaluation confidence per API request, with most development having zero LLM cost."

## Future Improvements

- [ ] Add retry-after header parsing for Groq API
- [ ] Support parallel execution when rate limits allow
- [ ] Add cache warming script (precompute common queries)
- [ ] Export cached results to evaluation report
- [ ] Add cache TTL (time-to-live) for stale entries
