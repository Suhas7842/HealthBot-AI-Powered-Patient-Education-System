# Agent Evaluation Instructions - Phase 4 Final Step

**Date Created**: 2026-08-21  
**Status**: Agent architecture complete, evaluation pending API quota  
**Task**: Run empirical agent evaluation and document results

---

## Context

### What Was Done (Phase 4)

HealthBot was transformed from a "RAG pipeline" to a "GenAI orchestration system":

**Before Phase 4**:
- Fixed 16-node pipeline
- Hardcoded keyword-based tool selection
- Only RAG + web search
- Interview positioning: "I built a RAG system"

**After Phase 4**:
- ReAct agent with LangGraph
- LLM-driven tool selection (agent decides which tools to call)
- 4 custom tools: RAG, Medical Calculator, PubMed API, Web Search
- Interview positioning: "I built GenAI orchestration with custom tools"

**Architecture Files**:
- `healthbot/agent_tools.py` - LangChain tool wrappers (4 tools)
- `healthbot/agent_graph.py` - ReAct agent workflow
- `healthbot/tools/medical_calculator.py` - BMI, dosage, kidney function
- `healthbot/tools/pubmed_api.py` - PubMed E-utilities integration
- `healthbot/evaluation/agent_eval.py` - Evaluation framework (20 test cases)
- `run_agent_evaluation.py` - Evaluation runner script

**Tests**: 132 passing (80 Phase 4 + 52 Phase 1-3)

**Git Commits**: 5 commits pushed to GitHub, all authored by Suhas (no Claude attribution)

---

## What Needs to Be Done

### The Problem

Phase 3 set the standard for empirical validation:
- RAG retrieval: **0.329 Recall@5** (measured, not claimed)
- Hybrid strategy: **320ms latency** (proven with data)
- Threshold validation: **100% pass rate** (empirically validated)

Phase 4 currently has:
- Tool selection: **Target ≥80%** (aspirational, not measured)
- Multi-tool usage: **Target ≥70%** (aspirational, not measured)

**This violates the CLAUDE.md principle**: "Verification is proof, not assumption."

### The Solution

Run **empirical agent evaluation** on 20 test cases to measure:
1. **Tool Selection Accuracy** - Agent chose correct tools
2. **Multi-Tool Usage Rate** - Used multiple tools when needed
3. **Precision/Recall/F1** - Quality metrics
4. **Failure Analysis** - What agent got wrong

Then update README with **actual measured data**, matching Phase 3 standard.

---

## Step-by-Step Instructions

### Step 1: Check API Quota

```bash
cd "c:\Users\rsuha\Downloads\HealthBot-AI-Powered-Patient-Education-System"

# Quick test - should succeed if quota reset
python -c "
from healthbot.agent_graph import run_agent_query
result = run_agent_query('What is diabetes?')
print('✓ API working! Quota available.')
print(f'Tools called: {result.get(\"tools_called\", [])}')
"
```

**Expected**: Should complete without 429 quota errors.

**If quota error**: Wait a few hours and try again. Gemini free tier: 20 requests/day.

---

### Step 2: Run Agent Evaluation

```bash
# This runs the agent on all 20 test cases
# Takes ~10-15 minutes (20 LLM calls + tool executions)
python run_agent_evaluation.py
```

**What This Does**:
1. Runs agent on 20 diverse test cases
2. Compares actual tool selection vs. expected tools
3. Calculates accuracy metrics
4. Saves results to `evaluation_results/phase4/agent_evaluation_results.json`
5. Creates README summary at `evaluation_results/phase4/README_summary.txt`

**Expected Output**:
```
======================================================================
EVALUATION COMPLETE
======================================================================

Tool Selection Accuracy:
- Exact Match: XX/20 (XX%)
- Partial Match: XX/20 (XX%)

Metrics:
- Precision: X.XXX
- Recall: X.XXX
- F1 Score: X.XXX

Tool Usage Patterns:
- Multi-Tool Usage Rate: XX%
- Single-Tool Accuracy: XX%
```

---

### Step 3: Review Results

**Check the results file**:
```bash
cat evaluation_results/phase4/agent_evaluation_results.json
```

**Key metrics to note**:
- `exact_match_rate` - % of queries where agent got tools exactly right
- `avg_precision` - % of called tools that were correct
- `avg_recall` - % of expected tools that were called
- `avg_f1_score` - Balanced accuracy metric
- `multi_tool_rate` - % of complex queries that used multiple tools
- `single_tool_accuracy` - % of simple queries that used exactly one correct tool

**Look for failures**:
```bash
# Check evaluations array for any with exact_match: false
cat evaluation_results/phase4/agent_evaluation_results.json | grep -A 5 "exact_match.*false"
```

---

### Step 4: Update README with Results

**Location**: `README.md` - Find the Phase 4 section

**Current Text** (aspirational targets):
```markdown
**📊 Agent Evaluation:**
- **Tool Selection Accuracy**: Target ≥80% (agent chooses optimal tools)
- **Multi-Tool Usage**: Target ≥70% on complex queries
- **Test Cases**: 20 agent-specific test cases measuring tool selection quality
```

**Replace With** (actual measured data):
```markdown
**📊 Agent Evaluation Results** (Empirical Validation):

**Test Cases**: 20 diverse queries (single-tool, multi-tool, tool diversity)

**Tool Selection Performance**:
- Exact Match Rate: XX/20 (XX%)
- Partial Match Rate: XX/20 (XX%)
- Precision: X.XXX (% of called tools that were correct)
- Recall: X.XXX (% of expected tools that were called)
- F1 Score: X.XXX (balanced accuracy)

**Tool Usage Patterns**:
- Multi-Tool Usage: XX% (agent used multiple tools for complex queries)
- Single-Tool Accuracy: XX% (correct tool for simple queries)

**Evaluation Date**: 2026-08-21  
**Methodology**: Agent executed on 20 test cases, tool selections compared against expected tools for each query type
```

**Also add comparison table** (find the existing architecture comparison and enhance it):

```markdown
| Aspect | Phase 3 (RAG Pipeline) | Phase 4 (GenAI Agent) | Evidence |
|--------|------------------------|----------------------|----------|
| Tool Selection | Hardcoded keywords | LLM reasoning | XX% accuracy measured |
| Tool Count | 2 (RAG + Web) | 4 (RAG + Calc + PubMed + Web) | 4 tools implemented |
| Multi-Tool | No | Yes | XX% multi-tool usage |
| Capabilities | Retrieval only | Retrieval + Computation + API | 3 capability types |
```

---

### Step 5: Commit Results

```bash
cd "c:\Users\rsuha\Downloads\HealthBot-AI-Powered-Patient-Education-System"

# Stage evaluation results
git add evaluation_results/phase4/
git add README.md

# Commit with measured results
git commit -m "feat: Add Phase 4 agent evaluation results

Empirical validation of agent tool selection:

Evaluation Results (20 test cases):
- Tool selection accuracy: XX% exact match, XX% partial match
- Precision: X.XXX, Recall: X.XXX, F1: X.XXX
- Multi-tool usage rate: XX%
- Single-tool accuracy: XX%

Methodology:
- Ran agent on 20 diverse test cases
- Measured tool selection vs. expected tools
- Calculated precision, recall, F1 metrics
- Analyzed multi-tool usage patterns

This completes Phase 4 empirical validation, matching the
Phase 3 standard of measured performance (not aspirational targets).
"

# Push to GitHub
git push origin main
```

---

## Expected Outcomes

### Good Results (≥75% accuracy)

**If Tool Selection ≥75%**:
- Agent is working correctly
- Tool selection is reliable
- Architecture validates the design
- **Interview story**: "I measured 78% tool selection accuracy on 20 test cases"

### Moderate Results (50-74% accuracy)

**If Tool Selection 50-74%**:
- Agent sometimes chooses wrong tools
- Still demonstrates GenAI orchestration (better than hardcoded)
- Shows honest evaluation (not cherry-picked results)
- **Interview story**: "Agent achieved 65% tool accuracy. I identified failure patterns: agent over-uses RAG, underuses calculator for numeric queries. This demonstrates empirical evaluation methodology."

### Poor Results (<50% accuracy)

**If Tool Selection <50%**:
- Agent prompt may need refinement
- Tool descriptions may be unclear
- Still proves architecture works (agent CAN call tools)
- **Interview story**: "Initial evaluation showed 45% accuracy. This revealed tool prompt engineering challenges - a realistic GenAI engineering problem. The infrastructure works; optimization is next phase."

**Any result is valuable** - it's empirical evidence, which is the goal.

---

## Success Criteria

✅ **Evaluation ran successfully** (20 test cases completed)  
✅ **Metrics calculated** (accuracy, precision, recall, F1)  
✅ **README updated** with actual measured data (not targets)  
✅ **Results committed** to GitHub  
✅ **Project has empirical validation** for both RAG (Phase 3) and Agent (Phase 4)

---

## Interview Defense After Evaluation

### Question: "How well does your agent select tools?"

**Before Evaluation**:
> "The agent should achieve around 80% accuracy based on the tool descriptions and prompts."

**After Evaluation**:
> "I measured XX% tool selection accuracy on 20 diverse test cases. The agent correctly identified when to use the calculator for numeric queries vs. RAG for medical knowledge in XX% of cases. Multi-tool orchestration worked in XX% of complex queries."

### Question: "How did you validate the agent works?"

**Current Answer**:
> "I have 80 unit tests for the agent infrastructure."

**Better Answer** (after evaluation):
> "I validated the agent with two approaches:
> 1. **Infrastructure testing**: 80 unit tests verify tool wrappers, agent graph, safety checks
> 2. **Behavioral testing**: 20-case empirical evaluation measuring tool selection accuracy (XX%), precision (X.XX), and multi-tool usage (XX%)
> 
> This matches my Phase 3 methodology where I empirically validated RAG retrieval (0.329 Recall@5) rather than assuming it worked."

---

## Troubleshooting

### Issue: Still Getting 429 Quota Errors

**Solution**: Check quota reset time
```bash
# Gemini free tier: 20 requests/day
# Quota resets 24 hours after FIRST request (not midnight)
# If yesterday's testing started at 2pm, quota resets today at 2pm
```

**Workaround**: Run evaluation on subset first
```python
# Edit run_agent_evaluation.py, line 37:
test_cases = get_test_cases()[:5]  # Test with 5 cases first
```

If 5 cases work, run full 20.

### Issue: Agent Not Calling Any Tools

**Check**: Agent graph is using the right model
```bash
# Should see gemini-2.0-flash in config
grep "GEMINI_MODEL" healthbot/config.py
```

**Check**: Tools are being loaded
```python
python -c "from healthbot.agent_tools import get_all_tools; print([t.name for t in get_all_tools()])"
# Should show: ['medical_rag_search', 'medical_calculator', 'pubmed_api_search', 'web_search']
```

### Issue: Evaluation Script Crashes

**Check**: Python environment
```bash
pytest tests/test_agent_graph.py -v
# Should pass: 11/11 agent graph tests
```

If tests pass but evaluation fails, there's a runtime issue (likely API).

---

## Files You'll Modify

1. **`README.md`** - Update Phase 4 section with actual metrics
2. **New**: `evaluation_results/phase4/agent_evaluation_results.json` - Full results
3. **New**: `evaluation_results/phase4/README_summary.txt` - Summary for README

---

## What This Completes

### Phase 3 Standard (Achieved)
- Built hybrid retrieval
- Ran experiments
- Got actual numbers: 0.329 Recall@5, 320ms, 100% pass rate
- Documented empirical evidence

### Phase 4 Standard (After This Task)
- Built agent with 4 tools
- Ran evaluation
- Got actual numbers: XX% tool accuracy, X.XX precision/recall
- Documented empirical evidence

**Both phases**: Architecture + Empirical Validation ✅

---

## Final Note

The goal is **empirical validation**, not perfect accuracy.

- 90% accuracy = Excellent agent design
- 70% accuracy = Good agent, shows real behavior
- 50% accuracy = Agent works, needs optimization
- ANY measured result = Empirical evidence (better than aspirational targets)

**What matters for interviews**: You measured it, not guessed it.

---

## Quick Start Command (Copy-Paste Tomorrow)

```bash
# Navigate to project
cd "c:\Users\rsuha\Downloads\HealthBot-AI-Powered-Patient-Education-System"

# Test API quota
python -c "from healthbot.agent_graph import run_agent_query; r = run_agent_query('What is diabetes?'); print(f'✓ API working! Tools: {r.get(\"tools_called\", [])}')"

# If API works, run full evaluation
python run_agent_evaluation.py

# Review results
cat evaluation_results/phase4/README_summary.txt

# Update README.md with actual numbers from summary

# Commit results
git add evaluation_results/phase4/ README.md
git commit -m "feat: Add Phase 4 agent evaluation results (XX% accuracy, X.XX F1)"
git push origin main
```

---

**Estimated Time**: 30-45 minutes (10-15 min evaluation + 15-30 min documentation)

**Outcome**: Phase 4 empirically validated with measured agent performance data.
