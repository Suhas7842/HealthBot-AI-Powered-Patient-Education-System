"""
Agent execution with caching and rate limit handling.

Separates agent execution from evaluation:
- Agent execution: Calls LLM (expensive, rate-limited)
- Evaluation: Analyzes agent outputs (cheap, unlimited)

This allows:
- Cache-first execution (0 API calls for cached queries)
- Rate limit resilience (bounded retries)
- Budget-aware execution (stop after N live calls)
- Mock mode (deterministic testing)
"""

import time
from typing import Dict, Any, Optional, Literal
from pathlib import Path

from healthbot.agent_graph import run_agent_query
from healthbot.evaluation.agent_cache import (
    load_from_cache,
    save_to_cache,
    get_cache_key,
    AGENT_EVAL_VERSION,
)
from healthbot.config import settings


# Rate limit configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0  # seconds
MAX_BACKOFF = 30.0  # seconds
INTER_REQUEST_DELAY = 1.0  # Minimum delay between live calls


class ExecutionResult:
    """
    Wrapper for agent execution results with status tracking.

    Statuses:
    - SUCCESS: Agent executed successfully
    - CACHED: Result loaded from cache
    - RATE_LIMITED: API rate limit reached
    - ERROR: Execution error
    - NOT_RUN: Skipped due to budget
    """

    def __init__(
        self,
        query: str,
        status: Literal["SUCCESS", "CACHED", "RATE_LIMITED", "ERROR", "NOT_RUN", "MOCK"],
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        self.query = query
        self.status = status
        self.result = result or {}
        self.error = error

    def is_evaluated(self) -> bool:
        """Returns True if this result can be used for evaluation."""
        return self.status in ["SUCCESS", "CACHED", "MOCK"]

    def get_tools_called(self):
        """Get tools called by agent."""
        return self.result.get("tools_called", [])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }


def execute_with_cache(
    query: str,
    model: str,
    patient_level: str = "beginner",
    use_cache: bool = True,
) -> ExecutionResult:
    """
    Execute agent query with cache-first strategy.

    Execution flow:
    1. Check cache (if use_cache=True)
    2. If cached, return cached result (0 API calls)
    3. If not cached, call live LLM with rate limit handling
    4. Save successful result to cache
    5. Return result with status

    Args:
        query: User query
        model: LLM model identifier
        patient_level: Education level
        use_cache: Whether to use cache (default True)

    Returns:
        ExecutionResult with status and result
    """
    # Step 1: Try cache first
    if use_cache:
        cached = load_from_cache(query, model, patient_level)
        if cached:
            return ExecutionResult(
                query=query,
                status="CACHED",
                result=cached,
            )

    # Step 2: Call live LLM with rate limit handling
    try:
        result = execute_with_retry(query, patient_level)

        # Save to cache
        save_to_cache(
            query=query,
            model=model,
            result=result,
            patient_level=patient_level,
            status="success",
        )

        return ExecutionResult(
            query=query,
            status="SUCCESS",
            result=result,
        )

    except RateLimitError as e:
        return ExecutionResult(
            query=query,
            status="RATE_LIMITED",
            error=str(e),
        )

    except Exception as e:
        return ExecutionResult(
            query=query,
            status="ERROR",
            error=str(e),
        )


def execute_with_retry(
    query: str,
    patient_level: str = "beginner",
    max_retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    """
    Execute agent with bounded exponential backoff for rate limits.

    Args:
        query: User query
        patient_level: Education level
        max_retries: Maximum retry attempts

    Returns:
        Agent result

    Raises:
        RateLimitError: If rate limited after retries
        Exception: Other execution errors
    """
    backoff = INITIAL_BACKOFF

    for attempt in range(max_retries + 1):
        try:
            # Add minimum delay between requests (except first attempt)
            if attempt > 0:
                time.sleep(min(backoff, MAX_BACKOFF))
                backoff *= 2  # Exponential backoff

            # Execute agent
            result = run_agent_query(query, patient_level)
            return result

        except Exception as e:
            error_str = str(e).lower()

            # Check if it's a rate limit error
            if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                if attempt < max_retries:
                    print(f"  Rate limited. Retry {attempt + 1}/{max_retries} in {backoff:.1f}s...")
                    continue
                else:
                    raise RateLimitError(f"Rate limited after {max_retries} retries")

            # Other errors: don't retry
            raise


class RateLimitError(Exception):
    """Raised when API rate limit is reached."""
    pass


def execute_mock(
    query: str,
    expected_tools: list,
) -> ExecutionResult:
    """
    Execute in mock mode with deterministic output.

    Mock mode tests evaluation logic without calling LLM.

    Args:
        query: User query
        expected_tools: Tools to mock as called

    Returns:
        ExecutionResult with MOCK status
    """
    # Create mock result
    mock_result = {
        "summary": f"[MOCK] Answer to: {query[:50]}...",
        "tools_called": expected_tools if isinstance(expected_tools, list) else expected_tools.get("required", []),
        "disclaimer_shown": True,
        "tool_call_trace": [f"[MOCK] Called: {t}" for t in (expected_tools if isinstance(expected_tools, list) else expected_tools.get("required", []))],
    }

    return ExecutionResult(
        query=query,
        status="MOCK",
        result=mock_result,
    )


def batch_execute(
    queries: list,
    model: str,
    patient_level: str = "beginner",
    use_cache: bool = True,
    live_budget: Optional[int] = None,
    mock_mode: bool = False,
    expected_tools_list: Optional[list] = None,
) -> list[ExecutionResult]:
    """
    Execute multiple queries with budget awareness.

    Args:
        queries: List of queries
        model: LLM model identifier
        patient_level: Education level
        use_cache: Whether to use cache
        live_budget: Maximum number of live API calls (None = unlimited)
        mock_mode: Use mock mode instead of real LLM
        expected_tools_list: Expected tools for mock mode

    Returns:
        List of ExecutionResults
    """
    results = []
    live_calls_made = 0

    for i, query in enumerate(queries):
        # Mock mode
        if mock_mode:
            expected = expected_tools_list[i] if expected_tools_list else []
            result = execute_mock(query, expected)
            results.append(result)
            continue

        # Check live budget
        if live_budget is not None and live_calls_made >= live_budget:
            # Budget exhausted - skip remaining uncached queries
            cached = load_from_cache(query, model, patient_level) if use_cache else None
            if cached:
                results.append(ExecutionResult(query=query, status="CACHED", result=cached))
            else:
                results.append(ExecutionResult(query=query, status="NOT_RUN"))
            continue

        # Execute with cache
        result = execute_with_cache(query, model, patient_level, use_cache)

        # Track live calls
        if result.status == "SUCCESS":
            live_calls_made += 1
            # Add inter-request delay
            if i < len(queries) - 1:  # Don't delay after last request
                time.sleep(INTER_REQUEST_DELAY)

        results.append(result)

    return results


if __name__ == "__main__":
    # Test cache module
    print("Testing agent executor...")

    test_query = "What is diabetes?"
    model = settings.OPENAI_MODEL if settings.LLM_PROVIDER != "gemini" else settings.GEMINI_MODEL

    print(f"\n1. Execute with cache (will call live LLM):")
    result1 = execute_with_cache(test_query, model)
    print(f"   Status: {result1.status}")
    print(f"   Tools: {result1.get_tools_called()}")

    print(f"\n2. Execute again (should hit cache):")
    result2 = execute_with_cache(test_query, model)
    print(f"   Status: {result2.status}")
    print(f"   Tools: {result2.get_tools_called()}")

    print(f"\n3. Mock mode:")
    result3 = execute_mock(test_query, ["medical_rag_search"])
    print(f"   Status: {result3.status}")
    print(f"   Tools: {result3.get_tools_called()}")
