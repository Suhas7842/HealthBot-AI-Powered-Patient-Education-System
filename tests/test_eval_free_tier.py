"""
Focused tests for free-tier evaluation strategy.

Tests:
1. Cache hit/miss behavior
2. Budget enforcement
3. Rate limit handling
4. Required tools logic
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from healthbot.evaluation.agent_cache import (
    get_cache_key,
    save_to_cache,
    load_from_cache,
    AGENT_EVAL_VERSION,
    CACHE_FILE,
)
from healthbot.evaluation.agent_executor import (
    batch_execute,
    ExecutionResult,
    RateLimitError,
)
from healthbot.evaluation.agent_eval import evaluate_tool_selection


class TestCacheHitMiss:
    """Test cache hit/miss behavior."""

    def test_cache_key_changes_with_query(self):
        """Different queries produce different cache keys."""
        key1 = get_cache_key("What is diabetes?", "model1")
        key2 = get_cache_key("What is cancer?", "model1")
        assert key1 != key2

    def test_cache_key_changes_with_model(self):
        """Different models produce different cache keys."""
        key1 = get_cache_key("What is diabetes?", "model1")
        key2 = get_cache_key("What is diabetes?", "model2")
        assert key1 != key2

    def test_cache_key_changes_with_version(self):
        """Different agent versions produce different cache keys."""
        # Cache key includes AGENT_EVAL_VERSION implicitly
        key1 = get_cache_key("What is diabetes?", "model1")

        # Simulate version change (in real usage, increment AGENT_EVAL_VERSION)
        with patch('healthbot.evaluation.agent_cache.AGENT_EVAL_VERSION', 'v2'):
            key2 = get_cache_key("What is diabetes?", "model1", agent_version='v2')

        assert key1 != key2

    def test_failed_calls_not_cached_as_success(self, tmp_path):
        """Failed LLM calls should not be cached as successful results."""
        # Override cache file to temp location
        with patch('healthbot.evaluation.agent_cache.CACHE_FILE', tmp_path / 'test_cache.jsonl'):
            # Save error result
            save_to_cache(
                query="test query",
                model="test_model",
                result={},
                status="error",
                error="Rate limit exceeded"
            )

            # Try to load - should return None because status != success
            cached = load_from_cache("test query", "test_model")
            assert cached is None


class TestBudgetEnforcement:
    """Test live call budget enforcement."""

    @patch('healthbot.evaluation.agent_executor.execute_with_cache')
    @patch('healthbot.evaluation.agent_executor.load_from_cache')
    def test_budget_respected(self, mock_load, mock_execute):
        """Batch execution respects live budget."""
        # Setup mocks
        mock_load.return_value = None  # No cache hits
        mock_execute.return_value = ExecutionResult(
            query="test",
            status="SUCCESS",
            result={"tools_called": ["medical_rag_search"]}
        )

        # Execute with budget of 3
        queries = ["q1", "q2", "q3", "q4", "q5"]
        results = batch_execute(
            queries=queries,
            model="test_model",
            use_cache=True,
            live_budget=3,
        )

        # Should make exactly 3 live calls
        assert mock_execute.call_count == 3

        # First 3 should be SUCCESS, rest should be NOT_RUN
        assert results[0].status == "SUCCESS"
        assert results[1].status == "SUCCESS"
        assert results[2].status == "SUCCESS"
        assert results[3].status == "NOT_RUN"
        assert results[4].status == "NOT_RUN"

    @patch('healthbot.evaluation.agent_executor.execute_with_cache')
    def test_cached_results_dont_count_toward_budget(self, mock_execute):
        """Cached results should not consume live budget."""
        # First 2 queries return CACHED, next 2 SUCCESS, rest NOT_RUN
        def execute_side_effect(query, model, patient_level, use_cache):
            if query in ["q1", "q2"]:
                return ExecutionResult(
                    query=query,
                    status="CACHED",
                    result={"tools_called": ["medical_rag_search"]}
                )
            return ExecutionResult(
                query=query,
                status="SUCCESS",
                result={"tools_called": ["medical_rag_search"]}
            )

        mock_execute.side_effect = execute_side_effect

        queries = ["q1", "q2", "q3", "q4", "q5"]
        results = batch_execute(
            queries=queries,
            model="test_model",
            use_cache=True,
            live_budget=2,  # Only 2 live calls allowed
        )

        # Should use cache for q1, q2 (don't count toward budget)
        # Should make live calls for q3, q4 (budget = 2)
        # Should mark q5 as NOT_RUN
        assert results[0].status == "CACHED"
        assert results[1].status == "CACHED"
        assert results[2].status == "SUCCESS"
        assert results[3].status == "SUCCESS"
        assert results[4].status == "NOT_RUN"


class TestRateLimitHandling:
    """Test rate limit detection and handling."""

    @patch('healthbot.evaluation.agent_executor.execute_with_cache')
    @patch('healthbot.evaluation.agent_executor.load_from_cache')
    def test_batch_stops_after_rate_limit(self, mock_load, mock_execute):
        """Batch execution should stop making live calls after rate limit."""
        mock_load.return_value = None  # No cache hits

        # First call succeeds, second hits rate limit
        mock_execute.side_effect = [
            ExecutionResult(query="q1", status="SUCCESS", result={"tools_called": []}),
            ExecutionResult(query="q2", status="RATE_LIMITED", error="429 Too Many Requests"),
        ]

        queries = ["q1", "q2", "q3", "q4"]
        results = batch_execute(
            queries=queries,
            model="test_model",
            use_cache=True,
            live_budget=10,  # High budget, but should stop at rate limit
        )

        # Should only call execute_with_cache twice (q1 and q2)
        assert mock_execute.call_count == 2

        # q1 succeeds, q2 rate limited, q3 and q4 not run
        assert results[0].status == "SUCCESS"
        assert results[1].status == "RATE_LIMITED"
        assert results[2].status == "NOT_RUN"
        assert results[3].status == "NOT_RUN"

    @patch('healthbot.evaluation.agent_executor.time.sleep')
    def test_retry_with_exponential_backoff(self, mock_sleep):
        """Rate limit should trigger exponential backoff retry."""
        from healthbot.evaluation.agent_executor import execute_with_retry, MAX_RETRIES

        # Mock the agent_graph module at the point of import
        with patch.dict('sys.modules', {'healthbot.agent_graph': MagicMock()}):
            import sys
            mock_module = sys.modules['healthbot.agent_graph']

            # First call fails with 429, second succeeds
            mock_module.run_agent_query = MagicMock(side_effect=[
                Exception("429 Too Many Requests"),
                {"tools_called": ["medical_rag_search"], "summary": "test"}
            ])

            result = execute_with_retry("test query", "beginner")

            # Should have retried once (total 2 attempts with MAX_RETRIES=1)
            assert mock_module.run_agent_query.call_count == 2

            # Should have slept once with 2s backoff
            assert mock_sleep.call_count == 1
            assert mock_sleep.call_args[0][0] == 2.0

    def test_rate_limit_after_max_retries(self):
        """Should raise RateLimitError after max retries."""
        from healthbot.evaluation.agent_executor import execute_with_retry, MAX_RETRIES

        # Mock the agent_graph module at the point of import
        with patch.dict('sys.modules', {'healthbot.agent_graph': MagicMock()}):
            import sys
            mock_module = sys.modules['healthbot.agent_graph']

            # All attempts fail with 429
            mock_module.run_agent_query = MagicMock(side_effect=Exception("429 Too Many Requests"))

            with pytest.raises(RateLimitError):
                execute_with_retry("test query", "beginner")

            # Should have tried MAX_RETRIES + 1 times (1 initial + 1 retry with default MAX_RETRIES=1)
            assert mock_module.run_agent_query.call_count == MAX_RETRIES + 1


class TestRequiredToolsLogic:
    """Test required/optional/inappropriate tool evaluation."""

    def test_all_required_tools_must_be_called(self):
        """When multiple tools are required, ALL must be called."""
        actual_tools = ["medical_calculator"]
        expected_tools = {
            "required": ["medical_calculator", "medical_rag_search"],
            "optional": [],
            "inappropriate": []
        }

        result = evaluate_tool_selection(actual_tools, expected_tools)

        # Should fail because medical_rag_search is also required
        assert result["has_required_tool"] == False
        assert result["exact_match"] == False

    def test_all_required_tools_present(self):
        """When all required tools are called, should pass."""
        actual_tools = ["medical_calculator", "medical_rag_search"]
        expected_tools = {
            "required": ["medical_calculator", "medical_rag_search"],
            "optional": [],
            "inappropriate": []
        }

        result = evaluate_tool_selection(actual_tools, expected_tools)

        # Should pass
        assert result["has_required_tool"] == True
        assert result["exact_match"] == True

    def test_optional_tools_not_required(self):
        """Optional tools should not be required."""
        actual_tools = ["medical_calculator"]
        expected_tools = {
            "required": ["medical_calculator"],
            "optional": ["medical_rag_search"],
            "inappropriate": []
        }

        result = evaluate_tool_selection(actual_tools, expected_tools)

        # Should pass without optional tool
        assert result["has_required_tool"] == True
        assert result["exact_match"] == True

    def test_inappropriate_tool_fails(self):
        """Using an inappropriate tool should fail the test."""
        actual_tools = ["medical_calculator", "web_search"]
        expected_tools = {
            "required": ["medical_calculator"],
            "optional": [],
            "inappropriate": ["web_search"]
        }

        result = evaluate_tool_selection(actual_tools, expected_tools)

        # Should fail due to inappropriate tool
        assert result["used_inappropriate_tool"] == True
        assert result["exact_match"] == False

    def test_empty_required_always_passes(self):
        """When no tools are required, any tool selection should pass required check."""
        actual_tools = ["medical_rag_search"]
        expected_tools = {
            "required": [],
            "optional": ["medical_rag_search", "pubmed_api_search"],
            "inappropriate": []
        }

        result = evaluate_tool_selection(actual_tools, expected_tools)

        # Should pass - no required tools
        assert result["has_required_tool"] == True

    def test_at_least_one_requires_one_tool(self):
        """When at_least_one is specified, at least one of those tools must be called."""
        # No tools called - should fail at_least_one
        actual_tools = []
        expected_tools = {
            "required": [],
            "at_least_one": ["pubmed_api_search", "medical_rag_search"],
            "optional": [],
            "inappropriate": []
        }

        result = evaluate_tool_selection(actual_tools, expected_tools)
        assert result["has_required_tool"] == True  # No required tools
        assert result["has_at_least_one"] == False  # Failed at_least_one

    def test_at_least_one_passes_with_any_alternative(self):
        """Any tool from at_least_one set should satisfy the requirement."""
        expected_tools = {
            "required": [],
            "at_least_one": ["pubmed_api_search", "medical_rag_search"],
            "optional": [],
            "inappropriate": []
        }

        # Try first alternative
        result1 = evaluate_tool_selection(["pubmed_api_search"], expected_tools)
        assert result1["has_at_least_one"] == True

        # Try second alternative
        result2 = evaluate_tool_selection(["medical_rag_search"], expected_tools)
        assert result2["has_at_least_one"] == True

    def test_at_least_one_fails_with_wrong_tool(self):
        """Calling a tool not in at_least_one set should fail."""
        actual_tools = ["web_search"]
        expected_tools = {
            "required": [],
            "at_least_one": ["pubmed_api_search", "medical_rag_search"],
            "optional": [],
            "inappropriate": []
        }

        result = evaluate_tool_selection(actual_tools, expected_tools)
        assert result["has_at_least_one"] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
