"""
Run Agent Evaluation - Phase 4 Empirical Validation

Supports multiple execution modes:
- LIVE: Run agent on all test cases (quota-aware, cache-first)
- CACHED: Evaluate only cached results (0 LLM calls)
- MOCK: Test evaluation logic with deterministic outputs (0 LLM calls)

Configuration:
- LIVE_BUDGET: Maximum number of live LLM calls (default: 5, conservative for free tier)
- USE_CACHE: Whether to use persistent cache (default: True)

This provides empirical evidence for agent performance while respecting free-tier constraints.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from healthbot.evaluation.agent_eval import (
    get_test_cases,
    evaluate_agent_performance,
    print_evaluation_summary,
)
from healthbot.evaluation.agent_executor import (
    batch_execute,
    ExecutionResult,
)
from healthbot.evaluation.agent_cache import get_cache_stats, AGENT_EVAL_VERSION
from healthbot.config import settings


def run_evaluation(
    mode: str = "live",
    live_budget: int = 20,
    use_cache: bool = True,
):
    """
    Run agent evaluation on all test cases.

    Args:
        mode: Execution mode ("live", "cached", "mock")
        live_budget: Maximum number of live LLM calls (for "live" mode)
        use_cache: Whether to use persistent cache (for "live" mode)
    """
    print("\n" + "="*70)
    print("AGENT EVALUATION - Phase 4 Empirical Validation")
    print("="*70)
    print(f"\nMode: {mode.upper()}")
    print(f"Agent Version: {AGENT_EVAL_VERSION}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Get model identifier
    if settings.LLM_PROVIDER == "gemini":
        model = settings.GEMINI_MODEL
    else:
        model = settings.OPENAI_MODEL

    print(f"Model: {model}")

    # Show cache stats if available
    if mode in ["live", "cached"]:
        cache_stats = get_cache_stats()
        print(f"\nCache: {cache_stats['successful_entries']} successful entries")

    test_cases = get_test_cases()
    print(f"\nTotal test cases: {len(test_cases)}")

    if mode == "live":
        print(f"Live budget: {live_budget} LLM calls")
        print(f"Use cache: {use_cache}")
        print("\nEstimated time: 1-2 minutes (with cache hits)")
    elif mode == "cached":
        print("\nEvaluating cached results only (0 LLM calls)")
    elif mode == "mock":
        print("\nMock mode - testing evaluation logic (0 LLM calls)")

    print("-"*70)

    # Extract queries and expected tools
    queries = [tc["query"] for tc in test_cases]
    expected_tools_list = [tc["expected_tools"] for tc in test_cases]

    # Execute based on mode
    if mode == "mock":
        # Mock mode: deterministic outputs
        execution_results = batch_execute(
            queries=queries,
            model=model,
            use_cache=False,
            mock_mode=True,
            expected_tools_list=expected_tools_list,
        )
    elif mode == "cached":
        # Cached mode: only use cache, no live calls
        execution_results = batch_execute(
            queries=queries,
            model=model,
            use_cache=True,
            live_budget=0,  # No live calls allowed
        )
    else:  # live
        # Live mode: cache-first with budget
        execution_results = batch_execute(
            queries=queries,
            model=model,
            use_cache=use_cache,
            live_budget=live_budget,
        )

    # Show progress
    for i, (exec_result, test_case) in enumerate(zip(execution_results, test_cases), 1):
        query = test_case["query"]
        category = test_case["category"]
        expected_tools = test_case["expected_tools"]

        print(f"\n[{i}/{len(test_cases)}] {category} [{exec_result.status}]")
        print(f"Query: {query[:60]}..." if len(query) > 60 else f"Query: {query}")

        if exec_result.is_evaluated():
            tools_called = exec_result.get_tools_called()
            print(f"Expected: {expected_tools}")
            print(f"Called: {tools_called if tools_called else 'None'}")
        elif exec_result.status == "NOT_RUN":
            print(f"Skipped (budget exhausted)")
        elif exec_result.status == "RATE_LIMITED":
            print(f"Rate limited: {exec_result.error}")
        elif exec_result.status == "ERROR":
            print(f"Error: {exec_result.error}")

    # Convert ExecutionResults to evaluation format
    results = []
    for exec_result, test_case in zip(execution_results, test_cases):
        results.append({
            "query": test_case["query"],
            "expected_tools": test_case["expected_tools"],
            "tools_called": exec_result.get_tools_called(),
            "category": test_case["category"],
            "success": exec_result.is_evaluated(),
        })

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

    # Calculate metrics (pass execution_results for status tracking)
    print("\nCalculating metrics...")
    summary = evaluate_agent_performance(results, execution_results=execution_results)
    summary["mode"] = mode
    summary["agent_version"] = AGENT_EVAL_VERSION
    summary["model"] = model

    # Print summary
    print_evaluation_summary(summary)

    # Save results
    output_dir = Path("evaluation_results/phase4")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "agent_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Results saved to: {results_file}")

    # Create summary for README
    readme_summary = f"""
## Agent Evaluation Results (Phase 4)

**Test Cases**: {summary['total_cases']} diverse queries

**Tool Selection Accuracy**:
- Exact Match: {summary['exact_match_count']}/{summary['total_cases']} ({summary['exact_match_rate']:.1%})
- Partial Match: {summary['partial_match_count']}/{summary['total_cases']} ({summary['partial_match_rate']:.1%})

**Metrics**:
- Precision: {summary['avg_precision']:.3f}
- Recall: {summary['avg_recall']:.3f}
- F1 Score: {summary['avg_f1_score']:.3f}

**Tool Usage Patterns**:
- Multi-Tool Usage Rate: {summary['multi_tool_rate']:.1%}
- Single-Tool Accuracy: {summary['single_tool_accuracy']:.1%}

**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d')}
"""

    readme_file = output_dir / "README_summary.txt"
    with open(readme_file, 'w') as f:
        f.write(readme_summary)

    print(f"[OK] README summary saved to: {readme_file}")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Review results above")
    print("2. Update README.md with actual metrics")
    print("3. Commit evaluation results to repository")
    print("\nAgent evaluation complete!")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agent evaluation")
    parser.add_argument(
        "--mode",
        choices=["live", "cached", "mock"],
        default="live",
        help="Execution mode: live (cache-first), cached (only cache), mock (deterministic)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=5,
        help="Maximum number of live LLM calls (live mode only). Default is conservative for free tier.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache (live mode only)",
    )

    args = parser.parse_args()

    try:
        summary = run_evaluation(
            mode=args.mode,
            live_budget=args.budget,
            use_cache=not args.no_cache,
        )
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
