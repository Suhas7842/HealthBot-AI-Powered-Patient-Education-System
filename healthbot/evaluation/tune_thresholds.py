"""
Evidence validation threshold tuning for HealthBot (Phase 3B).

Empirically determines optimal thresholds for evidence validation by testing
different combinations and measuring impact on retrieval quality and pass rate.

**Current Thresholds (hardcoded in nodes.py:236-239):**
- MIN_DOCS = 3
- MIN_AVG_SCORE = 0.015
- MIN_SOURCES = 2

**Goal:**
Find thresholds that:
- Pass 95%+ of valid medical queries (high recall)
- Filter low-quality retrievals (low false positive rate)
- Balance precision vs. coverage

**Usage:**
    python -m healthbot.evaluation.tune_thresholds [--validate]

**Options:**
    --validate: Check current thresholds only (no full tuning run)
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from healthbot.evaluation.test_suite import MEDICAL_TEST_CASES
from healthbot.logger import logger
from healthbot.tools import ToolSelector


class ThresholdTuner:
    """Tunes evidence validation thresholds empirically."""

    # Test matrix
    THRESHOLDS_TO_TEST = {
        "MIN_DOCS": [2, 3, 4, 5],
        "MIN_AVG_SCORE": [0.010, 0.015, 0.020, 0.025, 0.030],
        "MIN_SOURCES": [1, 2, 3],
    }

    # Current production thresholds
    CURRENT_THRESHOLDS = {
        "MIN_DOCS": 3,
        "MIN_AVG_SCORE": 0.015,
        "MIN_SOURCES": 2,
    }

    def __init__(self, sample_size: int | None = None):
        """
        Initialize threshold tuner.

        Args:
            sample_size: Number of test cases to use (None = all 50)
        """
        self.sample_size = sample_size
        self.test_cases = MEDICAL_TEST_CASES

        if sample_size and sample_size < len(self.test_cases):
            import random
            self.test_cases = random.sample(self.test_cases, sample_size)

        self.tool_selector = ToolSelector()
        self.results = []

    def validate_current_thresholds(self) -> dict:
        """
        Validate current production thresholds.

        Returns:
            Dictionary with pass rate and statistics for current thresholds
        """
        logger.info("Validating current production thresholds...")
        logger.info(f"Current: MIN_DOCS={self.CURRENT_THRESHOLDS['MIN_DOCS']}, "
                   f"MIN_AVG_SCORE={self.CURRENT_THRESHOLDS['MIN_AVG_SCORE']}, "
                   f"MIN_SOURCES={self.CURRENT_THRESHOLDS['MIN_SOURCES']}")

        result = self._test_threshold_combination(
            min_docs=self.CURRENT_THRESHOLDS["MIN_DOCS"],
            min_avg_score=self.CURRENT_THRESHOLDS["MIN_AVG_SCORE"],
            min_sources=self.CURRENT_THRESHOLDS["MIN_SOURCES"],
        )

        return result

    def run_full_tuning(self) -> dict:
        """
        Run full threshold tuning across test matrix.

        Returns:
            Dictionary with all results and recommendations
        """
        logger.info(f"Starting full threshold tuning on {len(self.test_cases)} test cases")
        logger.info(f"Test matrix: {len(self.THRESHOLDS_TO_TEST['MIN_DOCS'])} x "
                   f"{len(self.THRESHOLDS_TO_TEST['MIN_AVG_SCORE'])} x "
                   f"{len(self.THRESHOLDS_TO_TEST['MIN_SOURCES'])} = "
                   f"{len(self.THRESHOLDS_TO_TEST['MIN_DOCS']) * len(self.THRESHOLDS_TO_TEST['MIN_AVG_SCORE']) * len(self.THRESHOLDS_TO_TEST['MIN_SOURCES'])} combinations")

        start_time = time.time()
        total_combinations = (
            len(self.THRESHOLDS_TO_TEST["MIN_DOCS"]) *
            len(self.THRESHOLDS_TO_TEST["MIN_AVG_SCORE"]) *
            len(self.THRESHOLDS_TO_TEST["MIN_SOURCES"])
        )

        combination_num = 0
        for min_docs in self.THRESHOLDS_TO_TEST["MIN_DOCS"]:
            for min_avg_score in self.THRESHOLDS_TO_TEST["MIN_AVG_SCORE"]:
                for min_sources in self.THRESHOLDS_TO_TEST["MIN_SOURCES"]:
                    combination_num += 1
                    logger.info(f"[{combination_num}/{total_combinations}] Testing: "
                               f"MIN_DOCS={min_docs}, MIN_AVG_SCORE={min_avg_score}, "
                               f"MIN_SOURCES={min_sources}")

                    result = self._test_threshold_combination(
                        min_docs=min_docs,
                        min_avg_score=min_avg_score,
                        min_sources=min_sources,
                    )
                    self.results.append(result)

        total_time = time.time() - start_time
        logger.info(f"Full tuning completed in {total_time:.1f}s")

        # Analyze results and generate recommendations
        recommendations = self._generate_recommendations()

        return {
            "summary": {
                "total_combinations": total_combinations,
                "test_cases": len(self.test_cases),
                "total_time_seconds": total_time,
            },
            "current_thresholds": self.CURRENT_THRESHOLDS,
            "all_results": self.results,
            "recommendations": recommendations,
        }

    def _test_threshold_combination(
        self, min_docs: int, min_avg_score: float, min_sources: int
    ) -> dict:
        """
        Test a single threshold combination.

        Args:
            min_docs: Minimum document count
            min_avg_score: Minimum RRF score threshold
            min_sources: Minimum unique sources

        Returns:
            Dictionary with results for this combination
        """
        passed = 0
        failed = 0
        failure_reasons = defaultdict(int)

        for case in self.test_cases:
            # Retrieve documents
            results = self.tool_selector.select_and_search(
                case["question"], k=5
            )

            if not results["success"] or not results["documents"]:
                failed += 1
                failure_reasons["retrieval_failed"] += 1
                continue

            docs = results["documents"]

            # Apply validation thresholds
            validation_result = self._validate_evidence(
                docs, min_docs, min_avg_score, min_sources
            )

            if validation_result["passed"]:
                passed += 1
            else:
                failed += 1
                failure_reasons[validation_result["reason"]] += 1

        pass_rate = passed / len(self.test_cases)

        return {
            "thresholds": {
                "MIN_DOCS": min_docs,
                "MIN_AVG_SCORE": min_avg_score,
                "MIN_SOURCES": min_sources,
            },
            "passed": passed,
            "failed": failed,
            "pass_rate": pass_rate,
            "failure_reasons": dict(failure_reasons),
        }

    def _validate_evidence(
        self, docs: list, min_docs: int, min_avg_score: float, min_sources: int
    ) -> dict:
        """
        Apply evidence validation logic (mimics nodes.py validate_evidence).

        Args:
            docs: List of retrieved documents
            min_docs: Minimum document count
            min_avg_score: Minimum RRF score threshold
            min_sources: Minimum unique sources

        Returns:
            Dictionary with passed (bool) and reason (str)
        """
        # Check 1: Minimum document count
        if len(docs) < min_docs:
            return {
                "passed": False,
                "reason": f"insufficient_docs (got {len(docs)}, need {min_docs})",
            }

        # Check 2: Average RRF score
        scores = [doc.get("score", 0.0) for doc in docs]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        if avg_score < min_avg_score:
            return {
                "passed": False,
                "reason": f"low_avg_score (got {avg_score:.4f}, need {min_avg_score})",
            }

        # Check 3: Source diversity
        unique_sources = set()
        for doc in docs:
            metadata = doc.get("metadata", {})
            pmid = metadata.get("pmid")
            title = metadata.get("title")
            if pmid:
                unique_sources.add(pmid)
            elif title:
                unique_sources.add(title)

        if len(unique_sources) < min_sources:
            return {
                "passed": False,
                "reason": f"insufficient_sources (got {len(unique_sources)}, need {min_sources})",
            }

        return {"passed": True, "reason": "passed_all_checks"}

    def _generate_recommendations(self) -> dict:
        """
        Analyze results and generate threshold recommendations.

        Returns:
            Dictionary with recommended thresholds and justification
        """
        if not self.results:
            return {"error": "No results to analyze"}

        # Find thresholds with best balance
        # Goal: 95%+ pass rate, highest avg_score threshold possible
        viable_thresholds = [
            r for r in self.results if r["pass_rate"] >= 0.95
        ]

        if not viable_thresholds:
            # Relax to 90% pass rate
            viable_thresholds = [
                r for r in self.results if r["pass_rate"] >= 0.90
            ]

        if not viable_thresholds:
            return {
                "error": "No thresholds meet minimum 90% pass rate",
                "best_by_pass_rate": max(self.results, key=lambda r: r["pass_rate"]),
            }

        # Among viable options, pick highest MIN_AVG_SCORE (most strict)
        best_threshold = max(
            viable_thresholds,
            key=lambda r: r["thresholds"]["MIN_AVG_SCORE"],
        )

        # Current threshold performance
        current_result = next(
            (
                r
                for r in self.results
                if r["thresholds"] == self.CURRENT_THRESHOLDS
            ),
            None,
        )

        return {
            "recommended": best_threshold["thresholds"],
            "recommended_pass_rate": best_threshold["pass_rate"],
            "recommended_failures": best_threshold["failure_reasons"],
            "current_pass_rate": current_result["pass_rate"] if current_result else None,
            "current_failures": current_result["failure_reasons"] if current_result else None,
            "justification": self._generate_justification(best_threshold, current_result),
        }

    def _generate_justification(self, recommended: dict, current: dict | None) -> str:
        """Generate human-readable justification for recommendation."""
        lines = []

        rec_thresholds = recommended["thresholds"]
        rec_pass_rate = recommended["pass_rate"]

        lines.append(f"Recommended thresholds:")
        lines.append(f"  MIN_DOCS={rec_thresholds['MIN_DOCS']}")
        lines.append(f"  MIN_AVG_SCORE={rec_thresholds['MIN_AVG_SCORE']}")
        lines.append(f"  MIN_SOURCES={rec_thresholds['MIN_SOURCES']}")
        lines.append(f"")
        lines.append(f"Pass rate: {rec_pass_rate:.1%} ({recommended['passed']}/{recommended['passed'] + recommended['failed']} cases)")

        if current:
            cur_pass_rate = current["pass_rate"]
            if rec_pass_rate > cur_pass_rate:
                diff = (rec_pass_rate - cur_pass_rate) * 100
                lines.append(f"Improvement: +{diff:.1f}% pass rate vs current thresholds")
            elif rec_pass_rate == cur_pass_rate:
                lines.append("Same pass rate as current thresholds")
            else:
                diff = (cur_pass_rate - rec_pass_rate) * 100
                lines.append(f"Tradeoff: -{diff:.1f}% pass rate for higher precision")

        lines.append(f"")
        lines.append(f"Failure breakdown:")
        for reason, count in recommended["failure_reasons"].items():
            lines.append(f"  - {reason}: {count} cases")

        return "\n".join(lines)


def save_results(results: dict, output_path: str = "threshold_tuning_results.json"):
    """Save tuning results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def print_validation_results(results: dict):
    """Print validation results for current thresholds."""
    print("\n" + "=" * 80)
    print("CURRENT THRESHOLD VALIDATION")
    print("=" * 80)
    print()
    print("Current Production Thresholds:")
    print(f"  MIN_DOCS: {results['thresholds']['MIN_DOCS']}")
    print(f"  MIN_AVG_SCORE: {results['thresholds']['MIN_AVG_SCORE']}")
    print(f"  MIN_SOURCES: {results['thresholds']['MIN_SOURCES']}")
    print()
    print(f"Pass Rate: {results['pass_rate']:.1%} ({results['passed']}/{results['passed'] + results['failed']} cases)")
    print()
    print("Failure Breakdown:")
    for reason, count in results["failure_reasons"].items():
        print(f"  - {reason}: {count} cases")
    print()
    print("=" * 80)


def print_tuning_results(results: dict):
    """Print full tuning results."""
    print("\n" + "=" * 80)
    print("THRESHOLD TUNING RESULTS")
    print("=" * 80)
    print()
    print(f"Total combinations tested: {results['summary']['total_combinations']}")
    print(f"Test cases: {results['summary']['test_cases']}")
    print(f"Total time: {results['summary']['total_time_seconds']:.1f}s")
    print()

    if "error" in results["recommendations"]:
        print(f"Error: {results['recommendations']['error']}")
        return

    print("RECOMMENDED THRESHOLDS")
    print("-" * 80)
    print(results["recommendations"]["justification"])
    print()

    # Show top 5 by pass rate
    print("Top 5 Threshold Combinations (by pass rate):")
    print("-" * 80)
    sorted_results = sorted(results["all_results"], key=lambda r: r["pass_rate"], reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        t = r["thresholds"]
        print(f"{i}. MIN_DOCS={t['MIN_DOCS']}, MIN_AVG_SCORE={t['MIN_AVG_SCORE']}, "
              f"MIN_SOURCES={t['MIN_SOURCES']}")
        print(f"   Pass rate: {r['pass_rate']:.1%}, Failures: {r['failed']}")

    print()
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Tune evidence validation thresholds empirically"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate current thresholds only (no full tuning)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of test cases to use (default: all 50)",
    )

    args = parser.parse_args()

    tuner = ThresholdTuner(sample_size=args.sample_size)

    try:
        if args.validate:
            # Validation mode: check current thresholds only
            results = tuner.validate_current_thresholds()
            print_validation_results(results)

            # Save results
            save_results(
                {"validation": results},
                "threshold_validation_results.json",
            )

        else:
            # Full tuning mode
            print("=" * 80)
            print("THRESHOLD TUNING")
            print("=" * 80)
            print()
            print("This will test multiple threshold combinations to find the optimal")
            print("balance between pass rate (coverage) and precision (quality).")
            print()
            sample_size = args.sample_size or len(MEDICAL_TEST_CASES)
            print(f"Test cases: {sample_size}")
            print(f"Combinations: {len(tuner.THRESHOLDS_TO_TEST['MIN_DOCS']) * len(tuner.THRESHOLDS_TO_TEST['MIN_AVG_SCORE']) * len(tuner.THRESHOLDS_TO_TEST['MIN_SOURCES'])}")
            print()
            print("This will take several minutes...")
            print("=" * 80)
            print()

            results = tuner.run_full_tuning()
            print_tuning_results(results)

            # Save results
            save_results(results)

            print("\nResults saved to threshold_tuning_results.json")

    except KeyboardInterrupt:
        print("\n\nTuning interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTuning failed: {e}")
        logger.error(f"Tuning error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
