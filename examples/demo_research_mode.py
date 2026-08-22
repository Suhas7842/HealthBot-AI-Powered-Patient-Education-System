"""
Demo: Medical Research Mode

Demonstrates the difference between normal and research query workflows.

Usage:
    python demo_research_mode.py
"""

from healthbot.routing import QueryClassifier
from healthbot.agent_graph import run_agent_query


def demo_research_detection():
    """Show how research queries are detected."""
    print("=" * 70)
    print("RESEARCH QUERY DETECTION DEMO")
    print("=" * 70)

    classifier = QueryClassifier()

    test_queries = [
        ("What is Type 2 diabetes?", False),
        ("What does recent research say about diabetes risk factors?", True),
        ("What are the symptoms of hypertension?", False),
        ("What are the major risk factors for heart disease and which are modifiable?", True),
        ("Compare recent studies on asthma treatment", True),
        ("How is COPD treated?", False),
    ]

    print("\nQuery Classification Results:\n")
    for query, expected_research in test_queries:
        is_research = classifier.is_research_query(query)
        query_type = "RESEARCH" if is_research else "NORMAL"
        status = "[OK]" if is_research == expected_research else "[FAIL]"

        print(f"{status:6} {query_type:10} | {query}")

    print("\n" + "=" * 70)


def demo_research_mode_workflow():
    """
    Demonstrate research mode workflow.

    NOTE: This requires LLM access and is for demonstration purposes only.
    """
    print("\n\nRESEARCH MODE WORKFLOW DEMO")
    print("=" * 70)

    # Example 1: Normal query
    print("\n1. NORMAL QUERY:")
    print("-" * 70)
    normal_query = "What is Type 2 diabetes?"
    print(f"Query: {normal_query}")
    print("Expected: Single-step RAG retrieval\n")

    try:
        result = run_agent_query(normal_query)
        print(f"Query Type: {result.get('query_type', 'unknown')}")
        print(f"Tools Called: {result.get('tools_called', [])}")
        print(f"Summary Length: {len(result.get('summary', ''))} chars")
        print("[OK] Normal query workflow completed")
    except Exception as e:
        print(f"⚠ Skipped (requires LLM): {e}")

    # Example 2: Research query
    print("\n\n2. RESEARCH QUERY:")
    print("-" * 70)
    research_query = "What does recent research say about diabetes risk factors?"
    print(f"Query: {research_query}")
    print("Expected: Multi-step evidence synthesis (RAG + PubMed)\n")

    try:
        result = run_agent_query(research_query)
        print(f"Query Type: {result.get('query_type', 'unknown')}")
        print(f"Tools Called: {result.get('tools_called', [])}")
        print(f"Summary Length: {len(result.get('summary', ''))} chars")
        print("[OK] Research query workflow completed")
    except Exception as e:
        print(f"⚠ Skipped (requires LLM): {e}")

    print("\n" + "=" * 70)


def demo_comparison():
    """Show conceptual difference between modes."""
    print("\n\nMODE COMPARISON")
    print("=" * 70)

    print("\nNORMAL MODE:")
    print("  Flow: Query -> RAG -> Answer")
    print("  Use case: Simple medical questions")
    print("  Example: 'What is diabetes?'")

    print("\nRESEARCH MODE:")
    print("  Flow: Query -> Decompose -> RAG + PubMed -> Synthesize -> Answer")
    print("  Use case: Complex questions requiring multiple evidence sources")
    print("  Example: 'What does recent research say about diabetes risk factors?'")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run detection demo (no LLM required)
    demo_research_detection()

    # Show conceptual comparison
    demo_comparison()

    # Run workflow demo (requires LLM)
    print("\n\nWARNING: The workflow demo requires LLM access.")
    print("It will attempt to run but will skip if LLM is unavailable.\n")

    response = input("Run workflow demo? (y/n): ")
    if response.lower() == 'y':
        demo_research_mode_workflow()
    else:
        print("Skipping workflow demo.")

    print("\n[OK] Demo complete!")
