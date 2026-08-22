"""
Live Agent Test - Demonstrates Phase 4 Agent with Real Tool Calls

Tests the agent with various queries to show tool selection in action.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from healthbot.agent_graph import run_agent_query


def test_calculator_query():
    """Test query that should trigger medical calculator."""
    print("\n" + "="*70)
    print("TEST 1: Calculator Query")
    print("="*70)

    query = "What's my BMI if I'm 70kg and 1.75m tall?"
    print(f"\nQuery: {query}")
    print("\nExpected: Agent should call medical_calculator tool")
    print("\nRunning agent...")

    try:
        result = run_agent_query(query)
        print(f"\n✅ Success!")
        print(f"\nTools Called: {result.get('tools_called', [])}")
        print(f"\nResponse Preview:")
        print(result.get('summary', 'No summary')[:300] + "...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_medical_query():
    """Test query that should trigger medical RAG."""
    print("\n" + "="*70)
    print("TEST 2: Medical Knowledge Query")
    print("="*70)

    query = "What causes Type 2 diabetes?"
    print(f"\nQuery: {query}")
    print("\nExpected: Agent should call medical_rag_search tool")
    print("\nRunning agent...")

    try:
        result = run_agent_query(query)
        print(f"\n✅ Success!")
        print(f"\nTools Called: {result.get('tools_called', [])}")
        print(f"\nResponse Preview:")
        print(result.get('summary', 'No summary')[:300] + "...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_multi_tool_query():
    """Test query that should trigger multiple tools."""
    print("\n" + "="*70)
    print("TEST 3: Multi-Tool Query")
    print("="*70)

    query = "Calculate my BMI for 85kg and 1.80m, and tell me if that's healthy"
    print(f"\nQuery: {query}")
    print("\nExpected: Agent should call medical_calculator + medical_rag_search")
    print("\nRunning agent...")

    try:
        result = run_agent_query(query)
        print(f"\n✅ Success!")
        print(f"\nTools Called: {result.get('tools_called', [])}")
        print(f"\nResponse Preview:")
        print(result.get('summary', 'No summary')[:300] + "...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_simple_info_query():
    """Test simple informational query."""
    print("\n" + "="*70)
    print("TEST 4: Simple Information Query")
    print("="*70)

    query = "What is asthma?"
    print(f"\nQuery: {query}")
    print("\nExpected: Agent should call medical_rag_search")
    print("\nRunning agent...")

    try:
        result = run_agent_query(query)
        print(f"\n✅ Success!")
        print(f"\nTools Called: {result.get('tools_called', [])}")
        print(f"\nResponse Preview:")
        print(result.get('summary', 'No summary')[:300] + "...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HEALTHBOT PHASE 4 AGENT - LIVE TEST")
    print("="*70)
    print("\nThis demonstrates the LLM agent selecting and calling YOUR custom tools")
    print("based on query analysis (not hardcoded routing).")
    print("\nYOUR Tools:")
    print("  1. medical_calculator - BMI, dosage, kidney function")
    print("  2. medical_rag_search - Local knowledge base (716 articles)")
    print("  3. pubmed_api_search - Search 35M+ papers")
    print("  4. web_search - Current health news")

    # Run tests
    test_calculator_query()
    test_medical_query()
    test_multi_tool_query()
    test_simple_info_query()

    print("\n" + "="*70)
    print("AGENT TESTS COMPLETE")
    print("="*70)
    print("\n✨ This demonstrates GenAI orchestration:")
    print("   - LLM DECIDES which tools to call (not hardcoded)")
    print("   - Tools are YOUR infrastructure (calculator, retriever, etc.)")
    print("   - Agent can use multiple tools per query")
    print("   - This is tool engineering, not prompt engineering")
