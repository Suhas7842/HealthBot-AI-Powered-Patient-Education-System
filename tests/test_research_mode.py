"""
Tests for Medical Research Mode (Phase 4 enhancement).

Validates that:
1. Research queries are detected correctly
2. Research prompt is used for research queries
3. Normal queries remain on normal path
4. Multi-tool orchestration works
5. System handles failures gracefully
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from healthbot.routing import QueryClassifier
from healthbot.agent_graph import run_agent_query


class TestResearchDetection:
    """Test research query detection logic."""

    def test_normal_query_detection(self):
        """Normal medical questions should NOT trigger research mode."""
        classifier = QueryClassifier()

        normal_queries = [
            "What is Type 2 diabetes?",
            "What are the symptoms of hypertension?",
            "How is asthma treated?",
            "What causes heart disease?",
        ]

        for query in normal_queries:
            is_research = classifier.is_research_query(query)
            assert is_research is False, f"Query '{query}' should be NORMAL, not research"

    def test_research_query_detection(self):
        """Research-style queries should trigger research mode."""
        classifier = QueryClassifier()

        research_queries = [
            "What does recent research say about diabetes risk factors?",
            "What are the major risk factors for Type 2 diabetes and which are modifiable?",
            "Compare recent studies on hypertension treatment",
            "What is the evidence for metformin in diabetes prevention?",
            "Compare established knowledge with recent literature on COPD",
        ]

        for query in research_queries:
            is_research = classifier.is_research_query(query)
            assert is_research is True, f"Query '{query}' should be RESEARCH, not normal"

    def test_research_keywords(self):
        """Specific research keywords should trigger research mode."""
        classifier = QueryClassifier()

        # These specific patterns should always trigger research mode
        assert classifier.is_research_query("recent research on diabetes") is True
        assert classifier.is_research_query("what does research say") is True
        assert classifier.is_research_query("evidence for statins") is True
        assert classifier.is_research_query("modifiable risk factors") is True
        assert classifier.is_research_query("compare studies") is True


class TestResearchModeWorkflow:
    """Test that research mode uses correct workflow."""

    @patch('healthbot.agent_graph.create_react_agent')
    @patch.dict('sys.modules', {'healthbot.agent_graph': MagicMock()})
    def test_research_query_uses_research_prompt(self, mock_create_agent):
        """Research queries should use research prompt, not standard prompt."""
        # Mock the agent executor
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [
                MagicMock(content="Research answer with citations", tool_calls=[])
            ]
        }
        mock_create_agent.return_value = mock_agent

        # Run research query
        query = "What does recent research say about diabetes risk factors?"
        result = run_agent_query(query)

        # Check that research mode was detected
        assert result.get("query_type") == "research"

        # Check that agent was invoked
        assert mock_agent.invoke.called

        # Get the messages passed to agent
        agent_input = mock_agent.invoke.call_args[0][0]
        messages = agent_input["messages"]

        # System message should contain research-specific instructions
        system_message = messages[0].content
        assert "research" in system_message.lower() or "decompose" in system_message.lower()

    @patch('healthbot.agent_graph.create_react_agent')
    @patch.dict('sys.modules', {'healthbot.agent_graph': MagicMock()})
    def test_normal_query_uses_standard_prompt(self, mock_create_agent):
        """Normal queries should use standard prompt, not research prompt."""
        # Mock the agent executor
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [
                MagicMock(content="Standard answer", tool_calls=[])
            ]
        }
        mock_create_agent.return_value = mock_agent

        # Run normal query
        query = "What is Type 2 diabetes?"
        result = run_agent_query(query)

        # Check that normal mode was detected
        assert result.get("query_type") == "normal"

        # Check that agent was invoked
        assert mock_agent.invoke.called

        # Get the messages passed to agent
        agent_input = mock_agent.invoke.call_args[0][0]
        messages = agent_input["messages"]

        # System message should be standard agent prompt (NOT research prompt)
        system_message = messages[0].content
        # Research prompt has "DECOMPOSE" in it, standard prompt doesn't
        assert "DECOMPOSE" not in system_message


class TestResearchModeExecution:
    """Test actual execution behavior (requires mocking agent tools)."""

    @patch('healthbot.agent_graph.create_react_agent')
    @patch.dict('sys.modules', {'healthbot.agent_graph': MagicMock()})
    def test_research_mode_tracks_query_type(self, mock_create_agent):
        """query_type should be set in state for research queries."""
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [
                MagicMock(content="Answer", tool_calls=[])
            ]
        }
        mock_create_agent.return_value = mock_agent

        # Test research query
        result = run_agent_query("What does recent research say about diabetes?")
        assert result["query_type"] == "research"

        # Test normal query
        result = run_agent_query("What is diabetes?")
        assert result["query_type"] == "normal"

    @patch('healthbot.agent_graph.create_react_agent')
    @patch.dict('sys.modules', {'healthbot.agent_graph': MagicMock()})
    def test_research_mode_allows_multiple_tools(self, mock_create_agent):
        """Research mode should allow agent to call multiple tools."""
        # Mock agent that calls multiple tools
        mock_tool_call_rag = {"name": "medical_rag_search"}
        mock_tool_call_pubmed = {"name": "pubmed_api_search"}

        mock_msg1 = MagicMock()
        mock_msg1.tool_calls = [mock_tool_call_rag]
        mock_msg1.content = ""

        mock_msg2 = MagicMock()
        mock_msg2.tool_calls = [mock_tool_call_pubmed]
        mock_msg2.content = ""

        mock_msg3 = MagicMock()
        mock_msg3.tool_calls = []
        mock_msg3.content = "Synthesized research answer"

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [mock_msg1, mock_msg2, mock_msg3]
        }
        mock_create_agent.return_value = mock_agent

        # Run research query
        result = run_agent_query("What does recent research say about diabetes risk factors?")

        # Should have called both tools
        assert "medical_rag_search" in result["tools_called"]
        assert "pubmed_api_search" in result["tools_called"]
        assert len(result["tools_called"]) >= 2


class TestResearchModeFailureHandling:
    """Test failure handling in research mode."""

    @patch('healthbot.agent_graph.create_react_agent')
    @patch.dict('sys.modules', {'healthbot.agent_graph': MagicMock()})
    def test_research_mode_handles_empty_response(self, mock_create_agent):
        """Research mode should handle empty agent response gracefully."""
        # Mock agent with empty response
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": []}
        mock_create_agent.return_value = mock_agent

        # Should not crash
        result = run_agent_query("What does recent research say about diabetes?")
        assert result["query_type"] == "research"
        assert "summary" in result
        assert "No response generated" in result["summary"]

    @patch('healthbot.agent_graph.create_react_agent')
    @patch.dict('sys.modules', {'healthbot.agent_graph': MagicMock()})
    def test_research_mode_adds_disclaimer(self, mock_create_agent):
        """Research mode should add medical disclaimer."""
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [
                MagicMock(content="Research findings", tool_calls=[])
            ]
        }
        mock_create_agent.return_value = mock_agent

        # Run research query
        result = run_agent_query("What does recent research say about diabetes?")

        # Should include disclaimer
        assert "Medical Disclaimer" in result["summary"]
        assert result["disclaimer_shown"] is True


@pytest.mark.integration
class TestResearchModeIntegration:
    """
    Integration tests for research mode.

    These tests are marked as integration and should be run separately
    when you have LLM access and want to verify end-to-end behavior.
    """

    def test_research_query_integration(self):
        """
        Full integration test with real agent.

        This test requires LLM access and is skipped by default.
        Run with: pytest tests/test_research_mode.py -m integration
        """
        pytest.skip("Integration test - run only when LLM access available")

        query = "What are the major risk factors for Type 2 diabetes and which are modifiable?"
        result = run_agent_query(query)

        # Verify research mode was used
        assert result["query_type"] == "research"

        # Verify tools were called
        assert len(result["tools_called"]) > 0

        # Verify response has content
        assert len(result["summary"]) > 100

        # Verify disclaimer
        assert "Medical Disclaimer" in result["summary"]
