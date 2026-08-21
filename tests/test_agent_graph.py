"""
Tests for agent graph (Phase 4).

Tests the ReAct agent workflow with mocked LLM and tools.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from healthbot.agent_graph import (
    create_agent_workflow,
    safety_check_node,
    route_safety,
    agent_node,
    run_agent_query,
)


class TestSafetyCheckNode:
    """Test safety check node."""

    @patch('healthbot.agent_graph.check_emergency')
    def test_safety_check_emergency_detected(self, mock_check):
        """Test safety check detects emergency."""
        mock_check.return_value = True

        state = {"topic": "I'm having chest pain"}
        result = safety_check_node(state)

        assert result["emergency_detected"]
        assert "911" in result["summary"]

    @patch('healthbot.agent_graph.check_emergency')
    def test_safety_check_no_emergency(self, mock_check):
        """Test safety check with normal query."""
        mock_check.return_value = False

        state = {"topic": "What is diabetes?"}
        result = safety_check_node(state)

        assert not result["emergency_detected"]
        assert result["summary"] is None


class TestRouteSafety:
    """Test safety routing logic."""

    def test_route_emergency(self):
        """Test routing when emergency detected."""
        state = {"emergency_detected": True}
        route = route_safety(state)
        assert route == "emergency"

    def test_route_safe(self):
        """Test routing when no emergency."""
        state = {"emergency_detected": False}
        route = route_safety(state)
        assert route == "safe"

    def test_route_missing_flag(self):
        """Test routing with missing emergency flag."""
        state = {}
        route = route_safety(state)
        assert route == "safe"  # Default to safe


class TestAgentNode:
    """Test agent node with mocked LLM."""

    @patch('healthbot.agent_graph.create_react_agent')
    @patch('healthbot.agent_graph.ChatGoogleGenerativeAI')
    @patch('healthbot.agent_graph.get_all_tools')
    def test_agent_node_generates_response(self, mock_tools, mock_llm, mock_agent):
        """Test agent node generates response."""
        # Mock tools
        mock_tools.return_value = []

        # Mock agent executor
        mock_executor = Mock()
        mock_message = Mock()
        mock_message.content = "Test agent response"
        mock_message.tool_calls = []  # No tools called in this test
        mock_executor.invoke.return_value = {
            "messages": [mock_message]
        }
        mock_agent.return_value = mock_executor

        state = {
            "topic": "What is diabetes?",
            "messages": [],
            "emergency_detected": False,
        }

        result = agent_node(state)

        assert "summary" in result
        assert "Test agent response" in result["summary"]
        assert "Medical Disclaimer" in result["summary"]
        assert result["disclaimer_shown"]

    @patch('healthbot.agent_graph.create_react_agent')
    @patch('healthbot.agent_graph.ChatGoogleGenerativeAI')
    @patch('healthbot.agent_graph.get_all_tools')
    def test_agent_node_tracks_tools_called(self, mock_tools, mock_llm, mock_agent):
        """Test agent node tracks which tools were called."""
        mock_tools.return_value = []

        # Mock agent with tool calls
        mock_executor = Mock()
        mock_message = Mock()
        mock_message.content = "Response"
        # Match real LangChain AIMessage structure
        mock_message.tool_calls = [
            {"name": "medical_rag_search"},
            {"name": "medical_calculator"},
        ]
        mock_executor.invoke.return_value = {
            "messages": [mock_message]
        }
        mock_agent.return_value = mock_executor

        state = {
            "topic": "Calculate my BMI",
            "messages": [],
            "emergency_detected": False,
        }

        result = agent_node(state)

        assert "tools_called" in result
        assert "medical_rag_search" in result["tools_called"]
        assert "medical_calculator" in result["tools_called"]


class TestAgentWorkflow:
    """Test complete agent workflow."""

    def test_create_agent_workflow(self):
        """Test agent workflow creation."""
        workflow = create_agent_workflow()
        assert workflow is not None

    @patch('healthbot.agent_graph.check_emergency')
    @patch('healthbot.agent_graph.create_react_agent')
    @patch('healthbot.agent_graph.ChatGoogleGenerativeAI')
    @patch('healthbot.agent_graph.get_all_tools')
    def test_workflow_emergency_path(self, mock_tools, mock_llm, mock_agent, mock_check):
        """Test workflow handles emergency path."""
        # Mock emergency detection
        mock_check.return_value = True

        workflow = create_agent_workflow()
        result = workflow.invoke({
            "topic": "chest pain",
            "patient_level": "beginner",
            "messages": [],
            "summary": None,
            "emergency_detected": False,
            "disclaimer_shown": False,
            "tools_called": [],
            "tool_results": [],
            "tool_call_trace": [],
            "agent_synthesis": None,
        })

        assert result["emergency_detected"]
        # Workflow should end at emergency, not call agent
        mock_agent.assert_not_called()


class TestRunAgentQuery:
    """Test convenience function for running queries."""

    @patch('healthbot.agent_graph.agent_app')
    def test_run_agent_query(self, mock_app):
        """Test run_agent_query convenience function."""
        mock_app.invoke.return_value = {
            "summary": "Test response",
            "tools_called": ["medical_rag_search"],
        }

        result = run_agent_query("What is diabetes?")

        assert "summary" in result
        assert result["summary"] == "Test response"
        mock_app.invoke.assert_called_once()

    @patch('healthbot.agent_graph.agent_app')
    def test_run_agent_query_with_patient_level(self, mock_app):
        """Test run_agent_query with patient level."""
        mock_app.invoke.return_value = {"summary": "Response"}

        result = run_agent_query("diabetes", patient_level="advanced")

        # Check that patient_level was passed in state
        call_args = mock_app.invoke.call_args[0][0]
        assert call_args["patient_level"] == "advanced"


# Mark all tests as Phase 4 agent graph tests
pytestmark = pytest.mark.agent_graph
