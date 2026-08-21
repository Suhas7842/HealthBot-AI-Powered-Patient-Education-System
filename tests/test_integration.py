"""Integration tests for agent workflow."""

import pytest
from unittest.mock import patch, Mock
from healthbot.agent_graph import run_agent_query, agent_app


@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests for full agent workflow."""

    @patch('healthbot.agent_graph.ChatGoogleGenerativeAI')
    @patch('healthbot.agent_graph.create_react_agent')
    def test_agent_workflow_end_to_end(self, mock_agent, mock_llm):
        """
        Integration test: Query → Agent → Tool → Result.

        Mocks LLM but tests real state management, safety, and tool coordination.
        """
        mock_executor = Mock()

        # Simulate agent calling medical_rag_search tool (real structure)
        mock_tool_message = Mock()
        mock_tool_message.tool_calls = [{"name": "medical_rag_search"}]

        # Final response from agent
        mock_final_message = Mock()
        mock_final_message.content = "Type 2 diabetes is characterized by insulin resistance..."
        mock_final_message.tool_calls = []

        mock_executor.invoke.return_value = {
            "messages": [mock_tool_message, mock_final_message]
        }
        mock_agent.return_value = mock_executor

        # Run query through full workflow
        result = run_agent_query("What causes Type 2 diabetes?")

        # Verify workflow completion
        assert result is not None
        assert "summary" in result
        assert result["summary"]

        # Verify tool tracking worked (tests Issue 1 fix)
        assert "tools_called" in result
        assert "medical_rag_search" in result["tools_called"]

        # Verify safety/disclaimer
        assert "disclaimer_shown" in result
        assert result["disclaimer_shown"] is True

    @patch('healthbot.agent_graph.ChatGoogleGenerativeAI')
    @patch('healthbot.agent_graph.create_react_agent')
    def test_agent_handles_multiple_tools(self, mock_agent, mock_llm):
        """
        Integration test: Agent can handle multiple tool calls in sequence.

        Tests multi-tool coordination capability.
        """
        mock_executor = Mock()

        # Simulate agent calling multiple tools
        mock_tool1 = Mock()
        mock_tool1.tool_calls = [{"name": "medical_calculator"}]

        mock_tool2 = Mock()
        mock_tool2.tool_calls = [{"name": "medical_rag_search"}]

        mock_final = Mock()
        mock_final.content = "Your BMI is 22.9 (normal range). Medical evidence shows..."
        mock_final.tool_calls = []

        mock_executor.invoke.return_value = {
            "messages": [mock_tool1, mock_tool2, mock_final]
        }
        mock_agent.return_value = mock_executor

        # Run query that would trigger multiple tools
        result = run_agent_query("What's my BMI if I'm 70kg and 1.75m, and is that healthy?")

        # Verify both tools were tracked
        assert result is not None
        assert "tools_called" in result
        assert len(result["tools_called"]) == 2
        assert "medical_calculator" in result["tools_called"]
        assert "medical_rag_search" in result["tools_called"]
