"""
Tests for agent tools (Phase 4).

Tests LangChain tool wrappers that expose custom tools to LLM agent.
"""

import pytest
from unittest.mock import Mock, patch

from healthbot.agent_tools import (
    medical_rag_search,
    medical_calculator,
    pubmed_api_search,
    web_search,
    get_all_tools,
    get_tool_descriptions,
    TOOL_METADATA,
)


class TestMedicalRAGSearchTool:
    """Test medical_rag_search tool wrapper."""

    @patch('healthbot.agent_tools.ToolSelector')
    def test_medical_rag_search_success(self, mock_tool_selector_class):
        """Test medical_rag_search returns results."""
        mock_selector = Mock()
        mock_selector.select_and_search.return_value = {
            "success": True,
            "documents": [
                {"text": "Diabetes doc 1", "metadata": {"pmid": "12345"}},
                {"text": "Diabetes doc 2", "metadata": {"pmid": "67890"}},
            ],
            "method": "hybrid",
        }
        mock_tool_selector_class.return_value = mock_selector

        result = medical_rag_search.invoke({"query": "What is diabetes?", "k": 5})

        assert result["success"]
        assert result["count"] == 2
        assert result["method"] == "hybrid"
        assert result["source"] == "local_knowledge_base_716_articles"

    @patch('healthbot.agent_tools.ToolSelector')
    def test_medical_rag_search_with_custom_k(self, mock_tool_selector_class):
        """Test medical_rag_search with custom k parameter."""
        mock_selector = Mock()
        mock_selector.select_and_search.return_value = {
            "success": True,
            "documents": [{"text": "Doc"}] * 3,
            "method": "semantic",
        }
        mock_tool_selector_class.return_value = mock_selector

        result = medical_rag_search.invoke({"query": "hypertension", "k": 3})

        assert result["success"]
        assert result["count"] == 3
        mock_selector.select_and_search.assert_called_once_with("hypertension", k=3)

    def test_medical_rag_search_is_langchain_tool(self):
        """Test medical_rag_search has LangChain tool attributes."""
        assert hasattr(medical_rag_search, 'name')
        assert hasattr(medical_rag_search, 'description')
        assert medical_rag_search.name == "medical_rag_search"
        assert "knowledge base" in medical_rag_search.description.lower()


class TestMedicalCalculatorTool:
    """Test medical_calculator tool wrapper."""

    @patch('healthbot.agent_tools.medical_calculator_tool')
    def test_bmi_calculation(self, mock_calc):
        """Test BMI calculation through tool wrapper."""
        mock_calc.return_value = {
            "success": True,
            "bmi": 22.9,
            "category": "normal",
        }

        result = medical_calculator.invoke({
            "calculation_type": "bmi",
            "weight_kg": 70.0,
            "height_m": 1.75,
        })

        assert result["success"]
        assert result["bmi"] == 22.9
        assert result["source"] == "medical_calculator_tool"
        assert result["tool_type"] == "computation"

    @patch('healthbot.agent_tools.medical_calculator_tool')
    def test_dosage_calculation(self, mock_calc):
        """Test dosage calculation through tool wrapper."""
        mock_calc.return_value = {
            "success": True,
            "total_dose_mg": 350.0,
        }

        result = medical_calculator.invoke({
            "calculation_type": "dosage",
            "weight_kg": 70.0,
            "dose_per_kg": 5.0,
        })

        assert result["success"]
        assert result["total_dose_mg"] == 350.0
        assert result["source"] == "medical_calculator_tool"

    @patch('healthbot.agent_tools.medical_calculator_tool')
    def test_creatinine_clearance_calculation(self, mock_calc):
        """Test creatinine clearance calculation through tool wrapper."""
        mock_calc.return_value = {
            "success": True,
            "crcl_ml_min": 97.2,
            "interpretation": "Normal kidney function",
        }

        result = medical_calculator.invoke({
            "calculation_type": "creatinine_clearance",
            "age": 40,
            "weight_kg": 70.0,
            "serum_creatinine_mg_dl": 1.0,
            "sex": "male",
        })

        assert result["success"]
        assert result["crcl_ml_min"] == 97.2

    def test_medical_calculator_is_langchain_tool(self):
        """Test medical_calculator has LangChain tool attributes."""
        assert hasattr(medical_calculator, 'name')
        assert hasattr(medical_calculator, 'description')
        assert medical_calculator.name == "medical_calculator"
        assert "calculation" in medical_calculator.description.lower()


class TestPubMedAPISearchTool:
    """Test pubmed_api_search tool wrapper."""

    @patch('healthbot.agent_tools.pubmed_api_tool')
    def test_pubmed_search_success(self, mock_pubmed):
        """Test PubMed search returns papers."""
        mock_pubmed.return_value = {
            "success": True,
            "count": 2,
            "papers": [
                {
                    "pmid": "12345678",
                    "title": "Diabetes Research Paper 1",
                    "abstract": "Abstract text...",
                },
                {
                    "pmid": "87654321",
                    "title": "Diabetes Research Paper 2",
                    "abstract": "Abstract text...",
                },
            ],
        }

        result = pubmed_api_search.invoke({
            "query": "diabetes treatment",
            "max_results": 2,
        })

        assert result["success"]
        assert result["count"] == 2
        assert len(result["papers"]) == 2
        assert result["source"] == "pubmed_api_35M_articles"
        assert result["tool_type"] == "external_api"

    @patch('healthbot.agent_tools.pubmed_api_tool')
    def test_pubmed_search_no_results(self, mock_pubmed):
        """Test PubMed search with no results."""
        mock_pubmed.return_value = {
            "success": True,
            "count": 0,
            "papers": [],
        }

        result = pubmed_api_search.invoke({
            "query": "nonexistent medical condition xyz",
            "max_results": 5,
        })

        assert result["success"]
        assert result["count"] == 0
        assert result["papers"] == []

    @patch('healthbot.agent_tools.pubmed_api_tool')
    def test_pubmed_search_api_error(self, mock_pubmed):
        """Test PubMed search handles API errors."""
        mock_pubmed.return_value = {
            "success": False,
            "error": "API connection failed",
            "papers": [],
        }

        result = pubmed_api_search.invoke({
            "query": "diabetes",
            "max_results": 5,
        })

        assert not result["success"]
        assert "error" in result

    def test_pubmed_api_search_is_langchain_tool(self):
        """Test pubmed_api_search has LangChain tool attributes."""
        assert hasattr(pubmed_api_search, 'name')
        assert hasattr(pubmed_api_search, 'description')
        assert pubmed_api_search.name == "pubmed_api_search"
        assert "pubmed" in pubmed_api_search.description.lower()


class TestWebSearchTool:
    """Test web_search tool wrapper."""

    @patch('healthbot.agent_tools.TavilyTool')
    def test_web_search_success(self, mock_tavily_class):
        """Test web search returns results."""
        mock_tavily = Mock()
        mock_tavily.search.return_value = {
            "results": [
                {"url": "https://example.com/1", "title": "Health Article 1"},
                {"url": "https://example.com/2", "title": "Health Article 2"},
            ]
        }
        mock_tavily_class.return_value = mock_tavily

        result = web_search.invoke({
            "query": "COVID-19 treatment",
            "max_results": 2,
        })

        assert result["success"]
        assert result["count"] == 2
        assert len(result["results"]) == 2
        assert result["source"] == "tavily_web_search"
        assert result["tool_type"] == "web_search"

    @patch('healthbot.agent_tools.TavilyTool')
    def test_web_search_error(self, mock_tavily_class):
        """Test web search handles errors gracefully."""
        mock_tavily_class.side_effect = Exception("Tavily API error")

        result = web_search.invoke({
            "query": "health news",
            "max_results": 5,
        })

        assert not result["success"]
        assert "error" in result
        assert result["results"] == []

    def test_web_search_is_langchain_tool(self):
        """Test web_search has LangChain tool attributes."""
        assert hasattr(web_search, 'name')
        assert hasattr(web_search, 'description')
        assert web_search.name == "web_search"
        assert "web" in web_search.description.lower()


class TestToolUtilities:
    """Test utility functions for tools."""

    def test_get_all_tools_returns_list(self):
        """Test get_all_tools returns list of tools."""
        tools = get_all_tools()

        assert isinstance(tools, list)
        assert len(tools) == 4  # 4 tools: RAG, calculator, PubMed, web

    def test_get_all_tools_returns_langchain_tools(self):
        """Test all tools have LangChain tool attributes."""
        tools = get_all_tools()

        for tool in tools:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'invoke')

    def test_get_tool_descriptions(self):
        """Test get_tool_descriptions returns descriptions."""
        descriptions = get_tool_descriptions()

        assert isinstance(descriptions, dict)
        assert len(descriptions) == 4
        assert "medical_rag_search" in descriptions
        assert "medical_calculator" in descriptions
        assert "pubmed_api_search" in descriptions
        assert "web_search" in descriptions

    def test_tool_metadata_structure(self):
        """Test TOOL_METADATA has expected structure."""
        assert isinstance(TOOL_METADATA, dict)
        assert len(TOOL_METADATA) == 4

        for tool_name, metadata in TOOL_METADATA.items():
            assert "category" in metadata
            assert "scope" in metadata
            assert "use_cases" in metadata

    def test_tool_metadata_categories(self):
        """Test tool metadata has correct categories."""
        assert TOOL_METADATA["medical_rag_search"]["category"] == "retrieval"
        assert TOOL_METADATA["medical_calculator"]["category"] == "computation"
        assert TOOL_METADATA["pubmed_api_search"]["category"] == "retrieval"
        assert TOOL_METADATA["web_search"]["category"] == "web_search"


class TestToolInvocation:
    """Test tools can be invoked with LangChain API."""

    @patch('healthbot.agent_tools.ToolSelector')
    def test_tool_invoke_with_dict(self, mock_tool_selector_class):
        """Test tool can be invoked with dictionary input."""
        mock_selector = Mock()
        mock_selector.select_and_search.return_value = {
            "success": True,
            "documents": [{"text": "Doc"}],
            "method": "semantic",
        }
        mock_tool_selector_class.return_value = mock_selector

        # LangChain tools accept dict input
        result = medical_rag_search.invoke({"query": "test", "k": 5})

        assert "success" in result

    @patch('healthbot.agent_tools.medical_calculator_tool')
    def test_calculator_invoke_with_partial_params(self, mock_calc):
        """Test calculator handles optional parameters."""
        mock_calc.return_value = {
            "success": True,
            "bmi": 22.9,
        }

        # Only provide BMI-relevant params
        result = medical_calculator.invoke({
            "calculation_type": "bmi",
            "weight_kg": 70.0,
            "height_m": 1.75,
            # Other params (age, sex, etc.) are None
        })

        assert result["success"]


# Mark all tests as Phase 4 agent tool tests
pytestmark = pytest.mark.agent_tools
