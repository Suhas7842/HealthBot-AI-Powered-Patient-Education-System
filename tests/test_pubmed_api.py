"""
Tests for PubMed API integration (Phase 4).

Tests PubMed E-utilities search with mocking to avoid real API calls during tests.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

# Try to import PubMed tools
try:
    from healthbot.tools.pubmed_api import (
        PubMedClient,
        PubMedAPIError,
        search_pubmed,
        pubmed_api_tool,
        BIOPYTHON_AVAILABLE,
    )
    PUBMED_AVAILABLE = True
except ImportError:
    PUBMED_AVAILABLE = False


@pytest.mark.skipif(not PUBMED_AVAILABLE, reason="BioPython not installed")
class TestPubMedClient:
    """Test PubMedClient class."""

    def test_client_initialization_valid_email(self):
        """Test client initializes with valid email."""
        client = PubMedClient(email="test@example.com")
        assert client is not None
        assert client.rate_limit == 0.34

    def test_client_initialization_invalid_email(self):
        """Test client rejects placeholder email."""
        with pytest.raises(ValueError, match="Valid email required"):
            PubMedClient(email="your_email@example.com")

    def test_client_initialization_empty_email(self):
        """Test client rejects empty email."""
        with pytest.raises(ValueError, match="Valid email required"):
            PubMedClient(email="")

    @patch('healthbot.tools.pubmed_api.Entrez')
    def test_search_returns_pmids(self, mock_entrez):
        """Test search returns list of PMIDs."""
        # Mock Entrez.esearch response
        mock_handle = MagicMock()
        mock_handle.__enter__ = Mock(return_value=mock_handle)
        mock_handle.__exit__ = Mock(return_value=False)

        mock_entrez.esearch.return_value = mock_handle
        mock_entrez.read.return_value = {"IdList": ["12345678", "87654321"]}

        client = PubMedClient(email="test@example.com")
        pmids = client.search("diabetes", max_results=2)

        assert len(pmids) == 2
        assert "12345678" in pmids
        assert "87654321" in pmids

    @patch('healthbot.tools.pubmed_api.Entrez')
    def test_search_no_results(self, mock_entrez):
        """Test search with no results returns empty list."""
        mock_handle = MagicMock()
        mock_handle.__enter__ = Mock(return_value=mock_handle)
        mock_handle.__exit__ = Mock(return_value=False)

        mock_entrez.esearch.return_value = mock_handle
        mock_entrez.read.return_value = {"IdList": []}

        client = PubMedClient(email="test@example.com")
        pmids = client.search("nonexistent condition xyz", max_results=5)

        assert pmids == []

    @patch('healthbot.tools.pubmed_api.Entrez')
    def test_search_api_error(self, mock_entrez):
        """Test search handles API errors gracefully."""
        mock_entrez.esearch.side_effect = Exception("API connection error")

        client = PubMedClient(email="test@example.com")

        with pytest.raises(PubMedAPIError, match="PubMed search failed"):
            client.search("diabetes", max_results=5)

    @patch('healthbot.tools.pubmed_api.Entrez')
    def test_fetch_empty_pmid_list(self, mock_entrez):
        """Test fetch with empty PMID list returns empty list."""
        client = PubMedClient(email="test@example.com")
        articles = client.fetch([])

        assert articles == []
        # Should not call Entrez.efetch
        mock_entrez.efetch.assert_not_called()

    @patch('healthbot.tools.pubmed_api.Entrez')
    def test_search_and_fetch_success(self, mock_entrez):
        """Test complete search and fetch workflow."""
        # Mock search
        search_handle = MagicMock()
        search_handle.__enter__ = Mock(return_value=search_handle)
        search_handle.__exit__ = Mock(return_value=False)
        mock_entrez.esearch.return_value = search_handle
        mock_entrez.read.return_value = {"IdList": ["12345678"]}

        # Mock fetch with minimal XML
        fetch_handle = MagicMock()
        fetch_handle.read.return_value = b"""<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345678</PMID>
                    <Article>
                        <ArticleTitle>Test Article Title</ArticleTitle>
                        <Abstract>
                            <AbstractText>Test abstract text.</AbstractText>
                        </Abstract>
                        <Journal>
                            <Title>Test Journal</Title>
                        </Journal>
                        <PubDate>
                            <Year>2024</Year>
                            <Month>Jan</Month>
                        </PubDate>
                        <AuthorList>
                            <Author>
                                <LastName>Smith</LastName>
                                <Initials>J</Initials>
                            </Author>
                        </AuthorList>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>"""
        mock_entrez.efetch.return_value = fetch_handle

        client = PubMedClient(email="test@example.com")
        result = client.search_and_fetch("diabetes", max_results=1)

        assert result["success"]
        assert result["count"] == 1
        assert len(result["papers"]) == 1

        paper = result["papers"][0]
        assert paper["pmid"] == "12345678"
        assert paper["title"] == "Test Article Title"
        assert "Test abstract" in paper["abstract"]
        assert paper["journal"] == "Test Journal"
        assert "Smith" in paper["authors"][0]

    @patch('healthbot.tools.pubmed_api.Entrez')
    def test_rate_limiting(self, mock_entrez):
        """Test rate limiting enforces delay between requests."""
        import time

        mock_handle = MagicMock()
        mock_handle.__enter__ = Mock(return_value=mock_handle)
        mock_handle.__exit__ = Mock(return_value=False)
        mock_entrez.esearch.return_value = mock_handle
        mock_entrez.read.return_value = {"IdList": ["12345678"]}

        client = PubMedClient(email="test@example.com", rate_limit=0.1)

        start_time = time.time()
        client.search("diabetes", max_results=1)
        client.search("hypertension", max_results=1)
        elapsed = time.time() - start_time

        # Should take at least rate_limit seconds (0.1s)
        assert elapsed >= 0.1


@pytest.mark.skipif(not PUBMED_AVAILABLE, reason="BioPython not installed")
class TestPubMedConvenienceFunctions:
    """Test convenience functions for PubMed API."""

    @patch('healthbot.tools.pubmed_api.PubMedClient')
    def test_search_pubmed_with_email(self, mock_client_class):
        """Test search_pubmed with provided email."""
        mock_client = Mock()
        mock_client.search_and_fetch.return_value = {
            "success": True,
            "count": 1,
            "papers": [{"pmid": "12345678"}]
        }
        mock_client_class.return_value = mock_client

        result = search_pubmed("diabetes", max_results=5, email="test@example.com")

        assert result["success"]
        assert result["count"] == 1
        mock_client.search_and_fetch.assert_called_once_with("diabetes", 5)

    @patch('healthbot.tools.pubmed_api.PubMedClient')
    @patch('healthbot.config.settings')
    def test_search_pubmed_with_config_email(self, mock_settings, mock_client_class):
        """Test search_pubmed uses email from config."""
        mock_settings.ENTREZ_EMAIL = "config@example.com"

        mock_client = Mock()
        mock_client.search_and_fetch.return_value = {"success": True, "papers": []}
        mock_client_class.return_value = mock_client

        result = search_pubmed("diabetes", max_results=5)

        assert result["success"]
        mock_client_class.assert_called_once()

    @patch('healthbot.tools.pubmed_api.PubMedClient')
    def test_search_pubmed_handles_import_error(self, mock_client_class):
        """Test search_pubmed handles BioPython import errors."""
        mock_client_class.side_effect = ImportError("BioPython not installed")

        result = search_pubmed("diabetes", max_results=5, email="test@example.com")

        assert not result["success"]
        assert "BioPython" in result["error"]
        assert result["papers"] == []

    @patch('healthbot.tools.pubmed_api.PubMedClient')
    def test_search_pubmed_handles_value_error(self, mock_client_class):
        """Test search_pubmed handles invalid email."""
        mock_client_class.side_effect = ValueError("Valid email required")

        result = search_pubmed("diabetes", max_results=5, email="")

        assert not result["success"]
        assert "Valid email required" in result["error"]


@pytest.mark.skipif(not PUBMED_AVAILABLE, reason="BioPython not installed")
class TestPubMedAgentTool:
    """Test PubMed API tool for agent integration."""

    @patch('healthbot.tools.pubmed_api.search_pubmed')
    def test_pubmed_api_tool(self, mock_search):
        """Test pubmed_api_tool delegates to search_pubmed."""
        mock_search.return_value = {
            "success": True,
            "count": 2,
            "papers": [
                {"pmid": "12345678", "title": "Paper 1"},
                {"pmid": "87654321", "title": "Paper 2"},
            ]
        }

        result = pubmed_api_tool("diabetes treatment", max_results=2)

        assert result["success"]
        assert result["count"] == 2
        assert len(result["papers"]) == 2
        mock_search.assert_called_once_with("diabetes treatment", 2)


@pytest.mark.skipif(PUBMED_AVAILABLE, reason="Test BioPython not available case")
class TestPubMedWithoutBioPython:
    """Test behavior when BioPython is not installed."""

    def test_import_fails_gracefully(self):
        """Test that missing BioPython is handled gracefully."""
        # This test runs only when BioPython is NOT available
        # The import at the top should have set PUBMED_AVAILABLE = False
        assert not PUBMED_AVAILABLE


# Mark all tests as Phase 4 PubMed tests
pytestmark = pytest.mark.pubmed
