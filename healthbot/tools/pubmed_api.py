"""
PubMed API Integration for HealthBot Phase 4.

Uses NCBI E-utilities (Entrez) to search PubMed database (35M+ articles).
This demonstrates external API integration beyond local embeddings.
"""

import time
from typing import Dict, Any, List, Optional
from xml.etree import ElementTree as ET

try:
    from Bio import Entrez
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False


class PubMedAPIError(Exception):
    """Exception raised for PubMed API errors."""
    pass


class PubMedClient:
    """
    Client for PubMed E-utilities API.

    Implements search and fetch operations with rate limiting
    and error handling per NCBI guidelines.
    """

    def __init__(self, email: str, tool_name: str = "HealthBot", rate_limit: float = 0.34):
        """
        Initialize PubMed client.

        Args:
            email: Required by NCBI for E-utilities access
            tool_name: Name of the tool (for NCBI logs)
            rate_limit: Minimum seconds between requests (default: 0.34 = ~3 req/sec)
        """
        if not BIOPYTHON_AVAILABLE:
            raise ImportError(
                "BioPython is required for PubMed API access. "
                "Install with: pip install biopython"
            )

        if not email or email == "your_email@example.com":
            raise ValueError(
                "Valid email required for NCBI E-utilities. "
                "Set ENTREZ_EMAIL in .env or config."
            )

        Entrez.email = email
        Entrez.tool = tool_name
        self.rate_limit = rate_limit
        self.last_request_time = 0.0

    def _rate_limit_wait(self):
        """Enforce rate limiting per NCBI guidelines (max 3 requests/second)."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)

        self.last_request_time = time.time()

    def search(self, query: str, max_results: int = 5) -> List[str]:
        """
        Search PubMed and return PMIDs.

        Args:
            query: Search query (supports PubMed advanced syntax)
            max_results: Maximum number of results to return

        Returns:
            List of PMIDs (PubMed IDs)

        Raises:
            PubMedAPIError: If API request fails
        """
        self._rate_limit_wait()

        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()

            pmids = record.get("IdList", [])
            return pmids

        except Exception as e:
            raise PubMedAPIError(f"PubMed search failed: {str(e)}")

    def fetch(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch article details for given PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of article dictionaries with title, abstract, authors, etc.

        Raises:
            PubMedAPIError: If API request fails
        """
        if not pmids:
            return []

        self._rate_limit_wait()

        try:
            # Fetch XML records
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(pmids),
                rettype="xml",
                retmode="xml"
            )
            xml_data = handle.read()
            handle.close()

            # Parse XML
            root = ET.fromstring(xml_data)
            articles = []

            for article_elem in root.findall(".//PubmedArticle"):
                article_data = self._parse_article(article_elem)
                if article_data:
                    articles.append(article_data)

            return articles

        except Exception as e:
            raise PubMedAPIError(f"PubMed fetch failed: {str(e)}")

    def _parse_article(self, article_elem: ET.Element) -> Optional[Dict[str, Any]]:
        """
        Parse article XML element into structured dictionary.

        Args:
            article_elem: XML Element for PubmedArticle

        Returns:
            Dictionary with article data, or None if parsing fails
        """
        try:
            medline = article_elem.find(".//MedlineCitation")
            article = medline.find(".//Article")

            # Extract PMID
            pmid_elem = medline.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else "Unknown"

            # Extract title
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "No title"

            # Extract abstract
            abstract_texts = article.findall(".//AbstractText")
            if abstract_texts:
                abstract_parts = []
                for abs_elem in abstract_texts:
                    label = abs_elem.get("Label", "")
                    text = abs_elem.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = " ".join(abstract_parts)
            else:
                abstract = "No abstract available"

            # Extract authors
            author_list = article.find(".//AuthorList")
            authors = []
            if author_list is not None:
                for author_elem in author_list.findall(".//Author"):
                    last_name = author_elem.findtext(".//LastName", "")
                    initials = author_elem.findtext(".//Initials", "")
                    if last_name:
                        authors.append(f"{last_name} {initials}".strip())

            # Extract publication date
            pub_date = article.find(".//PubDate")
            if pub_date is not None:
                year = pub_date.findtext(".//Year", "")
                month = pub_date.findtext(".//Month", "")
                if year and month:
                    publication_date = f"{month} {year}"
                elif year:
                    publication_date = year
                else:
                    publication_date = "Date unknown"
            else:
                publication_date = "Date unknown"

            # Extract journal
            journal_elem = article.find(".//Journal")
            if journal_elem is not None:
                journal_title = journal_elem.findtext(".//Title", "Unknown journal")
            else:
                journal_title = "Unknown journal"

            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors[:5] if authors else [],  # First 5 authors
                "publication_date": publication_date,
                "journal": journal_title,
            }

        except Exception as e:
            # Log parsing error but don't fail entire batch
            return None

    def search_and_fetch(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search PubMed and fetch article details in one call.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            Dictionary with success flag and list of articles
        """
        try:
            # Search for PMIDs
            pmids = self.search(query, max_results)

            if not pmids:
                return {
                    "success": True,
                    "query": query,
                    "count": 0,
                    "papers": [],
                    "message": "No results found",
                }

            # Fetch article details
            articles = self.fetch(pmids)

            return {
                "success": True,
                "query": query,
                "count": len(articles),
                "papers": articles,
            }

        except PubMedAPIError as e:
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "papers": [],
            }


def search_pubmed(
    query: str,
    max_results: int = 5,
    email: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search PubMed via E-utilities API (convenience function).

    Args:
        query: Search query (supports PubMed advanced syntax)
        max_results: Maximum papers to return (default: 5)
        email: Email for NCBI (required, or set in config)

    Returns:
        Dictionary with:
            - success: bool
            - query: str
            - count: int
            - papers: list[dict] with pmid, title, abstract, authors, etc.

    Example:
        >>> result = search_pubmed("diabetes treatment", max_results=3)
        >>> if result["success"]:
        ...     for paper in result["papers"]:
        ...         print(f"{paper['title']} (PMID: {paper['pmid']})")
    """
    # Get email from environment if not provided
    if not email:
        from healthbot.config import settings
        email = settings.ENTREZ_EMAIL

    try:
        client = PubMedClient(email=email)
        return client.search_and_fetch(query, max_results)
    except (ImportError, ValueError) as e:
        return {
            "success": False,
            "query": query,
            "error": str(e),
            "papers": [],
        }


def pubmed_api_tool(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    PubMed API tool entry point for agent integration.

    This function is exposed to the LLM agent as a callable tool.

    Args:
        query: Medical search query
        max_results: Maximum papers to return

    Returns:
        Search results with papers
    """
    return search_pubmed(query, max_results)
