"""
Tools integration for HealthBot: RAG retrieval and Tavily search.
Provides fallback mechanisms for comprehensive medical information.
"""

from langchain_community.tools.tavily_search import TavilySearchResults

from healthbot.config import settings
from healthbot.logger import logger
from healthbot.retrieval.retriever import HybridRetriever

# Known medical conditions covered by our knowledge base
KNOWN_CONDITIONS = [
    "diabetes",
    "hypertension",
    "asthma",
    "heart disease",
    "coronary",
    "depression",
    "anxiety",
    "arthritis",
    "migraine",
    "allergies",
    "influenza",
    "flu",
    "cold",
    "obesity",
    "cancer",
    "stroke",
    "copd",
    "pneumonia",
    "gastritis",
    "kidney disease",
    "thyroid",
]


class RAGTool:
    """Tool for retrieving information from cloud vector database."""

    def __init__(self):
        """Initialize RAG tool with cloud or local retriever."""
        try:
            # Try Pinecone cloud first (production)
            if settings.USE_CLOUD_VECTOR_DB and settings.PINECONE_API_KEY:
                from healthbot.retrieval.pinecone_store import PineconeVectorStore

                self.retriever = PineconeVectorStore()
                logger.info("RAG tool initialized with Pinecone (cloud)")
            else:
                # Fallback to local ChromaDB with configurable reranking
                self.retriever = HybridRetriever(use_reranker=settings.USE_RERANKER)
                reranker_status = "enabled" if settings.USE_RERANKER else "disabled"
                logger.info(f"RAG tool initialized with ChromaDB (local, reranker={reranker_status})")
            self.available = True
        except Exception as e:
            logger.error(f"Failed to initialize RAG tool: {e}")
            self.available = False

    def search(self, query: str, k: int = 5) -> dict:
        """
        Search medical knowledge base using hybrid retrieval.

        Args:
            query: Medical question or search query
            k: Number of results to return (default 5)

        Returns:
            Dictionary with:
            - success: bool (whether search completed)
            - documents: list of retrieved documents with text and metadata
            - error: str (if search failed)
        """
        if not self.available:
            return {
                "success": False,
                "error": "RAG tool not available. Vector store may not be built.",
                "documents": [],
            }

        try:
            logger.info(f"RAG search: '{query}'")

            # Check if using Pinecone (has similarity_search) or HybridRetriever (has retrieve)
            if hasattr(self.retriever, "similarity_search"):
                # Pinecone cloud
                results = self.retriever.similarity_search(query, k=k)
                documents = results  # Already in correct format
            else:
                # Local HybridRetriever
                documents = self.retriever.retrieve(query, k=k)

            return {
                "success": True,
                "method": "rag",
                "query": query,
                "num_results": len(documents),
                "documents": documents,
            }

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return {"success": False, "error": str(e), "documents": []}

    def format_results(self, results: dict) -> str:
        """
        Format RAG search results as text.

        Args:
            results: Results dictionary from search()

        Returns:
            Formatted text string
        """
        if not results["success"]:
            return f"RAG search failed: {results.get('error', 'Unknown error')}"

        if not results["documents"]:
            return "No relevant information found in knowledge base."

        # Format based on retriever type
        if hasattr(self.retriever, "format_context"):
            # HybridRetriever has format_context
            return self.retriever.format_context(results["documents"])
        else:
            # Pinecone - format manually
            context_parts = []
            for i, doc in enumerate(results["documents"], 1):
                context_parts.append(
                    f"[Source {i}] Score: {doc.get('score', 0):.3f}\n"
                    f"{doc.get('text', '')}"
                )
            return "\n\n".join(context_parts)


class TavilyTool:
    """Tool for web search using Tavily (fallback for rare conditions)."""

    def __init__(self):
        """Initialize Tavily search tool."""
        if settings.TAVILY_API_KEY:
            try:
                self.tavily = TavilySearchResults(
                    max_results=settings.SEARCH_RESULTS, api_key=settings.TAVILY_API_KEY
                )
                self.available = True
                logger.info("Tavily tool initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Tavily: {e}")
                self.available = False
        else:
            logger.warning("Tavily API key not configured")
            self.available = False

    def search(self, query: str) -> dict:
        """
        Search web using Tavily.

        Args:
            query: Search query

        Returns:
            Dictionary with results and metadata
        """
        if not self.available:
            return {"success": False, "error": "Tavily not configured", "documents": []}

        try:
            logger.info(f"Tavily search: '{query}'")
            results = self.tavily.invoke({"query": query})

            # Format results
            documents = []
            for result in results:
                documents.append(
                    {
                        "text": result.get("content", ""),
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "score": result.get("score", 0.0),
                        "method": "tavily",
                    }
                )

            return {
                "success": True,
                "method": "tavily",
                "query": query,
                "num_results": len(documents),
                "documents": documents,
            }

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return {"success": False, "error": str(e), "documents": []}

    def format_results(self, results: dict) -> str:
        """
        Format Tavily search results as text.

        Args:
            results: Results dictionary from search()

        Returns:
            Formatted text string
        """
        if not results["success"]:
            return f"Web search failed: {results.get('error', 'Unknown error')}"

        if not results["documents"]:
            return "No web results found."

        context_parts = []
        for i, doc in enumerate(results["documents"], 1):
            context_parts.append(
                f"[Web Source {i}] {doc.get('title', 'Unknown')}\n"
                f"URL: {doc.get('url', 'N/A')}\n"
                f"{doc.get('text', '')}..."
            )

        return "\n\n".join(context_parts)


class ToolSelector:
    """Intelligently selects between RAG and Tavily based on query."""

    def __init__(self):
        """Initialize tool selector with both tools."""
        self.rag_tool = RAGTool()
        self.tavily_tool = TavilyTool()

    def is_known_condition(self, query: str) -> bool:
        """
        Check if query is about a condition in our knowledge base.

        Args:
            query: User query

        Returns:
            True if condition is known, False otherwise
        """
        query_lower = query.lower()
        return any(condition in query_lower for condition in KNOWN_CONDITIONS)

    def select_and_search(self, query: str, k: int = 5) -> dict:
        """
        Select appropriate tool and perform search.

        Args:
            query: Search query
            k: Number of results for RAG

        Returns:
            Search results with method metadata
        """
        # Prefer RAG for known conditions
        if self.is_known_condition(query) and self.rag_tool.available:
            logger.info(f"Using RAG for known condition: '{query}'")
            results = self.rag_tool.search(query, k=k)

            # Fallback to Tavily if RAG fails or returns nothing
            if not results["success"] or not results["documents"]:
                logger.info("RAG returned no results, falling back to Tavily")
                if self.tavily_tool.available:
                    return self.tavily_tool.search(query)

            return results

        # Use Tavily for rare/new conditions
        elif self.tavily_tool.available:
            logger.info(f"Using Tavily for rare condition: '{query}'")
            return self.tavily_tool.search(query)

        # Fallback to RAG if Tavily not available
        elif self.rag_tool.available:
            logger.info("Tavily unavailable, using RAG as fallback")
            return self.rag_tool.search(query, k=k)

        # No tools available
        else:
            logger.error("No search tools available")
            return {
                "success": False,
                "error": "No search tools available",
                "documents": [],
            }

    def format_results(self, results: dict) -> str:
        """
        Format results based on method used.

        Args:
            results: Results dictionary from select_and_search()

        Returns:
            Formatted text string
        """
        if not results["success"]:
            return f"Search failed: {results.get('error', 'Unknown error')}"

        method = results.get("method", "unknown")
        if method == "rag":
            return self.rag_tool.format_results(results)
        elif method == "tavily":
            return self.tavily_tool.format_results(results)
        else:
            return "No results available."


def main():
    """Test tool selector."""
    selector = ToolSelector()

    print("=" * 80)
    print("TOOL SELECTOR TEST")
    print("=" * 80)

    # Test 1: Known condition (should use RAG)
    query1 = "What are the symptoms of diabetes?"
    print(f"\nTest 1: '{query1}'")
    results1 = selector.select_and_search(query1, k=3)
    print(f"Method: {results1.get('method', 'unknown')}")
    print(f"Success: {results1['success']}")
    print(f"Results: {results1.get('num_results', 0)}")

    # Test 2: Rare condition (should use Tavily or fallback to RAG)
    query2 = "What is Fabry disease?"
    print(f"\nTest 2: '{query2}'")
    results2 = selector.select_and_search(query2, k=3)
    print(f"Method: {results2.get('method', 'unknown')}")
    print(f"Success: {results2['success']}")
    print(f"Results: {results2.get('num_results', 0)}")

    print("\n" + "=" * 80)
    print("FORMATTED CONTEXT")
    print("=" * 80)
    context = selector.format_results(results1)
    print(context[:500] + "...")
    print("=" * 80)


if __name__ == "__main__":
    main()
