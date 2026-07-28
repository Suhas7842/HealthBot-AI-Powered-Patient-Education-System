"""
Hybrid retrieval combining semantic search (Pinecone) and keyword search (BM25).
Uses reciprocal rank fusion to combine results from both methods.
"""

from rank_bm25 import BM25Okapi

from healthbot.data.processor import DocumentProcessor
from healthbot.logger import logger
from healthbot.retrieval.pinecone_store import PineconeVectorStore


class HybridRetriever:
    """Combines semantic and keyword-based retrieval for better results."""

    def __init__(self):
        """Initialize hybrid retriever with Pinecone vector store and BM25."""
        # Initialize Pinecone vector store for semantic search
        self.vector_store = PineconeVectorStore()

        # Load documents for BM25 indexing from local data
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build BM25 index from local processed documents."""
        logger.info("Building BM25 keyword index from local data")

        try:
            # Load and process documents from local parquet file for BM25
            processor = DocumentProcessor()
            chunks = processor.process_knowledge_base()

            if not chunks:
                logger.warning("No documents found for BM25 indexing")
                self.bm25 = None
                self.bm25_documents = []
                self.bm25_metadatas = []
                return

            # Extract text and metadata from chunks
            self.bm25_documents = [chunk["text"] for chunk in chunks]
            self.bm25_metadatas = [
                {
                    "pmid": chunk.get("pmid", ""),
                    "title": chunk.get("title", ""),
                    "condition": chunk.get("condition", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                }
                for chunk in chunks
            ]

            # Tokenize documents for BM25
            tokenized_docs = [doc.lower().split() for doc in self.bm25_documents]
            self.bm25 = BM25Okapi(tokenized_docs)

            logger.info(f"BM25 index built with {len(self.bm25_documents)} documents")

        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            logger.warning(
                "BM25 will not be available - falling back to semantic search only"
            )
            self.bm25 = None
            self.bm25_documents = []
            self.bm25_metadatas = []

    def keyword_search(self, query: str, k: int = 10) -> list[dict]:
        """
        Perform BM25 keyword-based search.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of documents with scores
        """
        if self.bm25 is None:
            logger.warning("BM25 index not available")
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top k results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :k
        ]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with non-zero scores
                results.append(
                    {
                        "text": self.bm25_documents[idx],
                        "metadata": self.bm25_metadatas[idx],
                        "score": float(scores[idx]),
                        "method": "bm25",
                    }
                )

        logger.info(f"BM25 search returned {len(results)} results")
        return results

    def semantic_search(self, query: str, k: int = 10) -> list[dict]:
        """
        Perform semantic vector search.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of documents with scores
        """
        results = self.vector_store.similarity_search(query, k=k)

        # Add method tag
        for result in results:
            result["method"] = "semantic"

        return results

    def reciprocal_rank_fusion(
        self, results_list: list[list[dict]], k: int = 60
    ) -> list[dict]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion.

        Args:
            results_list: List of result lists from different methods
            k: Constant for RRF formula (default 60)

        Returns:
            Combined and reranked results
        """
        # Track all unique documents by text content
        doc_scores = {}

        for results in results_list:
            for rank, doc in enumerate(results, 1):
                doc_text = doc["text"]

                # RRF score: 1 / (k + rank)
                rrf_score = 1.0 / (k + rank)

                if doc_text not in doc_scores:
                    doc_scores[doc_text] = {
                        "document": doc,
                        "rrf_score": 0,
                        "methods": [],
                    }

                doc_scores[doc_text]["rrf_score"] += rrf_score
                doc_scores[doc_text]["methods"].append(doc.get("method", "unknown"))

        # Sort by RRF score
        sorted_docs = sorted(
            doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True
        )

        # Format results
        combined_results = []
        for item in sorted_docs:
            doc = item["document"].copy()
            doc["rrf_score"] = item["rrf_score"]
            doc["methods"] = list(set(item["methods"]))  # Unique methods
            combined_results.append(doc)

        return combined_results

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """
        Hybrid retrieval combining semantic and keyword search.

        Args:
            query: Search query
            k: Number of final results to return

        Returns:
            List of top-k documents from combined retrieval
        """
        logger.info(f"Hybrid retrieval for query: '{query}'")

        # Perform both types of search (retrieve 2x for better fusion coverage)
        semantic_results = self.semantic_search(query, k=k * 2)
        keyword_results = self.keyword_search(query, k=k * 2)

        # Combine using Reciprocal Rank Fusion to balance both methods
        combined_results = self.reciprocal_rank_fusion(
            [semantic_results, keyword_results]
        )

        # Return top k after reranking
        final_results = combined_results[:k]

        logger.info(f"Hybrid retrieval returned {len(final_results)} results")

        # Log method distribution
        method_counts = {}
        for doc in final_results:
            methods_str = "+".join(sorted(doc.get("methods", ["unknown"])))
            method_counts[methods_str] = method_counts.get(methods_str, 0) + 1
        logger.info(f"Method distribution: {method_counts}")

        return final_results

    def format_context(self, documents: list[dict]) -> str:
        """
        Format retrieved documents as context for LLM.

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant information found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.get("metadata", {})
            title = metadata.get("title", "Unknown")
            pmid = metadata.get("pmid", "Unknown")
            condition = metadata.get("condition", "")

            context_parts.append(
                f"[Source {i}] {title}\n"
                f"PMID: {pmid} | Condition: {condition}\n"
                f"{doc['text']}\n"
            )

        return "\n\n".join(context_parts)


def main():
    """Test hybrid retrieval."""
    retriever = HybridRetriever()

    # Check Pinecone stats
    stats = retriever.vector_store.get_stats()
    if stats["total_vectors"] == 0:
        print("\nPinecone index is empty. Upload documents first:")
        print("  python -m healthbot.retrieval.pinecone_store")
        return

    print(f"\nPinecone stats: {stats['total_vectors']} vectors")
    print(f"BM25 index: {len(retriever.bm25_documents)} documents\n")

    # Test query
    query = "What are the main causes and risk factors of type 2 diabetes?"

    print("=" * 80)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 80)
    print(f"Query: {query}\n")

    # Retrieve documents
    results = retriever.retrieve(query, k=5)

    # Display results
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  RRF Score: {doc.get('rrf_score', 0):.4f}")
        print(f"  Methods: {', '.join(doc.get('methods', []))}")
        print(f"  Condition: {doc.get('metadata', {}).get('condition', 'Unknown')}")
        print(f"  Text: {doc['text'][:150]}...")

    print("\n" + "=" * 80)
    print("FORMATTED CONTEXT FOR LLM")
    print("=" * 80)
    context = retriever.format_context(results[:3])
    print(context[:500] + "...")
    print("=" * 80)


if __name__ == "__main__":
    main()
