"""
Cross-encoder reranker for improving retrieval precision.

Why reranking?
- Bi-encoders (semantic search) optimize for recall - finding candidate documents
- Cross-encoders optimize for precision - accurately scoring query-document relevance
- Hybrid approach: retrieve many candidates with bi-encoder, rerank with cross-encoder

This adds ~40ms latency but significantly improves ranking quality.
"""

import time

from sentence_transformers import CrossEncoder

from healthbot.logger import logger


class CrossEncoderReranker:
    """
    Reranks retrieved documents using a cross-encoder model.

    Cross-encoders jointly encode query and document, allowing for more accurate
    relevance scoring at the cost of higher computational expense.

    Model: ms-marco-MiniLM-L-12-v2
    - Trained on Microsoft MARCO passage ranking dataset
    - 12-layer MiniLM (lightweight but effective)
    - Returns relevance score between query and passage
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
                Default: ms-marco-MiniLM-L-12-v2 (384MB, fast inference)
                Alternatives:
                - cross-encoder/ms-marco-MiniLM-L-6-v2 (smaller, faster)
                - cross-encoder/ms-marco-TinyBERT-L-2-v2 (smallest)
        """
        logger.info(f"Loading cross-encoder reranker: {model_name}")
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        logger.info(f"Cross-encoder loaded successfully")

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
        score_field: str = "rerank_score",
    ) -> list[dict]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: User query text
            documents: List of document dicts with "text" field
            top_k: Number of top documents to return after reranking
            score_field: Field name to store reranking score

        Returns:
            Top-K documents sorted by reranking score (highest first)

        Example:
            >>> reranker = CrossEncoderReranker()
            >>> docs = [{"text": "diabetes treatment..."}, {"text": "cancer info..."}]
            >>> reranked = reranker.rerank("What is diabetes?", docs, top_k=3)
            >>> reranked[0]["rerank_score"]
            0.85  # High relevance score
        """
        if not documents:
            logger.warning("No documents to rerank")
            return []

        start_time = time.time()

        # Prepare query-document pairs for cross-encoder
        pairs = []
        for doc in documents:
            text = doc.get("text", "")
            if not text:
                logger.warning(f"Document missing text field: {doc}")
                text = ""
            pairs.append([query, text])

        # Score all pairs
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(f"Cross-encoder prediction failed: {e}")
            # Fallback: return original documents with score 0
            for doc in documents:
                doc[score_field] = 0.0
            return documents[:top_k]

        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc[score_field] = float(score)

        # Sort by reranking score (descending)
        reranked_docs = sorted(documents, key=lambda d: d.get(score_field, 0), reverse=True)

        # Get top-k
        top_docs = reranked_docs[:top_k]

        latency = time.time() - start_time
        logger.info(
            f"Reranked {len(documents)} docs to top-{top_k} in {latency*1000:.1f}ms"
        )
        logger.info(f"Top score: {top_docs[0].get(score_field, 0):.3f}")
        logger.info(f"Bottom score: {top_docs[-1].get(score_field, 0):.3f}")

        return top_docs

    def rerank_with_original_scores(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
        original_score_field: str = "score",
        rerank_score_field: str = "rerank_score",
        hybrid_weight: float = 1.0,
    ) -> list[dict]:
        """
        Rerank documents using weighted combination of original and reranking scores.

        This allows preserving signal from the initial retrieval (e.g., BM25 or RRF)
        while incorporating cross-encoder precision.

        Final score = hybrid_weight * rerank_score + (1 - hybrid_weight) * original_score

        Args:
            query: User query text
            documents: List of document dicts
            top_k: Number of top documents to return
            original_score_field: Field containing original retrieval score
            rerank_score_field: Field to store reranking score
            hybrid_weight: Weight for rerank score (0=use original only, 1=use rerank only)

        Returns:
            Top-K documents sorted by hybrid score
        """
        if not documents:
            return []

        # First get reranking scores
        reranked = self.rerank(query, documents, top_k=len(documents), score_field=rerank_score_field)

        # Normalize scores to [0, 1] range for fair combination
        def normalize_scores(docs, field):
            scores = [d.get(field, 0) for d in docs]
            if not scores:
                return docs
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                for d in docs:
                    d[f"{field}_normalized"] = 0.5
            else:
                for d in docs:
                    d[f"{field}_normalized"] = (d.get(field, 0) - min_score) / (max_score - min_score)
            return docs

        reranked = normalize_scores(reranked, original_score_field)
        reranked = normalize_scores(reranked, rerank_score_field)

        # Compute hybrid score
        for doc in reranked:
            original_norm = doc.get(f"{original_score_field}_normalized", 0)
            rerank_norm = doc.get(f"{rerank_score_field}_normalized", 0)
            doc["hybrid_score"] = (
                hybrid_weight * rerank_norm + (1 - hybrid_weight) * original_norm
            )

        # Sort by hybrid score
        reranked = sorted(reranked, key=lambda d: d.get("hybrid_score", 0), reverse=True)

        return reranked[:top_k]


def demo_reranker():
    """Demonstrate reranker with sample medical queries."""
    print("=" * 80)
    print("CROSS-ENCODER RERANKER DEMO")
    print("=" * 80)

    # Sample documents (diabetes vs unrelated)
    documents = [
        {"text": "Type 2 diabetes is characterized by insulin resistance and high blood sugar.", "id": "doc1"},
        {"text": "Regular exercise helps manage diabetes by improving insulin sensitivity.", "id": "doc2"},
        {"text": "The Great Wall of China is over 13,000 miles long.", "id": "doc3"},  # Irrelevant
        {"text": "Diabetes medications include metformin, insulin, and GLP-1 agonists.", "id": "doc4"},
        {"text": "Paris is the capital city of France.", "id": "doc5"},  # Irrelevant
    ]

    query = "How is Type 2 diabetes treated?"

    print(f"\nQuery: {query}")
    print(f"\nDocuments to rerank: {len(documents)}")

    # Initialize reranker
    reranker = CrossEncoderReranker()

    # Rerank
    reranked = reranker.rerank(query, documents, top_k=3)

    print("\n" + "=" * 80)
    print("RERANKING RESULTS")
    print("=" * 80)

    for i, doc in enumerate(reranked, 1):
        score = doc.get("rerank_score", 0)
        text = doc["text"][:60] + "..." if len(doc["text"]) > 60 else doc["text"]
        print(f"\nRank {i} (score: {score:.3f})")
        print(f"  {text}")

    print("\n" + "=" * 80)
    print("Notice: Irrelevant documents (Great Wall, Paris) ranked lower!")
    print("=" * 80)


if __name__ == "__main__":
    demo_reranker()
