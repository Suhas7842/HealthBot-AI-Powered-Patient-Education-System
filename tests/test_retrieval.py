"""
Unit tests for retrieval components.

Tests BM25, semantic search, RRF fusion, and hybrid retrieval.
"""

import pytest

from healthbot.retrieval.retriever import HybridRetriever


class TestHybridRetriever:
    """Test suite for HybridRetriever."""

    @pytest.fixture
    def retriever(self):
        """Create a HybridRetriever instance for testing."""
        return HybridRetriever(use_reranker=False)

    def test_retriever_initialization(self, retriever):
        """Test that retriever initializes correctly."""
        assert retriever is not None
        assert retriever.vector_store is not None
        assert retriever.bm25 is not None
        assert retriever.bm25_documents is not None
        assert len(retriever.bm25_documents) > 0

    def test_bm25_search(self, retriever):
        """Test BM25 keyword search functionality."""
        query = "diabetes treatment"
        results = retriever.keyword_search(query, k=5)

        assert isinstance(results, list)
        assert len(results) <= 5
        assert len(results) > 0  # Should find diabetes-related documents

        # Check result structure
        for doc in results:
            assert "text" in doc
            assert "metadata" in doc
            assert "score" in doc
            assert "method" in doc
            assert doc["method"] == "bm25"
            assert doc["score"] > 0  # BM25 returns non-zero scores

    def test_semantic_search(self, retriever):
        """Test semantic vector search functionality."""
        query = "What causes high blood sugar?"
        results = retriever.semantic_search(query, k=5)

        assert isinstance(results, list)
        assert len(results) <= 5
        assert len(results) > 0

        # Check result structure
        for doc in results:
            assert "text" in doc
            assert "metadata" in doc
            assert "score" in doc
            assert "method" in doc
            assert doc["method"] == "semantic"

    def test_reciprocal_rank_fusion(self, retriever):
        """Test RRF fusion combines results correctly."""
        # Create mock results
        results1 = [
            {"text": "doc1", "method": "method1", "metadata": {}},
            {"text": "doc2", "method": "method1", "metadata": {}},
        ]
        results2 = [
            {"text": "doc2", "method": "method2", "metadata": {}},  # Duplicate
            {"text": "doc3", "method": "method2", "metadata": {}},
        ]

        fused = retriever.reciprocal_rank_fusion([results1, results2], k=60)

        assert isinstance(fused, list)
        assert len(fused) == 3  # 3 unique documents

        # Doc2 should rank highest (appears in both lists)
        assert fused[0]["text"] == "doc2"
        assert "rrf_score" in fused[0]
        assert fused[0]["rrf_score"] > 0

        # Should have combined methods
        assert "methods" in fused[0]
        assert len(fused[0]["methods"]) == 2  # From both methods

    def test_hybrid_retrieve(self, retriever):
        """Test full hybrid retrieval pipeline."""
        query = "symptoms of Type 2 diabetes"
        results = retriever.retrieve(query, k=5)

        assert isinstance(results, list)
        assert len(results) == 5

        # Check all results have required fields
        for doc in results:
            assert "text" in doc
            assert "metadata" in doc
            assert "rrf_score" in doc
            assert "methods" in doc
            assert isinstance(doc["methods"], list)

        # First result should have highest RRF score
        scores = [doc["rrf_score"] for doc in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_handling(self, retriever):
        """Test that empty queries are handled gracefully."""
        results = retriever.retrieve("", k=5)
        # Should return some results (even if not highly relevant)
        assert isinstance(results, list)

    def test_medical_terminology_query(self, retriever):
        """Test retrieval with medical terminology."""
        query = "myocardial infarction pathophysiology"
        results = retriever.retrieve(query, k=5)

        assert len(results) > 0
        # Should retrieve heart disease related documents
        # (Some documents should contain relevant medical terms)

    def test_common_language_query(self, retriever):
        """Test retrieval with common language."""
        query = "heart attack causes"
        results = retriever.retrieve(query, k=5)

        assert len(results) > 0
        # Hybrid retrieval should handle both medical and common terms

    def test_rrf_deduplication(self, retriever):
        """Test that RRF properly deduplicates across methods."""
        query = "diabetes"
        results = retriever.retrieve(query, k=5)

        # Extract all text contents
        texts = [doc["text"] for doc in results]

        # Should have no duplicate texts
        assert len(texts) == len(set(texts))


class TestRetrievalMetrics:
    """Test retrieval evaluation metrics."""

    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        from healthbot.evaluation.metrics import recall_at_k

        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc2", "doc5", "doc7"]

        recall = recall_at_k(retrieved, relevant, k=5)
        assert recall == pytest.approx(2 / 3)  # Found 2 out of 3 relevant

    def test_mean_reciprocal_rank(self):
        """Test MRR calculation."""
        from healthbot.evaluation.metrics import mean_reciprocal_rank

        # First relevant doc at position 2
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc2"]

        mrr = mean_reciprocal_rank(retrieved, relevant)
        assert mrr == pytest.approx(0.5)  # 1/2

    def test_hit_rate(self):
        """Test Hit Rate calculation."""
        from healthbot.evaluation.metrics import hit_rate

        # Has at least one relevant doc
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc2"]
        assert hit_rate(retrieved, relevant, k=3) == 1.0

        # No relevant docs
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc5"]
        assert hit_rate(retrieved, relevant, k=3) == 0.0

    def test_ndcg_at_k(self):
        """Test nDCG@K calculation."""
        from healthbot.evaluation.metrics import ndcg_at_k

        # Perfect ranking: both relevant docs at top
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2"]
        ndcg = ndcg_at_k(retrieved, relevant, k=3)
        assert ndcg == pytest.approx(1.0)  # Perfect score

        # Worst ranking: relevant docs at bottom
        retrieved = ["doc3", "doc4", "doc1", "doc2"]
        relevant = ["doc1", "doc2"]
        ndcg = ndcg_at_k(retrieved, relevant, k=4)
        assert ndcg < 1.0  # Lower score due to poor ranking

    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        from healthbot.evaluation.metrics import precision_at_k

        # 2 out of 5 retrieved docs are relevant
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc2", "doc5"]

        precision = precision_at_k(retrieved, relevant, k=5)
        assert precision == pytest.approx(0.4)  # 2/5


class TestFormatContext:
    """Test context formatting for LLM."""

    def test_format_context(self):
        """Test that context is formatted correctly."""
        retriever = HybridRetriever(use_reranker=False)

        documents = [
            {
                "text": "Diabetes is a metabolic disorder.",
                "metadata": {
                    "pmid": "12345",
                    "title": "Understanding Diabetes",
                    "condition": "diabetes",
                },
            },
            {
                "text": "High blood sugar is the main symptom.",
                "metadata": {
                    "pmid": "12346",
                    "title": "Diabetes Symptoms",
                    "condition": "diabetes",
                },
            },
        ]

        context = retriever.format_context(documents)

        assert isinstance(context, str)
        assert len(context) > 0

        # Should include document text
        assert "Diabetes is a metabolic disorder" in context
        assert "High blood sugar is the main symptom" in context

        # Should include metadata
        assert "12345" in context or "Understanding Diabetes" in context
