"""
Unit tests for cross-encoder reranker.

Tests cross-encoder scoring, ranking, and integration.
"""

import pytest

from healthbot.retrieval.reranker import CrossEncoderReranker


class TestCrossEncoderReranker:
    """Test suite for CrossEncoderReranker."""

    @pytest.fixture
    def reranker(self):
        """Create a reranker instance for testing."""
        return CrossEncoderReranker()

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {"text": "Type 2 diabetes is treated with lifestyle changes and medication."},
            {"text": "The Eiffel Tower is located in Paris, France."},  # Irrelevant
            {
                "text": "Diabetes treatment includes metformin, insulin, and dietary modifications."
            },
            {"text": "Regular exercise helps manage diabetes symptoms."},
            {"text": "The Great Wall of China is over 13,000 miles long."},  # Irrelevant
        ]

    def test_reranker_initialization(self, reranker):
        """Test that reranker initializes correctly."""
        assert reranker is not None
        assert reranker.model is not None
        assert reranker.model_name is not None

    def test_rerank_basic(self, reranker, sample_documents):
        """Test basic reranking functionality."""
        query = "How is diabetes treated?"
        reranked = reranker.rerank(query, sample_documents, top_k=3)

        assert isinstance(reranked, list)
        assert len(reranked) == 3

        # Check all have rerank scores
        for doc in reranked:
            assert "rerank_score" in doc
            assert isinstance(doc["rerank_score"], (int, float))

        # Scores should be in descending order
        scores = [doc["rerank_score"] for doc in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_relevance_scoring(self, reranker, sample_documents):
        """Test that relevant documents score higher than irrelevant ones."""
        query = "diabetes treatment options"
        reranked = reranker.rerank(query, sample_documents, top_k=5)

        # Relevant documents should be ranked higher
        top_doc = reranked[0]
        bottom_doc = reranked[-1]

        # Top document should be about diabetes
        assert (
            "diabetes" in top_doc["text"].lower()
            or "treatment" in top_doc["text"].lower()
        )

        # Top doc should score higher than bottom doc
        assert top_doc["rerank_score"] > bottom_doc["rerank_score"]

    def test_rerank_empty_documents(self, reranker):
        """Test handling of empty document list."""
        query = "test query"
        reranked = reranker.rerank(query, [], top_k=5)

        assert isinstance(reranked, list)
        assert len(reranked) == 0

    def test_rerank_top_k_limiting(self, reranker, sample_documents):
        """Test that top_k correctly limits results."""
        query = "diabetes"

        # Request fewer than available
        reranked = reranker.rerank(query, sample_documents, top_k=2)
        assert len(reranked) == 2

        # Request more than available
        reranked = reranker.rerank(query, sample_documents, top_k=10)
        assert len(reranked) == len(sample_documents)

    def test_rerank_preserves_document_data(self, reranker):
        """Test that reranking preserves original document data."""
        docs = [
            {"text": "Document about diabetes", "metadata": {"id": "doc1"}},
            {"text": "Another diabetes document", "metadata": {"id": "doc2"}},
        ]

        query = "diabetes"
        reranked = reranker.rerank(query, docs, top_k=2)

        # Check that original fields are preserved
        for doc in reranked:
            assert "text" in doc
            assert "metadata" in doc
            assert "rerank_score" in doc
            assert "id" in doc["metadata"]

    def test_rerank_score_field_customization(self, reranker, sample_documents):
        """Test that score field name can be customized."""
        query = "diabetes"
        custom_field = "custom_score"

        reranked = reranker.rerank(
            query, sample_documents, top_k=3, score_field=custom_field
        )

        for doc in reranked:
            assert custom_field in doc
            assert isinstance(doc[custom_field], (int, float))

    def test_rerank_with_medical_query(self, reranker):
        """Test reranking with medical terminology."""
        docs = [
            {"text": "Hyperglycemia is elevated blood glucose levels."},
            {"text": "High blood sugar occurs in diabetes."},
            {"text": "Basketball is a popular sport."},  # Irrelevant
        ]

        # Medical term query
        query = "hyperglycemia causes"
        reranked = reranker.rerank(query, docs, top_k=3)

        # Document with "hyperglycemia" should rank highly
        assert "hyperglycemia" in reranked[0]["text"].lower()

    def test_rerank_semantic_similarity(self, reranker):
        """Test that semantically similar documents rank well."""
        docs = [
            {"text": "Insulin resistance is a hallmark of Type 2 diabetes."},
            {"text": "Type 2 diabetes involves impaired insulin function."},
            {"text": "The capital of France is Paris."},  # Semantically different
        ]

        query = "insulin problems in diabetes"
        reranked = reranker.rerank(query, docs, top_k=3)

        # Top 2 should be about insulin/diabetes
        top_2_texts = " ".join([doc["text"].lower() for doc in reranked[:2]])
        assert "insulin" in top_2_texts
        assert "diabetes" in top_2_texts

    def test_rerank_with_original_scores(self, reranker):
        """Test hybrid scoring with original and rerank scores."""
        docs = [
            {"text": "Diabetes treatment", "score": 0.8},
            {"text": "Diabetes causes", "score": 0.6},
            {"text": "Irrelevant document", "score": 0.9},  # High original, low rerank
        ]

        query = "diabetes treatment"

        # Use hybrid scoring
        reranked = reranker.rerank_with_original_scores(
            query,
            docs,
            top_k=3,
            original_score_field="score",
            hybrid_weight=0.7,  # 70% rerank, 30% original
        )

        assert len(reranked) == 3
        # Check that hybrid_score was calculated
        for doc in reranked:
            assert "hybrid_score" in doc
            assert "rerank_score" in doc


class TestRerankerIntegration:
    """Test reranker integration with HybridRetriever."""

    def test_retriever_with_reranker(self):
        """Test that retriever can use reranker."""
        from healthbot.retrieval.retriever import HybridRetriever

        # Initialize with reranker enabled
        retriever = HybridRetriever(use_reranker=True)

        assert retriever.use_reranker is True
        assert retriever.reranker is not None

    def test_retriever_without_reranker(self):
        """Test that retriever works without reranker."""
        from healthbot.retrieval.retriever import HybridRetriever

        # Initialize without reranker
        retriever = HybridRetriever(use_reranker=False)

        assert retriever.use_reranker is False
        assert retriever.reranker is None

    def test_reranked_retrieval(self):
        """Test end-to-end retrieval with reranking."""
        from healthbot.retrieval.retriever import HybridRetriever

        retriever = HybridRetriever(use_reranker=True)
        query = "What are the symptoms of diabetes?"

        results = retriever.retrieve(query, k=5)

        assert len(results) == 5

        # Should have rerank scores
        for doc in results:
            assert "rerank_score" in doc

        # Rerank scores should be ordered
        rerank_scores = [doc["rerank_score"] for doc in results]
        assert rerank_scores == sorted(rerank_scores, reverse=True)


class TestRerankerPerformance:
    """Test reranker performance characteristics."""

    @pytest.fixture
    def reranker(self):
        """Create a reranker instance for testing."""
        return CrossEncoderReranker()

    def test_rerank_latency_reasonable(self, reranker):
        """Test that reranking completes in reasonable time."""
        import time

        docs = [{"text": f"Document {i} about medical topics"} for i in range(20)]
        query = "medical information"

        start = time.time()
        reranked = reranker.rerank(query, docs, top_k=5)
        latency = time.time() - start

        # Should complete in under 1 second for 20 docs
        assert latency < 1.0
        assert len(reranked) == 5

    def test_rerank_handles_long_documents(self, reranker):
        """Test that reranker handles long documents."""
        long_text = "Diabetes is a metabolic disorder. " * 100  # Long document

        docs = [
            {"text": long_text},
            {"text": "Short document about diabetes."},
        ]

        query = "diabetes"
        reranked = reranker.rerank(query, docs, top_k=2)

        # Should complete without errors
        assert len(reranked) == 2
        for doc in reranked:
            assert "rerank_score" in doc
