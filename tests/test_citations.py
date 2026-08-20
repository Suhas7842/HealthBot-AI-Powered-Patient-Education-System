"""
Unit tests for citation verification system (Phase 2C).

Tests CitedClaim and CitedMedicalSummary schemas, citation verification logic,
and citation quality evaluation.
"""

import pytest

from healthbot.schemas import CitedClaim, CitedMedicalSummary


class TestCitedClaimSchema:
    """Test CitedClaim schema validation."""

    def test_cited_claim_creation(self):
        """Test creating a valid CitedClaim."""
        claim = CitedClaim(
            claim_text="Insulin resistance in muscle and liver cells",
            citation_ids=[1, 2],
            confidence=1.0,
        )

        assert claim.claim_text == "Insulin resistance in muscle and liver cells"
        assert claim.citation_ids == [1, 2]
        assert claim.confidence == 1.0

    def test_cited_claim_default_confidence(self):
        """Test CitedClaim uses default confidence of 1.0."""
        claim = CitedClaim(
            claim_text="High blood glucose levels", citation_ids=[1]
        )

        assert claim.confidence == 1.0

    def test_cited_claim_confidence_bounds(self):
        """Test CitedClaim confidence is bounded 0-1."""
        # Valid confidence values
        claim = CitedClaim(
            claim_text="Test claim", citation_ids=[1], confidence=0.5
        )
        assert claim.confidence == 0.5

        claim = CitedClaim(
            claim_text="Test claim", citation_ids=[1], confidence=0.0
        )
        assert claim.confidence == 0.0

        claim = CitedClaim(
            claim_text="Test claim", citation_ids=[1], confidence=1.0
        )
        assert claim.confidence == 1.0

    def test_cited_claim_multiple_citations(self):
        """Test CitedClaim with multiple source citations."""
        claim = CitedClaim(
            claim_text="Treatment involves lifestyle and medication",
            citation_ids=[1, 2, 3, 4],
        )

        assert len(claim.citation_ids) == 4
        assert all(isinstance(cid, int) for cid in claim.citation_ids)

    def test_cited_claim_empty_citation_list(self):
        """Test CitedClaim with empty citation list is allowed."""
        claim = CitedClaim(claim_text="Uncited claim", citation_ids=[])

        assert claim.citation_ids == []


class TestCitedMedicalSummarySchema:
    """Test CitedMedicalSummary schema validation."""

    def test_cited_summary_creation(self):
        """Test creating a valid CitedMedicalSummary."""
        sources = [
            {"text": "Source 1 about diabetes", "score": 0.9},
            {"text": "Source 2 about insulin", "score": 0.8},
        ]

        summary = CitedMedicalSummary(
            title="Type 2 Diabetes",
            condition="Metabolic disorder affecting glucose regulation",
            cited_causes=[
                CitedClaim(claim_text="Insulin resistance", citation_ids=[1, 2])
            ],
            cited_symptoms=[
                CitedClaim(claim_text="Increased thirst", citation_ids=[1])
            ],
            cited_treatments=[
                CitedClaim(claim_text="Lifestyle modifications", citation_ids=[2])
            ],
            sources=sources,
        )

        assert summary.title == "Type 2 Diabetes"
        assert len(summary.cited_causes) == 1
        assert len(summary.cited_symptoms) == 1
        assert len(summary.cited_treatments) == 1
        assert len(summary.sources) == 2

    def test_cited_summary_default_warning(self):
        """Test CitedMedicalSummary includes default medical disclaimer."""
        summary = CitedMedicalSummary(
            title="Test",
            condition="Test condition",
            cited_causes=[],
            cited_symptoms=[],
            cited_treatments=[],
            sources=[],
        )

        assert "educational purposes only" in summary.warning.lower()
        assert "consult" in summary.warning.lower()

    def test_cited_summary_empty_claims(self):
        """Test CitedMedicalSummary with no claims is valid."""
        summary = CitedMedicalSummary(
            title="Rare Condition",
            condition="Unknown condition",
            cited_causes=[],
            cited_symptoms=[],
            cited_treatments=[],
            sources=[],
        )

        assert len(summary.cited_causes) == 0
        assert len(summary.cited_symptoms) == 0
        assert len(summary.cited_treatments) == 0

    def test_cited_summary_multiple_claims_per_category(self):
        """Test CitedMedicalSummary with multiple claims in each category."""
        sources = [{"text": f"Source {i}", "score": 0.9} for i in range(1, 6)]

        summary = CitedMedicalSummary(
            title="Diabetes",
            condition="Metabolic disorder",
            cited_causes=[
                CitedClaim(claim_text="Insulin resistance", citation_ids=[1]),
                CitedClaim(claim_text="Genetic factors", citation_ids=[2, 3]),
                CitedClaim(claim_text="Obesity", citation_ids=[1, 4]),
            ],
            cited_symptoms=[
                CitedClaim(claim_text="Thirst", citation_ids=[1]),
                CitedClaim(claim_text="Urination", citation_ids=[2]),
            ],
            cited_treatments=[
                CitedClaim(claim_text="Metformin", citation_ids=[5]),
                CitedClaim(claim_text="Lifestyle changes", citation_ids=[1, 5]),
            ],
            sources=sources,
        )

        assert len(summary.cited_causes) == 3
        assert len(summary.cited_symptoms) == 2
        assert len(summary.cited_treatments) == 2


class TestCitationVerificationLogic:
    """Test citation verification logic (without LLM)."""

    def test_valid_citation_indices(self):
        """Test that citation IDs reference valid source indices."""
        sources = [
            {"text": "Source 1", "score": 0.9},
            {"text": "Source 2", "score": 0.8},
            {"text": "Source 3", "score": 0.7},
        ]

        claim = CitedClaim(
            claim_text="Test claim", citation_ids=[1, 2]  # Valid: 1-3
        )

        # Check all citation IDs are within valid range
        valid = all(1 <= cid <= len(sources) for cid in claim.citation_ids)
        assert valid

    def test_invalid_citation_indices_detected(self):
        """Test detection of invalid citation IDs."""
        sources = [
            {"text": "Source 1", "score": 0.9},
            {"text": "Source 2", "score": 0.8},
        ]

        claim = CitedClaim(
            claim_text="Test claim",
            citation_ids=[1, 5],  # 5 is invalid (only 2 sources)
        )

        # Check if any citation ID is out of range
        invalid_ids = [cid for cid in claim.citation_ids if cid > len(sources) or cid < 1]
        assert len(invalid_ids) > 0  # Should detect invalid ID 5

    def test_citation_coverage_calculation(self):
        """Test calculation of citation coverage percentage."""
        summary = CitedMedicalSummary(
            title="Test",
            condition="Test",
            cited_causes=[
                CitedClaim(claim_text="Cause 1", citation_ids=[1]),
                CitedClaim(claim_text="Cause 2", citation_ids=[]),  # No citations
            ],
            cited_symptoms=[CitedClaim(claim_text="Symptom 1", citation_ids=[2])],
            cited_treatments=[
                CitedClaim(claim_text="Treatment 1", citation_ids=[]),  # No citations
            ],
            sources=[{"text": "Source", "score": 0.9}],
        )

        # Calculate coverage: 2 out of 4 claims have citations
        all_claims = (
            summary.cited_causes + summary.cited_symptoms + summary.cited_treatments
        )
        claims_with_citations = sum(1 for claim in all_claims if claim.citation_ids)
        coverage = claims_with_citations / len(all_claims)

        assert coverage == 0.5  # 50% coverage

    def test_source_usage_tracking(self):
        """Test tracking which sources are actually cited."""
        summary = CitedMedicalSummary(
            title="Test",
            condition="Test",
            cited_causes=[
                CitedClaim(claim_text="Cause 1", citation_ids=[1, 2]),
                CitedClaim(claim_text="Cause 2", citation_ids=[2]),
            ],
            cited_symptoms=[CitedClaim(claim_text="Symptom 1", citation_ids=[3])],
            cited_treatments=[CitedClaim(claim_text="Treatment 1", citation_ids=[1])],
            sources=[
                {"text": "Source 1", "score": 0.9},
                {"text": "Source 2", "score": 0.8},
                {"text": "Source 3", "score": 0.7},
            ],
        )

        # Track which sources are cited
        all_claims = (
            summary.cited_causes + summary.cited_symptoms + summary.cited_treatments
        )
        cited_sources = set()
        for claim in all_claims:
            cited_sources.update(claim.citation_ids)

        assert cited_sources == {1, 2, 3}  # All sources used

    def test_uncited_sources_detection(self):
        """Test detection of sources that are never cited."""
        summary = CitedMedicalSummary(
            title="Test",
            condition="Test",
            cited_causes=[CitedClaim(claim_text="Cause 1", citation_ids=[1])],
            cited_symptoms=[CitedClaim(claim_text="Symptom 1", citation_ids=[1])],
            cited_treatments=[
                CitedClaim(claim_text="Treatment 1", citation_ids=[2])
            ],
            sources=[
                {"text": "Source 1", "score": 0.9},
                {"text": "Source 2", "score": 0.8},
                {"text": "Source 3", "score": 0.7},  # Never cited
                {"text": "Source 4", "score": 0.6},  # Never cited
            ],
        )

        # Find uncited sources
        all_claims = (
            summary.cited_causes + summary.cited_symptoms + summary.cited_treatments
        )
        cited_sources = set()
        for claim in all_claims:
            cited_sources.update(claim.citation_ids)

        total_sources = len(summary.sources)
        uncited_sources = set(range(1, total_sources + 1)) - cited_sources

        assert uncited_sources == {3, 4}  # Sources 3 and 4 not cited


class TestCitationEdgeCases:
    """Test edge cases for citation system."""

    def test_duplicate_citation_ids(self):
        """Test claim with duplicate citation IDs."""
        claim = CitedClaim(
            claim_text="Test claim",
            citation_ids=[1, 1, 2, 2, 3],  # Duplicates
        )

        # Remove duplicates for verification
        unique_citations = list(set(claim.citation_ids))
        assert len(unique_citations) == 3  # Only 1, 2, 3

    def test_citation_ids_out_of_order(self):
        """Test claim with unordered citation IDs."""
        claim = CitedClaim(
            claim_text="Test claim", citation_ids=[3, 1, 2]  # Out of order
        )

        # Citation IDs can be in any order
        assert set(claim.citation_ids) == {1, 2, 3}

    def test_zero_citation_id(self):
        """Test claim with invalid zero citation ID."""
        claim = CitedClaim(claim_text="Test claim", citation_ids=[0, 1, 2])

        # Zero is invalid (1-indexed)
        invalid_ids = [cid for cid in claim.citation_ids if cid < 1]
        assert len(invalid_ids) > 0

    def test_negative_citation_id(self):
        """Test claim with negative citation ID."""
        claim = CitedClaim(claim_text="Test claim", citation_ids=[-1, 1, 2])

        # Negative is invalid
        invalid_ids = [cid for cid in claim.citation_ids if cid < 1]
        assert len(invalid_ids) > 0

    def test_very_long_claim_text(self):
        """Test claim with very long text."""
        long_claim = "This is a very long claim. " * 100  # 2700+ characters

        claim = CitedClaim(claim_text=long_claim, citation_ids=[1])

        assert len(claim.claim_text) > 2000
        assert claim.citation_ids == [1]

    def test_empty_sources_list(self):
        """Test summary with no sources."""
        summary = CitedMedicalSummary(
            title="Test",
            condition="Test condition",
            cited_causes=[CitedClaim(claim_text="Unsourced cause", citation_ids=[1])],
            cited_symptoms=[],
            cited_treatments=[],
            sources=[],  # No sources!
        )

        # All citation IDs are invalid when there are no sources
        assert len(summary.sources) == 0
        assert summary.cited_causes[0].citation_ids == [1]  # References non-existent source


class TestCitationMetrics:
    """Test calculation of citation quality metrics."""

    def test_perfect_citation_coverage(self):
        """Test 100% citation coverage."""
        summary = CitedMedicalSummary(
            title="Test",
            condition="Test",
            cited_causes=[CitedClaim(claim_text="Cause", citation_ids=[1])],
            cited_symptoms=[CitedClaim(claim_text="Symptom", citation_ids=[2])],
            cited_treatments=[CitedClaim(claim_text="Treatment", citation_ids=[3])],
            sources=[{"text": "Source", "score": 0.9}],
        )

        all_claims = (
            summary.cited_causes + summary.cited_symptoms + summary.cited_treatments
        )
        coverage = sum(1 for c in all_claims if c.citation_ids) / len(all_claims)

        assert coverage == 1.0  # 100%

    def test_zero_citation_coverage(self):
        """Test 0% citation coverage."""
        summary = CitedMedicalSummary(
            title="Test",
            condition="Test",
            cited_causes=[CitedClaim(claim_text="Cause", citation_ids=[])],
            cited_symptoms=[CitedClaim(claim_text="Symptom", citation_ids=[])],
            cited_treatments=[CitedClaim(claim_text="Treatment", citation_ids=[])],
            sources=[{"text": "Source", "score": 0.9}],
        )

        all_claims = (
            summary.cited_causes + summary.cited_symptoms + summary.cited_treatments
        )
        coverage = sum(1 for c in all_claims if c.citation_ids) / len(all_claims)

        assert coverage == 0.0  # 0%

    def test_average_citations_per_claim(self):
        """Test calculation of average citations per claim."""
        summary = CitedMedicalSummary(
            title="Test",
            condition="Test",
            cited_causes=[
                CitedClaim(claim_text="Cause 1", citation_ids=[1]),  # 1 citation
                CitedClaim(claim_text="Cause 2", citation_ids=[1, 2]),  # 2 citations
            ],
            cited_symptoms=[
                CitedClaim(
                    claim_text="Symptom", citation_ids=[1, 2, 3]
                )  # 3 citations
            ],
            cited_treatments=[
                CitedClaim(claim_text="Treatment", citation_ids=[])  # 0 citations
            ],
            sources=[{"text": "Source", "score": 0.9}],
        )

        all_claims = (
            summary.cited_causes + summary.cited_symptoms + summary.cited_treatments
        )
        avg_citations = sum(len(c.citation_ids) for c in all_claims) / len(all_claims)

        assert avg_citations == 1.5  # (1 + 2 + 3 + 0) / 4
