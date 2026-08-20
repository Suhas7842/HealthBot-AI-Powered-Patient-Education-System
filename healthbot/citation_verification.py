"""
Citation verification for HealthBot (Phase 2C).

Verifies that cited claims are actually supported by the referenced sources.
Provides claim-level explainability for medical information.
"""

from healthbot.logger import logger
from healthbot.models import LLMWrapper
from healthbot.schemas import CitedClaim, CitedMedicalSummary


class CitationVerifier:
    """
    Verify claim-to-source attribution for explainability.

    Uses LLM-as-judge to verify that each claim is supported by its cited sources.
    Critical for medical information where provenance tracking is required.
    """

    def __init__(self):
        """Initialize citation verifier with LLM."""
        self.llm = LLMWrapper()

    def verify_claim(self, claim: CitedClaim, sources: list[dict]) -> dict:
        """
        Verify a single claim against its cited sources.

        Args:
            claim: CitedClaim with claim_text and citation_ids
            sources: List of source documents (1-indexed to match citation_ids)

        Returns:
            Dictionary with verification results:
                - claim: str (the claim text)
                - supported: bool (whether claim is supported)
                - confidence: float (verification confidence 0-1)
                - verdict: str (SUPPORTED | PARTIALLY_SUPPORTED | NOT_SUPPORTED)
                - explanation: str (reasoning)
                - cited_source_count: int (number of sources cited)

        Example:
            >>> claim = CitedClaim(
            ...     claim_text="Insulin resistance in muscle and liver cells",
            ...     citation_ids=[1, 2]
            ... )
            >>> result = verifier.verify_claim(claim, sources)
            >>> result['supported']
            True
        """
        # Extract cited source texts
        cited_texts = []
        valid_citations = []

        for idx in claim.citation_ids:
            if 1 <= idx <= len(sources):
                cited_texts.append(sources[idx - 1]["text"])  # Convert to 0-indexed
                valid_citations.append(idx)
            else:
                logger.warning(
                    f"Invalid citation ID {idx} in claim (only {len(sources)} sources available)"
                )

        if not cited_texts:
            return {
                "claim": claim.claim_text,
                "supported": False,
                "confidence": 0.0,
                "verdict": "NOT_SUPPORTED",
                "explanation": "No valid source citations provided",
                "cited_source_count": 0,
            }

        # Build verification prompt
        sources_text = "\n\n".join(
            [f"[Source {valid_citations[i]}]\n{text}" for i, text in enumerate(cited_texts)]
        )

        prompt = f"""You are verifying medical claim citations for accuracy and explainability.

**Claim**: "{claim.claim_text}"

**Cited Sources**:
{sources_text}

**Task**: Determine if the claim is supported by the cited sources.

**Verdict Options**:
- **SUPPORTED**: Claim is directly stated or clearly implied by the sources
- **PARTIALLY_SUPPORTED**: Claim has some support but requires inference or is incomplete
- **NOT_SUPPORTED**: Claim contradicts or is not mentioned in the sources

**Respond with**:
1. Verdict: (SUPPORTED | PARTIALLY_SUPPORTED | NOT_SUPPORTED)
2. Confidence: (0.0-1.0) - How confident are you in this verdict?
3. Explanation: Brief reasoning (2-3 sentences)

**Format your response as**:
Verdict: [verdict]
Confidence: [0.0-1.0]
Explanation: [explanation]
"""

        try:
            from langchain_core.messages import HumanMessage

            response = self.llm.invoke([HumanMessage(content=prompt)])

            # Parse response (simple line-based parsing)
            lines = response.strip().split("\n")
            verdict = "UNKNOWN"
            confidence = 0.5
            explanation = "Failed to parse verification response"

            for line in lines:
                if line.startswith("Verdict:"):
                    verdict = line.split(":", 1)[1].strip()
                elif line.startswith("Confidence:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                        confidence = max(0.0, min(1.0, confidence))  # Clamp 0-1
                    except ValueError:
                        confidence = 0.5
                elif line.startswith("Explanation:"):
                    explanation = line.split(":", 1)[1].strip()

            supported = verdict in ["SUPPORTED", "PARTIALLY_SUPPORTED"]

            return {
                "claim": claim.claim_text,
                "supported": supported,
                "confidence": confidence,
                "verdict": verdict,
                "explanation": explanation,
                "cited_source_count": len(valid_citations),
            }

        except Exception as e:
            logger.error(f"Citation verification failed: {e}")
            return {
                "claim": claim.claim_text,
                "supported": None,  # Unknown
                "confidence": 0.0,
                "verdict": "ERROR",
                "explanation": f"Verification error: {str(e)}",
                "cited_source_count": len(valid_citations),
            }

    def verify_summary(self, summary: CitedMedicalSummary) -> dict:
        """
        Verify all claims in a cited medical summary.

        Args:
            summary: CitedMedicalSummary with cited_causes, cited_symptoms, cited_treatments

        Returns:
            Dictionary with aggregate verification results:
                - total_claims: int
                - supported_claims: int
                - partially_supported_claims: int
                - unsupported_claims: int
                - verification_score: float (0-1, % of supported claims)
                - details: list of per-claim verification results
                - by_category: dict with breakdown by claim type

        Example:
            >>> result = verifier.verify_summary(cited_summary)
            >>> result['verification_score']
            0.92  # 92% of claims supported
        """
        # Collect all claims
        all_claims = []
        claim_categories = []

        for claim in summary.cited_causes:
            all_claims.append(claim)
            claim_categories.append("cause")

        for claim in summary.cited_symptoms:
            all_claims.append(claim)
            claim_categories.append("symptom")

        for claim in summary.cited_treatments:
            all_claims.append(claim)
            claim_categories.append("treatment")

        if not all_claims:
            return {
                "total_claims": 0,
                "supported_claims": 0,
                "partially_supported_claims": 0,
                "unsupported_claims": 0,
                "verification_score": 0.0,
                "details": [],
                "by_category": {},
            }

        # Verify each claim
        results = []
        supported_count = 0
        partially_supported_count = 0
        unsupported_count = 0

        logger.info(f"Verifying {len(all_claims)} claims...")

        for i, (claim, category) in enumerate(zip(all_claims, claim_categories), 1):
            logger.debug(f"Verifying claim {i}/{len(all_claims)}: {claim.claim_text[:50]}...")

            result = self.verify_claim(claim, summary.sources)
            result["category"] = category
            results.append(result)

            if result["verdict"] == "SUPPORTED":
                supported_count += 1
            elif result["verdict"] == "PARTIALLY_SUPPORTED":
                partially_supported_count += 1
            elif result["verdict"] == "NOT_SUPPORTED":
                unsupported_count += 1

        # Calculate verification score (count both supported and partially_supported as "acceptable")
        acceptable_count = supported_count + partially_supported_count
        verification_score = acceptable_count / len(all_claims) if all_claims else 0.0

        # Group by category
        by_category = {"cause": [], "symptom": [], "treatment": []}
        for result in results:
            category = result["category"]
            by_category[category].append(result)

        # Calculate category scores
        category_scores = {}
        for category, cat_results in by_category.items():
            if cat_results:
                cat_supported = sum(
                    1
                    for r in cat_results
                    if r["verdict"] in ["SUPPORTED", "PARTIALLY_SUPPORTED"]
                )
                category_scores[category] = {
                    "total": len(cat_results),
                    "supported": cat_supported,
                    "score": cat_supported / len(cat_results),
                }
            else:
                category_scores[category] = {"total": 0, "supported": 0, "score": 0.0}

        return {
            "total_claims": len(all_claims),
            "supported_claims": supported_count,
            "partially_supported_claims": partially_supported_count,
            "unsupported_claims": unsupported_count,
            "verification_score": verification_score,
            "details": results,
            "by_category": category_scores,
        }

    def format_verification_report(self, verification_result: dict) -> str:
        """
        Format verification results as human-readable report.

        Args:
            verification_result: Output from verify_summary()

        Returns:
            Formatted text report
        """
        report = []
        report.append("=" * 80)
        report.append("CITATION VERIFICATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary stats
        report.append("📊 Overall Statistics:")
        report.append(f"  • Total Claims: {verification_result['total_claims']}")
        report.append(f"  • Supported: {verification_result['supported_claims']}")
        report.append(
            f"  • Partially Supported: {verification_result['partially_supported_claims']}"
        )
        report.append(f"  • Not Supported: {verification_result['unsupported_claims']}")
        report.append(
            f"  • Verification Score: {verification_result['verification_score']:.2%}"
        )
        report.append("")

        # By category
        report.append("📋 By Category:")
        for category, scores in verification_result["by_category"].items():
            if scores["total"] > 0:
                report.append(
                    f"  • {category.capitalize()}: {scores['supported']}/{scores['total']} "
                    f"({scores['score']:.2%})"
                )
        report.append("")

        # Detailed results (only show unsupported/partially)
        unsupported_details = [
            d
            for d in verification_result["details"]
            if d["verdict"] in ["NOT_SUPPORTED", "PARTIALLY_SUPPORTED"]
        ]

        if unsupported_details:
            report.append("⚠️  Claims Needing Review:")
            for detail in unsupported_details:
                report.append(f"\n  [{detail['category'].upper()}] {detail['claim']}")
                report.append(f"    Verdict: {detail['verdict']}")
                report.append(f"    Confidence: {detail['confidence']:.2f}")
                report.append(f"    Explanation: {detail['explanation']}")
                report.append(f"    Sources Cited: {detail['cited_source_count']}")
        else:
            report.append("✅ All claims are well-supported by their cited sources!")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)
