"""
Pydantic models for structured LLM outputs.
Ensures type-safe, validated responses from the language model.
"""

from pydantic import BaseModel, Field


class MedicalSummary(BaseModel):
    """Structured medical information summary."""

    title: str = Field(description="Condition or topic name")
    condition: str = Field(description="Brief condition description")
    causes: list[str] = Field(description="List of causes or risk factors")
    symptoms: list[str] = Field(description="Common symptoms")
    treatment: list[str] = Field(description="Treatment options")
    warning: str = Field(
        default="This information is for educational purposes only. Always consult a qualified healthcare professional for medical advice.",
        description="Medical disclaimer",
    )


class QuizQuestion(BaseModel):
    """Structured quiz question with multiple choice options."""

    question: str = Field(description="The quiz question text")
    choices: list[str] = Field(description="Four answer choices (A, B, C, D)")
    correct_answer: str = Field(description="The correct answer letter (A, B, C, or D)")
    explanation: str = Field(description="Explanation of why the answer is correct")


class QuizEvaluation(BaseModel):
    """Structured quiz answer evaluation."""

    score: str = Field(description="Letter grade: A, B, C, D, or F")
    feedback: str = Field(description="Detailed feedback on the answer")
    improvements: str = Field(description="Suggestions for improvement")


class RetrievedDocument(BaseModel):
    """Metadata for a retrieved medical document."""

    title: str = Field(description="Article or document title")
    abstract: str = Field(description="Article abstract or excerpt")
    source: str = Field(description="Source database (e.g., PubMed)")
    pmid: str = Field(description="PubMed ID or document identifier")
    relevance_score: float = Field(description="Retrieval relevance score (0-1)")


class CitedClaim(BaseModel):
    """A single medical claim with source citations (Phase 2C)."""

    claim_text: str = Field(description="The specific claim or statement")
    citation_ids: list[int] = Field(
        description="List of source indices supporting this claim (1-indexed)"
    )
    confidence: float = Field(
        description="Confidence in citation accuracy (0-1)",
        default=1.0,
        ge=0.0,
        le=1.0,
    )


class CitedMedicalSummary(BaseModel):
    """Medical summary with claim-level citations for explainability (Phase 2C)."""

    title: str = Field(description="Condition or topic name")
    condition: str = Field(description="Brief condition description")

    # Claims with source attribution
    cited_causes: list[CitedClaim] = Field(
        description="Causes with source citations (each claim references specific sources)"
    )
    cited_symptoms: list[CitedClaim] = Field(
        description="Symptoms with source citations"
    )
    cited_treatments: list[CitedClaim] = Field(
        description="Treatments with source citations"
    )

    # Source reference map
    sources: list[dict] = Field(
        description="List of source documents with metadata (index matches citation_ids)"
    )

    warning: str = Field(
        default="This information is for educational purposes only. Always consult a qualified healthcare professional for medical advice.",
        description="Medical disclaimer",
    )
