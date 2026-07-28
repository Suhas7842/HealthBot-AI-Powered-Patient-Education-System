"""
State definition for HealthBot LangGraph workflow.
Tracks conversation state, RAG context, and execution metadata.
"""

from typing import TypedDict

from langchain_core.messages import BaseMessage


class PatientState(TypedDict):
    """
    Complete state for HealthBot conversation workflow.

    Tracks user interaction, retrieval context, LLM outputs, and observability metrics.
    """

    # Core conversation state
    messages: list[BaseMessage]
    topic: str
    patient_level: str  # "beginner", "intermediate", "advanced"

    # Content state
    summary: str | None
    quiz: str | None
    quiz_answer: str | None
    grade: str | None

    # RAG retrieval tracking
    retrieved_docs: list[dict]  # List of {title, abstract, source, pmid, score}
    retrieval_scores: list[float]  # Relevance scores for retrieved docs
    rag_context: str  # Formatted context string for LLM

    # Observability and metrics
    confidence_score: float  # LLM confidence (0-1)
    tool_calls: int  # Number of tool invocations
    node_latencies: dict[str, float]  # Per-node execution time
    token_usage: dict[str, int]  # {"prompt_tokens": X, "completion_tokens": Y}

    # Safety tracking
    emergency_detected: bool
    disclaimer_shown: bool
