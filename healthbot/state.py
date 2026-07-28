"""
State definition for HealthBot LangGraph workflow.
Tracks conversation state, RAG context, and execution metadata.
"""

from typing import TypedDict, List, Optional, Dict
from langchain_core.messages import BaseMessage


class PatientState(TypedDict):
    """
    Complete state for HealthBot conversation workflow.

    Tracks user interaction, retrieval context, LLM outputs, and observability metrics.
    """

    # Core conversation state
    messages: List[BaseMessage]
    topic: str
    patient_level: str  # "beginner", "intermediate", "advanced"

    # Content state
    summary: Optional[str]
    quiz: Optional[str]
    quiz_answer: Optional[str]
    grade: Optional[str]

    # RAG retrieval tracking
    retrieved_docs: List[Dict]  # List of {title, abstract, source, pmid, score}
    retrieval_scores: List[float]  # Relevance scores for retrieved docs
    rag_context: str  # Formatted context string for LLM

    # Observability and metrics
    confidence_score: float  # LLM confidence (0-1)
    tool_calls: int  # Number of tool invocations
    node_latencies: Dict[str, float]  # Per-node execution time
    token_usage: Dict[str, int]  # {"prompt_tokens": X, "completion_tokens": Y}

    # Safety tracking
    emergency_detected: bool
    disclaimer_shown: bool
