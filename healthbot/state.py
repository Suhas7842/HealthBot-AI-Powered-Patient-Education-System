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

    # Reranking tracking
    reranker_used: bool  # Whether reranking was applied
    rerank_latency_ms: float | None  # Reranking latency for observability

    # Query classification
    query_intent: str | None  # "informational", "diagnostic", "treatment", "preventive"
    query_complexity: str | None  # "simple", "moderate", "complex"

    # Conversation context
    previous_topic: str | None  # Last discussed topic
    conversation_turns: int  # Number of turns in current conversation
    last_summary: str | None  # Summary from previous turn for context
    is_follow_up: bool  # Whether current query is a follow-up

    # Agent-specific tracking (Phase 4)
    agent_plan: str | None  # Agent's reasoning/plan for tool selection
    tools_called: list[str]  # List of tools agent called (for observability)
    tool_results: list[dict]  # Results from each tool call
    agent_synthesis: str | None  # Agent's synthesized response
    reasoning_steps: list[str]  # Agent's step-by-step reasoning
    multi_step_research: bool  # Flag for complex research tasks
    tool_selection_rationale: str | None  # Why agent chose specific tools
