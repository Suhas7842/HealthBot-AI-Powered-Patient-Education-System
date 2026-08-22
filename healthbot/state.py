"""
State definition for HealthBot GenAI agent workflow.

Tracks only actively used state fields for agent-based workflow (Phase 4).
Unused fields from Phase 3 pipeline removed to avoid premature abstractions.
"""

from typing import TypedDict

from langchain_core.messages import BaseMessage


class PatientState(TypedDict):
    """
    State for HealthBot agent workflow.

    Includes only fields that are actively populated and used by the agent system.
    Fields removed: RAG pipeline-specific, unused observability, conversation memory,
    educational features (quiz), and speculative agent planning fields.
    """

    # Core input
    topic: str  # User's query
    patient_level: str  # "beginner", "intermediate", "advanced"

    # Agent conversation
    messages: list[BaseMessage]  # LangChain message history

    # Output
    summary: str | None  # Agent's final response

    # Safety
    emergency_detected: bool  # Whether emergency keywords detected
    disclaimer_shown: bool  # Whether medical disclaimer was shown

    # Query classification
    query_type: str | None  # "normal" or "research" - tracks which workflow was used

    # Agent tool orchestration tracking
    tools_called: list[str]  # Names of tools agent called
    tool_results: list[dict]  # Results from each tool call
    tool_call_trace: list[str]  # Trace of which tools were called (not internal reasoning)
    agent_synthesis: str | None  # Agent's synthesized answer
