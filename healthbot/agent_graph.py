"""
Agent Graph for HealthBot Phase 4 - GenAI Orchestration.

This implements a ReAct (Reason + Act) agent using LangGraph.

KEY ARCHITECTURE:
- LLM agent with tool calling capability
- Dynamic tool selection (LLM decides which tools to call)
- Multi-step reasoning and research
- Agent orchestrates YOUR custom tools (not generating from training data)

This is kept separate from graph.py (Phase 1-3 pipeline) to:
1. Preserve backward compatibility
2. Allow side-by-side comparison
3. Demonstrate both architectures
"""

from typing import Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from healthbot.config import settings
from healthbot.state import PatientState
from healthbot.agent_tools import get_all_tools
from healthbot.prompts_agent import get_agent_prompt
from healthbot.safety import check_emergency


def create_agent_workflow() -> StateGraph:
    """
    Create agent-based workflow for HealthBot.

    Workflow:
    1. Safety check (emergency detection)
    2. Agent node (LLM with tool calling)
    3. End

    Returns:
        Compiled LangGraph workflow
    """
    workflow = StateGraph(PatientState)

    # Add nodes
    workflow.add_node("safety_check", safety_check_node)
    workflow.add_node("agent", agent_node)

    # Set entry point
    workflow.set_entry_point("safety_check")

    # Add conditional edges
    workflow.add_conditional_edges(
        "safety_check",
        route_safety,
        {
            "emergency": END,  # End if emergency detected
            "safe": "agent",  # Continue to agent if safe
        }
    )

    # Agent goes to END
    workflow.add_edge("agent", END)

    return workflow.compile()


def safety_check_node(state: PatientState) -> Dict[str, Any]:
    """
    Check for medical emergencies before processing query.

    Args:
        state: Current conversation state

    Returns:
        Updated state with emergency detection results
    """
    from healthbot.safety import get_emergency_response

    topic = state.get("topic", "")

    # Check for emergency keywords
    is_emergency = check_emergency(topic)

    return {
        "emergency_detected": is_emergency,
        "summary": get_emergency_response() if is_emergency else None,
    }


def route_safety(state: PatientState) -> str:
    """Route based on emergency detection."""
    if state.get("emergency_detected", False):
        return "emergency"
    return "safe"


def agent_node(state: PatientState) -> Dict[str, Any]:
    """
    Main agent node with tool calling capability.

    This is where the LLM agent:
    1. Receives user query
    2. Reasons about which tools to use
    3. Calls tools (YOUR custom tools)
    4. Synthesizes results
    5. Returns cited response

    Args:
        state: Current conversation state

    Returns:
        Updated state with agent response
    """
    # Get topic and messages
    topic = state.get("topic", "")
    messages = state.get("messages", [])

    # Initialize LLM with tool calling (support both Gemini and OpenAI/Groq)
    if settings.LLM_PROVIDER == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0.0,
            google_api_key=settings.GOOGLE_API_KEY,
        )
    else:  # openai or groq
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )

    # Get all tools
    tools = get_all_tools()

    # Create ReAct agent
    # This is the KEY: LangGraph's prebuilt ReAct agent handles tool calling loop
    # Note: System prompt is added via messages, not state_modifier
    agent_executor = create_react_agent(
        llm,
        tools,
    )

    # Prepare input for agent
    if not messages:
        # First turn: use topic as initial message
        agent_input = {
            "messages": [
                SystemMessage(content=get_agent_prompt()),
                HumanMessage(content=topic)
            ]
        }
    else:
        # Follow-up turn: include conversation history
        agent_input = {
            "messages": messages + [HumanMessage(content=topic)]
        }

    # Invoke agent
    # Agent will automatically:
    # 1. Reason about task
    # 2. Decide which tools to call
    # 3. Call tools (can be multiple, sequential)
    # 4. Synthesize results
    result = agent_executor.invoke(agent_input)

    # Extract agent response and metadata
    agent_messages = result.get("messages", [])
    final_message = agent_messages[-1] if agent_messages else None

    # Track tools called (for observability)
    tools_called = []
    tool_results = []
    reasoning_steps = []

    # Parse intermediate steps if available
    for msg in agent_messages:
        if hasattr(msg, 'additional_kwargs'):
            tool_calls = msg.additional_kwargs.get('tool_calls', [])
            if tool_calls:
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get('name', 'unknown')
                        tools_called.append(tool_name)
                        reasoning_steps.append(f"Called tool: {tool_name}")

    # Extract final response
    summary = final_message.content if final_message else "No response generated"

    # Add medical disclaimer if not emergency
    if not state.get("emergency_detected", False):
        summary += "\n\n**Medical Disclaimer:** This is educational information, not medical advice. Please consult a healthcare provider for medical decisions."

    return {
        "summary": summary,
        "messages": agent_messages,
        "tools_called": tools_called,
        "tool_results": tool_results,
        "reasoning_steps": reasoning_steps,
        "agent_synthesis": summary,
        "disclaimer_shown": True,
    }


# Create and export the agent workflow
agent_app = create_agent_workflow()


def run_agent_query(query: str, patient_level: str = "beginner") -> Dict[str, Any]:
    """
    Convenience function to run a single query through the agent.

    Args:
        query: User question
        patient_level: Education level (beginner/intermediate/advanced)

    Returns:
        Dictionary with agent response and metadata

    Example:
        >>> result = run_agent_query("What's my BMI if I'm 70kg and 1.75m tall?")
        >>> print(result["summary"])
    """
    initial_state = {
        "topic": query,
        "patient_level": patient_level,
        "messages": [],
        "retrieved_docs": [],
        "retrieval_scores": [],
        "rag_context": "",
        "confidence_score": 0.0,
        "tool_calls": 0,
        "node_latencies": {},
        "token_usage": {},
        "emergency_detected": False,
        "disclaimer_shown": False,
        "reranker_used": False,
        "rerank_latency_ms": None,
        "query_intent": None,
        "query_complexity": None,
        "previous_topic": None,
        "conversation_turns": 0,
        "last_summary": None,
        "is_follow_up": False,
        "agent_plan": None,
        "tools_called": [],
        "tool_results": [],
        "agent_synthesis": None,
        "reasoning_steps": [],
        "multi_step_research": False,
        "tool_selection_rationale": None,
        "summary": None,
        "quiz": None,
        "quiz_answer": None,
        "grade": None,
    }

    # Run through agent workflow
    result = agent_app.invoke(initial_state)

    return result
