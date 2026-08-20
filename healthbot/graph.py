"""
LangGraph workflow orchestration for HealthBot.
Defines the complete patient education pipeline as a state machine.
"""

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from healthbot.logger import logger
from healthbot.nodes import (
    ask_for_new_topic,
    check_safety_node,
    collect_patient_answer,
    collect_patient_topic,
    create_quiz_question,
    evaluate_quiz_response,
    generate_grounded_summary,
    no_evidence_fallback,
    present_grade_to_patient,
    present_quiz_to_patient,
    present_summary_to_patient,
    retrieve_medical_knowledge,
    validate_evidence,
    wait_for_quiz_ready,
)
from healthbot.state import PatientState


def decide_safety_path(state: PatientState) -> Literal["emergency_exit", "retrieve"]:
    """
    Decide whether to exit due to emergency or continue to retrieval.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    if state.get("emergency_detected", False):
        logger.warning("Emergency detected - exiting workflow")
        return "emergency_exit"
    return "retrieve"


def decide_evidence_path(
    state: PatientState,
) -> Literal["generate_summary", "no_evidence_fallback"]:
    """
    Decide whether to generate summary or use fallback based on evidence quality.

    This prevents hallucination by validating retrieved context before generation.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    if state.get("evidence_valid", True):  # Default to True for backward compatibility
        logger.info("Evidence validation passed - proceeding to generation")
        return "generate_summary"
    else:
        reason = state.get("validation_reason", "unknown")
        logger.warning(f"Evidence validation failed: {reason} - using fallback")
        return "no_evidence_fallback"


def decide_continue(state: PatientState) -> Literal["collect_topic", "end"]:
    """
    Decide whether to continue with new topic or end conversation.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    if state.get("continue", False):
        logger.info("User wants to continue with new topic")
        return "collect_topic"
    else:
        logger.info("User ending conversation")
        return "end"


# Build the state graph
workflow = StateGraph(PatientState)

# Add all nodes
workflow.add_node("collect_topic", collect_patient_topic)
workflow.add_node("check_safety", check_safety_node)
workflow.add_node("retrieve", retrieve_medical_knowledge)
workflow.add_node("validate_evidence", validate_evidence)
workflow.add_node("generate_summary", generate_grounded_summary)
workflow.add_node("no_evidence_fallback", no_evidence_fallback)
workflow.add_node("present_summary", present_summary_to_patient)
workflow.add_node("wait_quiz", wait_for_quiz_ready)
workflow.add_node("generate_quiz", create_quiz_question)
workflow.add_node("present_quiz", present_quiz_to_patient)
workflow.add_node("collect_answer", collect_patient_answer)
workflow.add_node("evaluate", evaluate_quiz_response)
workflow.add_node("present_grade", present_grade_to_patient)
workflow.add_node("ask_continue", ask_for_new_topic)
workflow.add_node(
    "emergency_exit", present_summary_to_patient
)  # Reuse for emergency message

# Define workflow edges
workflow.add_edge(START, "collect_topic")
workflow.add_edge("collect_topic", "check_safety")

# Safety routing: emergency keywords trigger immediate exit, normal queries proceed
workflow.add_conditional_edges(
    "check_safety",
    decide_safety_path,
    {"emergency_exit": "emergency_exit", "retrieve": "retrieve"},
)

# Emergency path: bypass RAG and end immediately with emergency message
workflow.add_edge("emergency_exit", END)

# Normal educational flow: retrieve → validate → generate → present → quiz → evaluate → repeat
# Evidence validation prevents hallucination by checking retrieval quality
workflow.add_edge("retrieve", "validate_evidence")

# Evidence routing: valid evidence proceeds to generation, insufficient triggers fallback
workflow.add_conditional_edges(
    "validate_evidence",
    decide_evidence_path,
    {
        "generate_summary": "generate_summary",  # High-quality evidence → full workflow
        "no_evidence_fallback": "no_evidence_fallback",  # Low-quality → safe fallback
    },
)

# Normal path: generate and present summary, then quiz
workflow.add_edge("generate_summary", "present_summary")

# Fallback path: skip quiz, go directly to ask_continue
workflow.add_edge("no_evidence_fallback", "ask_continue")
workflow.add_edge("present_summary", "wait_quiz")
workflow.add_edge("wait_quiz", "generate_quiz")  # Quiz based on presented summary
workflow.add_edge("generate_quiz", "present_quiz")
workflow.add_edge("present_quiz", "collect_answer")
workflow.add_edge("collect_answer", "evaluate")  # Evaluate user's quiz response
workflow.add_edge("evaluate", "present_grade")
workflow.add_edge("present_grade", "ask_continue")

# Loop or exit: user chooses to learn another topic or end conversation
workflow.add_conditional_edges(
    "ask_continue",
    decide_continue,
    {
        "collect_topic": "collect_topic",  # Loop back to start for new topic
        "end": END,
    },
)

# Compile graph with memory
memory = MemorySaver()
healthbot_app = workflow.compile(checkpointer=memory)

logger.info("HealthBot workflow compiled successfully")


def run_healthbot():
    """
    Run the HealthBot application with a new conversation.
    """
    from langchain_core.runnables import RunnableConfig

    print("\n" + "=" * 80)
    print("🏥 HEALTHBOT - AI-Powered Patient Education System")
    print("=" * 80)
    print("\nWelcome! I'm here to help you understand medical conditions.")
    print("I can provide educational information about common health topics.")
    print("\n⚠️  Remember: This is for educational purposes only.")
    print("Always consult a healthcare professional for medical advice.")
    print("=" * 80)

    # Configuration
    config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "1"})

    # Initial state
    initial_state = {
        "topic": "",
        "patient_level": "beginner",
        "messages": [],
        "summary": "",
        "quiz": "",
        "quiz_answer": "",
        "quiz_ground_truth": "",
        "grade": "",
        "retrieved_docs": [],
        "retrieval_scores": [],
        "rag_context": "",
        "confidence_score": 0.0,
        "tool_calls": 0,
        "node_latencies": {},
        "token_usage": {},
        "emergency_detected": False,
        "disclaimer_shown": False,
        "evidence_valid": True,  # Tracks evidence validation status
        "validation_reason": "",  # Explains validation result
    }

    try:
        # Run the workflow
        result = healthbot_app.invoke(initial_state, config)

        # Print execution summary
        print("\n" + "=" * 80)
        print("EXECUTION SUMMARY")
        print("=" * 80)

        # Node latencies
        if result.get("node_latencies"):
            print("\nNode Execution Times:")
            for node, latency in result["node_latencies"].items():
                print(f"  • {node}: {latency:.2f}s")

            total_latency = sum(result["node_latencies"].values())
            print(f"\n  Total: {total_latency:.2f}s")

        # Tool calls
        tool_calls = result.get("tool_calls", 0)
        print(f"\nTool Calls: {tool_calls}")

        # Confidence
        confidence = result.get("confidence_score", 0.0)
        print(f"Confidence Score: {confidence:.2f}")

        # Token usage (if available)
        if result.get("token_usage"):
            print(f"\nEstimated Tokens: {sum(result['token_usage'].values())}")

        print("=" * 80)
        print("\nThank you for using HealthBot! 🏥")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nConversation interrupted by user.")
        logger.info("User interrupted conversation")
    except Exception as e:
        print(f"\n\nError: {e}")
        logger.error(f"Workflow error: {e}", exc_info=True)


def visualize_graph(output_path: str = "docs/healthbot_graph.png"):
    """
    Generate a visual representation of the workflow graph.

    Args:
        output_path: Path to save the graph image
    """
    try:
        from IPython.display import Image, display

        # Generate graph visualization
        graph_image = healthbot_app.get_graph().draw_mermaid_png()

        # Save to file
        with open(output_path, "wb") as f:
            f.write(graph_image)

        print(f"Graph visualization saved to: {output_path}")

        # Display if in Jupyter
        try:
            display(Image(graph_image))
        except:
            pass

    except Exception as e:
        logger.error(f"Failed to visualize graph: {e}")
        print(f"Could not generate graph visualization: {e}")


if __name__ == "__main__":
    run_healthbot()
