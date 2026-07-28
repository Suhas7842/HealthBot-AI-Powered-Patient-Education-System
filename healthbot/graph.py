"""
LangGraph workflow orchestration for HealthBot.
Defines the complete patient education pipeline as a state machine.
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from healthbot.state import PatientState
from healthbot.nodes import (
    collect_patient_topic,
    check_safety_node,
    retrieve_medical_knowledge,
    generate_grounded_summary,
    present_summary_to_patient,
    wait_for_quiz_ready,
    create_quiz_question,
    present_quiz_to_patient,
    collect_patient_answer,
    evaluate_quiz_response,
    present_grade_to_patient,
    ask_for_new_topic
)
from healthbot.logger import logger


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
workflow.add_node("generate_summary", generate_grounded_summary)
workflow.add_node("present_summary", present_summary_to_patient)
workflow.add_node("wait_quiz", wait_for_quiz_ready)
workflow.add_node("generate_quiz", create_quiz_question)
workflow.add_node("present_quiz", present_quiz_to_patient)
workflow.add_node("collect_answer", collect_patient_answer)
workflow.add_node("evaluate", evaluate_quiz_response)
workflow.add_node("present_grade", present_grade_to_patient)
workflow.add_node("ask_continue", ask_for_new_topic)
workflow.add_node("emergency_exit", present_summary_to_patient)  # Reuse for emergency message

# Define workflow edges
workflow.add_edge(START, "collect_topic")
workflow.add_edge("collect_topic", "check_safety")

# Conditional: emergency or continue
workflow.add_conditional_edges(
    "check_safety",
    decide_safety_path,
    {
        "emergency_exit": "emergency_exit",
        "retrieve": "retrieve"
    }
)

# Emergency path
workflow.add_edge("emergency_exit", END)

# Normal flow
workflow.add_edge("retrieve", "generate_summary")
workflow.add_edge("generate_summary", "present_summary")
workflow.add_edge("present_summary", "wait_quiz")
workflow.add_edge("wait_quiz", "generate_quiz")
workflow.add_edge("generate_quiz", "present_quiz")
workflow.add_edge("present_quiz", "collect_answer")
workflow.add_edge("collect_answer", "evaluate")
workflow.add_edge("evaluate", "present_grade")
workflow.add_edge("present_grade", "ask_continue")

# Conditional: continue or end
workflow.add_conditional_edges(
    "ask_continue",
    decide_continue,
    {
        "collect_topic": "collect_topic",
        "end": END
    }
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

    print("\n" + "="*80)
    print("🏥 HEALTHBOT - AI-Powered Patient Education System")
    print("="*80)
    print("\nWelcome! I'm here to help you understand medical conditions.")
    print("I can provide educational information about common health topics.")
    print("\n⚠️  Remember: This is for educational purposes only.")
    print("Always consult a healthcare professional for medical advice.")
    print("="*80)

    # Configuration
    config = RunnableConfig(
        recursion_limit=100,
        configurable={"thread_id": "1"}
    )

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
        "disclaimer_shown": False
    }

    try:
        # Run the workflow
        result = healthbot_app.invoke(initial_state, config)

        # Print execution summary
        print("\n" + "="*80)
        print("EXECUTION SUMMARY")
        print("="*80)

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

        print("="*80)
        print("\nThank you for using HealthBot! 🏥")
        print("="*80)

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
