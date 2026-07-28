"""
LangGraph node functions for HealthBot workflow.
Each node performs a specific step in the patient education pipeline.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from healthbot.logger import log_node_execution, logger
from healthbot.models import LLMWrapper
from healthbot.prompts import (
    EVALUATION_PROMPT,
    QUIZ_PROMPT,
    SUMMARY_PROMPT,
    SYSTEM_PROMPT,
)
from healthbot.safety import check_emergency, get_emergency_response
from healthbot.schemas import MedicalSummary, QuizEvaluation, QuizQuestion
from healthbot.state import PatientState
from healthbot.tools import ToolSelector

# Initialize components (shared across nodes)
llm_wrapper = LLMWrapper()
tool_selector = ToolSelector()


@log_node_execution("collect_topic")
def collect_patient_topic(state: PatientState) -> dict:
    """
    Collect the health topic from patient and initialize conversation.

    Args:
        state: Current workflow state

    Returns:
        Updated state with topic and initial messages
    """
    topic = state.get("topic", "")

    if not topic:
        topic = input("\nWhich health condition would you like to learn about? ")

    # Initialize state
    initial_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"I want to learn about: {topic}"),
    ]

    return {
        "topic": topic,
        "messages": initial_messages,
        "patient_level": "beginner",  # Default level
        "tool_calls": 0,
        "node_latencies": {},
        "emergency_detected": False,
        "disclaimer_shown": False,
        "confidence_score": 0.0,
    }


@log_node_execution("check_safety")
def check_safety_node(state: PatientState) -> dict:
    """
    Check if query contains emergency keywords.

    Args:
        state: Current workflow state

    Returns:
        Updated state with emergency detection result
    """
    topic = state.get("topic", "")
    is_emergency = check_emergency(topic)

    if is_emergency:
        logger.warning(f"Emergency detected in query: '{topic}'")
        emergency_message = get_emergency_response()

        return {
            "emergency_detected": True,
            "messages": state["messages"] + [AIMessage(content=emergency_message)],
            "summary": emergency_message,
        }

    return {"emergency_detected": False}


@log_node_execution("retrieve_knowledge")
def retrieve_medical_knowledge(state: PatientState) -> dict:
    """
    Retrieve relevant medical information using RAG or Tavily.

    Args:
        state: Current workflow state

    Returns:
        Updated state with retrieved documents and context
    """
    topic = state.get("topic", "")

    logger.info(f"Retrieving knowledge for: '{topic}'")

    # Use tool selector to get best results
    results = tool_selector.select_and_search(topic, k=5)

    if not results["success"] or not results["documents"]:
        logger.warning("No results found for query")
        return {
            "retrieved_docs": [],
            "retrieval_scores": [],
            "rag_context": "No relevant medical information was found for this topic.",
            "tool_calls": state.get("tool_calls", 0) + 1,
        }

    # Extract documents and scores
    documents = results["documents"]
    scores = [doc.get("score", 0.0) for doc in documents]

    # Format context
    context = tool_selector.format_results(results)

    logger.info(
        f"Retrieved {len(documents)} documents (avg score: {sum(scores) / len(scores):.3f})"
    )

    return {
        "retrieved_docs": documents,
        "retrieval_scores": scores,
        "rag_context": context,
        "tool_calls": state.get("tool_calls", 0) + 1,
    }


@log_node_execution("generate_summary")
def generate_grounded_summary(state: PatientState) -> dict:
    """
    Generate patient-friendly summary grounded in retrieved documents.

    Args:
        state: Current workflow state

    Returns:
        Updated state with generated summary
    """
    topic = state.get("topic", "")
    context = state.get("rag_context", "")

    # Build prompt
    prompt = SUMMARY_PROMPT.format(
        topic=topic, rag_context=context, schema=MedicalSummary.model_json_schema()
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

    # Generate structured summary
    try:
        summary_obj = llm_wrapper.invoke_structured(messages, MedicalSummary)

        # Format as readable text
        summary_text = f"""
**{summary_obj.title}**

**What is it?**
{summary_obj.condition}

**Causes:**
{chr(10).join(f"• {cause}" for cause in summary_obj.causes)}

**Symptoms:**
{chr(10).join(f"• {symptom}" for symptom in summary_obj.symptoms)}

**Treatment:**
{chr(10).join(f"• {treatment}" for treatment in summary_obj.treatment)}

{summary_obj.warning}
"""

        # Estimate tokens
        token_count = llm_wrapper.estimate_tokens(summary_text)

        return {
            "summary": summary_text.strip(),
            "token_usage": {"summary_tokens": token_count},
            "confidence_score": 0.85,  # High confidence for structured output
            "disclaimer_shown": True,
        }

    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        fallback = f"Unable to generate summary for {topic}. Please consult a healthcare professional."
        return {"summary": fallback, "confidence_score": 0.0}


@log_node_execution("present_summary")
def present_summary_to_patient(state: PatientState) -> dict:
    """
    Display the generated summary to the patient.

    Args:
        state: Current workflow state

    Returns:
        State (no updates)
    """
    summary = state.get("summary", "")

    print("\n" + "=" * 80)
    print("MEDICAL INFORMATION SUMMARY")
    print("=" * 80)
    print(summary)
    print("=" * 80)

    return {}


@log_node_execution("wait_for_quiz")
def wait_for_quiz_ready(state: PatientState) -> dict:
    """
    Pause and ask if patient is ready for the quiz.

    Args:
        state: Current workflow state

    Returns:
        State (no updates)
    """
    input("\nPress Enter when you're ready for the comprehension quiz...")
    return {}


@log_node_execution("generate_quiz")
def create_quiz_question(state: PatientState) -> dict:
    """
    Generate a multiple-choice quiz question from the summary.

    Args:
        state: Current workflow state

    Returns:
        Updated state with quiz question
    """
    summary = state.get("summary", "")

    # Build prompt
    prompt = QUIZ_PROMPT.format(
        summary=summary, schema=QuizQuestion.model_json_schema()
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

    # Generate quiz
    try:
        quiz_obj = llm_wrapper.invoke_structured(messages, QuizQuestion)

        # Format quiz as text
        quiz_text = f"""
{quiz_obj.question}

A) {quiz_obj.choices[0]}
B) {quiz_obj.choices[1]}
C) {quiz_obj.choices[2]}
D) {quiz_obj.choices[3]}
"""

        return {
            "quiz": quiz_text.strip(),
            "quiz_ground_truth": f"{quiz_obj.correct_answer}: {quiz_obj.explanation}",
        }

    except Exception as e:
        logger.error(f"Failed to generate quiz: {e}")
        return {"quiz": "Unable to generate quiz question.", "quiz_ground_truth": ""}


@log_node_execution("present_quiz")
def present_quiz_to_patient(state: PatientState) -> dict:
    """
    Display the quiz question to the patient.

    Args:
        state: Current workflow state

    Returns:
        State (no updates)
    """
    quiz = state.get("quiz", "")

    print("\n" + "=" * 80)
    print("COMPREHENSION QUIZ")
    print("=" * 80)
    print(quiz)
    print("=" * 80)

    return {}


@log_node_execution("collect_quiz_answer")
def collect_patient_answer(state: PatientState) -> dict:
    """
    Collect the patient's quiz answer.

    Args:
        state: Current workflow state

    Returns:
        Updated state with patient's answer
    """
    answer = input("\nYour answer (A, B, C, or D): ").strip().upper()

    return {"quiz_answer": answer}


@log_node_execution("evaluate_quiz")
def evaluate_quiz_response(state: PatientState) -> dict:
    """
    Grade the patient's quiz answer using LLM.

    Args:
        state: Current workflow state

    Returns:
        Updated state with grade and feedback
    """
    quiz = state.get("quiz", "")
    ground_truth = state.get("quiz_ground_truth", "")
    patient_answer = state.get("quiz_answer", "")
    summary = state.get("summary", "")

    # Build prompt
    prompt = EVALUATION_PROMPT.format(
        question=quiz,
        correct_answer=ground_truth,
        patient_answer=patient_answer,
        schema=QuizEvaluation.model_json_schema(),
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

    # Generate evaluation
    try:
        eval_obj = llm_wrapper.invoke_structured(messages, QuizEvaluation)

        grade_text = f"""
**Grade: {eval_obj.score}**

{eval_obj.feedback}

**How to improve:**
{eval_obj.improvements}
"""

        return {"grade": grade_text.strip()}

    except Exception as e:
        logger.error(f"Failed to evaluate quiz: {e}")
        return {"grade": "Unable to evaluate your answer."}


@log_node_execution("present_grade")
def present_grade_to_patient(state: PatientState) -> dict:
    """
    Display the quiz grade and feedback to the patient.

    Args:
        state: Current workflow state

    Returns:
        State (no updates)
    """
    grade = state.get("grade", "")

    print("\n" + "=" * 80)
    print("QUIZ EVALUATION")
    print("=" * 80)
    print(grade)
    print("=" * 80)

    return {}


@log_node_execution("ask_continue")
def ask_for_new_topic(state: PatientState) -> dict:
    """
    Ask if patient wants to learn about another topic.

    Args:
        state: Current workflow state

    Returns:
        Updated state with continuation decision
    """
    choice = (
        input("\nWould you like to learn about another health topic? (yes/no): ")
        .strip()
        .lower()
    )

    if choice == "yes":
        # Reset state for new topic
        return {
            "topic": "",
            "summary": "",
            "quiz": "",
            "quiz_answer": "",
            "quiz_ground_truth": "",
            "grade": "",
            "retrieved_docs": [],
            "retrieval_scores": [],
            "rag_context": "",
            "messages": [],
            "continue": True,
        }
    else:
        return {
            "continue": False,
            "messages": state.get("messages", [])
            + [AIMessage(content="Thank you for using HealthBot. Stay healthy!")],
        }
