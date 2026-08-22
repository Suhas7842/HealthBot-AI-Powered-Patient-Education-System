"""
Streamlit UI for HealthBot - AI-Powered Patient Education System.
Provides interactive chat interface with metrics dashboard.
"""

import time

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from healthbot.evaluation.metrics import HealthBotMetrics
from healthbot.models import LLMWrapper
from healthbot.prompts import (
    QUIZ_PROMPT,
    SUMMARY_PROMPT,
    SYSTEM_PROMPT,
)
from healthbot.safety import MEDICAL_DISCLAIMER, check_emergency, get_emergency_response
from healthbot.schemas import MedicalSummary, QuizQuestion
from healthbot.tools import ToolSelector

# Page config
st.set_page_config(
    page_title="HealthBot - AI Medical Education",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "summary" not in st.session_state:
    st.session_state.summary = None
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "metrics_tracker" not in st.session_state:
    st.session_state.metrics_tracker = HealthBotMetrics()
if "tool_selector" not in st.session_state:
    st.session_state.tool_selector = ToolSelector()
if "llm_wrapper" not in st.session_state:
    st.session_state.llm_wrapper = LLMWrapper()

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        color: #856404;
    }
    .emergency-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-header">🏥 HealthBot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-Powered Patient Education System</div>',
    unsafe_allow_html=True,
)

# Sidebar - Metrics Dashboard
with st.sidebar:
    st.header("📊 System Metrics")

    # Calculate metrics
    metrics_tracker = st.session_state.metrics_tracker
    if metrics_tracker.run_history:
        metrics = metrics_tracker.calculate_metrics(recent_n=10)

        st.subheader("Recent Performance")

        # Latency
        st.metric("Avg Latency", f"{metrics['latency']['mean']:.2f}s", delta=None)

        # Retrieval
        st.metric("RAG Hit Rate", f"{metrics['retrieval']['rag_hit_rate'] * 100:.1f}%")

        # Cost
        st.metric(
            "Avg Cost/Query",
            f"${metrics['cost']['estimated_cost_usd'] / max(metrics['usage']['total_runs'], 1):.4f}",
        )

        # Total runs
        st.metric("Total Queries", metrics["usage"]["total_runs"])

        # Show details
        with st.expander("📈 Detailed Metrics"):
            st.write("**Latency**")
            st.write(f"- Median: {metrics['latency']['median']:.2f}s")
            st.write(f"- P95: {metrics['latency']['p95']:.2f}s")
            st.write(f"- P99: {metrics['latency']['p99']:.2f}s")

            st.write("**Retrieval**")
            st.write(f"- Mean Score: {metrics['retrieval']['mean_score']:.3f}")

            st.write("**Cost**")
            st.write(f"- Total Tokens: {metrics['cost']['total_tokens']:,}")
            st.write(f"- Total Cost: ${metrics['cost']['estimated_cost_usd']:.4f}")
    else:
        st.info("No metrics available yet. Start a conversation to generate metrics!")

    st.divider()

    # About
    with st.expander("ℹ️ About HealthBot"):
        st.write("""
        **HealthBot** is an AI-powered patient education system that provides
        accurate, evidence-based medical information using RAG (Retrieval-Augmented Generation).

        **Features:**
        - 500+ PubMed articles
        - Hybrid retrieval (semantic + keyword)
        - Emergency detection
        - Interactive quizzes
        - Source citations
        """)

# Main content area
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📝 Quiz", "📚 Sources"])

# Tab 1: Chat Interface
with tab1:
    # Warning box
    st.markdown(
        f"""
    <div class="warning-box">
        <strong>⚠️ MEDICAL DISCLAIMER</strong><br><br>
        {MEDICAL_DISCLAIMER.strip()}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Chat messages container
    chat_container = st.container()

    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input (always at bottom)
    if prompt := st.chat_input("Ask about a medical condition..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Check for emergency
        if check_emergency(prompt):
            emergency_response = get_emergency_response()
            st.session_state.messages.append(
                {"role": "assistant", "content": emergency_response}
            )

            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(
                        f"""
                    <div class="emergency-box">
                        {emergency_response}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

        else:
            # Generate response
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Retrieving medical information..."):
                        start_time = time.time()

                        # Retrieve context
                        tool_selector = st.session_state.tool_selector
                        results = tool_selector.select_and_search(prompt, k=5)

                        if results["success"] and results["documents"]:
                            # Store sources for later
                            st.session_state.retrieved_docs = results["documents"]

                            # Generate summary
                            context = tool_selector.format_results(results)

                            llm_prompt = SUMMARY_PROMPT.format(
                                topic=prompt,
                                rag_context=context,
                                schema=MedicalSummary.model_json_schema(),
                            )

                            messages = [
                                SystemMessage(content=SYSTEM_PROMPT),
                                HumanMessage(content=llm_prompt),
                            ]

                            llm_wrapper = st.session_state.llm_wrapper
                            summary_obj = llm_wrapper.invoke_structured(
                                messages, MedicalSummary
                            )

                            # Format response
                            response_text = f"""**{summary_obj.title}**

**What is it?**
{summary_obj.condition}

**Causes:**
{chr(10).join(f"• {cause}" for cause in summary_obj.causes)}

**Symptoms:**
{chr(10).join(f"• {symptom}" for symptom in summary_obj.symptoms)}

**Treatment:**
{chr(10).join(f"• {treatment}" for treatment in summary_obj.treatment)}

---
{summary_obj.warning}
"""

                            st.markdown(response_text)
                            st.session_state.summary = summary_obj

                            # Log metrics
                            latency = time.time() - start_time
                            avg_score = sum(
                                doc.get("score", 0) for doc in results["documents"]
                            ) / len(results["documents"])

                            metrics_tracker = st.session_state.metrics_tracker
                            metrics_tracker.log_run(
                                {
                                    "topic": prompt,
                                    "total_latency": latency,
                                    "retrieval_score": avg_score,
                                    "num_retrieved_docs": len(results["documents"]),
                                    "tool_calls": 1,
                                    "confidence_score": 0.85,
                                    "emergency_detected": False,
                                    "used_rag": True,
                                }
                            )

                            # Show execution time
                            st.caption(
                                f"⚡ Generated in {latency:.2f}s"
                            )

                        else:
                            error_msg = "I couldn't find relevant medical information. Please try rephrasing your question."
                            st.error(error_msg)
                            response_text = error_msg

                        st.session_state.messages.append(
                            {"role": "assistant", "content": response_text}
                        )

# Tab 2: Quiz
with tab2:
    st.subheader("📝 Comprehension Quiz")

    if st.session_state.summary:
        if st.button("Generate Quiz Question", type="primary"):
            with st.spinner("Creating quiz..."):
                # Generate quiz
                summary_text = str(st.session_state.summary.condition)

                quiz_prompt = QUIZ_PROMPT.format(
                    summary=summary_text, schema=QuizQuestion.model_json_schema()
                )

                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=quiz_prompt),
                ]

                llm_wrapper = st.session_state.llm_wrapper
                quiz_obj = llm_wrapper.invoke_structured(messages, QuizQuestion)

                st.session_state.quiz = quiz_obj

        # Display quiz
        if st.session_state.quiz:
            quiz = st.session_state.quiz

            st.markdown(f"**Question:** {quiz.question}")

            answer = st.radio(
                "Select your answer:",
                options=["A", "B", "C", "D"],
                format_func=lambda x: f"{x}) {quiz.choices['ABCD'.index(x)]}",
            )

            if st.button("Submit Answer"):
                correct = answer == quiz.correct_answer

                if correct:
                    st.success(f"✅ Correct! {quiz.explanation}")
                else:
                    st.error(
                        f"❌ Incorrect. The correct answer is {quiz.correct_answer}."
                    )
                    st.info(f"💡 {quiz.explanation}")
    else:
        st.info("Start a conversation in the Chat tab to generate a quiz!")

# Tab 3: Sources
with tab3:
    st.subheader("📚 Source Documents")

    if hasattr(st.session_state, "retrieved_docs") and st.session_state.retrieved_docs:
        for i, doc in enumerate(st.session_state.retrieved_docs, 1):
            with st.expander(
                f"Source {i}: {doc.get('metadata', {}).get('title', 'Unknown')[:80]}..."
            ):
                st.write(f"**Relevance Score:** {doc.get('score', 0):.3f}")
                st.write(f"**PMID:** {doc.get('metadata', {}).get('pmid', 'N/A')}")
                st.write(
                    f"**Condition:** {doc.get('metadata', {}).get('condition', 'N/A')}"
                )
                st.divider()
                st.write(doc["text"])
    else:
        st.info(
            "No sources available yet. Ask a question in the Chat tab to retrieve medical information!"
        )

# Footer
st.divider()
st.caption("Built with LangGraph, ChromaDB, and OpenAI | v2.0.0")
