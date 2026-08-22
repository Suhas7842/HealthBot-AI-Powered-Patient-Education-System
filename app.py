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

# Sidebar - User Panel
with st.sidebar:
    st.header("ℹ️ About HealthBot")

    st.write("""
    **HealthBot** provides evidence-based health education using medical literature
    and research from trusted sources.

    **What HealthBot does:**
    - Answers health questions using 700+ curated medical articles
    - Provides clear explanations with source citations
    - Detects medical emergencies and provides guidance
    - Offers interactive learning with quizzes

    **Important:**
    - ✅ Educational information only
    - ❌ Not a substitute for professional medical advice
    - ❌ Not for diagnosis or treatment
    - 🚨 For emergencies, call your local emergency number
    """)

    st.divider()

    # Usage stats (user-friendly)
    metrics_tracker = st.session_state.metrics_tracker
    if metrics_tracker.run_history:
        total_queries = len(metrics_tracker.run_history)
        st.metric("Questions Asked", total_queries)
        st.caption("Your conversation history this session")

    st.divider()

    # Help section
    with st.expander("💡 How to Use"):
        st.write("""
        **Getting Started:**
        1. Type your health question in the chat below
        2. Review the answer and source citations
        3. Check the **Sources** tab to see the medical literature
        4. Try the **Quiz** tab to test your understanding

        **Example Questions:**
        - "What is Type 2 diabetes?"
        - "What are the symptoms of hypertension?"
        - "How can I prevent heart disease?"

        **Tips:**
        - Be specific in your questions
        - Check the sources for detailed information
        - Remember: this is educational, not medical advice
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
    st.subheader("📝 Test Your Understanding")
    st.caption("Interactive quiz based on your health topic")

    if st.session_state.summary:
        if st.button("Generate Quiz Question", type="primary"):
            with st.spinner("Creating your quiz..."):
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
                "Choose your answer:",
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
        st.info("Ask a health question in the Chat tab, then come back here to test your understanding!")

# Tab 3: Sources
with tab3:
    st.subheader("📚 Medical Sources")
    st.caption("Evidence used to answer your question")

    if hasattr(st.session_state, "retrieved_docs") and st.session_state.retrieved_docs:
        for i, doc in enumerate(st.session_state.retrieved_docs, 1):
            with st.expander(
                f"📄 Source {i}: {doc.get('metadata', {}).get('title', 'Unknown')[:80]}..."
            ):
                # Show user-friendly metadata only
                pmid = doc.get('metadata', {}).get('pmid', 'N/A')
                if pmid != 'N/A':
                    st.caption(f"PubMed ID: {pmid}")

                condition = doc.get('metadata', {}).get('condition', '')
                if condition:
                    st.caption(f"Related to: {condition}")

                st.divider()
                st.write(doc["text"])
    else:
        st.info(
            "Ask a health question in the Chat tab to see the medical sources used for the answer."
        )

# Footer
st.divider()
st.caption("HealthBot — Evidence-Based Health Education | Always consult healthcare professionals for medical decisions")
