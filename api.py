"""
FastAPI backend for HealthBot.
Provides RESTful API endpoints for medical education system.
"""

import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from healthbot.tools import ToolSelector
from healthbot.models import LLMWrapper
from healthbot.schemas import MedicalSummary, QuizQuestion
from healthbot.safety import check_emergency, get_emergency_response
from healthbot.prompts import SYSTEM_PROMPT, SUMMARY_PROMPT, QUIZ_PROMPT
from healthbot.evaluation.metrics import HealthBotMetrics
from healthbot.logger import logger

# Initialize FastAPI app
app = FastAPI(
    title="HealthBot API",
    description="AI-Powered Patient Education System with RAG",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
tool_selector = ToolSelector()
llm_wrapper = LLMWrapper()
metrics_tracker = HealthBotMetrics()


# Request/Response Models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User's health question", min_length=1)
    include_sources: bool = Field(default=True, description="Include source citations")


class SourceDocument(BaseModel):
    """Model for source document metadata."""
    title: str
    text: str
    score: float
    pmid: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Generated medical information")
    summary: Optional[MedicalSummary] = Field(None, description="Structured summary")
    sources: List[SourceDocument] = Field(default_factory=list, description="Source documents")
    metadata: dict = Field(default_factory=dict, description="Execution metadata")


class QuizRequest(BaseModel):
    """Request model for quiz generation."""
    summary: str = Field(..., description="Medical summary to create quiz from")


class QuizResponse(BaseModel):
    """Response model for quiz endpoint."""
    quiz: QuizQuestion
    metadata: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    components: dict


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""
    metrics: dict
    period: dict


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "HealthBot API",
        "version": "2.0.0",
        "description": "AI-Powered Patient Education System",
        "endpoints": {
            "chat": "/chat",
            "quiz": "/quiz",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Returns system status and component availability.
    """
    # Check components
    components = {
        "rag_tool": tool_selector.rag_tool.available,
        "tavily_tool": tool_selector.tavily_tool.available,
        "llm": True,  # If we got here, LLM config is loaded
        "vector_store": tool_selector.rag_tool.available
    }

    all_healthy = all(components.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="2.0.0",
        components=components
    )


@app.post("/chat", response_model=ChatResponse, tags=["Medical Education"])
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Main chat endpoint for medical education.

    Retrieves relevant medical information and generates patient-friendly summary.

    Args:
        request: Chat request with user message

    Returns:
        ChatResponse with generated content and sources
    """
    start_time = time.time()
    logger.info(f"Chat request: '{request.message}'")

    try:
        # Check for emergency
        if check_emergency(request.message):
            logger.warning(f"Emergency detected: '{request.message}'")
            return ChatResponse(
                response=get_emergency_response(),
                sources=[],
                metadata={
                    "emergency_detected": True,
                    "latency": time.time() - start_time
                }
            )

        # Retrieve context
        results = tool_selector.select_and_search(request.message, k=5)

        if not results["success"] or not results["documents"]:
            raise HTTPException(
                status_code=404,
                detail="No relevant medical information found"
            )

        # Extract and format context
        documents = results["documents"]
        context = tool_selector.format_results(results)

        # Generate structured summary
        prompt = SUMMARY_PROMPT.format(
            topic=request.message,
            rag_context=context,
            schema=MedicalSummary.model_json_schema()
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        summary_obj = llm_wrapper.invoke_structured(messages, MedicalSummary)

        # Format text response
        response_text = f"""**{summary_obj.title}**

**What is it?**
{summary_obj.condition}

**Causes:**
{chr(10).join(f'• {cause}' for cause in summary_obj.causes)}

**Symptoms:**
{chr(10).join(f'• {symptom}' for symptom in summary_obj.symptoms)}

**Treatment:**
{chr(10).join(f'• {treatment}' for treatment in summary_obj.treatment)}

{summary_obj.warning}
"""

        # Prepare source documents
        source_docs = []
        if request.include_sources:
            for doc in documents[:5]:
                source_docs.append(SourceDocument(
                    title=doc.get("metadata", {}).get("title", "Unknown"),
                    text=doc["text"][:300] + "...",  # Truncate for API response
                    score=doc.get("score", 0.0),
                    pmid=doc.get("metadata", {}).get("pmid")
                ))

        # Calculate latency
        latency = time.time() - start_time

        # Log metrics in background
        def log_metrics():
            metrics_tracker.log_run({
                "topic": request.message,
                "total_latency": latency,
                "retrieval_score": sum(doc.get("score", 0) for doc in documents) / len(documents),
                "num_retrieved_docs": len(documents),
                "tool_calls": 1,
                "confidence_score": 0.85,
                "emergency_detected": False,
                "used_rag": True
            })

        background_tasks.add_task(log_metrics)

        logger.info(f"Chat response generated in {latency:.2f}s")

        return ChatResponse(
            response=response_text.strip(),
            summary=summary_obj,
            sources=source_docs,
            metadata={
                "latency": latency,
                "num_sources": len(documents),
                "method": results.get("method", "unknown"),
                "emergency_detected": False
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz", response_model=QuizResponse, tags=["Medical Education"])
async def generate_quiz(request: QuizRequest):
    """
    Generate a quiz question from a medical summary.

    Args:
        request: Quiz request with summary text

    Returns:
        QuizResponse with generated question
    """
    start_time = time.time()
    logger.info("Quiz generation requested")

    try:
        # Generate quiz
        prompt = QUIZ_PROMPT.format(
            summary=request.summary,
            schema=QuizQuestion.model_json_schema()
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        quiz_obj = llm_wrapper.invoke_structured(messages, QuizQuestion)

        latency = time.time() - start_time
        logger.info(f"Quiz generated in {latency:.2f}s")

        return QuizResponse(
            quiz=quiz_obj,
            metadata={
                "latency": latency
            }
        )

    except Exception as e:
        logger.error(f"Quiz endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def get_metrics(recent_n: Optional[int] = None):
    """
    Get system performance metrics.

    Args:
        recent_n: Only return metrics for last N runs (optional)

    Returns:
        MetricsResponse with computed metrics
    """
    try:
        metrics = metrics_tracker.calculate_metrics(recent_n=recent_n)

        if "error" in metrics:
            raise HTTPException(status_code=404, detail=metrics["error"])

        return MetricsResponse(
            metrics=metrics,
            period=metrics.get("period", {})
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
