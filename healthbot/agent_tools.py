"""
Agent Tools for HealthBot Phase 4 - GenAI Orchestration.

Exposes custom tools to LLM agent via LangChain tool calling API.

KEY POINT: These are wrappers around YOUR infrastructure:
- medical_rag_search → YOUR hybrid retriever (Phase 1)
- medical_calculator → YOUR calculator tool (Phase 4)
- pubmed_api_search → YOUR PubMed client (Phase 4)
- web_search → YOUR Tavily wrapper (Phase 2)

The LLM acts as a ROUTER that decides which of YOUR tools to call.
This is NOT a ChatGPT wrapper - the value is in YOUR tool engineering.
"""

from typing import Dict, Any, Literal
from langchain_core.tools import tool

from healthbot.tools import (
    ToolSelector,
    TavilyTool,
    medical_calculator_tool,
    pubmed_api_tool,
)


@tool
def medical_rag_search(query: str, k: int = 5) -> Dict[str, Any]:
    """
    Search medical knowledge base (716 PubMed articles on common conditions).

    Use this tool for:
    - Medical conditions (diabetes, hypertension, asthma, etc.)
    - Symptoms and causes
    - Treatment information
    - General medical education questions

    Args:
        query: Medical search query
        k: Number of documents to retrieve (default: 5)

    Returns:
        Dictionary with:
            - success: bool
            - documents: list of medical documents
            - method: retrieval method used (semantic/bm25/hybrid)
            - scores: relevance scores

    Example:
        Query: "What causes Type 2 diabetes?"
        Returns documents from local knowledge base with citations.
    """
    tool_selector = ToolSelector()
    result = tool_selector.select_and_search(query, k=k)

    return {
        "success": result.get("success", False),
        "documents": result.get("documents", []),
        "method": result.get("method", "unknown"),
        "count": len(result.get("documents", [])),
        "source": "local_knowledge_base_716_articles",
    }


@tool
def medical_calculator(
    calculation_type: Literal["bmi", "dosage", "creatinine_clearance"],
    weight_kg: float = None,
    height_m: float = None,
    dose_per_kg: float = None,
    age: int = None,
    serum_creatinine_mg_dl: float = None,
    sex: Literal["male", "female"] = None,
) -> Dict[str, Any]:
    """
    Perform medical calculations (BMI, dosage, kidney function).

    Use this tool for:
    - BMI calculations
    - Medication dosage calculations
    - Kidney function assessment (creatinine clearance)

    Calculation types:

    1. BMI (Body Mass Index):
       - Required: weight_kg, height_m
       - Returns: bmi, category (underweight/normal/overweight/obese)

    2. Dosage (Medication):
       - Required: weight_kg, dose_per_kg
       - Returns: total_dose_mg

    3. Creatinine Clearance (Kidney function):
       - Required: age, weight_kg, serum_creatinine_mg_dl, sex
       - Returns: crcl_ml_min, interpretation

    Args:
        calculation_type: Type of calculation ("bmi", "dosage", "creatinine_clearance")
        weight_kg: Patient weight in kilograms
        height_m: Patient height in meters
        dose_per_kg: Medication dose per kilogram
        age: Patient age in years
        serum_creatinine_mg_dl: Serum creatinine in mg/dL
        sex: Patient sex ("male" or "female")

    Returns:
        Calculation results with interpretation

    Example:
        Query: "What's my BMI if I'm 70kg and 1.75m tall?"
        Call: medical_calculator("bmi", weight_kg=70, height_m=1.75)
        Returns: {"bmi": 22.9, "category": "normal", ...}
    """
    # Build params dict from provided arguments
    params = {}
    if weight_kg is not None:
        params["weight_kg"] = weight_kg
    if height_m is not None:
        params["height_m"] = height_m
    if dose_per_kg is not None:
        params["dose_per_kg"] = dose_per_kg
    if age is not None:
        params["age"] = age
    if serum_creatinine_mg_dl is not None:
        params["serum_creatinine_mg_dl"] = serum_creatinine_mg_dl
    if sex is not None:
        params["sex"] = sex

    result = medical_calculator_tool(calculation_type, **params)

    # Add source attribution
    result["source"] = "medical_calculator_tool"
    result["tool_type"] = "computation"

    return result


@tool
def pubmed_api_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search PubMed database directly via API (35M+ research papers).

    Use this tool for:
    - Recent research and studies
    - Specific medical research questions
    - Comparative evidence across studies
    - Research that may not be in local knowledge base

    Args:
        query: Medical research query (supports PubMed advanced syntax)
        max_results: Maximum papers to return (default: 5)

    Returns:
        Dictionary with:
            - success: bool
            - count: int
            - papers: list of dicts with:
                - pmid: PubMed ID
                - title: Paper title
                - abstract: Paper abstract
                - authors: List of authors
                - publication_date: Publication date
                - journal: Journal name

    Example:
        Query: "Compare recent studies on diabetes treatment"
        Returns: List of recent research papers from PubMed.

    Note: Requires ENTREZ_EMAIL configured. Rate limited to 3 requests/second.
    """
    result = pubmed_api_tool(query, max_results)

    # Add source attribution
    if result.get("success"):
        result["source"] = "pubmed_api_35M_articles"
        result["tool_type"] = "external_api"

    return result


@tool
def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search the web for current health information and news.

    Use this tool for:
    - Recent health news and outbreaks
    - Current treatment guidelines
    - Health information not in medical databases
    - General health questions outside medical literature

    Args:
        query: Health-related search query
        max_results: Maximum results to return (default: 5)

    Returns:
        Dictionary with:
            - success: bool
            - results: list of web search results
            - query: original query

    Example:
        Query: "Recent COVID-19 treatment updates"
        Returns: Current web information about COVID treatments.

    Note: Requires TAVILY_API_KEY configured.
    """
    try:
        tavily = TavilyTool()
        result = tavily.search(query, max_results=max_results)

        return {
            "success": True,
            "results": result.get("results", []),
            "query": query,
            "count": len(result.get("results", [])),
            "source": "tavily_web_search",
            "tool_type": "web_search",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "results": [],
            "source": "tavily_web_search",
        }


def get_all_tools() -> list:
    """
    Get all available tools for agent.

    Returns:
        List of LangChain tool objects
    """
    return [
        medical_rag_search,
        medical_calculator,
        pubmed_api_search,
        web_search,
    ]


def get_tool_descriptions() -> Dict[str, str]:
    """
    Get descriptions of all tools for documentation.

    Returns:
        Dictionary mapping tool names to descriptions
    """
    tools = get_all_tools()
    return {
        tool.name: tool.description
        for tool in tools
    }


# Tool metadata for evaluation and monitoring
TOOL_METADATA = {
    "medical_rag_search": {
        "category": "retrieval",
        "scope": "local_knowledge_base",
        "article_count": 716,
        "use_cases": ["medical_conditions", "symptoms", "treatments"],
    },
    "medical_calculator": {
        "category": "computation",
        "scope": "medical_calculations",
        "calculations": ["bmi", "dosage", "creatinine_clearance"],
        "use_cases": ["numerical_queries", "health_metrics"],
    },
    "pubmed_api_search": {
        "category": "retrieval",
        "scope": "external_api",
        "article_count": "35M+",
        "use_cases": ["research_questions", "recent_studies", "comparative_evidence"],
    },
    "web_search": {
        "category": "web_search",
        "scope": "general_web",
        "use_cases": ["current_news", "recent_outbreaks", "general_health_info"],
    },
}
