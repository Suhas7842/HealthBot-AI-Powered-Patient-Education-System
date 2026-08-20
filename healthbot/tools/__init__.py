"""
HealthBot tools package (Phase 4).

This package contains tool modules for the GenAI orchestration system:
- tool_selector: ToolSelector class for hybrid retrieval (Phase 1-3)
- medical_calculator: BMI, dosage, creatinine clearance calculations
- pubmed_api: PubMed E-utilities integration
"""

# Phase 1-3: Tool selector (hybrid retrieval)
from healthbot.tools.tool_selector import ToolSelector, TavilyTool

# Phase 4: Medical calculator
from healthbot.tools.medical_calculator import (
    calculate_bmi,
    calculate_dosage,
    calculate_creatinine_clearance,
    medical_calculator_tool,
)

# Phase 4: PubMed API
from healthbot.tools.pubmed_api import (
    PubMedClient,
    search_pubmed,
    pubmed_api_tool,
)

__all__ = [
    # Phase 1-3
    "ToolSelector",
    "TavilyTool",
    # Phase 4: Calculator
    "calculate_bmi",
    "calculate_dosage",
    "calculate_creatinine_clearance",
    "medical_calculator_tool",
    # Phase 4: PubMed
    "PubMedClient",
    "search_pubmed",
    "pubmed_api_tool",
]
