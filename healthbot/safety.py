"""
Medical safety module for emergency detection and disclaimers.
Ensures responsible handling of health-related queries.
"""

# Emergency keywords that require immediate medical attention
EMERGENCY_KEYWORDS: list[str] = [
    "chest pain",
    "heart attack",
    "difficulty breathing",
    "can't breathe",
    "cannot breathe",
    "stroke",
    "severe bleeding",
    "heavy bleeding",
    "bleeding won't stop",
    "unconscious",
    "unresponsive",
    "suicide",
    "suicidal",
    "kill myself",
    "severe head injury",
    "broken bone",
    "severe burn",
    "poisoning",
    "overdose",
    "seizure",
    "severe allergic reaction",
    "anaphylaxis",
    "choking",
]

MEDICAL_DISCLAIMER = """
⚠️ **MEDICAL DISCLAIMER**

This information is provided for educational purposes only and is not a substitute
for professional medical advice, diagnosis, or treatment. Always seek the advice of
your physician or other qualified health provider with any questions you may have
regarding a medical condition. Never disregard professional medical advice or delay
in seeking it because of something you have read here.
"""

EMERGENCY_ALERT = """
🚨 **EMERGENCY ALERT** 🚨

Based on your query, you may be experiencing a medical emergency.

**IMMEDIATE ACTION REQUIRED:**
- In the USA: Call 911
- In India: Call 102 or 108
- In other countries: Contact your local emergency services immediately

Do NOT wait for an online response. Seek immediate medical attention.
"""


def check_emergency(query: str) -> bool:
    """
    Check if a query contains emergency keywords.

    Args:
        query: User's input query

    Returns:
        True if emergency keywords detected, False otherwise
    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in EMERGENCY_KEYWORDS)


def format_with_disclaimer(response: str) -> str:
    """
    Append medical disclaimer to a response.

    Args:
        response: The main response text

    Returns:
        Response with disclaimer appended
    """
    return f"{response}\n\n{MEDICAL_DISCLAIMER}"


def get_emergency_response() -> str:
    """
    Get the emergency alert message.

    Returns:
        Emergency alert message
    """
    return EMERGENCY_ALERT
