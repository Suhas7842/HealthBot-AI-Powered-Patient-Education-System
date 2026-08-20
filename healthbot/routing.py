"""
Query classification and intelligent routing for HealthBot.

Classifies queries by intent and complexity to optimize retrieval strategy.
"""

import re
from enum import Enum

from healthbot.logger import logger
from healthbot.models import LLMWrapper


class QueryIntent(Enum):
    """Query intent classification for medical queries."""

    INFORMATIONAL = "informational"  # "What is X?" - General overview
    DIAGNOSTIC = "diagnostic"  # "What causes X?" - Symptoms, causes, risk factors
    TREATMENT = "treatment"  # "How to treat X?" - Treatment options, medications
    PREVENTIVE = "preventive"  # "How to prevent X?" - Prevention strategies
    FOLLOW_UP = "follow_up"  # Based on conversation history
    UNKNOWN = "unknown"  # Unable to classify


class QueryComplexity(Enum):
    """Query complexity level based on structure and scope."""

    SIMPLE = "simple"  # Single concept, straightforward ("What is diabetes?")
    MODERATE = "moderate"  # Multiple concepts ("diabetes symptoms and causes")
    COMPLEX = "complex"  # Multi-part or comparison ("Type 1 vs Type 2 diabetes")


class QueryClassifier:
    """
    Classify queries to optimize retrieval and generation.

    Uses fast rule-based classification (no LLM calls) for production speed.
    """

    def __init__(self):
        """Initialize query classifier with pattern matching rules."""
        self.llm = None  # Lazy load only if needed for rewriting

        # Intent classification patterns
        self.informational_patterns = [
            r"\bwhat is\b",
            r"\bdefine\b",
            r"\bexplain\b",
            r"\btell me about\b",
            r"\binformation about\b",
            r"\boverview\b",
            r"\bintroduce\b",
            r"\bdescribe\b",
        ]

        self.diagnostic_patterns = [
            r"\bsymptoms?\b",
            r"\bcauses?\b",
            r"\brisk factors?\b",
            r"\bsigns? of\b",
            r"\bhow to diagnose\b",
            r"\bwhat causes\b",
            r"\bdiagnosis\b",
            r"\brecognize\b",
            r"\bidentify\b",
        ]

        self.treatment_patterns = [
            r"\btreatment\b",
            r"\bhow to treat\b",
            r"\bhow is .* treated\b",
            r"\bcure\b",
            r"\bmanage\b",
            r"\bmedication\b",
            r"\btherapy\b",
            r"\bintervention\b",
            r"\bremedy\b",
            r"\bhow do i treat\b",
        ]

        self.preventive_patterns = [
            r"\bprevent\b",
            r"\bavoid\b",
            r"\breduce risk\b",
            r"\bprevention\b",
            r"\bhow to not get\b",
            r"\bprotect against\b",
            r"\bhow can i avoid\b",
        ]

        # Follow-up detection patterns
        self.follow_up_indicators = [
            r"\bit\b",
            r"\bthis\b",
            r"\bthat\b",
            r"\bthese\b",
            r"\bthose\b",
            r"\btell me more\b",
            r"\bexplain further\b",
            r"\bgo on\b",
            r"\bcontinue\b",
            r"\bwhat about\b",
            r"\bhow about\b",
            r"\bwhat else\b",
            r"^(and|but|also|additionally)\b",
        ]

        # Complexity indicators
        self.complexity_indicators = [
            r"\band\b",
            r"\bor\b",
            r"\bvs\b",
            r"\bversus\b",
            r"\bdifference between\b",
            r"\bcompare\b",
            r"\brelationship between\b",
            r"\bboth\b",
            r"\brisk factors?\b",
        ]

    def classify_intent_fast(self, query: str) -> QueryIntent:
        """
        Fast rule-based intent classification using pattern matching.

        Args:
            query: User's medical query

        Returns:
            QueryIntent enum value

        Example:
            >>> classifier = QueryClassifier()
            >>> classifier.classify_intent_fast("What is Type 2 diabetes?")
            QueryIntent.INFORMATIONAL
            >>> classifier.classify_intent_fast("How is hypertension treated?")
            QueryIntent.TREATMENT
        """
        query_lower = query.lower()

        # Check preventive first (most specific)
        if any(re.search(p, query_lower) for p in self.preventive_patterns):
            logger.debug(f"Classified as PREVENTIVE: {query}")
            return QueryIntent.PREVENTIVE

        # Check treatment
        if any(re.search(p, query_lower) for p in self.treatment_patterns):
            logger.debug(f"Classified as TREATMENT: {query}")
            return QueryIntent.TREATMENT

        # Check diagnostic
        if any(re.search(p, query_lower) for p in self.diagnostic_patterns):
            logger.debug(f"Classified as DIAGNOSTIC: {query}")
            return QueryIntent.DIAGNOSTIC

        # Check informational
        if any(re.search(p, query_lower) for p in self.informational_patterns):
            logger.debug(f"Classified as INFORMATIONAL: {query}")
            return QueryIntent.INFORMATIONAL

        # Default to informational for safety (broad overview is safest)
        logger.debug(f"Defaulted to INFORMATIONAL: {query}")
        return QueryIntent.INFORMATIONAL

    def classify_complexity(self, query: str) -> QueryComplexity:
        """
        Classify query complexity based on structure and word count.

        Args:
            query: User's medical query

        Returns:
            QueryComplexity enum value

        Complexity Heuristics:
            - SIMPLE: Single concept, < 8 words, no complexity indicators
            - MODERATE: Multiple concepts or 8-15 words, 1 indicator
            - COMPLEX: Multi-part question, > 15 words, or 2+ indicators

        Example:
            >>> classifier = QueryClassifier()
            >>> classifier.classify_complexity("What is diabetes?")
            QueryComplexity.SIMPLE
            >>> classifier.classify_complexity("What are diabetes symptoms and treatments?")
            QueryComplexity.MODERATE
            >>> classifier.classify_complexity("Difference between Type 1 and Type 2 diabetes?")
            QueryComplexity.COMPLEX
        """
        query_lower = query.lower()

        # Count complexity indicators
        complexity_score = sum(
            1 for indicator in self.complexity_indicators if re.search(indicator, query_lower)
        )

        # Count words
        word_count = len(query.split())

        # Classify based on combined heuristics
        if complexity_score >= 2 or word_count > 15:
            logger.debug(f"Classified as COMPLEX: {query} (score={complexity_score}, words={word_count})")
            return QueryComplexity.COMPLEX
        elif complexity_score == 1 or word_count > 8:
            logger.debug(f"Classified as MODERATE: {query} (score={complexity_score}, words={word_count})")
            return QueryComplexity.MODERATE
        else:
            logger.debug(f"Classified as SIMPLE: {query} (score={complexity_score}, words={word_count})")
            return QueryComplexity.SIMPLE

    def get_retrieval_params(
        self, intent: QueryIntent, complexity: QueryComplexity
    ) -> dict:
        """
        Get optimal retrieval parameters based on query classification.

        Args:
            intent: Query intent (informational, diagnostic, treatment, preventive)
            complexity: Query complexity (simple, moderate, complex)

        Returns:
            Dictionary with optimized retrieval parameters:
                - k: Number of documents to retrieve
                - score_threshold: Minimum relevance score (for evidence validation)

        Retrieval Strategy:
            - INFORMATIONAL: k=7 (comprehensive overview needs more sources)
            - TREATMENT: k=5, higher threshold (precision critical for medical advice)
            - DIAGNOSTIC: k=6, higher threshold (symptoms + causes)
            - PREVENTIVE: k=5 (prevention strategies)
            - COMPLEX: k+2 (multi-part questions need more coverage)

        Example:
            >>> classifier = QueryClassifier()
            >>> params = classifier.get_retrieval_params(
            ...     QueryIntent.TREATMENT,
            ...     QueryComplexity.SIMPLE
            ... )
            >>> params
            {'k': 5, 'score_threshold': 0.020}
        """
        # Start with defaults
        params = {"k": 5, "score_threshold": 0.015}

        # Adjust based on intent
        if intent == QueryIntent.INFORMATIONAL:
            params["k"] = 7  # Comprehensive overview needs more sources
            logger.debug("INFORMATIONAL → k=7 (comprehensive coverage)")

        elif intent == QueryIntent.TREATMENT:
            params["k"] = 5
            params["score_threshold"] = 0.020  # Higher precision for treatment advice
            logger.debug("TREATMENT → k=5, threshold=0.020 (precision critical)")

        elif intent == QueryIntent.DIAGNOSTIC:
            params["k"] = 6  # Symptoms + causes need good coverage
            params["score_threshold"] = 0.020  # Higher precision for diagnosis
            logger.debug("DIAGNOSTIC → k=6, threshold=0.020 (symptoms + causes)")

        elif intent == QueryIntent.PREVENTIVE:
            params["k"] = 5  # Prevention strategies
            logger.debug("PREVENTIVE → k=5 (prevention strategies)")

        # Adjust based on complexity
        if complexity == QueryComplexity.COMPLEX:
            params["k"] += 2  # Multi-part questions need more sources
            logger.debug(f"COMPLEX → k+2 (now k={params['k']})")

        return params

    def is_follow_up_query(self, query: str, previous_topic: str | None) -> bool:
        """
        Detect if query is a follow-up to previous conversation topic.

        Args:
            query: Current user query
            previous_topic: Topic from previous turn (None if first turn)

        Returns:
            True if query appears to be a follow-up, False otherwise

        Follow-up Indicators:
            - Pronouns: "it", "this", "that", "these", "those"
            - Continuation phrases: "tell me more", "explain further", "what about"
            - Short queries (<5 words) without question words
            - Starting with "and", "but", "also"

        Example:
            >>> classifier = QueryClassifier()
            >>> classifier.is_follow_up_query("What are the symptoms?", "diabetes")
            True
            >>> classifier.is_follow_up_query("How do I treat it?", "diabetes")
            True
            >>> classifier.is_follow_up_query("What is diabetes?", None)
            False
        """
        if not previous_topic:
            return False

        query_lower = query.lower()

        # Check for follow-up indicator patterns
        has_follow_up_pattern = any(
            re.search(pattern, query_lower) for pattern in self.follow_up_indicators
        )

        # Check for short implicit follow-ups
        word_count = len(query.split())
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        has_question_word = any(qw in query_lower for qw in question_words)

        # Short query without question word often implies follow-up
        is_short_implicit = word_count < 5 and not has_question_word

        is_follow_up = has_follow_up_pattern or is_short_implicit

        if is_follow_up:
            logger.info(f"Detected follow-up query: '{query}' (previous: {previous_topic})")

        return is_follow_up

    def rewrite_with_context(
        self, query: str, previous_topic: str, conversation_summary: str = ""
    ) -> str:
        """
        Rewrite follow-up query to be self-contained using conversation context.

        Args:
            query: Follow-up query with pronouns or implicit references
            previous_topic: Topic from previous conversation turn
            conversation_summary: Brief summary of previous turn (optional)

        Returns:
            Rewritten query with explicit context (no pronouns)

        Examples:
            - "What are the symptoms?" → "What are the symptoms of diabetes?"
            - "How do I treat it?" → "How do I treat diabetes?"
            - "Tell me more" → "Tell me more about diabetes"

        Note:
            Uses LLM for context-aware rewriting. Adds ~200ms latency but improves
            retrieval quality for follow-up questions significantly.
        """
        # Lazy load LLM only when needed
        if self.llm is None:
            self.llm = LLMWrapper()

        prompt = f"""Rewrite the follow-up query to be self-contained by incorporating context from the previous conversation.

**Previous Topic**: {previous_topic}

**Conversation Summary**: {conversation_summary or "User asked about " + previous_topic}

**Follow-up Query**: {query}

**Task**: Rewrite the query to be explicit and self-contained. Replace pronouns ("it", "this", "that") with the actual topic. Do not add extra information beyond making the query explicit.

**Examples**:
- "What causes it?" → "What causes {previous_topic}?"
- "Tell me more about symptoms" → "Tell me more about symptoms of {previous_topic}"
- "How do I treat that?" → "How do I treat {previous_topic}?"
- "What about prevention?" → "What about prevention of {previous_topic}?"

**Return ONLY the rewritten query, nothing else. No explanations, no extra text.**"""

        try:
            from langchain_core.messages import HumanMessage

            rewritten = self.llm.invoke([HumanMessage(content=prompt)])
            rewritten = rewritten.strip()

            logger.info(f"Query rewriting: '{query}' → '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            # Fallback: simple string replacement
            fallback = query.replace(" it ", f" {previous_topic} ")
            fallback = fallback.replace(" this ", f" {previous_topic} ")
            fallback = fallback.replace(" that ", f" {previous_topic} ")
            logger.info(f"Using fallback rewriting: '{query}' → '{fallback}'")
            return fallback


# Singleton instance for convenience
_classifier = None


def get_classifier() -> QueryClassifier:
    """Get or create singleton QueryClassifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier()
    return _classifier
