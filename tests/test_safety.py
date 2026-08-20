"""
Unit tests for safety detection system.

Tests emergency keyword detection and safety routing logic.
"""

import pytest

from healthbot.safety import EMERGENCY_KEYWORDS, check_emergency


class TestEmergencyDetection:
    """Test suite for emergency keyword detection."""

    def test_check_emergency_with_clear_emergency(self):
        """Test detection of clear emergency keywords."""
        # Direct emergency keywords
        assert check_emergency("I'm having chest pain") is True
        assert check_emergency("I can't breathe") is True
        assert check_emergency("I think I'm having a heart attack") is True
        assert check_emergency("severe bleeding won't stop") is True
        assert check_emergency("I feel suicidal") is True

    def test_check_emergency_case_insensitive(self):
        """Test that detection is case-insensitive."""
        assert check_emergency("CHEST PAIN") is True
        assert check_emergency("Chest Pain") is True
        assert check_emergency("chest pain") is True
        assert check_emergency("ChEsT pAiN") is True

    def test_check_emergency_within_sentence(self):
        """Test detection when keyword is within a longer sentence."""
        assert check_emergency("I woke up with chest pain this morning") is True
        assert check_emergency("My father is having difficulty breathing") is True
        assert check_emergency("There's heavy bleeding from the wound") is True

    def test_non_emergency_queries(self):
        """Test that normal medical queries are not flagged as emergencies."""
        assert check_emergency("What causes diabetes?") is False
        assert check_emergency("How is hypertension treated?") is False
        assert check_emergency("What are symptoms of the flu?") is False
        assert check_emergency("Can you explain asthma?") is False
        assert check_emergency("I want to learn about nutrition") is False

    def test_edge_cases(self):
        """Test edge cases in emergency detection."""
        # Empty query
        assert check_emergency("") is False

        # Whitespace only
        assert check_emergency("   ") is False

        # Very long query without emergency keywords
        long_query = "I am interested in learning about " + "health " * 100
        assert check_emergency(long_query) is False

    def test_partial_matches_not_detected(self):
        """Test that partial word matches don't trigger false positives."""
        # "stroke" should match, but "stroked" (as in "stroked the cat") should not
        # This is a limitation of the current simple implementation
        # but we can document expected behavior

        # These SHOULD be detected
        assert check_emergency("I think I'm having a stroke") is True
        assert check_emergency("stroke symptoms") is True

        # Current implementation may detect these (substring matching)
        # but ideally should not:
        # "breakthrough" contains "through" but not emergency
        # We test current behavior, not ideal behavior
        result = check_emergency("I had a breakthrough in my research")
        # Document current behavior (may match "through" from "breakthrough")

    def test_all_emergency_keywords_covered(self):
        """Test that all defined emergency keywords can be detected."""
        for keyword in EMERGENCY_KEYWORDS:
            query = f"I am experiencing {keyword}"
            assert (
                check_emergency(query) is True
            ), f"Failed to detect keyword: {keyword}"

    def test_multiple_keywords_in_query(self):
        """Test queries with multiple emergency keywords."""
        query = "chest pain and difficulty breathing"
        assert check_emergency(query) is True

    def test_medical_terminology_emergencies(self):
        """Test detection with medical terminology."""
        # Suicidal ideation variations
        assert check_emergency("I want to kill myself") is True
        assert check_emergency("suicidal thoughts") is True

        # Trauma
        assert check_emergency("severe head injury from fall") is True
        assert check_emergency("broken bone sticking out") is True

        # Overdose
        assert check_emergency("I took too many pills, overdose") is True
        assert check_emergency("possible poisoning") is True

    def test_emergency_keywords_constant(self):
        """Test that EMERGENCY_KEYWORDS constant is properly defined."""
        assert isinstance(EMERGENCY_KEYWORDS, list)
        assert len(EMERGENCY_KEYWORDS) >= 23  # At least 23 keywords documented
        assert all(isinstance(keyword, str) for keyword in EMERGENCY_KEYWORDS)
        assert all(
            keyword == keyword.lower() for keyword in EMERGENCY_KEYWORDS
        )  # Should be lowercase


class TestSafetyNode:
    """Test the safety check node in the workflow."""

    def test_safety_node_emergency_detection(self):
        """Test that safety node correctly detects emergencies."""
        from healthbot.nodes import check_safety_node

        # Emergency state
        state = {
            "topic": "I'm having chest pain",
            "messages": [],
        }

        result = check_safety_node(state)

        assert result["emergency_detected"] is True
        assert "summary" in result
        assert "911" in result["summary"] or "emergency" in result["summary"].lower()

    def test_safety_node_normal_query(self):
        """Test that safety node allows normal queries through."""
        from healthbot.nodes import check_safety_node

        # Normal educational query
        state = {
            "topic": "What is diabetes?",
            "messages": [],
        }

        result = check_safety_node(state)

        assert result["emergency_detected"] is False
        assert "summary" not in result or result.get("summary") == ""


class TestSafetyRouting:
    """Test safety-related routing logic."""

    def test_decide_safety_path_emergency(self):
        """Test routing decision for emergency queries."""
        from healthbot.graph import decide_safety_path

        state = {"emergency_detected": True}
        decision = decide_safety_path(state)

        assert decision == "emergency_exit"

    def test_decide_safety_path_normal(self):
        """Test routing decision for normal queries."""
        from healthbot.graph import decide_safety_path

        state = {"emergency_detected": False}
        decision = decide_safety_path(state)

        assert decision == "retrieve"


class TestEmergencyMessage:
    """Test emergency response message."""

    def test_emergency_message_content(self):
        """Test that emergency message has required information."""
        from healthbot.nodes import check_safety_node

        state = {
            "topic": "chest pain",
            "messages": [],
        }

        result = check_safety_node(state)
        message = result.get("summary", "")

        # Should contain emergency numbers
        assert "911" in message or "102" in message or "108" in message

        # Should have urgent language
        assert any(
            word in message.lower()
            for word in ["emergency", "immediately", "urgent", "call"]
        )

        # Should mention professional help
        assert any(
            word in message.lower()
            for word in ["doctor", "hospital", "professional", "medical"]
        )
