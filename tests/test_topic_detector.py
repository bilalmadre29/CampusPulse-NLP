"""Unit tests for topic detector."""

import pytest
from src.topic_detector import detect_topic, detect_topics_batch


def test_detect_teaching_topic():
    feedback = "The professor provides clear explanations and solves our syllabus doubts."
    result = detect_topic(feedback)
    assert result["topic"] == "Teaching"
    assert result["is_identified"] is True
    assert any(kw in ["professor", "syllabus", "doubts", "explanation"] for kw in result["matched_keywords"])


def test_detect_examinations_topic():
    feedback = "The exam schedule had adequate gaps and the question paper was well balanced."
    result = detect_topic(feedback)
    assert result["topic"] == "Examinations"
    assert result["is_identified"] is True


def test_detect_infrastructure_topic():
    feedback = "Computer lab workstations have high speed internet and clean desks."
    result = detect_topic(feedback)
    assert result["topic"] == "Infrastructure"
    assert result["is_identified"] is True


def test_detect_administration_topic():
    feedback = "The accounts counter was crowded and fee receipt generation took three hours."
    result = detect_topic(feedback)
    assert result["topic"] == "Administration"
    assert result["is_identified"] is True


def test_unknown_topic_fallback():
    # Feedback without any college domain keywords
    feedback = "It was raining heavily today and the sky was completely grey."
    result = detect_topic(feedback)
    assert result["topic"] == "Other / Uncategorized"
    assert result["is_identified"] is False
    assert result["score"] == 0.0


def test_empty_and_invalid_inputs():
    assert detect_topic("")["topic"] == "Other / Uncategorized"
    assert detect_topic("   ")["topic"] == "Other / Uncategorized"
    assert detect_topic(None)["topic"] == "Other / Uncategorized"
    assert detect_topic(12345)["topic"] == "Other / Uncategorized"


def test_batch_topic_detection():
    batch = [
        "Great lecture by the professor.",
        "Midterm exam timetable was published.",
        "Unrelated generic sentence about apples and oranges."
    ]
    results = detect_topics_batch(batch)
    assert len(results) == 3
    assert results[0]["topic"] == "Teaching"
    assert results[1]["topic"] == "Examinations"
    assert results[2]["topic"] == "Other / Uncategorized"


def test_detect_multi_topic():
    """Verify that feedback mentioning multiple domains identifies multi-topic signals."""
    # Sentence 1 from user
    feedback1 = "The fees are too high, but the teachers are helpful."
    res1 = detect_topic(feedback1)
    assert res1["is_identified"] is True
    assert res1["is_multi_topic"] is True
    assert "Teaching" in res1["all_topics"]
    assert "Administration" in res1["all_topics"]
    assert res1["topic"] in ["Teaching & Administration", "Administration & Teaching"]

    # Sentence 2 from user
    feedback2 = "The fees are affordable and the teachers are excellent."
    res2 = detect_topic(feedback2)
    assert res2["is_identified"] is True
    assert res2["is_multi_topic"] is True
    assert "Teaching" in res2["all_topics"]
    assert "Administration" in res2["all_topics"]
    assert res2["topic"] in ["Teaching & Administration", "Administration & Teaching"]

    # Multi-topic with Infrastructure & Teaching
    feedback3 = "The professor is knowledgeable, but the library wifi is disconnected."
    res3 = detect_topic(feedback3)
    assert res3["is_identified"] is True
    assert res3["is_multi_topic"] is True
    assert "Teaching" in res3["all_topics"]
    assert "Infrastructure" in res3["all_topics"]
