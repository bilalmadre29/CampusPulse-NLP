"""Unit tests for batch CSV processing and analytics."""

import pandas as pd
import pytest
from src.analytics import analyze_feedback_df, detect_text_column


def test_detect_common_text_columns():
    df1 = pd.DataFrame({"feedback": ["Good class"], "id": [1]})
    assert detect_text_column(df1) == "feedback"

    df2 = pd.DataFrame({"comment": ["Nice lab"], "score": [5]})
    assert detect_text_column(df2) == "comment"

    df3 = pd.DataFrame({"student_feedback": ["Exams fair"]})
    assert detect_text_column(df3) == "student_feedback"


def test_valid_batch_analytics():
    data = {
        "feedback_text": [
            "The professor is very knowledgeable and kind.",
            "Wifi in the campus library is fast.",
            "Exam timetable was published on the portal.",
            "Accounts counter is terribly slow and staff is rude."
        ]
    }
    df = pd.DataFrame(data)
    results = analyze_feedback_df(df, "feedback_text")

    assert results["total_records"] == 4
    assert len(results["analyzed_df"]) == 4
    assert "Predicted_Sentiment" in results["analyzed_df"].columns
    assert "Detected_Topic" in results["analyzed_df"].columns
    assert "Confidence_%" in results["analyzed_df"].columns
    assert "Is_Uncertain" in results["analyzed_df"].columns
    assert "Uncertainty_Reason" in results["analyzed_df"].columns
    assert not results["topic_sentiment_crosstab"].empty


def test_handling_missing_nan_and_empty_rows():
    data = {
        "text": [
            "Good teaching and clear notes.",
            None,
            float("nan"),
            "   ",
            "Projectors never work in class 201."
        ]
    }
    df = pd.DataFrame(data)
    results = analyze_feedback_df(df, "text")

    # Only 2 non-empty rows should be analyzed
    assert results["total_records"] == 2
    assert len(results["analyzed_df"]) == 2
    assert results["warning"] is None


def test_all_empty_dataframe():
    df = pd.DataFrame({"feedback": [None, "   ", ""]})
    results = analyze_feedback_df(df, "feedback")

    assert results["total_records"] == 0
    assert results["analyzed_df"].empty
    assert results["warning"] is not None


def test_missing_column_error_handling():
    df = pd.DataFrame({"non_existent_column": ["sample text"]})
    with pytest.raises(ValueError, match="was not found"):
        analyze_feedback_df(df, "target_text_column")


def test_repeated_uploads_independence():
    # Process batch A
    df_a = pd.DataFrame({"feedback": ["Great faculty", "Broken desks"]})
    res_a = analyze_feedback_df(df_a, "feedback")

    # Process batch B
    df_b = pd.DataFrame({"feedback": ["Fee portal payment failed"]})
    res_b = analyze_feedback_df(df_b, "feedback")

    assert res_a["total_records"] == 2
    assert res_b["total_records"] == 1
    # Verify no state leak between calls
    assert len(res_b["analyzed_df"]) == 1


def test_batch_multi_topic_detection():
    """Verify that batch processing properly preserves multi-topic categories in the analyzed dataframe."""
    df = pd.DataFrame({
        "feedback_text": [
            "The fees are too high, but the teachers are helpful.",
            "The fees are affordable and the teachers are excellent.",
            "The computer lab wifi is fast and clean."
        ]
    })
    results = analyze_feedback_df(df, "feedback_text")
    topics = results["analyzed_df"]["Detected_Topic"].tolist()

    assert "Teaching & Administration" in topics[0] or "Administration & Teaching" in topics[0]
    assert "Teaching & Administration" in topics[1] or "Administration & Teaching" in topics[1]
    assert topics[2] == "Infrastructure"
