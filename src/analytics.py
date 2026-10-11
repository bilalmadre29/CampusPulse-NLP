"""Feedback Analytics Module for CampusPulse NLP.

Performs batch analytics strictly on user-submitted feedback data:
- Sentiment & Topic distributions
- Topic-Sentiment cross-tabulations
- Actionable negative problem areas and positive highlights extracted directly from input
- CSV ingestion and column validation
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from src.sentiment_model import get_sentiment_model
from src.topic_detector import detect_topic
from src.preprocessor import clean_text


COMMON_TEXT_COLUMNS = [
    "feedback", "feedback_text", "comment", "comments",
    "review", "reviews", "text", "student_feedback", "response"
]


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the most likely text column containing student feedback."""
    cols_lower = {col.lower().strip(): col for col in df.columns}
    for candidate in COMMON_TEXT_COLUMNS:
        if candidate in cols_lower:
            return cols_lower[candidate]

    # Fallback to the first string/object column
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("str"):
            return col
    return None


def analyze_feedback_df(df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
    """Analyze a DataFrame containing student feedback.

    Validates columns, sanitizes null/empty entries, applies NLP pipeline,
    and returns aggregated analytics and the enriched DataFrame.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' was not found in the uploaded file.")

    model = get_sentiment_model()

    # Create a working copy and sanitize inputs
    working_df = df.copy()
    working_df["_raw_text"] = working_df[text_column].fillna("").astype(str)

    # Filter out empty or whitespace-only rows
    valid_mask = working_df["_raw_text"].str.strip().str.len() > 0
    if not valid_mask.any():
        return {
            "total_records": 0,
            "sentiment_counts": {},
            "sentiment_percentages": {},
            "topic_counts": {},
            "topic_percentages": {},
            "topic_sentiment_crosstab": pd.DataFrame(),
            "common_negative_feedback": [],
            "common_positive_feedback": [],
            "analyzed_df": pd.DataFrame(),
            "warning": "No valid text entries found in the selected column."
        }

    valid_df = working_df[valid_mask].copy()

    # Run predictions row by row
    sentiments = []
    confidences = []
    uncertainties = []
    uncertainty_reasons = []
    topics = []
    topic_scores = []
    matched_kws = []

    for text in valid_df["_raw_text"]:
        # Sentiment prediction
        sent_res = model.predict(text)
        sentiments.append(sent_res["sentiment"])
        confidences.append(sent_res["confidence"])
        uncertainties.append(sent_res["is_uncertain"])
        uncertainty_reasons.append(sent_res["uncertainty_reason"])

        # Topic detection
        topic_res = detect_topic(text)
        if topic_res.get("is_multi_topic") and topic_res.get("all_topics"):
            top_name = " & ".join(topic_res["all_topics"])
        else:
            top_name = topic_res.get("topic", "Other / Uncategorized")
        topics.append(top_name)
        topic_scores.append(topic_res["score"])
        matched_kws.append(", ".join(topic_res["matched_keywords"][:4]))

    # Enrich DataFrame
    valid_df["Predicted_Sentiment"] = sentiments
    valid_df["Confidence_%"] = confidences
    valid_df["Is_Uncertain"] = uncertainties
    valid_df["Uncertainty_Reason"] = uncertainty_reasons
    valid_df["Detected_Topic"] = topics
    valid_df["Topic_Score"] = topic_scores
    valid_df["Matched_Keywords"] = matched_kws

    # Clean temporary column
    valid_df = valid_df.drop(columns=["_raw_text"])

    total = len(valid_df)

    # Sentiment distribution
    sentiment_counts = valid_df["Predicted_Sentiment"].value_counts().to_dict()
    sentiment_pcts = {
        k: round((v / total) * 100, 1) for k, v in sentiment_counts.items()
    }

    # Topic distribution
    topic_counts = valid_df["Detected_Topic"].value_counts().to_dict()
    topic_pcts = {
        k: round((v / total) * 100, 1) for k, v in topic_counts.items()
    }

    # Cross-tabulation: Topic vs Sentiment
    crosstab_df = pd.crosstab(
        valid_df["Detected_Topic"],
        valid_df["Predicted_Sentiment"],
        margins=True,
        margins_name="Total"
    )

    # Extract top actionable positive feedback (sorted by confidence)
    pos_df = valid_df[valid_df["Predicted_Sentiment"] == "positive"].sort_values(
        by="Confidence_%", ascending=False
    )
    common_pos = []
    for _, row in pos_df.head(5).iterrows():
        common_pos.append({
            "text": str(row[text_column]),
            "topic": row["Detected_Topic"],
            "confidence": row["Confidence_%"]
        })

    # Extract top critical negative feedback (sorted by confidence)
    neg_df = valid_df[valid_df["Predicted_Sentiment"] == "negative"].sort_values(
        by="Confidence_%", ascending=False
    )
    common_neg = []
    for _, row in neg_df.head(5).iterrows():
        common_neg.append({
            "text": str(row[text_column]),
            "topic": row["Detected_Topic"],
            "confidence": row["Confidence_%"]
        })

    return {
        "total_records": total,
        "sentiment_counts": sentiment_counts,
        "sentiment_percentages": sentiment_pcts,
        "topic_counts": topic_counts,
        "topic_percentages": topic_pcts,
        "topic_sentiment_crosstab": crosstab_df,
        "common_negative_feedback": common_neg,
        "common_positive_feedback": common_pos,
        "analyzed_df": valid_df,
        "warning": None
    }
