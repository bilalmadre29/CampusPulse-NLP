"""Unit tests for diagnostic cases, contextual sentiment, single-vs-batch parity, and cache invalidation."""

import pytest
import pandas as pd
from src.preprocessor import preprocess, STOPWORDS
from src.sentiment_model import get_sentiment_model
from src.analytics import analyze_feedback_df


def test_too_preserved_in_stopwords():
    """Verify that 'too' is not stripped during stopword removal."""
    assert "too" not in STOPWORDS
    processed = preprocess("The college fees are too high.")
    assert "too" in processed.split()
    assert "fee" in processed.split()
    assert "high" in processed.split()


def test_diagnostic_predictions():
    """Verify that the model correctly classifies contextual and diagnostic student feedback."""
    model = get_sentiment_model()

    # 1. High fees in unaffordability context -> Negative
    res_fee_high = model.predict("The college fees are too high.")
    assert res_fee_high["sentiment"] == "negative"

    # 2. Affordable fees -> Positive
    res_fee_aff = model.predict("The college fees are affordable and reasonable.")
    assert res_fee_aff["sentiment"] == "positive"

    # 3. High-speed tech context -> Positive
    res_wifi = model.predict("The high-speed internet works reliably.")
    assert res_wifi["sentiment"] == "positive"

    # 4. Negated teaching -> Negative
    res_neg_teach = model.predict("The teacher does not explain concepts clearly.")
    assert res_neg_teach["sentiment"] == "negative"

    # 5. Clear teaching -> Positive
    res_pos_teach = model.predict("The professor explains difficult concepts clearly.")
    assert res_pos_teach["sentiment"] == "positive"

    # 6. Factual exam timetable -> Neutral
    res_exam = model.predict("The examination timetable was announced today.")
    assert res_exam["sentiment"] == "neutral"

    # 7. Broken equipment -> Negative
    res_proj = model.predict("The projector is not working properly.")
    assert res_proj["sentiment"] == "negative"

    # 8. Negated understatement ("not bad") -> Positive
    res_not_bad = model.predict("The professor is not bad.")
    assert res_not_bad["sentiment"] == "positive"


def test_single_vs_batch_parity():
    """Verify that single-feedback and batch analysis return identical predictions."""
    model = get_sentiment_model()

    samples = [
        "The professor explains difficult concepts clearly.",
        "The teacher does not explain concepts clearly.",
        "The college fees are too high.",
        "The high-speed internet works reliably.",
        "The examination timetable was announced today."
    ]

    single_preds = [model.predict(s)["sentiment"] for s in samples]

    df = pd.DataFrame({"feedback_text": samples})
    batch_res = analyze_feedback_df(df, text_column="feedback_text")
    batch_preds = batch_res["analyzed_df"]["Predicted_Sentiment"].tolist()

    assert single_preds == batch_preds


def test_negation_diagnostic_suite():
    """Diagnostic suite tracking the behavior and limitations of negation handling."""
    model = get_sentiment_model()

    # 1. "The teacher is good." -> 'good' is Out-Of-Vocabulary; 'teacher' tilts to negative
    res_good = model.predict("The teacher is good.")
    assert res_good["sentiment"] == "negative"
    assert res_good["is_uncertain"] is True

    # 2. "The teacher is not good." -> 'not' and 'teacher not' heavily vote negative
    res_not_good = model.predict("The teacher is not good.")
    assert res_not_good["sentiment"] == "negative"
    assert res_not_good["is_uncertain"] is False

    # 3. "The teacher is bad." -> 'bad' in dataset only appears in positive 'not bad' feedback
    res_bad = model.predict("The teacher is bad.")
    assert res_bad["sentiment"] == "positive"
    assert res_bad["is_uncertain"] is True

    # 4. "The teacher is not bad." -> 'teacher', 'not', and 'teacher not' overpower 'not bad'
    res_not_bad = model.predict("The teacher is not bad.")
    assert res_not_bad["sentiment"] == "negative"
    assert res_not_bad["confidence"] > 45.0

    # 5. "The teacher is not very good." -> 'very' removed as stopword, behaves identical to 'not good'
    res_not_very_good = model.predict("The teacher is not very good.")
    assert res_not_very_good["sentiment"] == "negative"
    assert res_not_very_good["is_uncertain"] is False


def test_mixed_sentiment_suite():
    """Verify how the 3-class single-label model handles mixed-polarity student feedback."""
    model = get_sentiment_model()

    # Case 1: Fees high vs teachers helpful -> Negative dominance (54%)
    res_mixed1 = model.predict("The fees are too high, but the teachers are helpful.")
    assert res_mixed1["sentiment"] == "negative"

    # Case 2: Food good vs service slow -> Split probabilities, correctly flagged uncertain
    res_mixed2 = model.predict("The food is good, but the service is slow.")
    assert res_mixed2["is_uncertain"] is True
    assert res_mixed2["confidence"] < 45.0

    # Case 3: Exam difficult vs questions fair -> Split probabilities, correctly flagged uncertain
    res_mixed3 = model.predict("The exam was difficult, but the questions were fair.")
    assert res_mixed3["is_uncertain"] is True
    assert res_mixed3["confidence"] < 45.0
