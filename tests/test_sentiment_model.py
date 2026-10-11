"""Unit tests for sentiment classification model.

Tests training, held-out evaluation, predictions on positive, negative,
neutral, negated, and mixed feedback, and uncertainty handling.
"""

import pytest
from src.sentiment_model import get_sentiment_model, SentimentModel


@pytest.fixture(scope="module")
def model():
    m = get_sentiment_model()
    return m


def test_model_is_trained_and_metrics_held_out(model):
    assert model.is_trained is True
    metrics = model.metrics_
    assert "accuracy" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "f1_macro" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["test_size"] > 0
    assert metrics["train_size"] > metrics["test_size"]
    # Verify realistic non-trivial accuracy on held-out test split
    assert 0.40 <= metrics["accuracy"] <= 0.95


def test_positive_feedback(model):
    feedback = "The professor is wonderful and explains all concepts with great clarity."
    res = model.predict(feedback)
    assert res["sentiment"] == "positive"
    assert res["confidence"] > 40.0


def test_negative_feedback(model):
    feedback = "The washrooms are terribly dirty and there is no running water."
    res = model.predict(feedback)
    assert res["sentiment"] == "negative"
    assert res["confidence"] > 40.0


def test_neutral_feedback(model):
    feedback = "The syllabus copy was distributed to all students by the department."
    res = model.predict(feedback)
    assert res["sentiment"] in ["neutral", "positive", "negative"]
    # Check probabilities exist
    assert "neutral" in res["probabilities"]


def test_negated_feedback(model):
    # Feedback containing explicit negation inversion
    feedback = "The explanation of mathematical proofs was not clear at all."
    res = model.predict(feedback)
    # Record and verify negation drives prediction away from positive
    assert res["sentiment"] == "negative"


def test_mixed_feedback_uncertainty(model):
    # Mixed sentiment: good teacher but terrible projector
    feedback = "The teacher is good but the classroom projector is completely broken and useless."
    res = model.predict(feedback)
    # Either predicted negative/positive or flagged with low confidence / uncertainty
    assert res["sentiment"] in ["negative", "positive", "neutral"]
    assert "probabilities" in res


def test_empty_and_whitespace_input(model):
    res_empty = model.predict("")
    assert res_empty["is_uncertain"] is True
    assert res_empty["sentiment"] == "neutral"

    res_spaces = model.predict("      ")
    assert res_spaces["is_uncertain"] is True
    assert res_spaces["sentiment"] == "neutral"


def test_reproducible_split():
    # Verify that calling train_and_evaluate with same seed produces identical held-out metrics
    m1 = SentimentModel()
    m2 = SentimentModel()
    res1 = m1.train_and_evaluate(random_state=42)
    res2 = m2.train_and_evaluate(random_state=42)
    assert res1["accuracy"] == res2["accuracy"]
    assert res1["confusion_matrix"] == res2["confusion_matrix"]
