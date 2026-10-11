"""Unit tests for text preprocessor."""

import pytest
from src.preprocessor import (
    clean_text,
    tokenize,
    remove_stopwords,
    lemmatize_tokens,
    preprocess,
    preprocess_pipeline,
    NEGATION_WORDS
)


def test_clean_text_normalization():
    raw = "  THE PROFESSOR's Lectures WERE GREAT!!! http://college.edu  "
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "!" not in cleaned
    assert cleaned.islower()
    assert "professor is lectures were great" in cleaned


def test_negation_contractions_expansion():
    # Test specific and general negation contractions
    text = "Students can't access wifi and won't attend if professor doesn't explain."
    cleaned = clean_text(text)
    assert "can not" in cleaned
    assert "will not" in cleaned
    assert "does not" in cleaned


def test_negations_preserved_in_stopwords():
    tokens = ["the", "teacher", "is", "not", "helpful", "and", "never", "arrives"]
    filtered = remove_stopwords(tokens)
    assert "not" in filtered
    assert "never" in filtered
    assert "the" not in filtered
    assert "is" not in filtered
    assert "and" not in filtered


def test_lemmatization_reduces_plurals():
    tokens = ["professors", "lectures", "computers", "examinations"]
    lemmas = lemmatize_tokens(tokens)
    assert "professor" in lemmas
    assert "lecture" in lemmas
    assert "computer" in lemmas
    assert "examination" in lemmas


def test_preprocess_end_to_end():
    sample = "The exams were NOT fair at all!"
    processed = preprocess(sample)
    # Negation "not" must be present in final string
    assert "not" in processed.split()
    assert "fair" in processed.split()


def test_preprocess_pipeline_dict():
    res = preprocess_pipeline("I didn't like the canteen food.")
    assert res["token_count"] > 0
    assert "not" in res["negations_preserved"]
    assert isinstance(res["lemmatized_tokens"], list)


def test_preprocess_empty_and_noise_inputs():
    assert preprocess("") == ""
    assert preprocess("    ") == ""
    assert preprocess("!@#$%^&*()") == ""
    assert preprocess(None) == ""
