"""Text Preprocessing Module for CampusPulse NLP.

Handles text normalization, tokenization, negation-preserving stop-word removal,
and lemmatization with automatic graceful fallbacks.
"""

import re
import string
from typing import List, Dict, Any

# Local NLP fallbacks
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer

    # Attempt to load stopwords
    try:
        _RAW_STOPWORDS = set(stopwords.words("english"))
    except Exception:
        _RAW_STOPWORDS = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
            "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
            "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
            "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
            "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
            "at", "by", "for", "with", "about", "against", "between", "into", "through",
            "during", "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "such", "only", "own", "same", "so",
            "than", "too", "very", "s", "t", "can", "will", "just", "should", "now"
        }

    # Attempt to initialize Lemmatizer
    try:
        _LEMMATIZER = WordNetLemmatizer()
        _LEMMATIZER.lemmatize("students")  # Test invocation
    except Exception:
        _LEMMATIZER = None

    _STEMMER = PorterStemmer()

except ImportError:
    _RAW_STOPWORDS = set()
    _LEMMATIZER = None
    _STEMMER = None

# Critical negation words that MUST be preserved for accurate sentiment classification
NEGATION_WORDS = {
    "not", "no", "nor", "neither", "never", "hardly", "barely",
    "scarcely", "without", "dont", "doesn't", "doesnt", "didnt",
    "didn't", "wasnt", "wasn't", "werent", "weren't", "wont",
    "won't", "wouldnt", "wouldn't", "cant", "can't", "couldnt",
    "couldn't", "shouldnt", "shouldn't", "hasnt", "hasn't",
    "havent", "haven't", "hadnt", "hadn't"
}

# Critical sentiment modifiers that convey excessive negative intensity (e.g. 'too high', 'too expensive')
SENTIMENT_MODIFIERS = {"too"}

# Final effective stopwords (standard stopwords minus sentiment negations and modifiers)
STOPWORDS = _RAW_STOPWORDS - NEGATION_WORDS - SENTIMENT_MODIFIERS


def clean_text(text: str) -> str:
    """Normalize text: convert to lowercase, expand common contractions,

    remove URLs, and strip non-alphanumeric punctuation while retaining negations.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower().strip()

    # Normalize standard contractions to preserve negation meaning
    # Specific negative contractions first
    specific_negations = [
        (r"\bcan['’]?t\b", "can not"),
        (r"\bcannot\b", "can not"),
        (r"\bwon['’]?t\b", "will not"),
        (r"\bshan['’]?t\b", "shall not"),
        (r"\bain['’]?t\b", "is not"),
        (r"\bdon['’]?t\b", "do not"),
        (r"\bdoesn['’]?t\b", "does not"),
        (r"\bdidn['’]?t\b", "did not"),
        (r"\bisn['’]?t\b", "is not"),
        (r"\baren['’]?t\b", "are not"),
        (r"\bwasn['’]?t\b", "was not"),
        (r"\bweren['’]?t\b", "were not"),
        (r"\bhasn['’]?t\b", "has not"),
        (r"\bhaven['’]?t\b", "have not"),
        (r"\bhadn['’]?t\b", "had not"),
        (r"\bwouldn['’]?t\b", "would not"),
        (r"\bshouldn['’]?t\b", "should not"),
        (r"\bcouldn['’]?t\b", "could not"),
    ]
    for pattern, replacement in specific_negations:
        text = re.sub(pattern, replacement, text)

    # General contractions
    contractions = {
        r"n['’]t\b": " not",
        r"['’]re\b": " are",
        r"['’]s\b": " is",
        r"['’]d\b": " would",
        r"['’]ll\b": " will",
        r"['’]ve\b": " have",
        r"['’]m\b": " am",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)

    # Remove URLs and web links
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove non-alphabetic characters (preserving spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> List[str]:
    """Tokenize cleaned text into individual word tokens."""
    if not text:
        return []
    # Fast, robust regex word tokenization
    return [token for token in text.split() if token]


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Filter out common stopwords while preserving negation and sentiment modifier tokens."""
    return [
        token for token in tokens
        if token not in STOPWORDS or token in NEGATION_WORDS or token in SENTIMENT_MODIFIERS
    ]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize tokens to base dictionary forms.

    Gracefully falls back to PorterStemmer or raw tokens if WordNet is unavailable.
    """
    if not tokens:
        return []

    if _LEMMATIZER is not None:
        try:
            return [_LEMMATIZER.lemmatize(token) for token in tokens]
        except Exception:
            pass

    if _STEMMER is not None:
        try:
            return [_STEMMER.stem(token) for token in tokens]
        except Exception:
            pass

    return tokens


def preprocess(text: str) -> str:
    """End-to-end preprocessing pipeline returning a space-separated normalized string."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    filtered = remove_stopwords(tokens)
    lemmatized = lemmatize_tokens(filtered)
    return " ".join(lemmatized)


def preprocess_pipeline(text: str) -> Dict[str, Any]:
    """Returns detailed intermediate representations for UI explanation and Viva demonstration."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    filtered = remove_stopwords(tokens)
    lemmatized = lemmatize_tokens(filtered)
    final_text = " ".join(lemmatized)

    return {
        "raw_text": text,
        "cleaned_text": cleaned,
        "tokens": tokens,
        "token_count": len(tokens),
        "filtered_tokens": filtered,
        "lemmatized_tokens": lemmatized,
        "final_preprocessed_text": final_text,
        "negations_preserved": [t for t in filtered if t in NEGATION_WORDS or t in SENTIMENT_MODIFIERS],
    }
