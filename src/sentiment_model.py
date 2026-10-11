"""Sentiment Classification Model for CampusPulse NLP.

Implements TF-IDF feature extraction and Logistic Regression classification.
Ensures rigorous data leakage prevention via isolated train/test splitting,
and computes held-out evaluation metrics (accuracy, precision, recall, F1, confusion matrix).
Includes uncertainty / confidence detection.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from src.preprocessor import preprocess

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "student_feedback.csv"
)

MODEL_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "sentiment_model.joblib"
)

# Uncertainty thresholds
UNCERTAINTY_MIN_PROB = 0.45
UNCERTAINTY_MARGIN = 0.08


class SentimentModel:
    """End-to-end TF-IDF + Logistic Regression sentiment classifier."""

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[LogisticRegression] = None
        self.classes_: list = []
        self.metrics_: Dict[str, Any] = {}
        self.is_trained: bool = False

    def train_and_evaluate(
        self,
        data_path: str = DEFAULT_DATA_PATH,
        test_size: float = 0.20,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train vectorizer and Logistic Regression on training split ONLY to prevent data leakage.

        Evaluate and return metrics strictly on the held-out test split.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training dataset not found at: {data_path}")

        df = pd.read_csv(data_path)
        if "feedback_text" not in df.columns or "sentiment" not in df.columns:
            raise ValueError("Dataset must contain 'feedback_text' and 'sentiment' columns.")

        # Drop any empty or null rows
        df = df.dropna(subset=["feedback_text", "sentiment"]).copy()
        df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()

        # Step 1: Preprocess all text before feature extraction
        df["processed_text"] = df["feedback_text"].apply(preprocess)

        # Step 2: Stratified Train / Test Split (reproducible, 80/20)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            df["processed_text"],
            df["sentiment"],
            test_size=test_size,
            random_state=random_state,
            stratify=df["sentiment"]
        )

        # Step 3: Fit TF-IDF Vectorizer ONLY on training set (Strict Data Leakage Prevention)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            sublinear_tf=True,
            min_df=1
        )
        X_train_tfidf = self.vectorizer.fit_transform(X_train_raw)

        # Step 4: Transform held-out test set using the fitted vectorizer
        X_test_tfidf = self.vectorizer.transform(X_test_raw)

        # Step 5: Fit Logistic Regression Classifier
        self.classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.2,
            random_state=random_state
        )
        self.classifier.fit(X_train_tfidf, y_train)
        self.classes_ = list(self.classifier.classes_)
        self.is_trained = True

        # Step 6: Evaluate on held-out test set
        y_pred = self.classifier.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        conf_mat = confusion_matrix(y_test, y_pred, labels=self.classes_)
        report = classification_report(y_test, y_pred, labels=self.classes_, output_dict=True, zero_division=0)

        self.metrics_ = {
            "test_size": len(y_test),
            "train_size": len(y_train),
            "accuracy": float(round(acc, 4)),
            "precision_macro": float(round(prec_macro, 4)),
            "recall_macro": float(round(rec_macro, 4)),
            "f1_macro": float(round(f1_macro, 4)),
            "precision_weighted": float(round(prec_weighted, 4)),
            "recall_weighted": float(round(rec_weighted, 4)),
            "f1_weighted": float(round(f1_weighted, 4)),
            "classes": self.classes_,
            "confusion_matrix": conf_mat.tolist(),
            "classification_report": report,
            "test_distribution": y_test.value_counts().to_dict(),
            "train_distribution": y_train.value_counts().to_dict()
        }

        # Save model and metrics artifact to cache
        try:
            joblib.dump({
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "classes_": self.classes_,
                "metrics_": self.metrics_
            }, MODEL_CACHE_PATH)
        except Exception:
            pass

        return self.metrics_

    def load_or_train(
        self,
        data_path: str = DEFAULT_DATA_PATH,
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """Load cached trained model if available and up-to-date; otherwise train fresh.

        Prevents serving stale models by verifying file modification timestamps
        of both the training dataset and the pipeline source modules.
        """
        if not force_retrain and os.path.exists(MODEL_CACHE_PATH):
            try:
                cache_mtime = os.path.getmtime(MODEL_CACHE_PATH)
                dataset_is_newer = os.path.exists(data_path) and os.path.getmtime(data_path) > cache_mtime

                src_dir = os.path.dirname(os.path.abspath(__file__))
                preproc_file = os.path.join(src_dir, "preprocessor.py")
                this_file = os.path.abspath(__file__)
                code_is_newer = (
                    (os.path.exists(preproc_file) and os.path.getmtime(preproc_file) > cache_mtime) or
                    (os.path.exists(this_file) and os.path.getmtime(this_file) > cache_mtime)
                )

                if not dataset_is_newer and not code_is_newer:
                    cached = joblib.load(MODEL_CACHE_PATH)
                    if all(k in cached for k in ["vectorizer", "classifier", "classes_", "metrics_"]):
                        self.vectorizer = cached["vectorizer"]
                        self.classifier = cached["classifier"]
                        self.classes_ = cached["classes_"]
                        self.metrics_ = cached["metrics_"]
                        self.is_trained = True
                        return self.metrics_
            except Exception:
                pass

        return self.train_and_evaluate(data_path=data_path)

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for a single feedback string with confidence and uncertainty analysis."""
        if not self.is_trained or self.vectorizer is None or self.classifier is None:
            self.load_or_train()

        if not isinstance(text, str) or not text.strip():
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "probabilities": {cls: 0.33 for cls in self.classes_},
                "is_uncertain": True,
                "uncertainty_reason": "Empty or blank input text provided.",
                "explanation": "No text provided to evaluate."
            }

        # Preprocess input
        processed = preprocess(text)
        if not processed:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "probabilities": {cls: 0.33 for cls in self.classes_},
                "is_uncertain": True,
                "uncertainty_reason": "Input contained only noise or stopwords.",
                "explanation": "Feedback reduced to empty tokens after stopword removal."
            }

        # Vectorize
        X_vec = self.vectorizer.transform([processed])
        probs = self.classifier.predict_proba(X_vec)[0]
        prob_dict = {cls: float(round(p, 4)) for cls, p in zip(self.classes_, probs)}

        # Sort probabilities to detect top candidates
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top_cls, top_prob = sorted_probs[0]
        second_cls, second_prob = sorted_probs[1] if len(sorted_probs) > 1 else ("", 0.0)

        # Uncertainty heuristics
        is_uncertain = False
        uncertainty_reasons = []

        if top_prob < UNCERTAINTY_MIN_PROB:
            is_uncertain = True
            uncertainty_reasons.append(f"Top probability ({top_prob*100:.1f}%) is below confidence threshold ({UNCERTAINTY_MIN_PROB*100:.0f}%).")

        margin = top_prob - second_prob
        if margin < UNCERTAINTY_MARGIN:
            is_uncertain = True
            uncertainty_reasons.append(f"Model is torn between '{top_cls}' and '{second_cls}' (margin: {margin*100:.1f}%).")

        return {
            "sentiment": top_cls,
            "confidence": float(round(top_prob * 100, 2)),
            "probabilities": prob_dict,
            "is_uncertain": is_uncertain,
            "uncertainty_reason": " ".join(uncertainty_reasons) if is_uncertain else "",
            "explanation": f"Highest probability: {top_cls.upper()} ({top_prob*100:.1f}%)."
        }

    def predict_batch(self, texts: list) -> list:
        """Batch predict sentiment for a list of text strings."""
        if not self.is_trained or self.vectorizer is None or self.classifier is None:
            self.load_or_train()

        return [self.predict(t) for t in texts]


# Global singleton instance
_GLOBAL_MODEL: Optional[SentimentModel] = None


def get_sentiment_model(force_retrain: bool = False) -> SentimentModel:
    """Retrieve or initialize the global sentiment model singleton."""
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None or force_retrain:
        _GLOBAL_MODEL = SentimentModel()
        _GLOBAL_MODEL.load_or_train(force_retrain=force_retrain)
    return _GLOBAL_MODEL
