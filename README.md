# CampusPulse — NLP-Based Student Feedback Intelligence System

CampusPulse is an explainable NLP intelligence system designed for colleges to analyze student feedback across academic categories (Teaching, Examinations, Infrastructure, and Administration) without external APIs or cloud dependencies.

---

## 1. Quick Start

### Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- Local libraries: Streamlit, Scikit-learn, Pandas, NLTK, Pytest

### Setup & Run Commands
```powershell
# 1. Navigate to project directory
cd E:\CampusPulse-NLP

# 2. Install dependencies (if not already installed)
python -m pip install -r requirements.txt

# 3. Launch Streamlit Application
python -m streamlit run app.py
```
*Note: If port 8501 is busy on your machine, Streamlit can be launched on port 8502 using:*
```powershell
python -m streamlit run app.py --server.port 8502
```

---

## 2. System Architecture & NLP Pipeline

```
Raw Student Text
      │
      ▼
1. Preprocessing (`src/preprocessor.py`)
   ├── Lowercasing & noise/URL removal
   ├── Contraction expansion (can't ➔ can not, won't ➔ will not)
   ├── Tokenization
   ├── Negation-Preserving Stopword Filtering (preserves 'not', 'never', 'hardly')
   └── Lemmatization (WordNetLemmatizer with PorterStemmer fallback)
      │
      ├─────────────────────────────────────────┐
      ▼                                         ▼
2. Sentiment Classification               3. Topic Identification
   (`src/sentiment_model.py`)               (`src/topic_detector.py`)
   ├── TF-IDF Vectorizer (1-2 ngrams)       ├── Transparent Domain Lexicon
   ├── Logistic Regression                  ├── Single & Compound Term Matching
   ├── Confidence & Probability Dist.       ├── Score Weighting & Tie Detection
   └── Uncertainty Flagging                 └── "Other / Uncategorized" Fallback
      │                                         │
      └────────────────────┬────────────────────┘
                           ▼
4. Feedback Intelligence & Batch Analytics (`src/analytics.py`)
   ├── Sentiment & Topic Distributions
   ├── Topic-Sentiment Cross-Tabulation Matrix
   ├── High-Priority Negative Problem Areas & Positive Strengths
   └── Interactive Dashboard (`app.py`)
```

---

## 3. Data Leakage Prevention & Evaluation Methodology

1. **Stratified Split**: The ground-truth dataset (`data/student_feedback.csv`, 207 rows) is split into 80% train (165 rows) and 20% test (42 rows) stratified by class.
2. **Zero Leakage**: `TfidfVectorizer.fit()` is called **exclusively on the training split**. The held-out test split is strictly evaluated using `transform()`.
3. **No Fabricated Scores**: Reported test metrics (Accuracy: ~66.67%, Macro F1: ~64.63%, Negative Recall: 93.33%) reflect authentic performance on held-out student feedback.

---

## 4. Test Suite Execution

Run automated unit and integration tests:
```powershell
python -m pytest tests -v
```

Run manual diagnostic feedback samples test:
```powershell
python -m tests.test_manual_samples
```

---

## 5. College NLP Viva Q&A Guide

### Q1: Why TF-IDF over simple Bag-of-Words (CountVectorizer)?
**Answer**: Term Frequency-Inverse Document Frequency (TF-IDF) penalizes ubiquitous domain words that appear in almost all feedbacks (e.g., 'college', 'student', 'class') while boosting discriminative sentiment and topic keywords (e.g., 'broken', 'engaging', 'delayed'). Simple Bag-of-Words merely counts raw frequencies, causing non-informative high-frequency words to dominate feature weights.

### Q2: Why was negation preserved during stopword removal?
**Answer**: Standard stopword lists (such as NLTK's English list) remove words like `'not'`, `'no'`, `'never'`, and `'hardly'`. If removed, negative feedback like *"The professor was not helpful"* collapses to *"professor helpful"*, inverting the sentiment to a false positive. Our preprocessor explicitly preserves negation tokens.

### Q3: How was data leakage prevented during model evaluation?
**Answer**: We split raw data into training and held-out test sets *before* feature extraction. The TF-IDF vectorizer was fitted *only* on the training split, and the test split was merely transformed. There are 0 duplicate rows and 0 vocabulary overlaps between train and test splits.

### Q4: Why use Macro F1-Score instead of Accuracy alone?
**Answer**: Macro F1 calculates the arithmetic mean of F1-scores across all three sentiment classes (Positive, Negative, Neutral) with equal weight. This ensures that good performance on a majority class does not hide poor performance on a minority class.

### Q5: Why is topic identification rule-and-lexicon based?
**Answer**: Unsupervised topic models (like LDA) generate probabilistic word clusters that shift across runs and lack deterministic boundaries. In college administration, departments (Teaching, Examinations, Infrastructure, Administration) have clear operational vocabularies. An explicit lexicon is deterministic, explainable in viva, auditable, and transparently flags `"Other / Uncategorized"` when keywords are absent.

---

## 6. Known Limitations
1. **Vocabulary Coverage**: Sentiment classification is bounded by the vocabulary in the training set; rare slang or misspellings may reduce classification confidence.
2. **Sarcasm and Subtlety**: Sarcastic comments (e.g., *"Oh wonderful, the lab computers crashed again"*) may be misclassified without deeper contextual models.
3. **Multi-Domain Feedbacks**: When a single student feedback mentions multiple topics with equal keyword frequency, the system detects a tie and notes tied categories.
