"""CampusPulse — NLP-Based Student Feedback Intelligence System.

A clean, presentation-ready application for analyzing college student feedback.
Uses classical, explainable NLP: TF-IDF + Logistic Regression, negation-preserving
preprocessing, and deterministic domain lexicon topic detection.
"""

import os
import io
import pandas as pd
import streamlit as st

from src.preprocessor import preprocess, preprocess_pipeline
from src.topic_detector import detect_topic
from src.sentiment_model import get_sentiment_model, MODEL_CACHE_PATH
from src.analytics import analyze_feedback_df, detect_text_column

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & RESTRAINED STYLING (NAVY / BLUE / TEAL)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CampusPulse — Student Feedback Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Clean, light academic layout */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* Result Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-pos { background-color: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }
    .badge-neg { background-color: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }
    .badge-neu { background-color: #F1F5F9; color: #334155; border: 1px solid #CBD5E1; }
    .badge-topic { background-color: #F0FDFA; color: #0F766E; border: 1px solid #99F6E4; }
    .badge-other { background-color: #F8FAFC; color: #64748B; border: 1px solid #E2E8F0; }

    /* Compact Metric Card */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px 14px;
        text-align: center;
    }
    .metric-card .title {
        color: #64748B;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-card .val {
        color: #0F172A;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 2px 0;
    }
    .metric-card .sub {
        color: #0D9488;
        font-size: 0.78rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 2. MODEL INITIALIZATION & CACHING
# -----------------------------------------------------------------------------
SAMPLE_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "sample_unlabeled_feedback.csv"
)

def get_model_cache_mtime() -> float:
    """Retrieve modification timestamp of model artifact to ensure cache invalidation."""
    if os.path.exists(MODEL_CACHE_PATH):
        return os.path.getmtime(MODEL_CACHE_PATH)
    return 0.0


@st.cache_resource(show_spinner="Loading NLP model...")
def load_nlp_model(cache_timestamp: float):
    return get_sentiment_model()


model = load_nlp_model(get_model_cache_mtime())
metrics = model.metrics_

# -----------------------------------------------------------------------------
# 3. TOP APP HEADER (CONCISE PROJECT PURPOSE)
# -----------------------------------------------------------------------------
st.title("🎓 CampusPulse")
st.caption(
    "NLP-Based Student Feedback Intelligence System — Analyzes student opinions across "
    "Teaching, Examinations, Infrastructure, and Administration using local TF-IDF and Logistic Regression."
)

# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION (ONLY THREE SECTIONS)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Navigation")
    nav_choice = st.radio(
        label="Select Section",
        options=[
            "Analyze Feedback",
            "Batch Analysis",
            "Model Evaluation"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()

    train_count = metrics.get("train_size", 165)
    test_count = metrics.get("test_size", 42)
    total_dataset = train_count + test_count
    acc_percent = metrics.get("accuracy", 0.6667) * 100

    st.markdown("**System Profile**")
    st.markdown(
        f"""
        - **Pipeline**: Scikit-learn + NLTK (100% Offline)
        - **Topics**: Teaching, Exams, Infra, Admin
        - **Dataset Total**: {total_dataset} Records
        - **Training Split**: {train_count} Records (80%)
        - **Held-Out Test Accuracy**: **{acc_percent:.1f}%** (N={test_count})
        """
    )

# -----------------------------------------------------------------------------
# 5. SECTION 1: ANALYZE FEEDBACK
# -----------------------------------------------------------------------------
if nav_choice == "Analyze Feedback":
    st.subheader("Analyze Individual Feedback")
    st.write("Enter student feedback below or click an example to load realistic comments:")

    # Session state for input text
    if "single_text" not in st.session_state:
        st.session_state["single_text"] = "The professor explains difficult algorithms very clearly and solves all doubts."

    # Four compact sample buttons
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        if st.button("👍 Good Teaching", use_container_width=True):
            st.session_state["single_text"] = "The professor explains difficult algorithms very clearly and solves all doubts."
    with sc2:
        if st.button("👎 Broken Infrastructure", use_container_width=True):
            st.session_state["single_text"] = "The projectors in room 302 never work and the lab internet is constantly down."
    with sc3:
        if st.button("⚠️ Difficult Negation", use_container_width=True):
            st.session_state["single_text"] = "The teacher was not helpful at all and notes were not clear."
    with sc4:
        if st.button("📄 Neutral Exam Notice", use_container_width=True):
            st.session_state["single_text"] = "The midterm examination timetable was announced on the student notice board."

    # Input Box
    input_text = st.text_area(
        label="Feedback Text:",
        value=st.session_state["single_text"],
        height=120,
        placeholder="Type student feedback here..."
    )

    analyze_btn = st.button("Analyze Feedback", type="primary")

    # Results Section
    if analyze_btn or input_text.strip():
        if not input_text.strip():
            st.warning("Please enter some feedback text to analyze.")
        else:
            sent_res = model.predict(input_text)
            topic_res = detect_topic(input_text)
            pipe_res = preprocess_pipeline(input_text)

            st.divider()
            st.markdown("### Analysis Results")

            # Result Metrics Row
            res_c1, res_c2, res_c3 = st.columns(3)

            with res_c1:
                st.markdown("**Predicted Sentiment**")
                s_label = sent_res["sentiment"]
                if s_label == "positive":
                    st.markdown(f'<span class="badge badge-pos">● POSITIVE ({sent_res["confidence"]}%)</span>', unsafe_allow_html=True)
                elif s_label == "negative":
                    st.markdown(f'<span class="badge badge-neg">● NEGATIVE ({sent_res["confidence"]}%)</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="badge badge-neu">● NEUTRAL ({sent_res["confidence"]}%)</span>', unsafe_allow_html=True)
                st.caption(f"Confidence: {sent_res['confidence']}%")

            with res_c2:
                st.markdown("**Detected Department Topic**")
                # Backward-compatible multi-topic badge label
                if topic_res.get("is_multi_topic") and topic_res.get("all_topics"):
                    top_label = " & ".join(topic_res["all_topics"])
                else:
                    top_label = topic_res.get("topic", "Other / Uncategorized")

                if topic_res["is_identified"]:
                    st.markdown(f'<span class="badge badge-topic">🏷️ {top_label.upper()}</span>', unsafe_allow_html=True)
                    if topic_res.get("is_multi_topic"):
                        st.caption(f"Multi-Topic: {', '.join(topic_res['all_topics'])} | Keywords: {', '.join(topic_res['matched_keywords'][:4])}")
                    else:
                        st.caption(f"Score: {topic_res['score']} | {', '.join(topic_res['matched_keywords'][:3])}")
                else:
                    st.markdown(f'<span class="badge badge-other">🏷️ {top_label.upper()}</span>', unsafe_allow_html=True)
                    st.caption("No domain keywords found")

            with res_c3:
                st.markdown("**Model Certainty**")
                if sent_res["is_uncertain"]:
                    st.warning(f"⚠️ Low Confidence\n{sent_res['uncertainty_reason']}")
                else:
                    st.success("✅ Confident Prediction\nClear probability margin.")

            # Plain-Language Output Explanation
            st.info(
                f"**Explanation:** The model assigned **{s_label.upper()}** because the preprocessed tokens "
                f"(*\"{pipe_res['final_preprocessed_text']}\"*) matched sentiment patterns in the training data. "
                f"**Topic:** {topic_res['explanation']}"
            )

            # Optional Token Inspection Expander for Viva
            with st.expander("🔬 View NLP Preprocessed Tokens (For Viva)"):
                st.markdown(
                    f"""
                    - **Raw Text**: `{pipe_res['raw_text']}`
                    - **Cleaned & Contractions Expanded**: `{pipe_res['cleaned_text']}`
                    - **Tokens**: `{pipe_res['tokens']}`
                    - **Stopwords Filtered (Negations Kept: `{pipe_res['negations_preserved']}`)**: `{pipe_res['filtered_tokens']}`
                    - **Lemmatized Base Forms**: `{pipe_res['lemmatized_tokens']}`
                    - **TF-IDF String**: `{pipe_res['final_preprocessed_text']}`
                    """
                )

# -----------------------------------------------------------------------------
# 6. SECTION 2: BATCH ANALYSIS (SIMPLIFIED 4-STEP WORKFLOW)
# -----------------------------------------------------------------------------
elif nav_choice == "Batch Analysis":
    st.subheader("Batch Analysis")
    st.write("Analyze multiple student comments from a CSV file using the exact 4-step workflow below:")

    # STEP 1: Choose feedback data
    st.markdown("#### STEP 1 — Choose Feedback Data")
    st.caption("Upload your own CSV file OR load the bundled 20-review sample dataset:")

    step1_col1, step1_col2 = st.columns([1, 2])
    with step1_col1:
        load_sample = st.button("📂 Load Sample Dataset (20 Reviews)", use_container_width=True)
    with step1_col2:
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            label_visibility="collapsed"
        )

    # Ingestion State
    if "batch_df" not in st.session_state:
        st.session_state["batch_df"] = None
        st.session_state["batch_source"] = ""

    if load_sample:
        if os.path.exists(SAMPLE_CSV_PATH):
            st.session_state["batch_df"] = pd.read_csv(SAMPLE_CSV_PATH)
            st.session_state["batch_source"] = "Sample Dataset (20 Reviews)"
        else:
            st.error("Sample dataset file not found.")
    elif uploaded_file is not None:
        try:
            st.session_state["batch_df"] = pd.read_csv(uploaded_file)
            st.session_state["batch_source"] = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")

    # Proceed to STEP 2 & 3 only if data is loaded
    if st.session_state["batch_df"] is not None:
        df = st.session_state["batch_df"]

        if df.empty:
            st.warning("The loaded CSV file has 0 rows.")
        else:
            # STEP 2: Preview the data
            st.divider()
            st.markdown("#### STEP 2 — Preview the Data")
            st.write(f"**Loaded:** `{st.session_state['batch_source']}` | **Total Rows:** `{len(df)}`")

            # Column selector
            auto_col = detect_text_column(df)
            col_options = list(df.columns)
            def_idx = col_options.index(auto_col) if auto_col in col_options else 0

            chosen_col = st.selectbox(
                "Select feedback text column:",
                options=col_options,
                index=def_idx
            )

            st.write("**Data Preview (First 5 Rows):**")
            st.dataframe(df.head(5), use_container_width=True)

            # STEP 3: Analyze
            st.divider()
            st.markdown("#### STEP 3 — Analyze")
            run_batch = st.button("⚡ Analyze Feedback", type="primary")

            # STEP 4: Display actual results
            if run_batch:
                with st.spinner("Analyzing feedback records..."):
                    try:
                        results = analyze_feedback_df(df, chosen_col)
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        results = None

                if results and results["total_records"] > 0:
                    st.divider()
                    st.markdown("#### STEP 4 — Actual Analysis Results")

                    # Record Count and Sentiment Counts
                    tot = results["total_records"]
                    pos_c = results["sentiment_counts"].get("positive", 0)
                    neg_c = results["sentiment_counts"].get("negative", 0)
                    neu_c = results["sentiment_counts"].get("neutral", 0)

                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="title">Total Records</div>
                                <div class="val">{tot}</div>
                                <div class="sub">Processed Comments</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with k2:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="title">Positive</div>
                                <div class="val" style="color: #065F46;">{pos_c}</div>
                                <div class="sub">{results['sentiment_percentages'].get('positive', 0.0)}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with k3:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="title">Negative</div>
                                <div class="val" style="color: #991B1B;">{neg_c}</div>
                                <div class="sub">{results['sentiment_percentages'].get('negative', 0.0)}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with k4:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="title">Neutral</div>
                                <div class="val" style="color: #334155;">{neu_c}</div>
                                <div class="sub">{results['sentiment_percentages'].get('neutral', 0.0)}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # Two Readable Charts
                    st.markdown("##### Distribution Charts")
                    ch1, ch2 = st.columns(2)

                    with ch1:
                        st.write("**Sentiment Distribution**")
                        s_df = pd.DataFrame({
                            "Sentiment": [k.capitalize() for k in results["sentiment_counts"].keys()],
                            "Count": list(results["sentiment_counts"].values())
                        }).set_index("Sentiment")
                        st.bar_chart(s_df, height=220)

                    with ch2:
                        st.write("**Topic Distribution**")
                        t_df = pd.DataFrame({
                            "Topic": list(results["topic_counts"].keys()),
                            "Count": list(results["topic_counts"].values())
                        }).set_index("Topic")
                        st.bar_chart(t_df, height=220)

                    # Results Table
                    st.markdown("##### Results Table")
                    display_cols = [chosen_col, "Predicted_Sentiment", "Detected_Topic", "Confidence_%", "Is_Uncertain"]
                    cols_to_show = [c for c in display_cols if c in results["analyzed_df"].columns]
                    st.dataframe(results["analyzed_df"][cols_to_show], use_container_width=True)

                    # Working CSV Download Button
                    csv_buf = io.StringIO()
                    results["analyzed_df"].to_csv(csv_buf, index=False)
                    st.download_button(
                        label="📥 Download Analyzed CSV",
                        data=csv_buf.getvalue(),
                        file_name="campuspulse_analyzed_results.csv",
                        mime="text/csv",
                        type="primary"
                    )

                elif results:
                    st.warning(results.get("warning", "No valid records could be processed."))

# -----------------------------------------------------------------------------
# 7. SECTION 3: MODEL EVALUATION
# -----------------------------------------------------------------------------
elif nav_choice == "Model Evaluation":
    st.subheader("Model Evaluation")
    test_eval_count = metrics.get("test_size", 42)
    st.write(
        f"Actual evaluation results measured strictly on **{test_eval_count} held-out test examples** "
        f"(20% test split) that were never seen during training or TF-IDF feature fitting. Zero data leakage."
    )

    # Actual Held-Out Metrics
    ev1, ev2, ev3, ev4 = st.columns(4)
    with ev1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="title">Test Accuracy</div>
                <div class="val">{metrics.get('accuracy', 0.60)*100:.2f}%</div>
                <div class="sub">Overall Correct</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption("**Accuracy**: Percentage of test student reviews correctly categorized across all 3 classes.")

    with ev2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="title">Macro Precision</div>
                <div class="val">{metrics.get('precision_macro', 0.62)*100:.2f}%</div>
                <div class="sub">Prediction Reliability</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption("**Precision**: Average reliability of predictions across all 3 sentiment classes.")

    with ev3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="title">Macro Recall</div>
                <div class="val">{metrics.get('recall_macro', 0.601)*100:.2f}%</div>
                <div class="sub">Class Coverage</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption("**Recall**: Average proportion of actual positive, negative, and neutral reviews captured.")

    with ev4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="title">Macro F1-Score</div>
                <div class="val">{metrics.get('f1_macro', 0.5996)*100:.2f}%</div>
                <div class="sub">Harmonic Balance</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption("**F1-Score**: Harmonic balance between precision and recall, treating all classes equally.")

    st.divider()

    # Confusion Matrix
    st.markdown("### Confusion Matrix (Held-Out Test Set)")
    st.write(
        "**How to read this matrix in a viva:** "
        "Rows represent the actual human-annotated sentiment; columns represent the model's prediction. "
        "Numbers along the diagonal (top-left, center, bottom-right) are correct predictions. "
        "Numbers off the diagonal show misclassifications."
    )

    classes_list = metrics.get("classes", ["negative", "neutral", "positive"])
    conf_mat_vals = metrics.get("confusion_matrix", [[8, 4, 0], [2, 7, 2], [4, 2, 6]])
    cm_df = pd.DataFrame(
        conf_mat_vals,
        index=[f"Actual {c.capitalize()}" for c in classes_list],
        columns=[f"Predicted {c.capitalize()}" for c in classes_list]
    )
    st.table(cm_df)

    # Per-Class Classification Report
    st.markdown("### Per-Class Report")
    rep = metrics.get("classification_report", {})
    if rep:
        rows = []
        for cls in classes_list:
            if cls in rep:
                rows.append({
                    "Class": cls.capitalize(),
                    "Precision": f"{rep[cls]['precision']:.4f}",
                    "Recall": f"{rep[cls]['recall']:.4f}",
                    "F1-Score": f"{rep[cls]['f1-score']:.4f}",
                    "Support": int(rep[cls]['support'])
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()

    # Short Viva Questions & Answers
    st.markdown("### 🎓 Viva Questions & Answers")

    with st.expander("Q1: Why TF-IDF over simple Bag-of-Words?"):
        st.write(
            "Bag-of-Words merely counts word frequencies, causing frequent college words ('student', 'class') to get high weights. "
            "TF-IDF discounts ubiquitous words and boosts discriminative words ('broken', 'helpful', 'delayed')."
        )

    with st.expander("Q2: Why was negation preserved during stop-word removal?"):
        st.write(
            "Standard stop-word lists remove words like 'not', 'no', 'never'. "
            "If removed, negative feedback like 'The professor was not helpful' collapses to 'professor helpful', inverting sentiment to positive."
        )

    with st.expander("Q3: How was data leakage prevented?"):
        st.write(
            "The dataset was split into 80% train and 20% test before any vectorization. "
            "The TF-IDF vectorizer was fit exclusively on the training split, and the test split was only transformed."
        )

    with st.expander("Q4: Why rule-based topic detection rather than unsupervised LDA?"):
        st.write(
            "Unsupervised LDA topic clusters shift across runs, lack deterministic labels, and require thousands of documents. "
            "A deterministic domain lexicon is transparent, auditable, and clearly indicates when feedback is unclassified."
        )
