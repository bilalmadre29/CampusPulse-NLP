"""Topic Identification Module for CampusPulse NLP.

Uses an explainable, transparent rule-and-lexicon matching engine to categorize student
feedback into:
- Teaching
- Examinations
- Infrastructure
- Administration
- Other / Uncategorized (when no domain signals are present)

NOTE FOR VIVA: This is an explicit rule-based domain keyword & n-gram matcher,
chosen for transparency, auditability, and deterministic explainability without black-box bias.
"""

import re
from typing import Dict, Any, List, Tuple


# Domain Lexicons with weights for single words and compound domain phrases
TOPIC_LEXICONS = {
    "Teaching": {
        "keywords": [
            "teacher", "teachers", "teaching", "faculty", "professor", "professors", "prof",
            "lecturer", "lecturers", "lecture", "lectures", "instructor", "instructors",
            "syllabus", "curriculum", "concepts", "explanation", "explain", "explaining",
            "doubt", "doubts", "pedagogy", "slides", "notes", "tutorial", "tutorials",
            "mentor", "mentoring", "coursework", "assignment", "assignments", "homework"
        ],
        "phrases": [
            "teaching style", "doubt clearing", "office hours", "project guide",
            "audio visual", "lab instructor", "guest lecture", "course content",
            "board teaching", "clear explanations", "fast pace"
        ]
    },
    "Examinations": {
        "keywords": [
            "exam", "exams", "examination", "examinations", "test", "tests", "quiz", "quizzes",
            "midterm", "mid-term", "finals", "grading", "grade", "grades", "marks", "marking",
            "paper", "papers", "invigilator", "invigilation", "revaluation", "recheck",
            "result", "results", "assessment", "assessments", "hall ticket", "supplementary", "viva",
            "practical exam", "theory exam", "answer key", "answer sheet"
        ],
        "phrases": [
            "question paper", "exam hall", "exam schedule", "exam timetable",
            "seating arrangement", "paper evaluation", "out of syllabus",
            "grace marks", "passing marks", "study leave", "continuous assessment",
            "results declaration", "result announcement"
        ]
    },
    "Infrastructure": {
        "keywords": [
            "wifi", "wi-fi", "internet", "lab", "labs", "laboratory", "laboratories",
            "computer", "computers", "pc", "pcs", "hardware", "library", "books",
            "reading room", "classroom", "classrooms", "bench", "benches", "desk", "desks",
            "projector", "projectors", "ac", "air conditioner", "air conditioning",
            "cooler", "ventilation", "washroom", "washrooms", "toilet", "toilets",
            "canteen", "cafeteria", "mess", "food", "hostel", "hostels", "bus", "transport",
            "parking", "sports", "gym", "water", "purifier"
        ],
        "phrases": [
            "computer lab", "high speed", "smart board", "seminar hall",
            "drinking water", "power cut", "broken fans", "clean washrooms",
            "campus network", "sports complex", "bus service"
        ]
    },
    "Administration": {
        "keywords": [
            "admin", "administration", "administrative", "office", "registrar", "counter",
            "accounts", "fee", "fees", "scholarship", "scholarships", "admission",
            "admissions", "bonafide", "transcript", "transcripts", "certificate",
            "certificates", "circular", "notice", "notices", "management", "staff",
            "clerk", "clerks", "grievance", "helpdesk", "refund", "deposit",
            "coordinator", "id card", "identity card"
        ],
        "phrases": [
            "accounts counter", "scholarship desk", "fee payment", "admission desk",
            "student section", "academic calendar", "grievance cell", "office staff",
            "official notice", "transfer certificate", "caution deposit"
        ]
    }
}


def detect_topic(text: str) -> Dict[str, Any]:
    """Detect the academic topic of a given student feedback.

    Returns:
        Dict with detected topic, confidence score, matched keywords,
        detailed scores for all categories, and whether a confident domain match was made.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "topic": "Other / Uncategorized",
            "score": 0.0,
            "matched_keywords": [],
            "scores_by_topic": {cat: 0 for cat in TOPIC_LEXICONS},
            "is_identified": False,
            "explanation": "Input text was empty or invalid."
        }

    lower_text = text.lower()
    # Normalize punctuation to spaces for keyword matching
    normalized_text = re.sub(r"[^a-z0-9\s-]", " ", lower_text)
    words = set(normalized_text.split())

    scores: Dict[str, float] = {}
    matched_by_cat: Dict[str, List[str]] = {}

    for category, lexicon in TOPIC_LEXICONS.items():
        score = 0.0
        matches = []

        # Check multi-word domain phrases first (weight: 2.5 per phrase match)
        for phrase in lexicon["phrases"]:
            if phrase in lower_text:
                score += 2.5
                matches.append(phrase)

        # Check individual domain keywords (weight: 1.0 per exact word match)
        for kw in lexicon["keywords"]:
            if " " in kw:
                if kw in lower_text:
                    score += 1.5
                    matches.append(kw)
            elif kw in words:
                score += 1.0
                matches.append(kw)

        scores[category] = score
        matched_by_cat[category] = matches

    # Determine highest scoring topic
    sorted_topics = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_topic, best_score = sorted_topics[0]

    # Threshold check: requires at least 1 keyword match (score >= 1.0)
    if best_score < 1.0:
        return {
            "topic": "Other / Uncategorized",
            "primary_topic": "Other / Uncategorized",
            "secondary_topics": [],
            "all_topics": [],
            "score": 0.0,
            "matched_keywords": [],
            "scores_by_topic": scores,
            "is_identified": False,
            "is_tied": False,
            "is_multi_topic": False,
            "explanation": "No specific academic domain keywords (Teaching, Examinations, Infrastructure, Administration) were detected in this feedback."
        }

    # Identify all topics with meaningful keyword support (score >= 1.0)
    matched_topics = [cat for cat, sc in sorted_topics if sc >= 1.0]
    is_multi_topic = len(matched_topics) > 1

    # Check for tied topic scores at the top
    is_tied = False
    tied_categories = []
    for cat, sc in sorted_topics[1:]:
        if sc == best_score and sc > 0:
            is_tied = True
            tied_categories.append(cat)

    # Multi-topic formatting: when multiple topics have keywords (score >= 1.0), display both e.g. "Teaching & Administration"
    if is_multi_topic:
        display_topic = " & ".join(matched_topics)
        all_matched_kws = []
        for cat in matched_topics:
            all_matched_kws.extend(matched_by_cat[cat])
        expl = f"Matched domain indicators: {', '.join(all_matched_kws[:5])} (Multi-topic detected: {display_topic})"
    else:
        display_topic = best_topic
        all_matched_kws = matched_by_cat[best_topic]
        expl = f"Matched domain indicators: {', '.join(all_matched_kws[:5])}"

    return {
        "topic": display_topic,
        "primary_topic": best_topic,
        "secondary_topics": [c for c in matched_topics if c != best_topic],
        "all_topics": matched_topics,
        "score": round(best_score, 2),
        "matched_keywords": all_matched_kws,
        "scores_by_topic": scores,
        "is_identified": True,
        "is_tied": is_tied,
        "tied_with": tied_categories if is_tied else [],
        "is_multi_topic": is_multi_topic,
        "explanation": expl
    }


def detect_topics_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """Detect topics for a batch of student feedback strings."""
    return [detect_topic(t) for t in texts]
