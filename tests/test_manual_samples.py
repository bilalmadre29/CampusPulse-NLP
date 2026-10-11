"""Explicit verification runner to test diverse feedbacks and record any misclassifications."""

from src.sentiment_model import get_sentiment_model
from src.topic_detector import detect_topic

def run_diagnostic():
    model = get_sentiment_model()

    test_cases = [
        ("The professor explains algorithms with great clarity and patience.", "positive", "Teaching"),
        ("The library is silent, spacious, and has plenty of reference books.", "positive", "Infrastructure"),
        ("The final exam schedule gave us zero preparation days between papers.", "negative", "Examinations"),
        ("Accounts office staff was extremely rude and made me wait for 3 hours.", "negative", "Administration"),
        ("The syllabus copy was distributed to all students by the faculty.", "neutral", "Teaching"),
        ("Seating arrangements for the midterm exam are posted on the board.", "neutral", "Examinations"),
        ("The explanation of mathematical proofs was not clear at all.", "negative", "Teaching"),
        ("The laboratory equipment is not working properly for the practical exam.", "negative", "Infrastructure"),
        ("The teacher is not bad, but the classroom projector is completely broken.", "negative", "Infrastructure"),
        ("Yesterday the college bus arrived at eight thirty in the morning.", "neutral", "Infrastructure"),
        ("The scholarship disbursement was processed quickly without any delay.", "positive", "Administration"),
        ("The invigilator was shouting at students during the examination.", "negative", "Examinations"),
        ("Today the sky was blue and birds were singing outside the campus.", "neutral", "Other / Uncategorized"),
    ]

    print(f"=== REAL FEEDBACK TEST SUITE ({len(test_cases)} CASES) ===\n")
    correct_sent = 0
    correct_topic = 0

    for i, (text, exp_sent, exp_topic) in enumerate(test_cases, 1):
        sent_res = model.predict(text)
        top_res = detect_topic(text)

        pred_sent = sent_res["sentiment"]
        pred_topic = top_res["topic"]
        conf = sent_res["confidence"]
        uncertain = sent_res["is_uncertain"]

        sent_match = (pred_sent == exp_sent)
        topic_match = (pred_topic == exp_topic)

        if sent_match:
            correct_sent += 1
        if topic_match:
            correct_topic += 1

        status = "PASS" if (sent_match and topic_match) else "MISMATCH"
        print(f"[{i}] \"{text}\"")
        print(f"    Expected: [{exp_sent.upper()} | {exp_topic}]")
        print(f"    Got:      [{pred_sent.upper()} ({conf}%) | {pred_topic}] (Uncertain={uncertain}) -> {status}")
        if not sent_match:
            print(f"    >>> Sentiment note: Expected {exp_sent}, got {pred_sent}. Probs: {sent_res['probabilities']}")
        if not topic_match:
            print(f"    >>> Topic note: Expected {exp_topic}, got {pred_topic}. Matched: {top_res['matched_keywords']}")
        print()

    print(f"Summary: Sentiment Accuracy = {correct_sent}/{len(test_cases)} ({correct_sent/len(test_cases)*100:.1f}%), Topic Accuracy = {correct_topic}/{len(test_cases)} ({correct_topic/len(test_cases)*100:.1f}%)")

if __name__ == "__main__":
    run_diagnostic()
