import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sentiment_model import get_sentiment_model

def main():
    m = get_sentiment_model()
    metrics = m.metrics_
    print("=== FINAL HELD-OUT TEST EVALUATION METRICS ===")
    print(f"Training Samples (80%): {metrics['train_size']}")
    print(f"Held-out Test Samples (20%): {metrics['test_size']}")
    print(f"Test Class Distribution: {metrics['test_distribution']}")
    print(f"Accuracy:          {metrics['accuracy']*100:.2f}%")
    print(f"Macro Precision:   {metrics['precision_macro']*100:.2f}%")
    print(f"Macro Recall:      {metrics['recall_macro']*100:.2f}%")
    print(f"Macro F1-Score:    {metrics['f1_macro']*100:.2f}%")
    print(f"Weighted F1-Score: {metrics['f1_weighted']*100:.2f}%")
    print(f"\nClasses: {metrics['classes']}")
    print("Confusion Matrix (Rows: Actual [Negative, Neutral, Positive], Cols: Predicted [Negative, Neutral, Positive]):")
    for row in metrics['confusion_matrix']:
        print(f"  {row}")
    print("\nPer-Class Classification Report:")
    for cls in metrics['classes']:
        rep = metrics['classification_report'][cls]
        print(f"  {cls.upper():<10} -> Precision: {rep['precision']:.4f}, Recall: {rep['recall']:.4f}, F1: {rep['f1-score']:.4f}, Support: {int(rep['support'])}")

if __name__ == "__main__":
    main()
