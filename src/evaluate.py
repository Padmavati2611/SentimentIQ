"""
SentimentIQ - Evaluation and Model Inspection Module
Generates detailed classification reports, confusion matrices, and model comparison.
"""

import json
import logging
import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.preprocessing import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_saved_model(model_path: Path = config.BEST_MODEL_PATH, test_csv: Path = config.SAMPLE_DATASET_PATH):
    """
    Evaluates a saved model pipeline against a dataset and displays metrics.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Run 'python src/train.py' first.")

    logger.info("Loading model pipeline from %s", model_path)
    pipeline = joblib.load(model_path)

    logger.info("Loading test dataset from %s", test_csv)
    df = pd.read_csv(test_csv)
    df = df.dropna(subset=["text", "sentiment"])
    df["cleaned_text"] = df["text"].apply(clean_text)

    X = df["cleaned_text"].values
    y_true = df["sentiment"].values

    y_pred = pipeline.predict(X)
    classes = config.TARGET_CLASSES

    print("\n" + "=" * 60)
    print("           SENTIMENTIQ - MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"Evaluated Model: {model_path.name}")
    print(f"Total Samples Evaluated: {len(df)}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print("Confusion Matrix:")
    header = f"{'Actual \\ Predicted':<20}" + "".join([f"{c:>12}" for c in classes])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{classes[i]:<20}" + "".join([f"{val:>12}" for val in row])
        print(row_str)
    print("=" * 60 + "\n")

    return {
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }


def get_stored_metrics() -> dict:
    """Reads the stored evaluation metrics from training."""
    if not config.METRICS_PATH.exists():
        return {}
    with open(config.METRICS_PATH, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    evaluate_saved_model()
