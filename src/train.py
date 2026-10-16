"""
SentimentIQ - Model Training Pipeline
Trains Multinomial Naive Bayes and Logistic Regression with TF-IDF.
Prevents data leakage by fitting vectorizer strictly on training split.
Saves model artifacts and comprehensive evaluation metrics.
"""

import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.preprocessing import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_and_validate_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Loads and validates the labeled sentiment dataset.
    Checks for missing values, duplicates, and verifies three valid classes.
    """
    logger.info("Loading dataset from: %s", csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate essential columns
    if "text" not in df.columns or "sentiment" not in df.columns:
        raise ValueError(f"Dataset must contain 'text' and 'sentiment' columns. Found: {list(df.columns)}")

    initial_count = len(df)
    logger.info("Initial rows loaded: %d", initial_count)

    # 1. Drop nulls or empty texts
    df = df.dropna(subset=["text", "sentiment"])
    df["text"] = df["text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    # 2. Standardize sentiment labels
    label_map = {
        "positive": "Positive",
        "pos": "Positive",
        "1": "Positive",
        "negative": "Negative",
        "neg": "Negative",
        "-1": "Negative",
        "neutral": "Neutral",
        "neu": "Neutral",
        "0": "Neutral",
    }
    df["sentiment"] = df["sentiment"].apply(lambda s: label_map.get(s.lower(), s))

    # Validate classes
    valid_classes = set(config.TARGET_CLASSES)
    df = df[df["sentiment"].isin(valid_classes)]

    # 3. Handle duplicates
    duplicates_count = df.duplicated(subset=["text"]).sum()
    if duplicates_count > 0:
        logger.info("Removing %d duplicate text entries", duplicates_count)
        df = df.drop_duplicates(subset=["text"], keep="first")

    logger.info("Validated dataset rows: %d (Dropped %d invalid/duplicate rows)", len(df), initial_count - len(df))

    # Display class distribution
    distribution = df["sentiment"].value_counts().to_dict()
    logger.info("Class distribution: %s", distribution)

    return df


def train_and_evaluate():
    """
    Full training, evaluation, comparison, and serialization workflow.
    """
    df = load_and_validate_dataset(config.SAMPLE_DATASET_PATH)

    # Preprocess text column using our sentiment-aware NLP preprocessor
    logger.info("Preprocessing text corpus...")
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Ensure no empty cleaned strings
    df = df[df["cleaned_text"].str.len() > 0]

    X = df["cleaned_text"].values
    y = df["sentiment"].values

    # Stratified Train-Test Split (80% train, 20% test)
    # Prevents data leakage: vectorizer is ONLY fit on X_train
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SPLIT_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )
    logger.info("Train set size: %d, Test set size: %d", len(X_train), len(X_test))

    # Model 1: Multinomial Naive Bayes Pipeline
    nb_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**config.TFIDF_CONFIG)),
        ("clf", MultinomialNB(alpha=1.0)),
    ])

    # Model 2: Logistic Regression Pipeline
    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**config.TFIDF_CONFIG)),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=config.RANDOM_SEED,
            class_weight="balanced"
        )),
    ])

    models = {
        "Multinomial Naive Bayes": nb_pipeline,
        "Logistic Regression": lr_pipeline,
    }

    eval_results = {}
    classes = config.TARGET_CLASSES

    for model_name, pipeline in models.items():
        logger.info("--- Training and Evaluating: %s ---", model_name)
        # Fit vectorizer and model on training set ONLY
        pipeline.fit(X_train, y_train)

        # Predict on holdout test split
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

        prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_test, y_pred, labels=classes)
        cr = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)

        logger.info("%s Accuracy: %.4f | Macro F1: %.4f", model_name, acc, f1_macro)

        eval_results[model_name] = {
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(rec_macro), 4),
            "f1_macro": round(float(f1_macro), 4),
            "precision_weighted": round(float(prec_weighted), 4),
            "recall_weighted": round(float(rec_weighted), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": classes,
            "classification_report": cr,
        }

    # Model Selection: Champion chosen based on highest Macro F1 Score
    nb_f1 = eval_results["Multinomial Naive Bayes"]["f1_macro"]
    lr_f1 = eval_results["Logistic Regression"]["f1_macro"]

    champion_name = "Logistic Regression" if lr_f1 >= nb_f1 else "Multinomial Naive Bayes"
    champion_pipeline = models[champion_name]

    logger.info("Champion Model Selected: %s (Macro F1: %.4f)", champion_name, eval_results[champion_name]["f1_macro"])

    # Model Persistence
    logger.info("Persisting models to: %s", config.MODELS_DIR)
    joblib.dump(nb_pipeline, config.NB_MODEL_PATH)
    joblib.dump(lr_pipeline, config.LR_MODEL_PATH)
    joblib.dump(champion_pipeline, config.BEST_MODEL_PATH)

    # Save complete evaluation metrics
    output_metrics = {
        "champion_model": champion_name,
        "selection_criterion": "Highest Macro F1-Score on Holdout Test Split",
        "classes": classes,
        "dataset_statistics": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "split_ratio": "80/20 Stratified",
            "random_seed": config.RANDOM_SEED,
            "class_distribution": df["sentiment"].value_counts().to_dict(),
        },
        "models": eval_results,
    }

    with open(config.METRICS_PATH, "w") as f:
        json.dump(output_metrics, f, indent=2)

    logger.info("Saved evaluation metrics to: %s", config.METRICS_PATH)
    logger.info("Training pipeline completed successfully.")
    return output_metrics


if __name__ == "__main__":
    train_and_evaluate()
