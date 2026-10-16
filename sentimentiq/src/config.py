"""
SentimentIQ - Configuration Settings
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset paths
SAMPLE_DATASET_PATH = DATA_DIR / "sample_reviews.csv"
BATCH_TEST_PATH = DATA_DIR / "batch_test_example.csv"

# Model artifact paths
BEST_MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
NB_MODEL_PATH = MODELS_DIR / "nb_model.joblib"
LR_MODEL_PATH = MODELS_DIR / "lr_model.joblib"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

# Training configuration
RANDOM_SEED = 42
TEST_SPLIT_SIZE = 0.20

# Class labels
TARGET_CLASSES = ["Negative", "Neutral", "Positive"]

# TF-IDF Vectorizer configuration
TFIDF_CONFIG = {
    "ngram_range": (1, 2),        # Unigrams and bigrams to capture "not good", "very clear"
    "max_features": 5000,         # Vocabulary size
    "min_df": 1,                  # Minimum document frequency
    "sublinear_tf": True,         # Apply sublinear scaling: 1 + log(tf)
    "use_idf": True,
    "norm": "l2"
}

# API configuration
MAX_TEXT_LENGTH = 5000            # Max characters allowed per input text
MAX_BATCH_SIZE = 1000             # Max rows allowed in batch CSV analysis
PORT = int(os.environ.get("PORT", 5001))
