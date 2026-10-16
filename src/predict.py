"""
SentimentIQ - Inference and Prediction Engine
Loads the serialized model pipeline once at startup.
Provides single-text and batch prediction with confidence scoring,
mixed-sentiment detection, vocabulary overlap inspection, and token explainability.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.preprocessing import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SentimentPredictor:
    """
    Production-ready sentiment predictor wrapper with caching, mixed-sentiment detection,
    out-of-vocabulary detection, and token-level explainability.
    """

    def __init__(self, model_path: Path = config.BEST_MODEL_PATH):
        self.model_path = model_path
        self.pipeline = None
        self.classes = config.TARGET_CLASSES
        self._load_model()

    def _load_model(self):
        """Loads serialized model pipeline from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Please execute 'python src/train.py' first."
            )
        logger.info("Loading sentiment model from %s...", self.model_path)
        self.pipeline = joblib.load(self.model_path)
        if hasattr(self.pipeline.named_steps.get("clf"), "classes_"):
            self.classes = list(self.pipeline.named_steps["clf"].classes_)
        logger.info("Model loaded successfully. Classes: %s", self.classes)

    def is_ready(self) -> bool:
        """Returns True if model pipeline is initialized."""
        return self.pipeline is not None

    def explain_prediction(self, raw_text: str, cleaned_text: str, predicted_class: str) -> List[Dict[str, Any]]:
        """
        Calculates token attribution for explainability.
        Finds which words/n-grams in the input text contributed most towards the predicted class.
        """
        try:
            tfidf = self.pipeline.named_steps["tfidf"]
            clf = self.pipeline.named_steps["clf"]

            vector = tfidf.transform([cleaned_text])
            feature_names = tfidf.get_feature_names_out()
            nonzero_indices = vector.nonzero()[1]

            if len(nonzero_indices) == 0:
                return []

            class_idx = list(clf.classes_).index(predicted_class)

            explanations = []
            if hasattr(clf, "coef_"):
                # Logistic Regression: coef_[class_idx, feature_idx] * tfidf_weight
                for idx in nonzero_indices:
                    feature = feature_names[idx]
                    weight = clf.coef_[class_idx, idx] * vector[0, idx]
                    explanations.append({
                        "token": feature,
                        "weight": round(float(weight), 4),
                        "polarity": "positive_signal" if weight > 0 else "counter_signal"
                    })
            elif hasattr(clf, "feature_log_prob_"):
                # Naive Bayes: feature_log_prob_[class_idx, feature_idx]
                for idx in nonzero_indices:
                    feature = feature_names[idx]
                    weight = clf.feature_log_prob_[class_idx, idx] * vector[0, idx]
                    explanations.append({
                        "token": feature,
                        "weight": round(float(weight), 4),
                        "polarity": "positive_signal" if weight > -5 else "neutral_signal"
                    })

            explanations = sorted(explanations, key=lambda x: x["weight"], reverse=True)
            return explanations[:6]
        except Exception as e:
            logger.warning("Explainability calculation skipped: %s", e)
            return []

    def predict_one(self, text: Optional[str]) -> Dict[str, Any]:
        """
        Classifies a single text string into Positive, Negative, or Neutral sentiment.
        Validates input, measures inference latency, checks vocabulary coverage and mixed sentiment.
        """
        start_time = time.perf_counter()

        if text is None or not isinstance(text, str):
            raise ValueError("Input text must be a valid non-empty string.")

        raw_trimmed = text.strip()
        if len(raw_trimmed) == 0:
            raise ValueError("Text input cannot be empty or only whitespace.")

        if len(raw_trimmed) > config.MAX_TEXT_LENGTH:
            raise ValueError(f"Text exceeds maximum character limit of {config.MAX_TEXT_LENGTH}.")

        cleaned = clean_text(raw_trimmed)
        if len(cleaned) == 0:
            cleaned = raw_trimmed.lower()

        tfidf = self.pipeline.named_steps["tfidf"]
        vector = tfidf.transform([cleaned])
        nonzero_count = int(vector.count_nonzero())
        is_oov = bool(nonzero_count == 0)

        # Run pipeline inference
        probabilities = self.pipeline.predict_proba([cleaned])[0]
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = str(self.classes[predicted_idx])
        confidence = float(probabilities[predicted_idx])

        # Map class probabilities
        prob_dict = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(self.classes, probabilities)
        }

        # Check for mixed sentiment / close margin
        sorted_probs = sorted([(cls, p) for cls, p in prob_dict.items()], key=lambda x: x[1], reverse=True)
        margin = float(sorted_probs[0][1] - sorted_probs[1][1])
        is_mixed_sentiment = bool(margin < 0.12 and not is_oov)

        # Token explainability
        influential_tokens = self.explain_prediction(raw_trimmed, cleaned, predicted_class)

        # Calibrated certainty labeling
        if is_oov:
            certainty = "prior baseline (unseen vocabulary)"
            explanation_msg = (
                f"Prediction '{predicted_class}' is based on baseline prior probabilities. "
                "The input contains terms outside the training vocabulary, so model certainty is low."
            )
        elif is_mixed_sentiment:
            certainty = "mixed / borderline sentiment"
            explanation_msg = (
                f"Close contest between '{sorted_probs[0][0]}' ({round(sorted_probs[0][1]*100, 1)}%) "
                f"and '{sorted_probs[1][0]}' ({round(sorted_probs[1][1]*100, 1)}%). The text contains opposing sentiment signals."
            )
        elif confidence >= 0.70:
            certainty = "high certainty"
            tokens_str = ", ".join([f"'{t['token']}'" for t in influential_tokens[:3]])
            explanation_msg = f"Classified as '{predicted_class}' with high certainty ({round(confidence * 100, 1)}%). Strongest signals: {tokens_str or 'overall phrase semantics'}."
        elif confidence >= 0.48:
            certainty = "moderate certainty"
            tokens_str = ", ".join([f"'{t['token']}'" for t in influential_tokens[:3]])
            explanation_msg = f"Classified as '{predicted_class}' with moderate certainty ({round(confidence * 100, 1)}%). Influential features: {tokens_str or 'overall phrase semantics'}."
        else:
            certainty = "marginal confidence"
            explanation_msg = f"Classified as '{predicted_class}' with marginal confidence ({round(confidence * 100, 1)}%)."

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "text": raw_trimmed,
            "cleaned_text": cleaned,
            "sentiment": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "explanation": explanation_msg,
            "influential_tokens": influential_tokens,
            "character_count": len(raw_trimmed),
            "word_count": len(raw_trimmed.split()),
            "is_mixed_sentiment": is_mixed_sentiment,
            "is_out_of_vocabulary": is_oov,
            "vocabulary_tokens_matched": nonzero_count,
            "certainty_level": certainty,
            "latency_ms": latency_ms,
        }

    def predict_batch(self, texts: List[str]) -> Dict[str, Any]:
        """
        Batch prediction for a list of texts (e.g. from a CSV upload).
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Input texts must be a non-empty list of strings.")

        if len(texts) > config.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum limit of {config.MAX_BATCH_SIZE} items.")

        results = []
        counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        total_valid = 0

        for idx, t in enumerate(texts):
            raw = str(t) if t is not None else ""
            if not raw.strip():
                continue

            try:
                res = self.predict_one(raw)
                res["row_index"] = idx + 1
                results.append(res)
                counts[res["sentiment"]] = counts.get(res["sentiment"], 0) + 1
                total_valid += 1
            except Exception as e:
                logger.warning("Error predicting row %d: %s", idx, e)

        if total_valid == 0:
            raise ValueError("No valid, non-empty text rows found to analyze.")

        percentages = {
            sentiment: round((count / total_valid) * 100, 1)
            for sentiment, count in counts.items()
        }

        return {
            "total_analyzed": total_valid,
            "counts": counts,
            "percentages": percentages,
            "items": results,
        }


_predictor_instance = None


def get_predictor() -> SentimentPredictor:
    """Returns singleton instance of SentimentPredictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SentimentPredictor()
    return _predictor_instance
