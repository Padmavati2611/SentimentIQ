"""
SentimentIQ - Flask REST API Application
Provides prediction endpoints, health checks, batch processing, and model metadata.
Can run standalone with Jinja2 templates or serve as the ML backend.
"""

import json
import logging
import os
import sys
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

# Add root directory to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src import config
from src.predict import get_predictor
from src.preprocessing import preprocessor

# Initialize Flask
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Enable Cross-Origin Resource Sharing for API callers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sentimentiq")

# Initialize model predictor on startup
try:
    predictor = get_predictor()
    logger.info("Predictor initialized successfully with model: %s", config.BEST_MODEL_PATH)
except Exception as e:
    logger.error("Failed to initialize predictor: %s", e)
    predictor = None


# Curated sample texts for quick testing across all three sentiment classes
SAMPLE_REVIEWS = [
    {
        "category": "Positive",
        "title": "Audio Headset Review",
        "text": "The sound quality is crystal clear with deep bass and excellent battery life. Absolutely love this headset!"
    },
    {
        "category": "Negative",
        "title": "Hardware Defect Complaint",
        "text": "Worst customer service experience of my life. The item arrived broken and nobody responded to my emails."
    },
    {
        "category": "Neutral",
        "title": "Logistics Delivery Notice",
        "text": "The package arrived on Tuesday as scheduled. The box was sealed and contents were inside."
    },
    {
        "category": "Positive",
        "title": "SaaS Platform Praise",
        "text": "Exceptional build quality and intuitive software. Completely transformed our team's daily productivity."
    },
    {
        "category": "Negative",
        "title": "Battery Life Complaint",
        "text": "This phone battery drains completely in less than four hours. Total waste of money and very disappointed."
    },
    {
        "category": "Neutral",
        "title": "Product Specification",
        "text": "The device dimensions are 6.1 inches by 2.8 inches with a weight of 174 grams."
    }
]


# ==========================================
# REST API Endpoints
# ==========================================

@app.route("/api/health", methods=["GET"])
def health_check():
    """
    GET /api/health
    Returns server status, model readiness, and version info.
    """
    ready = predictor is not None and predictor.is_ready()
    return jsonify({
        "status": "healthy" if ready else "degraded",
        "model_loaded": ready,
        "model_path": str(config.BEST_MODEL_PATH.name),
        "classes": config.TARGET_CLASSES,
        "version": "1.0.0",
        "service": "SentimentIQ ML Engine"
    }), (200 if ready else 503)


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    """
    GET /api/model-info
    Returns champion model metadata, hyperparameters, and documented evaluation metrics.
    """
    if not config.METRICS_PATH.exists():
        return jsonify({
            "error": "Metrics not found. Please execute model training first.",
            "champion_model": "Logistic Regression (TF-IDF)",
            "classes": config.TARGET_CLASSES
        }), 404

    with open(config.METRICS_PATH, "r") as f:
        metrics_data = json.load(f)

    # Add runtime configuration metadata
    metrics_data["tfidf_configuration"] = {
        "ngram_range": list(config.TFIDF_CONFIG["ngram_range"]),
        "max_features": config.TFIDF_CONFIG["max_features"],
        "sublinear_tf": config.TFIDF_CONFIG["sublinear_tf"],
        "stopword_filtering": "Enabled with negation preservation"
    }
    metrics_data["supported_classes"] = config.TARGET_CLASSES
    return jsonify(metrics_data), 200


@app.route("/api/samples", methods=["GET"])
def get_sample_reviews():
    """
    GET /api/samples
    Returns pre-configured sample reviews representing Positive, Negative, and Neutral.
    """
    return jsonify({"samples": SAMPLE_REVIEWS}), 200


@app.route("/api/predict", methods=["POST"])
def predict_single():
    """
    POST /api/predict
    Predict sentiment for a single text input.
    Body: {"text": "..."}
    """
    if not predictor or not predictor.is_ready():
        return jsonify({"error": "Model is not loaded. Please ensure training artifacts exist."}), 503

    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON request body. Expected {'text': '...'}"}), 400

    text = data.get("text")
    if text is None or not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Text field cannot be empty. Please provide text to analyze."}), 400

    if len(text.strip()) > config.MAX_TEXT_LENGTH:
        return jsonify({
            "error": f"Text exceeds maximum character limit of {config.MAX_TEXT_LENGTH}. Current length: {len(text.strip())}"
        }), 400

    try:
        result = predictor.predict_one(text)
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception("Inference error: %s", e)
        return jsonify({"error": "Internal inference error occurred while processing the text."}), 500


@app.route("/api/predict-batch", methods=["POST"])
def predict_batch():
    """
    POST /api/predict-batch
    Predict sentiment for a list of texts (e.g. parsed CSV rows).
    Body: {"texts": ["text1", "text2", ...]}
    """
    if not predictor or not predictor.is_ready():
        return jsonify({"error": "Model is not loaded."}), 503

    data = request.get_json(silent=True)
    if not data or "texts" not in data or not isinstance(data["texts"], list):
        return jsonify({"error": "Invalid request body. Expected {'texts': ['string1', ...]}"}), 400

    texts = data["texts"]
    if len(texts) == 0:
        return jsonify({"error": "The 'texts' list cannot be empty."}), 400

    if len(texts) > config.MAX_BATCH_SIZE:
        return jsonify({
            "error": f"Batch size exceeds maximum allowed of {config.MAX_BATCH_SIZE} items."
        }), 400

    try:
        batch_result = predictor.predict_batch(texts)
        return jsonify(batch_result), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.exception("Batch prediction error: %s", e)
        return jsonify({"error": "Internal error occurred while processing batch predictions."}), 500


@app.route("/api/preprocess-test", methods=["POST"])
def test_preprocessing():
    """
    POST /api/preprocess-test
    Interactive NLP pipeline inspection: returns before/after steps of text cleaning.
    Body: {"text": "..."}
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    steps = preprocessor.get_pipeline_steps(text)
    return jsonify(steps), 200


@app.route("/api/download-sample-csv", methods=["GET"])
def download_sample_csv():
    """
    GET /api/download-sample-csv
    Allows users to download the sample batch CSV for testing.
    """
    if not config.BATCH_TEST_PATH.exists():
        return jsonify({"error": "Sample batch CSV not found."}), 404
    return send_from_directory(
        directory=str(config.DATA_DIR),
        path="batch_test_example.csv",
        as_attachment=True,
        mimetype="text/csv"
    )


# Standalone Web Template Route (for direct Flask deployment)
@app.route("/", methods=["GET"])
def index():
    """Renders the standalone Flask HTML template."""
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5005))
    logger.info("Starting SentimentIQ Flask API on http://0.0.0.0:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
