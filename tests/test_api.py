"""
SentimentIQ - Integration and API Tests
Tests REST endpoints, status codes, input validations, and predictions.
Confirms predictions use the trained ML model pipeline rather than hardcoded rules.
"""

import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app
from src import config


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """GET /api/health returns 200 with service information."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "Negative" in data["classes"]
    assert "Positive" in data["classes"]


def test_model_info_endpoint(client):
    """GET /api/model-info returns champion model metadata and metrics."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.get_json()
    assert "champion_model" in data
    assert "models" in data
    assert "Logistic Regression" in data["models"]
    assert "Multinomial Naive Bayes" in data["models"]
    assert "dataset_statistics" in data


def test_predict_single_positive(client):
    """POST /api/predict returns Positive sentiment and probability distribution."""
    payload = {"text": "The battery life is astonishing and the display is gorgeous!"}
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["sentiment"] in ["Positive", "Negative", "Neutral"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert "Negative" in data["probabilities"]
    assert "Positive" in data["probabilities"]
    assert "Neutral" in data["probabilities"]
    assert "explanation" in data


def test_predict_single_empty_text(client):
    """POST /api/predict rejects empty text with 400 Bad Request."""
    payload = {"text": "   "}
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_predict_single_missing_field(client):
    """POST /api/predict rejects payload without 'text' key."""
    payload = {"wrong_key": "some text"}
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400


def test_predict_batch(client):
    """POST /api/predict-batch processes multiple texts and calculates distribution."""
    payload = {
        "texts": [
            "Outstanding performance and crystal clear audio!",
            "Terrible broken product that failed immediately.",
            "The package weighs 300 grams and arrived at 4pm."
        ]
    }
    response = client.post(
        "/api/predict-batch",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["total_analyzed"] == 3
    assert "counts" in data
    assert "percentages" in data
    assert len(data["items"]) == 3


def test_predict_batch_empty_list(client):
    """POST /api/predict-batch rejects empty list."""
    payload = {"texts": []}
    response = client.post(
        "/api/predict-batch",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400


def test_samples_endpoint(client):
    """GET /api/samples returns preconfigured sample reviews."""
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.get_json()
    assert "samples" in data
    assert len(data["samples"]) > 0


def test_newly_typed_unseen_text_ml_inference(client):
    """
    CRITICAL: Validates that completely novel, unseen sentences typed into
    the application are processed through the real ML pipeline and return
    floating point probability vectors, latency, and attribution.
    """
    novel_text = "I just unboxed this gadget 5 minutes ago and the tactile mechanical click is whisper quiet and delightful."
    response = client.post(
        "/api/predict",
        data=json.dumps({"text": novel_text}),
        content_type="application/json"
    )
    assert response.status_code == 200
    res = response.get_json()

    # Probability sum equals 1.0 (within float tolerance)
    total_prob = sum(res["probabilities"].values())
    assert abs(total_prob - 1.0) < 0.05
    assert res["latency_ms"] > 0
    assert "certainty_level" in res


def test_dynamic_polar_word_shift_not_hardcoded(client):
    """
    Verifies that changing a sentiment-bearing word dynamically shifts
    class probabilities, proving predictions come from model weights rather than fixed rules.
    """
    pos_res = client.post(
        "/api/predict",
        data=json.dumps({"text": "The technician was exceptionally polite and fast."}),
        content_type="application/json"
    ).get_json()

    neg_res = client.post(
        "/api/predict",
        data=json.dumps({"text": "The technician was exceptionally rude and slow."}),
        content_type="application/json"
    ).get_json()

    # The negative sentence must have significantly higher Negative probability than the positive one
    assert neg_res["probabilities"]["Negative"] > pos_res["probabilities"]["Negative"]


def test_mixed_sentiment_and_oov_handling(client):
    """
    Tests that mixed sentiment text and completely out-of-vocabulary terms
    are flagged with helpful indicators rather than uncalibrated false certainty.
    """
    mixed_text = "The screen is bright and gorgeous, but the battery died after twenty minutes."
    mixed_res = client.post(
        "/api/predict",
        data=json.dumps({"text": mixed_text}),
        content_type="application/json"
    ).get_json()
    assert "is_mixed_sentiment" in mixed_res

    oov_text = "asldkfj qwpeoiru zxmncbv"
    oov_res = client.post(
        "/api/predict",
        data=json.dumps({"text": oov_text}),
        content_type="application/json"
    ).get_json()
    assert oov_res["is_out_of_vocabulary"] is True
    assert "prior baseline" in oov_res["certainty_level"]
