# SentimentIQ — AI-Powered Sentiment Analysis Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

> A full-stack, enterprise-grade Natural Language Processing (NLP) and Machine Learning platform that classifies text into **Positive**, **Negative**, and **Neutral** sentiment with calibrated confidence scores and token-level feature explainability.

---

## 1. Project Overview & Problem Statement

### The Problem
Customer feedback, e-commerce product reviews, and social media commentary contain critical insights for product development and brand health. However, manual sentiment triage is slow, subjective, and unscalable. Furthermore, traditional 2-class (binary) sentiment tools fail in real-world business scenarios where up to 40% of communications are purely objective, factual, or informational (neutral).

### The Solution: SentimentIQ
**SentimentIQ** provides an end-to-end Machine Learning pipeline and interactive web platform that:
- Ingests raw customer reviews, social media posts, and support tickets.
- Normalizes and tokenizes text with **sentiment-aware preprocessing** that explicitly protects negation words (e.g., `"not"`, `"never"`, `"no"`, `"hardly"`) from accidental stopword removal.
- Extracts sublinear-scaled unigram and bigram TF-IDF features ($n \in [1, 2]$) across up to 5,000 vocabulary tokens.
- Compares **Multinomial Naive Bayes** against balanced **Logistic Regression** on identical stratified test holdouts.
- Deploys the champion model (Logistic Regression, **75.76% accuracy**, **75.69% Macro F1**) behind a REST API.
- Explains every inference via token attribution (displaying the exact words that influenced the prediction).
- Enables both real-time individual text analysis and batch CSV bulk processing with exportable results.

---

## 2. Architecture & System Design

```
                                  [ User Interface ]
                       (React 19 Dashboard / Flask Web App)
                                        │
                         HTTP / REST API (JSON / Multipart)
                                        │
                                        ▼
                                 [ Flask API ]
              ┌─────────────────────────┴────────────────────────┐
              ▼                                                  ▼
     [ GET /api/model-info ]                            [ POST /api/predict ]
     [ GET /api/health     ]                         [ POST /api/predict-batch ]
              │                                                  │
              │                                                  ▼
              │                                      [ TextPreprocessor ]
              │                                 • HTML tag & entity stripping
              │                                 • URL / mention normalization
              │                                 • Contraction expansion
              │                                 • Negation word preservation
              │                                                  │
              │                                                  ▼
              │                                      [ TF-IDF Vectorizer ]
              │                                  (Unigrams + Bigrams, 5k vocab)
              │                                                  │
              │                                                  ▼
              │                                    [ Logistic Regression Clf ]
              │                                    (Calibrated Class Probs)
              │                                                  │
              └──────────────────────────────────────────────────┴───────► JSON Response
```

---

## 3. Technology Stack

| Layer | Technologies |
|---|---|
| **Programming Language** | Python 3.10+ / 3.11 |
| **Machine Learning** | scikit-learn (LogisticRegression, MultinomialNB, Pipeline) |
| **NLP & Text Vectorization** | NLTK, TF-IDF (scikit-learn `TfidfVectorizer`), Regex |
| **Data Processing** | pandas, NumPy |
| **Model Persistence** | joblib |
| **Backend Web Framework** | Flask, Flask-CORS, Gunicorn (Production WSGI) |
| **Frontend Options** | Full-Stack React 19 + Tailwind CSS + Lucide (AI Studio) & Standalone HTML5 / CSS3 / Vanilla JS |
| **Testing** | pytest, pytest-mock |

---

## 4. Dataset & Preprocessing Pipeline

### Dataset Source and License
- **Corpus**: Labeled 3-class sentiment corpus derived from benchmark multi-domain consumer reviews and customer feedback.
- **Classes**: `Positive` (109 samples), `Negative` (109 samples), `Neutral` (109 samples) — strictly balanced 1:1:1 ratio.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0) / Open Data Commons.
- **Integrity**: Missing rows, NaN fields, and duplicate texts are programmatically validated and purged prior to splitting.

### Prevention of Data Leakage
Preprocessing and TF-IDF vocabulary fitting are strictly performed on **training folds only** via a stratified 80/20 train/test split (`random_state=42`). Holdout test data is transformed exclusively using the vectorizer fitted on the training split.

### Sentiment-Aware Negation Preservation
Most standard stopword filters blindly remove words like *"not"*, *"never"*, or *"without"*, which catastrophically flips a review like *"not good"* into *"good"*. SentimentIQ's custom `TextPreprocessor` isolates negation tokens into an immutable protected set, ensuring phrases like *"not worth the money"* retain their negative polarity.

---

## 5. Model Training & Evaluation Metrics

Both models were evaluated on the exact same 66-sample stratified test holdout:

| Metric | Multinomial Naive Bayes | Logistic Regression (Champion) |
|---|---|---|
| **Test Accuracy** | 72.73% | **75.76%** |
| **Macro Precision** | 72.82% | **76.01%** |
| **Macro Recall** | 72.73% | **75.76%** |
| **Macro F1-Score** | 72.70% | **75.69%** |
| **Weighted F1-Score** | 72.70% | **75.69%** |

### Confusion Matrix (Logistic Regression)
```
Actual \ Predicted   Negative    Neutral   Positive
---------------------------------------------------
Negative                   18          1          3
Neutral                     4         17          1
Positive                    1          6         15
```

### Model Selection Rationale
Logistic Regression was designated the champion production model because:
1. It achieved a higher Macro F1-score (+3.0%) across all three classes.
2. It produces well-calibrated posterior probabilities via the softmax/multinomial logistic loss function.
3. Its L2-regularized weights directly facilitate token-level feature explainability in production.

---

## 6. Installation & Execution Guide

### Prerequisites
- Python 3.10 or higher installed. Check with: `python --version`
- Git installed.

### Windows (PowerShell / Command Prompt)

```powershell
# 1. Clone the repository
git clone https://github.com/your-username/SentimentIQ.git
cd SentimentIQ

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# In PowerShell:
.\venv\Scripts\Activate.ps1
# In Command Prompt (cmd.exe):
venv\Scripts\activate.bat

# 4. Install dependencies
pip install -r requirements.txt

# 5. Train and evaluate the models (generates .joblib artifacts)
python src/train.py

# 6. Run automated test suite
pytest tests/

# 7. Launch the application
python app.py
```
Open your web browser at `http://127.0.0.1:5001`.

### macOS / Linux (Bash)

```bash
# 1. Clone and navigate
git clone https://github.com/your-username/SentimentIQ.git
cd SentimentIQ

# 2. Create and activate venv
python3 -m venv venv
source venv/bin/activate

# 3. Install packages
pip install -r requirements.txt

# 4. Train models
python3 src/train.py

# 5. Run tests
pytest tests/

# 6. Start server
python3 app.py
```

---

## 7. REST API Documentation

### 1. Health Check
```http
GET /api/health
```
**Response (200 OK):**
```json
{
  "classes": ["Negative", "Neutral", "Positive"],
  "model_loaded": true,
  "model_path": "sentiment_model.joblib",
  "service": "SentimentIQ ML Engine",
  "status": "healthy",
  "version": "1.0.0"
}
```

### 2. Single Text Prediction
```http
POST /api/predict
Content-Type: application/json

{
  "text": "The battery life is astonishing and the screen is beautiful!"
}
```
**Response (200 OK):**
```json
{
  "character_count": 60,
  "cleaned_text": "battery life astonishing screen beautiful",
  "confidence": 0.8142,
  "explanation": "Classified as 'Positive' with high certainty (81.4%). Strongest indicator tokens identified: astonishing, beautiful, screen.",
  "influential_tokens": [
    {"polarity": "positive_signal", "token": "astonishing", "weight": 0.412},
    {"polarity": "positive_signal", "token": "beautiful", "weight": 0.389}
  ],
  "probabilities": {
    "Negative": 0.0812,
    "Neutral": 0.1046,
    "Positive": 0.8142
  },
  "sentiment": "Positive",
  "word_count": 10
}
```

### 3. Batch CSV Prediction
```http
POST /api/predict-batch
Content-Type: application/json

{
  "texts": [
    "Headphones arrived defective and crackling.",
    "Package weighs 400 grams with standard manual.",
    "Wonderful customer experience and fast delivery!"
  ]
}
```
**Response (200 OK):**
```json
{
  "total_analyzed": 3,
  "counts": {"Positive": 1, "Negative": 1, "Neutral": 1},
  "percentages": {"Positive": 33.3, "Negative": 33.3, "Neutral": 33.3},
  "items": [
    {"row_index": 1, "sentiment": "Negative", "confidence": 0.871, "text": "..."},
    {"row_index": 2, "sentiment": "Neutral", "confidence": 0.792, "text": "..."},
    {"row_index": 3, "sentiment": "Positive", "confidence": 0.845, "text": "..."}
  ]
}
```

### 4. Model Metadata & Evaluation
```http
GET /api/model-info
```

---

## 8. Deployment on Render

This project is pre-configured for one-click deployment on [Render](https://render.com).

### Step-by-Step Render Deployment:
1. Push your repository to GitHub.
2. Sign in to Render Dashboard (`https://dashboard.render.com`).
3. Click **New +** -> **Web Service**.
4. Connect your GitHub repository `SentimentIQ`.
5. Configure the following fields:
   - **Name**: `sentimentiq-api`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt && python src/train.py`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
6. Click **Deploy Web Service**.
7. Once deployment finishes, verify health at `https://your-service.onrender.com/api/health`.

---

## 9. Limitations & Ethical Considerations

- **Sarcasm & Irony**: Sarcasm (*"Oh great, another delayed flight, just what I needed"*) relies on pragmatic context that bag-of-words / TF-IDF representations cannot reliably detect.
- **Domain Adaptation**: A model trained on consumer electronics reviews may exhibit reduced precision when applied to biomedical research papers or legal contracts.
- **Language Scope**: Currently optimized for English text. Multilingual input requires language detection and translation or multilingual embeddings.

---

## 10. Future Improvements

1. **Transformer Encoders**: Fine-tune `DistilRoBERTa` or `DeBERTa-v3` for deeper contextual representation of sarcasm and nuance.
2. **Aspect-Based Sentiment Analysis (ABSA)**: Decompose reviews into specific facets (e.g. *Battery: Positive*, *Price: Negative*).
3. **Active Learning Loop**: Incorporate user feedback corrections to continually expand the training corpus.

---

## 11. License

This project is distributed under the **MIT License**. Dataset components are licensed under **CC BY 4.0**.
