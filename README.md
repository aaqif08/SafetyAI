# SafetyAI — Incident Intelligence Platform

An ML-powered platform for industrial incident classification, severity assessment, and process safety insights.

SafetyAI analyzes workplace incident narratives and predicts incident mechanisms, injury types, affected body parts, and severity. It combines machine learning with rule-based evidence extraction to support incident analysis and process safety management.

> **Research context:** This project implements concepts associated with the research by Afsheen et al. (2026) and Farook et al. (2026). Reported research results are distinguished from application-level performance and should be validated against the actual implementation and evaluation data.

## Features

- **Incident classification:** Predicts incident mechanism, injury type, and affected body part.
- **Severity assessment:** Categorizes incidents into four severity levels.
- **PSM mapping:** Maps incident evidence to relevant Process Safety Management elements.
- **Explainability:** Displays keyword evidence supporting predictions.
- **Batch processing:** Classifies up to 20 incident narratives per request.
- **Pipeline visualization:** Streams processing stages through Server-Sent Events (SSE).
- **Interactive dashboard:** Includes Home, Classifier, Insights, and About pages.
- **Risk analytics:** Visualizes severity distributions, risk proxies, and model performance.
- **Deployment support:** Includes Docker configuration and a standalone frontend demo.

## Architecture

```text
safetyai/
├── frontend/
│   └── index.html
├── backend/
│   ├── main.py
│   └── requirements.txt
├── model/
│   └── classifier.py
├── Dockerfile
└── README.md
```

### Processing Pipeline

```text
Incident Narrative + Structured Metadata
                  |
                  v
          Text Preprocessing
                  |
                  v
       TF-IDF Feature Extraction
                  |
                  v
       Optional SVD Dimensionality Reduction
                  |
                  v
        Trained Classification Models
                  |
                  v
       Rule-Based Evidence Integration
                  |
                  v
  Mechanism | Injury | Body Part | Severity
                  |
                  v
       PSM Mapping and Evidence
                  |
                  v
          Dashboard / REST API
```

### Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, HTML, Recharts |
| Backend | Python, FastAPI |
| Machine learning | scikit-learn |
| Text processing | TF-IDF, SVD |
| Semantic embeddings | Sentence Transformers |
| Class balancing | imbalanced-learn (SMOTE) |
| API streaming | Server-Sent Events (SSE) |
| Deployment | Docker |

*The technologies listed above describe the intended architecture. Verify that each component is implemented and active in the current codebase.*

## Getting Started

### Prerequisites

- Python 3.10 or a compatible version supported by the project dependencies
- pip
- Docker (optional)

### Option 1: Run the Standalone Demo

Open `frontend/index.html` in a browser.

The standalone demo is intended to run without a backend. Features requiring API inference may be unavailable when the backend is disconnected.

### Option 2: Run the Full Application

**Step 1: Install dependencies**

```bash
pip install -r backend/requirements.txt
```

**Step 2: Start the FastAPI server**

Run from the project root:

```bash
uvicorn backend.main:app --reload --port 8000
```

**Step 3: Open the frontend**

Open `frontend/index.html` in your browser.

If the frontend cannot connect to the API, check the backend status, API URL, and browser CORS settings.

**Step 4: Explore the API documentation**

With the backend running, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Option 3: Run with Docker

Build the image:

```bash
docker build -t safetyai .
```

Start the container:

```bash
docker run --rm -p 8000:8000 safetyai
```

Open http://localhost:8000/docs to inspect the API.

*These Docker commands assume that the Dockerfile exposes or serves the API on port 8000. Verify that its configuration matches these commands.*

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/classify` | Classify a single incident narrative |
| `POST` | `/batch-classify` | Classify up to 20 narratives |
| `GET` | `/classify/stream` | Stream the processing pipeline |
| `GET` | `/examples` | Retrieve example incident narratives |
| `GET` | `/stats` | Retrieve model performance statistics |
| `GET` | `/psm-elements` | Retrieve PSM keyword mappings |

### Example Request

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "narrative": "Worker fell from scaffold at construction site and fractured leg",
    "hospitalized": 1,
    "amputation": 0
  }'
```

### Example Response

```json
{
  "success": true,
  "data": {
    "mechanism": {
      "predicted": "Fall",
      "confidence": 0.855,
      "top3": []
    },
    "injury_type": {
      "predicted": "Fracture",
      "confidence": 0.72,
      "top3": []
    },
    "body_part": {
      "predicted": "Leg/Foot",
      "confidence": 0.68,
      "top3": []
    },
    "severity": {
      "predicted": "Major",
      "confidence": 0.829
    },
    "psm_mapping": [],
    "keyword_evidence": [],
    "inference_time_ms": 12
  }
}
```

*Illustrative response structure only. Replace the empty arrays and example confidence values with an actual API response before presenting this as a verified output.*

## Machine Learning Methodology

| Component | Purpose |
|---|---|
| TF-IDF | Converts incident text into numerical features |
| SVD | Reduces feature dimensionality |
| Sentence-BERT | Generates semantic text representations |
| SMOTE | Addresses class imbalance during training |
| Logistic Regression | Predicts incident categories |
| Rule-based engine | Extracts keyword evidence and supports PSM mapping |

The pipeline combines statistical text representations, supervised classification, and rule-based analysis. The classifier learns patterns from labeled data, while the rule engine applies predefined mappings.

## Research Results

The following figures are reported results from the referenced studies, not independently verified benchmarks of the deployed application.

| Model | Study | Accuracy | F1-score |
|---|---|---:|---:|
| Mixed-feature Logistic Regression | OSHA | 99.6% | 0.99 |
| Linear SVC | OSHA | 93.3% | 0.93 |
| XGBoost | EMARS | 97.3% | 0.97 |
| Multi-Layer Perceptron | EMARS | 97.0% | 0.97 |
| Random Forest | EMARS | 96.8% | 0.96 |
| Prior benchmark (Kurian et al., 2020) | Previous study | 88.48% | Not reported |

The difference between 99.6% and 88.48% is **11.12 percentage points**. This comparison is meaningful only if the datasets, label definitions, evaluation procedures, and train/test conditions are comparable.

## Future Work

- Fine-tune DistilBERT or RoBERTa on labeled incident reports.
- Support CSV uploads and large-scale batch analysis.
- Integrate SHAP or LIME for instance-level model explanations.
- Develop configurable risk thresholds and monitoring alerts.
- Explore multilingual incident classification.
- Introduce authentication, access controls, and audit logging.
- Evaluate generalization on independent datasets and unseen incidents.

## References

1. Afsheen, A. et al. (2026). *Enhancing Process Safety through Machine Learning*. Sathyabama Institute.
2. Farook, A. et al. (2026). *Applying AI and ML to Analyse Major Accident Reports*. Sathyabama Institute.
3. Kurian, D. et al. (2020). “Using ML to Categorize Incident Reports.” *Journal of Loss Prevention in the Process Industries*, 64, 104069.
4. Reimers, N. and Gurevych, I. (2019). “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.” *Proceedings of EMNLP-IJCNLP 2019*.

## Disclaimer

SafetyAI is a research and analytical prototype. Its predictions are intended to support incident review, not replace qualified safety professionals, formal risk assessments, regulatory reporting, or emergency-response procedures.

## Contributors

Developed as a research-driven application for machine-learning-assisted industrial incident analysis.

