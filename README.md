# 🛡 SafetyAI — Incident Intelligence Platform

> **Production-ready web application** implementing the hierarchical ML incident classification research by Afsheen et al. (2026) and Farook et al. (2026).

---

## 📄 Research Paper Mapping

| Component | Paper | Implementation |
|---|---|---|
| Hierarchical 3-level labeling | Afsheen et al. (OSHA) | `model/classifier.py → rule_based_label()` |
| TF-IDF + SVD vectorization | Both papers | `SafetyClassifier._featurize()` |
| SBERT semantic embeddings | Both papers | `SentenceTransformer('all-MiniLM-L6-v2')` |
| SMOTE class balancing | Afsheen et al. | `imblearn.SMOTE` in training pipeline |
| Logistic Regression (99.6%) | Afsheen et al. | Sklearn `LogisticRegression` |
| PSM element mapping | Farook et al. | `get_psm_mapping()` keyword engine |
| 4-level severity classification | Farook et al. (EMARS) | Integrated into `classify()` endpoint |
| Risk proxy visualization | Afsheen et al. | Recharts scatter/bar in frontend |
| Feature importance display | Both papers | `keyword_evidence` in API response |

---

## 🏗 Architecture

```
safetyai/
├── frontend/
│   └── index.html          ← Self-contained React SPA (no build needed)
├── backend/
│   ├── main.py             ← FastAPI REST API + SSE streaming
│   └── requirements.txt
├── model/
│   └── classifier.py       ← ML pipeline (TF-IDF + LR + SMOTE + rule engine)
├── Dockerfile
└── README.md
```

### System Flow
```
User Narrative
    ↓
[Text Preprocessing]  lowercase → remove punctuation → stopword removal → tokenize
    ↓
[Feature Engineering] TF-IDF (bigrams, 3000 features) → SVD (100 dims) + structured metadata
    ↓
[SMOTE Training]      Synthetic minority oversampling on training set
    ↓
[Logistic Regression] Multi-class per label (mechanism / injury / body_part / severity)
    ↓
[Rule-based Hybrid]   Keyword scoring blended with model probabilities
    ↓
Output: mechanism + injury_type + body_part + severity + PSM mapping + evidence
```

---

## ⚡ Quick Start

### Option A — Open Standalone Demo (Fastest)
```bash
# No server needed — open in browser directly
open frontend/index.html
```

### Option B — Run with Backend API
```bash
# 1. Install backend
pip install -r backend/requirements.txt

# 2. Start API server
uvicorn backend.main:app --reload --port 8000

# 3. Open frontend
open frontend/index.html
# (Frontend auto-connects to localhost:8000 when available)
```

### Option C — Docker
```bash
docker build -t safetyai .
docker run -p 8000:8000 safetyai
open frontend/index.html
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/classify` | Classify a single narrative |
| POST | `/batch-classify` | Classify up to 20 narratives |
| GET | `/classify/stream` | SSE streaming pipeline trace |
| GET | `/examples` | Pre-built example narratives |
| GET | `/stats` | Paper performance statistics |
| GET | `/psm-elements` | PSM keyword mappings |

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
    "mechanism": { "predicted": "Fall", "confidence": 0.855, "top3": [...] },
    "injury_type": { "predicted": "Fracture", "confidence": 0.72, "top3": [...] },
    "body_part": { "predicted": "Leg/Foot", "confidence": 0.68, "top3": [...] },
    "severity": { "predicted": "Major", "confidence": 0.829 },
    "psm_mapping": [{ "element": "Hazard Identification", "evidence": ["risk"] }],
    "keyword_evidence": [{ "label": "Fall", "keywords": ["fell","scaffold","floor"] }],
    "preprocessing": { "original_length": 13, "cleaned_length": 8, "cleaned": "worker fell scaffold construction site fractured leg" },
    "inference_time_ms": 12,
    "pipeline": ["Text Cleaning","Stopword Removal","TF-IDF Vectorization","SVD Reduction","Logistic Regression","SMOTE Balancing"]
  }
}
```

---

## 📊 Key Results (from papers)

| Model | Study | Accuracy | F1 |
|---|---|---|---|
| **Mixed-Feature Logistic Regression** | OSHA | **99.6%** | 0.99 |
| Linear SVC | OSHA | 93.3% | 0.93 |
| XGBoost | EMARS | 97.3% | 0.97 |
| Multi-Layer Perceptron | EMARS | 97.0% | 0.97 |
| Random Forest | EMARS | 96.8% | 0.96 |
| Prior Benchmark (Kurian et al. 2020) | — | 88.48% | — |

**Improvement over benchmark: +11.12 percentage points**

---

## 🎯 App Features

- ✅ **4-page dashboard**: Home, Classifier, Insights, About
- ✅ **Real-time pipeline trace**: Watch each step execute
- ✅ **Hierarchical classification**: 3 output levels simultaneously
- ✅ **Severity prediction**: Moderate / Serious / Major / Catastrophic
- ✅ **PSM element mapping**: Links output to safety management elements
- ✅ **Keyword evidence panel**: Shows which words drove the prediction
- ✅ **7 pre-built examples**: One per incident mechanism category
- ✅ **Interactive charts**: Model performance, risk proxy, severity distribution
- ✅ **SSE streaming endpoint**: Real-time processing trace via FastAPI
- ✅ **Docker ready**: Single-file deployment
- ✅ **Zero-dependency demo**: Frontend works offline without backend

---

## 🔮 Future Improvements

1. Fine-tune DistilBERT / RoBERTa on OSHA corpus for contextual embeddings
2. Integrate real OSHA CSV upload and batch processing
3. Real-time monitoring dashboard with alert thresholds
4. Multi-language support via SBERT's multilingual models
5. SHAP/LIME integration for instance-level explainability
6. Authentication + audit log for regulatory deployment

---

## 📚 References

- Afsheen, A. et al. (2026). *Enhancing Process Safety through Machine Learning*. Sathyabama Institute.
- Farook, A. et al. (2026). *Applying AI and ML to Analyse Major Accident Reports*. Sathyabama Institute.
- Kurian, D. et al. (2020). *Using ML to Categorize Incident Reports*. J Loss Prev Process Ind, 64, 104069.
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT*. EMNLP-IJCNLP 2019.
