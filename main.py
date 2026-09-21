"""
SafetyAI FastAPI Backend
========================
REST API for the hierarchical incident classification system.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import json, sys, os, asyncio, time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.classifier import (
    get_classifier, MECHANISM_KEYWORDS, INJURY_KEYWORDS,
    BODY_PART_KEYWORDS, SEVERITY_KEYWORDS, PSM_KEYWORDS,
    TEMPLATES, SEVERITY_TEMPLATES, preprocess_text, rule_based_label
)

app = FastAPI(
    title="SafetyAI – Incident Classifier",
    description="Hierarchical ML classification of workplace incident narratives (Afsheen & Farook, 2026)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schemas ─────────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    narrative: str = Field(..., min_length=10, max_length=2000,
        example="Worker fell from scaffold while installing roof panels and fractured leg")
    hospitalized: Optional[int] = Field(0, ge=0, le=1)
    amputation: Optional[int] = Field(0, ge=0, le=1)
    model: Optional[str] = Field("hybrid", description="hybrid | rule | tfidf")


class BatchRequest(BaseModel):
    narratives: list[str] = Field(..., max_length=20)


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Pre-warm the classifier on startup."""
    get_classifier()
    print("✅ SafetyAI classifier ready")


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "app": "SafetyAI", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "classifier_ready": get_classifier().is_trained}


@app.post("/classify")
def classify(req: ClassifyRequest):
    """
    Classify an incident narrative using the hierarchical ML pipeline.
    Returns mechanism, injury type, body part, severity, PSM mapping, and feature evidence.
    """
    clf = get_classifier()
    try:
        result = clf.predict(req.narrative, req.hospitalized, req.amputation)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-classify")
def batch_classify(req: BatchRequest):
    """Classify multiple narratives at once (max 20)."""
    clf = get_classifier()
    results = []
    for narrative in req.narratives:
        try:
            r = clf.predict(narrative)
            results.append({"narrative": narrative[:80] + "...", "result": r, "error": None})
        except Exception as e:
            results.append({"narrative": narrative[:80] + "...", "result": None, "error": str(e)})
    return {"success": True, "count": len(results), "data": results}


@app.get("/classify/stream")
async def classify_stream(narrative: str, hospitalized: int = 0, amputation: int = 0):
    """
    SSE streaming endpoint — shows each pipeline step as it completes.
    """
    async def event_generator():
        steps = [
            ("preprocessing", 0.3, lambda n: {
                "step": "preprocessing",
                "original_tokens": len(n.split()),
                "cleaned": preprocess_text(n),
                "cleaned_tokens": len(preprocess_text(n).split()),
                "removed": len(n.split()) - len(preprocess_text(n).split()),
            }),
            ("feature_extraction", 0.4, lambda n: {
                "step": "feature_extraction",
                "method": "TF-IDF (bigrams, max_features=3000) + SVD (100 components)",
                "keyword_matches": len([kw for kws in MECHANISM_KEYWORDS.values() for kw in kws if kw in n.lower()]),
                "top_keywords": [kw for kws in MECHANISM_KEYWORDS.values() for kw in kws if kw in n.lower()][:8],
            }),
            ("model_inference", 0.5, None),
        ]

        for step_name, delay, fn in steps:
            await asyncio.sleep(delay)
            if fn:
                payload = fn(narrative)
            else:
                clf = get_classifier()
                payload = {"step": "model_inference", **clf.predict(narrative, hospitalized, amputation)}
            yield f"data: {json.dumps(payload)}\n\n"

        yield "data: {\"step\": \"complete\"}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/examples")
def get_examples():
    """Return pre-built example narratives for each mechanism category."""
    examples = []
    for mechanism, templates in TEMPLATES.items():
        examples.append({
            "mechanism": mechanism,
            "narrative": templates[0],
            "hospitalized": 1 if mechanism in ["Fire/Explosion", "Exposure"] else 0,
            "amputation": 1 if mechanism == "Caught In/Between" else 0,
        })
    for severity, templates in SEVERITY_TEMPLATES.items():
        examples.append({
            "mechanism": f"Severity Demo – {severity}",
            "narrative": templates[0],
            "hospitalized": 1 if severity in ["Major", "Catastrophic"] else 0,
            "amputation": 0,
        })
    return {"success": True, "count": len(examples), "data": examples}


@app.get("/stats")
def get_stats():
    """Return dataset statistics and model performance summary from the paper."""
    return {
        "success": True,
        "data": {
            "paper_stats": {
                "osha_records": 99601,
                "date_range": "Jan 2015 – Mar 2025",
                "best_accuracy": 0.996,
                "best_model": "Mixed-Feature Logistic Regression",
                "benchmark_accuracy": 0.8848,
                "benchmark_paper": "Kurian et al. (2020)",
            },
            "emars_stats": {
                "records": 1239,
                "industries": ["Petrochemical", "Oil & Gas", "Chemical Manufacturing", "Storage"],
                "best_accuracy": 0.973,
                "best_model": "XGBoost",
            },
            "model_comparison": [
                {"model": "Mixed-Feature Logistic Regression", "accuracy": 0.996, "f1": 0.99, "study": "OSHA"},
                {"model": "Linear SVC", "accuracy": 0.933, "f1": 0.93, "study": "OSHA"},
                {"model": "Logistic Regression (SBERT)", "accuracy": 0.890, "f1": 0.88, "study": "OSHA"},
                {"model": "Multinomial Naive Bayes", "accuracy": 0.760, "f1": 0.76, "study": "OSHA"},
                {"model": "XGBoost", "accuracy": 0.973, "f1": 0.97, "study": "EMARS"},
                {"model": "Multi-Layer Perceptron", "accuracy": 0.970, "f1": 0.97, "study": "EMARS"},
                {"model": "Random Forest", "accuracy": 0.968, "f1": 0.96, "study": "EMARS"},
                {"model": "Support Vector Machine", "accuracy": 0.941, "f1": 0.94, "study": "EMARS"},
                {"model": "Logistic Regression", "accuracy": 0.825, "f1": 0.82, "study": "EMARS"},
                {"model": "Naive Bayes", "accuracy": 0.750, "f1": 0.75, "study": "EMARS"},
            ],
            "risk_proxy": [
                {"mechanism": "Caught In/Between", "frequency": 22, "avg_severity": 4.1, "risk_proxy": 90},
                {"mechanism": "Struck-By", "frequency": 18, "avg_severity": 3.8, "risk_proxy": 68},
                {"mechanism": "Fall", "frequency": 28, "avg_severity": 3.2, "risk_proxy": 90},
                {"mechanism": "Exposure", "frequency": 10, "avg_severity": 3.5, "risk_proxy": 35},
                {"mechanism": "Fire/Explosion", "frequency": 6, "avg_severity": 4.8, "risk_proxy": 29},
                {"mechanism": "Vehicle Incident", "frequency": 8, "avg_severity": 3.6, "risk_proxy": 29},
                {"mechanism": "Overexertion", "frequency": 20, "avg_severity": 2.5, "risk_proxy": 50},
            ],
            "injury_heatmap": [
                {"mechanism": "Caught In/Between", "injury": "Amputation", "count": 85},
                {"mechanism": "Caught In/Between", "injury": "Cut/Laceration", "count": 40},
                {"mechanism": "Fall", "injury": "Fracture", "count": 70},
                {"mechanism": "Fall", "injury": "Head Injury", "count": 30},
                {"mechanism": "Struck-By", "injury": "Head Injury", "count": 55},
                {"mechanism": "Exposure", "injury": "Burn", "count": 45},
                {"mechanism": "Fire/Explosion", "injury": "Burn", "count": 60},
                {"mechanism": "Overexertion", "injury": "Pain/Strain", "count": 90},
                {"mechanism": "Vehicle Incident", "injury": "Fracture", "count": 35},
            ]
        }
    }


@app.get("/psm-elements")
def get_psm_elements():
    """Return PSM element keyword mappings for display."""
    return {
        "success": True,
        "data": [
            {"element": k, "keywords": v[:6]} for k, v in PSM_KEYWORDS.items()
        ]
    }
