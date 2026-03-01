"""
main.py — FastAPI application for CGPA Prediction.
MLOps-style: single responsibility, dependency injection, proper error handling.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from api.schemas import (
    StudentInput,
    PredictionResponse,
    HealthResponse,
    FeatureImportanceResponse,
    FeatureImportanceItem,
)
from api.predictor import CGPAPredictor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CGPA Prediction API",
    description=(
        "Predict a student's semester GPA using an ensemble ML model "
        "trained on real college survey data. Built with FastAPI + scikit-learn."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow React dev server ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    try:
        predictor = CGPAPredictor.get()
        logger.info(f"Model loaded: {predictor.model_name} | Features: {len(predictor.features)}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "CGPA Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health():
    predictor = CGPAPredictor.get()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=predictor.model_name,
        features_count=len(predictor.features),
    )


@app.get("/feature-importance", response_model=FeatureImportanceResponse, tags=["Model"])
async def feature_importance():
    """Returns ranked feature importances from the trained model."""
    predictor = CGPAPredictor.get()
    items = [
        FeatureImportanceItem(**fi) for fi in predictor.feature_importance
    ]
    return FeatureImportanceResponse(features=items)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(student: StudentInput):
    """
    Predict CGPA for a student based on academic and lifestyle inputs.

    Returns predicted CGPA, confidence interval, grade band, risk level, and key insights.
    """
    try:
        predictor = CGPAPredictor.get()
        result = predictor.predict(student.model_dump())
        logger.info(f"Prediction made → CGPA: {result['predicted_cgpa']} | Risk: {result['risk_level']}")
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
