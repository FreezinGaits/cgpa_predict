"""
main.py — FastAPI application for CGPA Prediction.
MLOps-style: single responsibility, dependency injection, proper error handling.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging
import tempfile
import os

from api.schemas import (
    StudentInput,
    PredictionResponse,
    HealthResponse,
    FeatureImportanceResponse,
    FeatureImportanceItem,
    GradeResponse,
)
from api.predictor import CGPAPredictor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CGPA Prediction API",
    description=(
        "Predict a student's semester GPA using a multi-modal ensemble ML model "
        "trained on real college survey data, audio intros (Whisper AI), and "
        "handwritten notes (Computer Vision). Built with FastAPI + scikit-learn."
    ),
    version="2.0.0",
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
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "grade_intro": "/grade-intro",
        "grade_handwriting": "/grade-handwriting",
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
    Predict CGPA for a student based on academic, lifestyle, and multi-modal inputs.

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


@app.post("/grade-intro", response_model=GradeResponse, tags=["Grading"])
async def grade_intro(file: UploadFile = File(...)):
    """
    Upload an audio file (MP3/WAV) of a student's self-introduction.
    Whisper AI will transcribe it and grade the quality (1–10).
    """
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.webm', '.ogg')):
        raise HTTPException(status_code=400, detail="Please upload an audio file (.mp3, .wav, .m4a)")

    tmp_path = None
    try:
        # Save uploaded file to temp
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Lazy-load whisper
        import whisper
        import re

        if not hasattr(app.state, "whisper_model"):
            logger.info("Loading Whisper model (base)...")
            app.state.whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded")

        model = app.state.whisper_model
        result = model.transcribe(tmp_path, language="en")
        transcript = result["text"].strip()

        # Grade the transcript (same logic as grade_introductions.py)
        if not transcript or len(transcript.strip()) < 10:
            return GradeResponse(grade=1, details={"transcript": transcript, "word_count": 0, "reason": "Too short or empty"})

        words = transcript.split()
        word_count = len(words)
        sentences = [s.strip() for s in re.split(r'[.!?]+', transcript) if len(s.strip()) > 3]
        sentence_count = max(len(sentences), 1)
        unique_words = len(set(w.lower() for w in words))
        vocab_richness = unique_words / max(word_count, 1)

        score = 0.0
        if word_count >= 80: score += 3.0
        elif word_count >= 50: score += 2.0
        elif word_count >= 25: score += 1.0

        if sentence_count >= 5: score += 2.0
        elif sentence_count >= 3: score += 1.5
        elif sentence_count >= 2: score += 1.0

        if vocab_richness >= 0.7: score += 2.0
        elif vocab_richness >= 0.5: score += 1.5
        elif vocab_richness >= 0.3: score += 1.0

        text_lower = transcript.lower()
        content_keywords = [
            "study", "learn", "university", "college", "semester", "engineering",
            "computer", "science", "goal", "interest", "project", "skill",
            "experience", "future", "career", "passion", "hobby", "technology",
            "develop", "coding", "program", "degree", "education", "work"
        ]
        keyword_hits = sum(1 for kw in content_keywords if kw in text_lower)
        if keyword_hits >= 5: score += 2.0
        elif keyword_hits >= 3: score += 1.5
        elif keyword_hits >= 1: score += 1.0

        score += 1.0
        final_grade = int(min(10, max(1, round(score))))

        logger.info(f"Intro graded: words={word_count}, grade={final_grade}/10")
        return GradeResponse(
            grade=final_grade,
            details={
                "transcript": transcript,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "vocab_richness": round(vocab_richness, 3),
                "keyword_hits": keyword_hits,
            }
        )
    except Exception as e:
        logger.error(f"Intro grading error: {e}")
        raise HTTPException(status_code=500, detail=f"Audio grading failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/grade-handwriting", response_model=GradeResponse, tags=["Grading"])
async def grade_handwriting(file: UploadFile = File(...)):
    """
    Upload an image (JPG/PNG) of handwritten notes.
    Computer Vision will analyze neatness, density, and structure (1–10).
    """
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        raise HTTPException(status_code=400, detail="Please upload an image file (.jpg, .png)")

    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        from PIL import Image, ImageStat, ImageFilter
        import numpy as np

        img = Image.open(tmp_path).convert("L")
        pixels = np.array(img)
        stat = ImageStat.Stat(img)

        content_density = float(np.mean(pixels < 128))
        contrast = float(stat.stddev[0])
        edges = np.array(img.filter(ImageFilter.FIND_EDGES))
        edge_density = float(np.mean(edges > 30))

        n_strips = 10
        h = pixels.shape[0]
        strip_h = h // n_strips
        strip_densities = []
        for i in range(n_strips):
            strip = pixels[i * strip_h : (i + 1) * strip_h, :]
            strip_densities.append(np.mean(strip < 128))
        non_empty_strips = sum(1 for d in strip_densities if d > 0.02)
        line_regularity = non_empty_strips / n_strips

        score = 0.0
        if content_density >= 0.25: score += 3.0
        elif content_density >= 0.15: score += 2.0
        elif content_density >= 0.08: score += 1.0

        if contrast >= 50: score += 2.0
        elif contrast >= 35: score += 1.5
        elif contrast >= 20: score += 1.0

        if edge_density >= 0.20: score += 2.0
        elif edge_density >= 0.12: score += 1.5
        elif edge_density >= 0.05: score += 1.0

        if line_regularity >= 0.8: score += 2.0
        elif line_regularity >= 0.5: score += 1.5
        elif line_regularity >= 0.3: score += 1.0

        score += 1.0
        final_grade = int(min(10, max(1, round(score))))

        logger.info(f"Handwriting graded: density={content_density:.3f}, grade={final_grade}/10")
        return GradeResponse(
            grade=final_grade,
            details={
                "content_density": round(content_density, 4),
                "contrast": round(contrast, 1),
                "edge_density": round(edge_density, 4),
                "line_regularity": round(line_regularity, 2),
            }
        )
    except Exception as e:
        logger.error(f"Handwriting grading error: {e}")
        raise HTTPException(status_code=500, detail=f"Image grading failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
