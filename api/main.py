"""
main.py — FastAPI application for CGPA Prediction.
MLOps-style: single responsibility, dependency injection, proper error handling.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import logging
import tempfile
import os
import io
import re
import math
import pathlib
import numpy as np
import pandas as pd

from api.schemas import (
    StudentInput,
    PredictionResponse,
    HealthResponse,
    FeatureImportanceResponse,
    FeatureImportanceItem,
    GradeResponse,
)
from api.predictor import CGPAPredictor
from api.auth import signup_user, login_user, get_current_user
from pydantic import BaseModel, Field
from typing import Optional

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

# ── CORS — allow all origins for deployment ──────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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




# ── Auth Routes ─────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., min_length=5)
    password: str = Field(..., min_length=4)
    role: str = Field(..., pattern="^(student|teacher)$")

class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/signup", tags=["Auth"])
async def signup(req: SignupRequest):
    """Register a new student or teacher."""
    return signup_user(req.name, req.email, req.password, req.role)


@app.post("/login", tags=["Auth"])
async def login(req: LoginRequest):
    """Login and receive a JWT token."""
    return login_user(req.email, req.password)


@app.get("/me", tags=["Auth"])
async def me(user: dict = Depends(get_current_user)):
    """Get current logged-in user info."""
    return {"user": user}


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


# ── Batch Prediction for Teachers ─────────────────────────────────────────────

# Same parsers as cgpa_prediction_v2.py
REJECT_WORDS = [
    "na","n/a","none","null","not","reappear","re-appear","reaper","back",
    "fail","supply","pending","got","know","sure","declared","yet",
    "available","received","first","1st","one","unknown","no","fresher",
    "4 sem","1year","awaited","yta","result"
]

def _is_reject(s):
    return any(w in s for w in REJECT_WORDS)

def _extract_gpa(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if _is_reject(s): return np.nan
    s = re.sub(r"sgpa|cgpa|grade|/10|out of 10", "", s)
    m = re.match(r"([\.\d]+)\s*/\s*10", s)
    if m: return float(m.group(1))
    nums = re.findall(r"[\d]+\.?[\d]*", s)
    if not nums: return np.nan
    v = float(nums[0])
    return v if 0 < v <= 10 else np.nan

def _extract_score(val, lo=0, hi=100):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if _is_reject(s): return np.nan
    nums = re.findall(r"[\d]+\.?[\d]*", s)
    if not nums: return np.nan
    vals = [float(x) for x in nums if lo <= float(x) <= hi]
    return np.mean(vals) if vals else np.nan

def _extract_pct(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    reject_pct = ["na","n/a","none","not","sure","know","covid","pass","a grade","a+","idk","-","."]
    if any(r == s or r in s.split() for r in reject_pct): return np.nan
    s = re.sub(r"percent|%|℅|℃", "", s)
    nums = re.findall(r"[\d]+\.?[\d]*", s)
    if not nums: return np.nan
    v = float(nums[0])
    if v > 100: return np.nan
    if v <= 1: v *= 100
    return v if 0 <= v <= 100 else np.nan

def _extract_hours(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if any(r in s for r in ["na","fix","nothing","depends","all day"]): return np.nan
    nums = [float(x) for x in re.findall(r"[\d]+\.?[\d]*", s) if float(x) <= 24]
    return np.mean(nums) if nums else np.nan

def _extract_backlogs(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if any(x in s for x in ["no","nil","none","zero","na","null","nill","-","0 backlogs"]):
        return 0.0
    nums = re.findall(r"[\d]+", s)
    return float(nums[0]) if nums else np.nan

def _extract_dist(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if any(r in s for r in ["na","hostel","walk","accommodation"]): return np.nan
    if "meter" in s:
        nm = re.findall(r"[\d]+\.?[\d]*", s)
        return float(nm[0]) / 1000 if nm else np.nan
    nums = [float(x) for x in re.findall(r"[\d]+\.?[\d]*", s) if float(x) < 1000]
    return np.mean(nums) if nums else np.nan

def _encode_complexity(val):
    if pd.isna(val): return np.nan
    s = str(val).lower()
    if "1" in s or "easy" in s: return 1
    if "2" in s or "medium" in s: return 2
    if "3" in s or "hard" in s: return 3
    return np.nan

def _encode_teacher_fb(val):
    if pd.isna(val): return np.nan
    s = str(val).lower()
    if "good" in s and "not" not in s: return 3
    if "confident" in s or "need" in s: return 2
    return 1

def _encode_participation(val):
    if pd.isna(val): return np.nan
    s = str(val).lower()
    if "moderator" in s: return 4
    if "shares" in s or "brings" in s or "statistic" in s: return 3
    if "listener" in s: return 2
    if "less active" in s: return 1
    return 2


@app.post("/batch-predict", tags=["Teacher"])
async def batch_predict(file: UploadFile = File(...)):
    """
    Upload a raw student CSV file. The system will:
    1. Clean all columns using the same parsers as cgpa_prediction_v2.py
    2. Engineer all 20 features
    3. Use the trained model to predict missing CGPA values
    4. Return a completed CSV with all CGPAs filled
    """
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    try:
        content = await file.read()
        df_raw = pd.read_csv(io.BytesIO(content))
        logger.info(f"Batch upload: {df_raw.shape[0]} rows, {df_raw.shape[1]} cols")

        # Apply parsers
        df = pd.DataFrame()
        df["midterm"]     = df_raw["Midterm_Score_Average"].apply(_extract_score)
        df["assign"]      = df_raw["Assignment_Score_Average"].apply(_extract_score)
        df["twelfth_pct"] = df_raw["Twelfth_Grade_Percentage"].apply(_extract_pct)
        df["tenth_pct"]   = df_raw["Tenth_Grade_Percentage"].apply(_extract_pct)
        df["study_hours"] = df_raw["Study_Hours_Per_Day"].apply(_extract_hours)
        df["attendance"]  = df_raw["Attendance_Percentage"].apply(_extract_pct)
        df["backlogs"]    = df_raw["Number_of_Backlogs"].apply(_extract_backlogs)
        df["stress"]      = df_raw["Mental_Stress_Score"].astype(str).str.strip()
        df["stress"]      = df["stress"].map({"0": 0, "1": 1}).astype(float)
        df["distance"]    = df_raw["Distance_From_Campus_KM"].apply(_extract_dist)
        df["complexity"]  = df_raw.iloc[:, 14].apply(_encode_complexity)
        df["teacher_fb"]  = df_raw.iloc[:, 15].apply(_encode_teacher_fb)
        df["participation"] = df_raw.iloc[:, 16].apply(_encode_participation)
        df["prev_prev_gpa"] = df_raw["CGPA of last to last Semester"].apply(_extract_gpa)
        df["CGPA"]        = df_raw["Previous_Semester_GPA"].apply(_extract_gpa)

        # Track original CGPA status
        original_cgpa = df["CGPA"].copy()
        had_cgpa = original_cgpa.notna()

        # Median impute features
        from sklearn.impute import SimpleImputer
        feature_cols = [c for c in df.columns if c != "CGPA"]
        imputer = SimpleImputer(strategy="median")
        df[feature_cols] = imputer.fit_transform(df[feature_cols])

        # Feature engineering
        df["midterm_norm"]   = df["midterm"].clip(0, 100)
        df["assign_norm"]    = df["assign"].clip(0, 100)
        df["academic_score"] = (df["midterm_norm"] + df["assign_norm"]) / 2
        df["school_avg"]     = (df["twelfth_pct"] + df["tenth_pct"]) / 2
        df["attend_stress"]  = df["attendance"] * (1 - df["stress"] * 0.1)
        df["backlogs_log"]   = np.log1p(df["backlogs"])
        df["has_prev_gpa"]   = (df["prev_prev_gpa"] > 0).astype(int)
        df["intro_grade"]    = 5.0  # Default — no audio available in CSV
        df["hw_grade"]       = 5.0  # Default — no image available in CSV

        FEATURES = [
            "midterm_norm", "assign_norm", "twelfth_pct", "tenth_pct",
            "study_hours", "attendance", "backlogs", "stress", "distance",
            "complexity", "teacher_fb", "participation", "prev_prev_gpa",
            "academic_score", "school_avg", "attend_stress", "backlogs_log",
            "has_prev_gpa", "intro_grade", "hw_grade",
        ]

        # Predict missing CGPAs
        predictor = CGPAPredictor.get()
        missing_mask = df["CGPA"].isna()
        n_missing = missing_mask.sum()
        n_original = had_cgpa.sum()

        if n_missing > 0:
            X_missing = df.loc[missing_mask, FEATURES]
            preds = np.clip(predictor.model.predict(X_missing), 0, 10)
            df.loc[missing_mask, "CGPA"] = np.round(preds, 2)

        # Build output CSV
        output = df_raw.copy()
        output["Predicted_CGPA"] = df["CGPA"].round(2)
        output["Was_Predicted"] = (~had_cgpa).map({True: "Yes", False: "No"})

        # Convert to CSV bytes
        buf = io.StringIO()
        output.to_csv(buf, index=False)
        buf.seek(0)

        logger.info(f"Batch done: {n_original} original, {n_missing} predicted")

        return StreamingResponse(
            io.BytesIO(buf.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="predicted_{file.filename}"',
                "X-Total-Rows": str(len(df)),
                "X-Original-CGPA": str(int(n_original)),
                "X-Predicted-CGPA": str(int(n_missing)),
                "Access-Control-Expose-Headers": "X-Total-Rows, X-Original-CGPA, X-Predicted-CGPA",
            },
        )
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"CSV missing required column: {e}")
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# ── Serve React Frontend (production) ─────────────────────────────────────────
FRONTEND_DIR = pathlib.Path(__file__).parent.parent / "frontend" / "dist"

if FRONTEND_DIR.exists():
    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    # Serve other static files in dist root (favicon, pcte-logo, etc.)
    @app.get("/{full_path:path}", tags=["Frontend"])
    async def serve_frontend(full_path: str):
        """Serve the React SPA — all non-API routes go to index.html."""
        file_path = FRONTEND_DIR / full_path
        if full_path and file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIR / "index.html"))
