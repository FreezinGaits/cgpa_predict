"""
predictor.py — Model loading, feature engineering, and prediction logic.
Mirrors the exact feature engineering done in cgpa_prediction_v2.py.
"""
import json
import math
import pathlib
import numpy as np
import pandas as pd
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "CGPA Project" / "best_cgpa_model_v2.pkl"
META_PATH  = BASE_DIR / "CGPA Project" / "model_meta_v2.json"

# RMSE from holdout evaluation — used for confidence interval
CV_RMSE = 0.5143

# ── Feature display names ─────────────────────────────────────────────────────
DISPLAY_NAMES = {
    "midterm_norm":    "Midterm Score",
    "assign_norm":     "Assignment Score",
    "twelfth_pct":     "12th Grade %",
    "tenth_pct":       "10th Grade %",
    "study_hours":     "Study Hours/Day",
    "attendance":      "Attendance %",
    "backlogs":        "No. of Backlogs",
    "stress":          "Mental Stress",
    "distance":        "Distance (km)",
    "complexity":      "Content Complexity",
    "teacher_fb":      "Teacher Feedback",
    "participation":   "Discussion Participation",
    "prev_prev_gpa":   "Historical GPA",
    "academic_score":  "Academic Performance",
    "school_avg":      "School Average",
    "backlogs_log":    "Backlog Impact (log)",
    "attend_stress":   "Attendance × Low Stress",
    "has_prev_gpa":    "Has Historical GPA",
    "intro_grade":     "Introduction Grade",
    "hw_grade":        "Handwriting Grade",
}

# ── Grade bands ───────────────────────────────────────────────────────────────
def get_grade_band(cgpa: float) -> tuple[str, str]:
    if cgpa >= 9.0:
        return "Outstanding", "CGPA in top tier — exceptional performance"
    elif cgpa >= 8.0:
        return "Excellent", "CGPA in distinction range"
    elif cgpa >= 7.0:
        return "Very Good", "CGPA above average — good standing"
    elif cgpa >= 6.0:
        return "Good", "CGPA at satisfactory level"
    elif cgpa >= 5.0:
        return "Average", "CGPA at passing level — room for improvement"
    elif cgpa >= 4.0:
        return "Below Average", "CGPA below satisfactory — needs attention"
    else:
        return "At Risk", "CGPA in critical range — immediate intervention needed"


def get_risk(cgpa: float) -> tuple[str, str]:
    if cgpa >= 7.5:
        return "Low", "#22c55e"
    elif cgpa >= 6.0:
        return "Moderate", "#f59e0b"
    else:
        return "High", "#ef4444"


class CGPAPredictor:
    _instance = None

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            self.meta = json.load(f)
        self.features = self.meta["features"]
        self.model_name = self.meta.get("model", "StackingRegressor")
        self._feature_importance = self._compute_feature_importance()

    @classmethod
    def get(cls) -> "CGPAPredictor":
        if cls._instance is None:
            cls._instance = CGPAPredictor()
        return cls._instance

    # ── Feature engineering — mirrors cgpa_prediction_v2.py exactly ──────────
    def _engineer(self, raw: dict) -> dict:
        midterm   = float(raw["midterm"])
        assign    = float(raw["assignment"])
        twelfth   = float(raw["twelfth_pct"])
        tenth     = float(raw["tenth_pct"])
        attend    = float(raw["attendance"])
        backlogs  = float(raw["backlogs"])
        stress    = float(raw["stress"])
        ppgpa     = raw.get("prev_prev_gpa")

        feats = {
            "midterm_norm":    min(midterm, 100.0),
            "assign_norm":     min(assign, 100.0),
            "twelfth_pct":     twelfth,
            "tenth_pct":       tenth,
            "study_hours":     float(raw["study_hours"]),
            "attendance":      attend,
            "backlogs":        backlogs,
            "stress":          stress,
            "distance":        float(raw["distance"]),
            "complexity":      float(raw["complexity"]),
            "teacher_fb":      float(raw["teacher_feedback"]),
            "participation":   float(raw["participation"]),
            "prev_prev_gpa":   float(ppgpa) if ppgpa is not None else np.nan,
            # Engineered
            "academic_score":  (min(midterm, 100) + min(assign, 100)) / 2,
            "school_avg":      (twelfth + tenth) / 2,
            "backlogs_log":    math.log1p(backlogs),
            "attend_stress":   attend * (10 - stress),
            "has_prev_gpa":    1.0 if ppgpa is not None else 0.0,
            # New v2 features — from audio/image grading
            "intro_grade":     float(raw.get("intro_grade", 5)),
            "hw_grade":        float(raw.get("hw_grade", 5)),
        }
        return feats

    def predict(self, raw: dict) -> dict:
        feats = self._engineer(raw)
        df = pd.DataFrame([{f: feats.get(f, np.nan) for f in self.features}])

        cgpa = float(np.clip(self.model.predict(df)[0], 0.0, 10.0))
        lower = round(max(0.0, cgpa - CV_RMSE), 2)
        upper = round(min(10.0, cgpa + CV_RMSE), 2)

        grade, grade_desc = get_grade_band(cgpa)
        risk, risk_color  = get_risk(cgpa)
        insights          = self._insights(feats, cgpa)

        return {
            "predicted_cgpa": round(cgpa, 2),
            "lower_bound":    lower,
            "upper_bound":    upper,
            "grade_band":     grade,
            "grade_description": grade_desc,
            "risk_level":     risk,
            "risk_color":     risk_color,
            "key_insights":   insights,
            "model_name":     self.model_name,
        }

    def _insights(self, feats: dict, cgpa: float) -> list[str]:
        tips = []
        if feats["backlogs"] > 0:
            tips.append(f"Active backlogs ({int(feats['backlogs'])}) are reducing predicted GPA — clearing them is top priority.")
        if feats["attendance"] < 75:
            tips.append(f"Attendance at {feats['attendance']:.0f}% is below 75% — this significantly impacts learning and GPA.")
        if feats["stress"] >= 7:
            tips.append(f"High mental stress score ({feats['stress']:.0f}/10) may be hurting academic performance.")
        if feats["academic_score"] >= 70:
            tips.append(f"Strong academic performance (midterm + assignment avg: {feats['academic_score']:.1f}/100) is the greatest positive signal.")
        if feats["study_hours"] < 2:
            tips.append(f"Low study hours ({feats['study_hours']:.1f} hrs/day) — increasing to 3+ hrs typically correlates with better GPA.")
        if feats["intro_grade"] >= 8:
            tips.append(f"Strong introduction grade ({int(feats['intro_grade'])}/10) — good communication skills correlate with higher CGPA.")
        elif feats["intro_grade"] <= 3:
            tips.append(f"Low introduction grade ({int(feats['intro_grade'])}/10) — improving communication and vocabulary can help.")
        if feats["hw_grade"] >= 8:
            tips.append(f"Excellent handwriting grade ({int(feats['hw_grade'])}/10) — organized note-taking is a positive signal.")
        elif feats["hw_grade"] <= 3:
            tips.append(f"Low handwriting grade ({int(feats['hw_grade'])}/10) — neater and more thorough notes may improve performance.")
        if feats["prev_prev_gpa"] is not np.nan and not math.isnan(feats.get("prev_prev_gpa", float("nan"))):
            ppgpa = feats["prev_prev_gpa"]
            diff = cgpa - ppgpa
            if diff > 0.3:
                tips.append(f"Improvement from historical GPA ({ppgpa:.2f}) — positive trajectory detected.")
            elif diff < -0.3:
                tips.append(f"Decline from historical GPA ({ppgpa:.2f}) — investigate contributing factors.")
        if feats["distance"] > 30:
            tips.append(f"Long commute ({feats['distance']:.0f} km) may affect attendance and study time.")
        if not tips:
            tips.append("Overall profile is balanced. Maintain consistency in attendance and academic work.")
        return tips[:5]

    def _compute_feature_importance(self) -> list[dict]:
        try:
            inner = self.model
            if hasattr(inner, "named_steps"):
                for step_name, step in inner.named_steps.items():

                    # Direct model (e.g. GradientBoosting, RF) — not stacking
                    if hasattr(step, "feature_importances_"):
                        fi = step.feature_importances_
                        if len(fi) == len(self.features):
                            return [
                                {
                                    "feature": f,
                                    "importance": round(float(fi[i]), 4),
                                    "display_name": DISPLAY_NAMES.get(f, f),
                                }
                                for i, f in enumerate(self.features)
                            ]

                    # StackingRegressor — use named_estimators_ (dict), NOT estimators_ (list)
                    # Average importances across all tree-based base models
                    if hasattr(step, "named_estimators_"):
                        all_fi = []
                        for est_name, est in step.named_estimators_.items():
                            if hasattr(est, "feature_importances_"):
                                fi = est.feature_importances_
                                if len(fi) == len(self.features):
                                    all_fi.append(fi)

                        if all_fi:
                            avg_fi = np.mean(all_fi, axis=0)
                            # Normalise so importances sum to 1
                            avg_fi = avg_fi / avg_fi.sum()
                            return [
                                {
                                    "feature": f,
                                    "importance": round(float(avg_fi[i]), 4),
                                    "display_name": DISPLAY_NAMES.get(f, f),
                                }
                                for i, f in enumerate(self.features)
                            ]
        except Exception:
            pass

        # Fallback: uniform importance
        n = len(self.features)
        return [
            {"feature": f, "importance": round(1/n, 4), "display_name": DISPLAY_NAMES.get(f, f)}
            for f in self.features
        ]

    @property
    def feature_importance(self) -> list[dict]:
        return sorted(self._feature_importance, key=lambda x: x["importance"], reverse=True)
