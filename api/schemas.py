from pydantic import BaseModel, Field
from typing import Optional


class StudentInput(BaseModel):
    midterm: float = Field(..., ge=0, le=100, description="Midterm exam score (0–100)")
    assignment: float = Field(..., ge=0, le=100, description="Assignment score average (0–100)")
    twelfth_pct: float = Field(..., ge=0, le=100, description="12th grade percentage")
    tenth_pct: float = Field(..., ge=0, le=100, description="10th grade percentage")
    study_hours: float = Field(..., ge=0, le=24, description="Study hours per day")
    attendance: float = Field(..., ge=0, le=100, description="Attendance percentage")
    backlogs: int = Field(..., ge=0, description="Number of backlogs/failed subjects")
    stress: float = Field(..., ge=0, le=10, description="Mental stress score (0–10)")
    distance: float = Field(..., ge=0, description="Distance from campus in km")
    complexity: int = Field(..., ge=1, le=3, description="Content complexity: 1=Easy, 2=Medium, 3=Hard")
    teacher_feedback: int = Field(..., ge=1, le=3, description="Teacher feedback: 1=Poor, 2=Average, 3=Good")
    participation: int = Field(..., ge=1, le=4, description="Discussion participation: 1=Less Active, 2=Good Listener, 3=Shares Stats, 4=Moderator")
    prev_prev_gpa: Optional[float] = Field(None, ge=0, le=10, description="CGPA of the semester before last (optional)")
    intro_grade: int = Field(5, ge=1, le=10, description="Introduction quality grade (1–10) from audio analysis")
    hw_grade: int = Field(5, ge=1, le=10, description="Handwriting quality grade (1–10) from image analysis")

    class Config:
        json_schema_extra = {
            "example": {
                "midterm": 40,
                "assignment": 17,
                "twelfth_pct": 80.0,
                "tenth_pct": 85.0,
                "study_hours": 3.0,
                "attendance": 82.0,
                "backlogs": 0,
                "stress": 2,
                "distance": 15.0,
                "complexity": 2,
                "teacher_feedback": 3,
                "participation": 2,
                "prev_prev_gpa": None,
                "intro_grade": 6,
                "hw_grade": 7
            }
        }


class PredictionResponse(BaseModel):
    predicted_cgpa: float
    lower_bound: float
    upper_bound: float
    grade_band: str
    grade_description: str
    risk_level: str
    risk_color: str
    key_insights: list[str]
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    features_count: int


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    display_name: str


class FeatureImportanceResponse(BaseModel):
    features: list[FeatureImportanceItem]
