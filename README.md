# 🎓 CGPA Prediction — MLOps Project

> Real student survey data · FastAPI · Stacking Ensemble ML · React Dashboard

---

## 📁 Project Structure

```
cgpa_predict/
├── CGPA Project/                  # Notebook & data (untouched)
│   ├── cgpa_prediction.ipynb      # Full ML pipeline notebook
│   ├── original_data.csv          # Raw survey data (~960 students)
│   ├── best_cgpa_model.pkl        # Trained Stacking Ensemble model
│   ├── model_meta.json            # Feature list + model name
│   └── README.md                  # Detailed documentation
│
├── api/                           # FastAPI backend
│   ├── __init__.py
│   ├── main.py                    # App, CORS, routes
│   ├── schemas.py                 # Pydantic request/response models
│   └── predictor.py               # Model loading + feature engineering
│
├── frontend/                      # React (Vite) dashboard
│   └── src/
│       ├── App.jsx                # Main app layout
│       ├── index.css              # Global styles
│       └── components/
│           ├── PredictionForm.jsx # Input form
│           ├── ResultCard.jsx     # Animated CGPA result
│           └── FeatureImportance.jsx # Bar chart
│
├── backup/                        # Original backup files
├── .venv/                         # Python virtual environment
└── pyproject.toml                 # Python dependencies (uv)
```

---

## 🚀 Running Locally

### 1. Start the API (Terminal 1)
```powershell
# From project root, with venv activated
.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload --port 8000
```
API will be live at: http://127.0.0.1:8000
Swagger docs at:   http://127.0.0.1:8000/docs

### 2. Start the Frontend (Terminal 2)
```powershell
cd frontend
npm install   # first time only
npm run dev
```
Frontend will be live at: http://localhost:5173

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Model status + metadata |
| POST | `/predict` | Predict CGPA from student inputs |
| GET | `/feature-importance` | Ranked feature importances |
| GET | `/docs` | Interactive Swagger UI |

### Example Request
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "midterm": 40, "assignment": 17,
    "twelfth_pct": 80, "tenth_pct": 85,
    "study_hours": 3, "attendance": 82,
    "backlogs": 0, "stress": 2, "distance": 15,
    "complexity": 2, "teacher_feedback": 3,
    "participation": 2, "prev_prev_gpa": null
  }'
```

### Example Response
```json
{
  "predicted_cgpa": 7.48,
  "lower_bound": 6.80,
  "upper_bound": 8.16,
  "grade_band": "Very Good",
  "grade_description": "CGPA above average — good standing",
  "risk_level": "Moderate",
  "risk_color": "#f59e0b",
  "key_insights": ["..."],
  "model_name": "Stacking Ensemble"
}
```

---

## 🤖 Model Summary

| Property | Value |
|----------|-------|
| Best Model | Stacking Ensemble (RF + ET + GB + XGB + LGB + Ridge) |
| Training Data | 583 real student samples |
| Features | 18 (13 raw + 5 engineered) |
| R² Score | 0.611 |
| RMSE | 0.876 |
| ±1.0 Accuracy | 85.5% |
| CV RMSE | 0.678 ± 0.177 (10-fold) |

---

## 📦 Tech Stack

**ML Pipeline:** Python · scikit-learn · XGBoost · LightGBM · pandas · joblib

**Backend:** FastAPI · Pydantic · Uvicorn

**Frontend:** React · Vite · Recharts · Axios

**Data:** Real Google Form survey data from college students
