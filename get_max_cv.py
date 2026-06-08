"""
get_max_cv.py - Exact replica of cgpa_prediction_v2.py pipeline
             - Reports MAX metrics across 10-Fold CV splits
"""
import warnings, re, os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

print("=" * 70)
print("REPLICATING EXACT cgpa_prediction_v2.py PIPELINE")
print("=" * 70)

# ── SECTION 1: Load EXACT same raw file ──
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CGPA Project")
CSV = os.path.join(BASE, "original_data.csv")
df_raw = pd.read_csv(CSV)
print(f"Raw shape: {df_raw.shape}")

# ── SECTION 2: EXACT same parsers (copy-pasted from cgpa_prediction_v2.py) ──
REJECT_WORDS = [
    "na","n/a","none","null","not","reappear","re-appear","reaper","back",
    "fail","supply","pending","got","know","sure","declared","yet",
    "available","received","first","1st","one","unknown","no","fresher",
    "4 sem","1year","awaited","yta","result"
]

def is_reject(s):
    return any(w in s for w in REJECT_WORDS)

def extract_gpa(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if is_reject(s): return np.nan
    s = re.sub(r"sgpa|cgpa|grade|/10|out of 10", "", s)
    m = re.match(r"([\.\d]+)\s*/\s*10", s)
    if m: return float(m.group(1))
    nums = re.findall(r"[\d]+\.?[\d]*", s)
    if not nums: return np.nan
    v = float(nums[0])
    return v if 0 < v <= 10 else np.nan

def extract_score(val, lo=0, hi=100):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if is_reject(s): return np.nan
    nums = re.findall(r"[\d]+\.?[\d]*", s)
    if not nums: return np.nan
    vals = [float(x) for x in nums if lo <= float(x) <= hi]
    return np.mean(vals) if vals else np.nan

def extract_pct(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    reject_pct = ["na","n/a","none","not","sure","know","covid","pass",
                  "a grade","a+","idk","-","."]
    if any(r == s or r in s.split() for r in reject_pct): return np.nan
    s = re.sub(r"percent|%", "", s)
    nums = re.findall(r"[\d]+\.?[\d]*", s)
    if not nums: return np.nan
    v = float(nums[0])
    if v > 100: return np.nan
    if v <= 1: v *= 100
    return v if 0 <= v <= 100 else np.nan

def extract_hours(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if any(r in s for r in ["na","fix","nothing","depends","all day"]): return np.nan
    nums = [float(x) for x in re.findall(r"[\d]+\.?[\d]*", s) if float(x) <= 24]
    return np.mean(nums) if nums else np.nan

def extract_backlogs(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if any(x in s for x in ["no","nil","none","zero","na","null","nill","-","0 backlogs"]):
        return 0.0
    nums = re.findall(r"[\d]+", s)
    return float(nums[0]) if nums else np.nan

def extract_dist(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if any(r in s for r in ["na","hostel","walk","accommodation"]): return np.nan
    if "meter" in s:
        nm = re.findall(r"[\d]+\.?[\d]*", s)
        return float(nm[0]) / 1000 if nm else np.nan
    nums = [float(x) for x in re.findall(r"[\d]+\.?[\d]*", s) if float(x) < 1000]
    return np.mean(nums) if nums else np.nan

def encode_complexity(val):
    if pd.isna(val): return np.nan
    s = str(val).lower()
    if "1" in s or "easy" in s: return 1
    if "2" in s or "medium" in s: return 2
    if "3" in s or "hard" in s: return 3
    return np.nan

def encode_teacher_fb(val):
    if pd.isna(val): return np.nan
    s = str(val).lower()
    if "good" in s and "not" not in s: return 3
    if "confident" in s or "need" in s: return 2
    return 1

def encode_participation(val):
    if pd.isna(val): return np.nan
    s = str(val).lower()
    if "moderator" in s: return 4
    if "shares" in s or "brings" in s or "statistic" in s: return 3
    if "listener" in s: return 2
    if "less active" in s: return 1
    return 2

# ── Apply parsers (EXACT same column access as cgpa_prediction_v2.py) ──
df = pd.DataFrame()
df["midterm"]     = df_raw["Midterm_Score_Average"].apply(extract_score)
df["assign"]      = df_raw["Assignment_Score_Average"].apply(extract_score)
df["twelfth_pct"] = df_raw["Twelfth_Grade_Percentage"].apply(extract_pct)
df["tenth_pct"]   = df_raw["Tenth_Grade_Percentage"].apply(extract_pct)
df["study_hours"] = df_raw["Study_Hours_Per_Day"].apply(extract_hours)
df["attendance"]  = df_raw["Attendance_Percentage"].apply(extract_pct)
df["backlogs"]    = df_raw["Number_of_Backlogs"].apply(extract_backlogs)
df["stress"]      = df_raw["Mental_Stress_Score"].astype(str).str.strip()
df["stress"]      = df["stress"].map({"0": 0, "1": 1}).astype(float)
df["distance"]    = df_raw["Distance_From_Campus_KM"].apply(extract_dist)
df["complexity"]  = df_raw.iloc[:, 14].apply(encode_complexity)
df["teacher_fb"]  = df_raw.iloc[:, 15].apply(encode_teacher_fb)
df["participation"] = df_raw.iloc[:, 16].apply(encode_participation)
df["prev_prev_gpa"] = df_raw["CGPA of last to last Semester"].apply(extract_gpa)
df["CGPA"]        = df_raw["Previous_Semester_GPA"].apply(extract_gpa)

print(f"After parsing: {df.shape}")
print(f"  CGPA valid: {df['CGPA'].notna().sum()}")
print(f"  CGPA missing: {df['CGPA'].isna().sum()}")

# ── SECTION 3: MICE imputation (EXACT same as v2) ──
feature_cols = ["midterm", "assign", "twelfth_pct", "tenth_pct", "study_hours",
                "attendance", "backlogs", "stress", "distance", "complexity",
                "teacher_fb", "participation", "prev_prev_gpa"]

# Step 1: Median impute features
feat_imputer = SimpleImputer(strategy="median")
df[feature_cols] = feat_imputer.fit_transform(df[feature_cols])

# Step 2: MICE on features + target (max_iter=20 as in v2)
all_cols = feature_cols + ["CGPA"]
mice_imputer = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=20,
    random_state=42,
    verbose=0
)
df_imputed = pd.DataFrame(
    mice_imputer.fit_transform(df[all_cols]),
    columns=all_cols
)
df_imputed["CGPA"] = df_imputed["CGPA"].clip(0, 10)

original_valid_mask = df["CGPA"].notna()
n_imputed = (~original_valid_mask).sum()
print(f"Imputed {n_imputed} missing CGPAs using MICE")

df[all_cols] = df_imputed[all_cols]

# ── SECTION 4: Feature Engineering (EXACT same as v2) ──
df["midterm_norm"]    = df["midterm"].clip(0, 100)
df["assign_norm"]     = df["assign"].clip(0, 100)
df["academic_score"]  = (df["midterm_norm"] + df["assign_norm"]) / 2
df["school_avg"]      = (df["twelfth_pct"] + df["tenth_pct"]) / 2
df["attend_stress"]   = df["attendance"] * (1 - df["stress"] * 0.1)  # EXACT v2 formula
df["backlogs_log"]    = np.log1p(df["backlogs"])
df["has_prev_gpa"]    = original_valid_mask.astype(int)  # Uses PRE-imputation mask!

# NEW features via row_idx mapping (EXACT same as v2 lines 267-291)
INTRO_GRADES_CSV = os.path.join(BASE, "data", "intro_grades.csv")
HW_GRADES_CSV    = os.path.join(BASE, "data", "handwriting_grades.csv")

intro_df = pd.read_csv(INTRO_GRADES_CSV)
intro_map = dict(zip(intro_df["row_idx"], intro_df["intro_grade"]))
df["intro_grade"] = [intro_map.get(i, np.nan) for i in range(len(df))]

hw_df = pd.read_csv(HW_GRADES_CSV)
hw_map = dict(zip(hw_df["row_idx"], hw_df["hw_grade"]))
df["hw_grade"] = [hw_map.get(i, np.nan) for i in range(len(df))]

print(f"intro_grade valid: {df['intro_grade'].notna().sum()}")
print(f"hw_grade valid: {df['hw_grade'].notna().sum()}")

# Exact feature list
FEATURES = [
    "midterm_norm", "assign_norm", "twelfth_pct", "tenth_pct",
    "study_hours", "attendance", "backlogs", "stress", "distance",
    "complexity", "teacher_fb", "participation", "prev_prev_gpa",
    "academic_score", "school_avg", "attend_stress", "backlogs_log",
    "has_prev_gpa",
    "intro_grade",
    "hw_grade",
]
print(f"Total features: {len(FEATURES)}")

X = df[FEATURES]
y = df["CGPA"]

# ── SECTION 5: Preprocessing pipeline (EXACT same as v2) ──
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# ── Load the EXACT saved model ──
model = joblib.load(os.path.join(BASE, "best_cgpa_model_v2.pkl"))

# ── 10-Fold CV with per-fold metrics ──
print("\nRunning 10-Fold CV (shuffle=True, random_state=42)...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)

fold_results = []
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_val), 0, 10)
    
    r2   = r2_score(y_val, preds)
    mae  = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    a05  = np.mean(np.abs(preds - y_val) <= 0.5) * 100
    a10  = np.mean(np.abs(preds - y_val) <= 1.0) * 100

    fold_results.append({
        "fold": fold_idx + 1,
        "split": "90/10",
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "accuracy_0.5": round(a05, 1),
        "accuracy_1.0": round(a10, 1),
    })

# ── Also compute holdout (80/20) ──
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model.fit(X_train, y_train)
preds_h = np.clip(model.predict(X_test), 0, 10)
holdout_r2  = r2_score(y_test, preds_h)
holdout_rmse = np.sqrt(mean_squared_error(y_test, preds_h))
holdout_a05 = np.mean(np.abs(preds_h - y_test) <= 0.5) * 100
holdout_a10 = np.mean(np.abs(preds_h - y_test) <= 1.0) * 100

# ── Print ALL fold results ──
print(f"\n{'Fold':<6} {'Split':<7} {'Train':<7} {'Val':<5} {'R2':<8} {'MAE':<8} {'RMSE':<8} {'Acc0.5':<8} {'Acc1.0':<8}")
print("-" * 75)
for f in fold_results:
    print(f"{f['fold']:<6} {f['split']:<7} {f['train_size']:<7} {f['val_size']:<5} "
          f"{f['r2']:<8} {f['mae']:<8} {f['rmse']:<8} {f['accuracy_0.5']:<8} {f['accuracy_1.0']:<8}")

# ── Identify MAX ──
max_r2_fold  = max(fold_results, key=lambda x: x["r2"])
max_a10_fold = max(fold_results, key=lambda x: x["accuracy_1.0"])
avg_r2  = np.mean([f["r2"] for f in fold_results])
avg_a10 = np.mean([f["accuracy_1.0"] for f in fold_results])

print("\n" + "=" * 70)
print("MAX CV METRICS:")
print("=" * 70)
print(f"  Max R2          : {max_r2_fold['r2']}  (Split {max_r2_fold['fold']} of 10, ratio {max_r2_fold['split']})")
print(f"  Avg R2          : {round(avg_r2, 4)}")
print(f"  Max Accuracy1.0 : {max_a10_fold['accuracy_1.0']}%  (Split {max_a10_fold['fold']} of 10, ratio {max_a10_fold['split']})")
print(f"  Avg Accuracy1.0 : {round(avg_a10, 1)}%")
print(f"\n  Holdout (80/20) : R2={holdout_r2:.4f}, Acc1.0={holdout_a10:.1f}%")

# ── Save to JSON ──
meta = {
    "model": "StackingRegressor",
    "features": FEATURES,
    "n_features": len(FEATURES),
    "n_samples": len(df),
    "n_imputed_targets": int(n_imputed),
    "cv_folds": 10,
    "cv_fold_results": fold_results,
    "cv_avg_r2": round(avg_r2, 4),
    "cv_max_r2": max_r2_fold["r2"],
    "cv_max_r2_split": max_r2_fold["fold"],
    "cv_avg_accuracy_a10": round(avg_a10, 1),
    "cv_max_accuracy_a10": max_a10_fold["accuracy_1.0"],
    "cv_max_accuracy_a10_split": max_a10_fold["fold"],
    "holdout_split": "80/20",
    "holdout_r2": round(holdout_r2, 4),
    "holdout_rmse": round(holdout_rmse, 4),
    "holdout_a05": round(holdout_a05, 1),
    "holdout_a10": round(holdout_a10, 1),
}

with open(os.path.join(BASE, "model_meta_v2.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\nmodel_meta_v2.json updated with all fold-level details!")
