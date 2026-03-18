import pandas as pd
import numpy as np
import re
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
import os

print("Starting update process...")
file_path = "CGPA Project/Academic_Score_Prediction.xlsx"
df_raw = pd.read_excel(file_path)

def extract_gpa(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if any(x in s.lower() for x in ['na', 'none', 'waiting', 'not', '-', '.']): return np.nan
    nums = re.findall(r"[\d]+\.?[\d]*", s)
    if not nums: return np.nan
    v = float(nums[0])
    if v > 10 and v <= 100: v /= 10
    return v if 0 <= v <= 10 else np.nan

def extract_pct(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if 'cgpa' in s or 'sgpa' in s: return np.nan
    nums = re.findall(r"[\d]+\.?[\d]*", s)
    if not nums: return np.nan
    v = float(nums[0])
    if v <= 10: v *= 10
    return v if 0 <= v <= 100 else np.nan

def extract_score(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    reject_pct = ["na","nil","null","none","no","fix","good","average","idk","-","."]
    if any(r == s or r in s.split() for r in reject_pct): return np.nan
    s = re.sub(r"percent|%|℅|℃", "", s)
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
    if any(x in s for x in ["no","nil","none","zero","na","null","nill","-","0 backlogs"]): return 0.0
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

df = pd.DataFrame()
df["midterm"]     = df_raw.iloc[:, 7].apply(extract_score)
df["assign"]      = df_raw.iloc[:, 10].apply(extract_score)
df["twelfth_pct"] = df_raw.iloc[:, 8].apply(extract_pct)
df["tenth_pct"]   = df_raw.iloc[:, 11].apply(extract_pct)
df["study_hours"] = df_raw.iloc[:, 9].apply(extract_hours)
df["attendance"]  = df_raw.iloc[:, 12].apply(extract_pct)
df["backlogs"]    = df_raw.iloc[:, 13].apply(extract_backlogs)
df["stress"]      = df_raw.iloc[:, 14].astype(str).str.strip()
df["stress"]      = df["stress"].map({"0": 0, "1": 1}).astype(float)
df["distance"]    = df_raw.iloc[:, 15].apply(extract_dist)
df["complexity"]  = df_raw.iloc[:, 16].apply(encode_complexity)
df["teacher_fb"]  = df_raw.iloc[:, 17].apply(encode_teacher_fb)
df["participation"] = df_raw.iloc[:, 18].apply(encode_participation)
df["prev_prev_gpa"] = df_raw.iloc[:, 5].apply(extract_gpa)
df["CGPA"]        = df_raw.iloc[:, 6].apply(extract_gpa)

# Impute features with median first
feature_cols = ["midterm", "assign", "twelfth_pct", "tenth_pct", "study_hours",
                "attendance", "backlogs", "stress", "distance", "complexity",
                "teacher_fb", "participation", "prev_prev_gpa"]

feat_imputer = SimpleImputer(strategy="median")
df[feature_cols] = feat_imputer.fit_transform(df[feature_cols])

# Convert original raw CGPA col missing check
original_missing_mask = df["CGPA"].isna()
print(f"Missing CGPAs to fill: {original_missing_mask.sum()}")

# IterativeImputer (MICE)
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

# Replace Only Missing Values in the Original DataFrame
target_col_name = df_raw.columns[6]
df_raw.loc[original_missing_mask, target_col_name] = df_imputed.loc[original_missing_mask, "CGPA"].round(2)

print(f"Update complete! {original_missing_mask.sum()} missing values were filled.")
df_raw.to_excel(file_path, index=False)
print(f"Successfully saved to {file_path}")
