"""
cgpa_prediction_v2.py — Enhanced CGPA Prediction Pipeline (Professor's feedback)
==================================================================================
Changes from v1:
  1. NO rows dropped — missing targets imputed with IterativeImputer (MICE)
  2. 10-Fold CV used for model selection (averaged metrics)
  3. Two NEW features: intro_grade (from Whisper STT) + hw_grade (from image analysis)
  4. Full re-evaluation with new features + expanded dataset

Run:  .venv\Scripts\python.exe "CGPA Project/cgpa_prediction_v2.py"
"""

import warnings, re, json, os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 110

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, StackingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

XGB_OK = LGB_OK = CAT_OK = False
try:
    import xgboost as xgb; XGB_OK = True; print("✅ XGBoost")
except ImportError: print("⚠️  XGBoost not found")
try:
    import lightgbm as lgb; LGB_OK = True; print("✅ LightGBM")
except ImportError: print("⚠️  LightGBM not found")
try:
    from catboost import CatBoostRegressor; CAT_OK = True; print("✅ CatBoost")
except ImportError: print("⚠️  CatBoost not found")

print("\n✅ All imports done")

def main():

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 1: Data Loading
    # ═══════════════════════════════════════════════════════════════════════════════
    BASE = os.path.dirname(os.path.abspath(__file__))
    CSV = os.path.join(BASE, "original_data.csv")
    df_raw = pd.read_csv(CSV)
    print(f"\nRaw shape: {df_raw.shape}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 2: Data Cleaning (SAME parsers as v1 — proven code)
    # ═══════════════════════════════════════════════════════════════════════════════
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

    print("✅ Parsers ready")

    # ── Apply parsers ──
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

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 3: KEY CHANGE #1 — Impute Missing Targets (NO rows dropped!)
    # Professor's feedback: "You shouldn't remove any row"
    # We use IterativeImputer (MICE) which predicts missing CGPA from all features
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("IMPUTING MISSING TARGETS (IterativeImputer / MICE)")
    print("=" * 70)

    # First impute features (median), then impute target using ALL columns
    feature_cols = ["midterm", "assign", "twelfth_pct", "tenth_pct", "study_hours",
                    "attendance", "backlogs", "stress", "distance", "complexity",
                    "teacher_fb", "participation", "prev_prev_gpa"]

    # Step 1: Impute features with median first (simple)
    feat_imputer = SimpleImputer(strategy="median")
    df[feature_cols] = feat_imputer.fit_transform(df[feature_cols])

    # Step 2: Impute CGPA using IterativeImputer on ALL columns (features + target)
    # This uses BayesianRidge regression iteratively to predict missing values
    print("Running IterativeImputer (MICE) to predict missing CGPA values...")
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

    # Clip imputed CGPA to valid range
    df_imputed["CGPA"] = df_imputed["CGPA"].clip(0, 10)

    # Track which rows were imputed
    original_valid_mask = df["CGPA"].notna()
    n_imputed = (~original_valid_mask).sum()
    n_original = original_valid_mask.sum()

    df[all_cols] = df_imputed[all_cols]
    df["is_imputed"] = (~original_valid_mask).astype(int)

    print(f"✅ Target imputation complete!")
    print(f"   Original valid CGPA: {n_original}")
    print(f"   Imputed CGPA:        {n_imputed}")
    print(f"   Total rows now:      {len(df)} (ZERO rows dropped!)")
    print(f"   Imputed CGPA range:  {df.loc[~original_valid_mask, 'CGPA'].min():.2f} – "
          f"{df.loc[~original_valid_mask, 'CGPA'].max():.2f}")
    print(f"   Imputed CGPA mean:   {df.loc[~original_valid_mask, 'CGPA'].mean():.2f}")

    # Compare distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(df.loc[original_valid_mask, "CGPA"], bins=25, alpha=0.7,
                 color="steelblue", label=f"Original (n={n_original})")
    axes[0].hist(df.loc[~original_valid_mask, "CGPA"], bins=25, alpha=0.7,
                 color="coral", label=f"Imputed (n={n_imputed})")
    axes[0].set_title("CGPA Distribution: Original vs Imputed", fontweight="bold")
    axes[0].set_xlabel("CGPA"); axes[0].legend()
    axes[1].boxplot([df.loc[original_valid_mask, "CGPA"].values,
                     df.loc[~original_valid_mask, "CGPA"].values],
                    labels=["Original", "Imputed"])
    axes[1].set_title("CGPA Box Plot Comparison", fontweight="bold")
    axes[1].set_ylabel("CGPA")
    plt.tight_layout(); plt.savefig(os.path.join(BASE, "data", "imputation_comparison.png"), dpi=150)
    plt.close()

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 4: Feature Engineering (same as v1 + NEW features)
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    # Derived features from v1
    df["midterm_norm"]    = df["midterm"].clip(0, 100)
    df["assign_norm"]     = df["assign"].clip(0, 100)
    df["academic_score"]  = (df["midterm_norm"] + df["assign_norm"]) / 2
    df["school_avg"]      = (df["twelfth_pct"] + df["tenth_pct"]) / 2
    df["attend_stress"]   = df["attendance"] * (1 - df["stress"] * 0.1)
    df["backlogs_log"]    = np.log1p(df["backlogs"])
    df["has_prev_gpa"]    = df["prev_prev_gpa"].notna().astype(int)

    # ── NEW FEATURE: intro_grade ──
    INTRO_GRADES_CSV = os.path.join(BASE, "data", "intro_grades.csv")
    if os.path.exists(INTRO_GRADES_CSV):
        intro_df = pd.read_csv(INTRO_GRADES_CSV)
        intro_map = dict(zip(intro_df["row_idx"], intro_df["intro_grade"]))
        df["intro_grade"] = [intro_map.get(i, np.nan) for i in range(len(df))]
        n_intro = df["intro_grade"].notna().sum()
        print(f"✅ Loaded intro_grade for {n_intro} students (mean: {df['intro_grade'].mean():.2f})")
    else:
        df["intro_grade"] = np.nan
        print("⚠️  intro_grades.csv not found — run grade_introductions.py first")
        print("   Feature will be median-imputed for now")

    # ── NEW FEATURE: hw_grade ──
    HW_GRADES_CSV = os.path.join(BASE, "data", "handwriting_grades.csv")
    if os.path.exists(HW_GRADES_CSV):
        hw_df = pd.read_csv(HW_GRADES_CSV)
        hw_map = dict(zip(hw_df["row_idx"], hw_df["hw_grade"]))
        df["hw_grade"] = [hw_map.get(i, np.nan) for i in range(len(df))]
        n_hw = df["hw_grade"].notna().sum()
        print(f"✅ Loaded hw_grade for {n_hw} students (mean: {df['hw_grade'].mean():.2f})")
    else:
        df["hw_grade"] = np.nan
        print("⚠️  handwriting_grades.csv not found — run grade_handwriting.py first")
        print("   Feature will be median-imputed for now")

    # Features list (v1 + 2 new)
    FEATURES = [
        "midterm_norm", "assign_norm", "twelfth_pct", "tenth_pct",
        "study_hours", "attendance", "backlogs", "stress", "distance",
        "complexity", "teacher_fb", "participation", "prev_prev_gpa",
        "academic_score", "school_avg", "attend_stress", "backlogs_log",
        "has_prev_gpa",
        "intro_grade",      # NEW
        "hw_grade",         # NEW
    ]
    print(f"\nTotal features: {len(FEATURES)} (18 original + 2 new)")

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 5: KEY CHANGE #2 — 10-Fold CV for Model Selection
    # Professor's feedback: "Use 10-fold CV and average metrics to find best model"
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("10-FOLD CROSS-VALIDATION MODEL SELECTION")
    print("=" * 70)

    X = df[FEATURES]
    y = df["CGPA"]

    # Preprocessing pipeline
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    def make_pipe(estimator):
        return Pipeline([("pre", preprocessor), ("m", estimator)])


    # All models to compare
    models = [
        ("Ridge",            Ridge(alpha=1.0)),
        ("Lasso",            Lasso(alpha=0.01, max_iter=5000)),
        ("ElasticNet",       ElasticNet(alpha=0.01, max_iter=5000)),
        ("KNN",              KNeighborsRegressor(n_neighbors=5)),
        ("SVR-RBF",          SVR(kernel="rbf", C=10, gamma="scale")),
        ("RandomForest",     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ("ExtraTrees",       ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)),
    ]
    if XGB_OK:
        models.append(("XGBoost", xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                                    random_state=42, verbosity=0)))
    if LGB_OK:
        models.append(("LightGBM", lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                                      random_state=42, verbose=-1)))
    if CAT_OK:
        models.append(("CatBoost", CatBoostRegressor(iterations=200, learning_rate=0.05,
                                                      random_state=42, verbose=0)))

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    cv_results = []
    print(f"\n{'Model':25s}  {'MAE':>8s}  {'RMSE':>8s}  {'R²':>8s}")
    print("-" * 55)

    for name, est in models:
        pipe = make_pipe(est)
        try:
            mae_scores  = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
            rmse_scores = np.sqrt(-cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1))
            r2_scores   = cross_val_score(pipe, X, y, cv=kf, scoring="r2", n_jobs=-1)

            cv_results.append({
                "Model": name,
                "MAE": round(mae_scores.mean(), 4),
                "MAE_std": round(mae_scores.std(), 4),
                "RMSE": round(rmse_scores.mean(), 4),
                "RMSE_std": round(rmse_scores.std(), 4),
                "R²": round(r2_scores.mean(), 4),
                "R²_std": round(r2_scores.std(), 4),
            })
            print(f"{name:25s}  {mae_scores.mean():8.4f}  {rmse_scores.mean():8.4f}  {r2_scores.mean():8.4f}")
        except Exception as e:
            print(f"{name:25s}  FAILED: {e}")

    cv_df = pd.DataFrame(cv_results).sort_values("R²", ascending=False)
    print("\n" + "=" * 70)
    print("10-FOLD CV RESULTS (RANKED BY R²)")
    print("=" * 70)
    print(cv_df.to_string(index=False))

    # Save results
    cv_df.to_csv(os.path.join(BASE, "data", "cv_model_comparison.csv"), index=False)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, met, col in zip(axes, ["R²", "MAE", "RMSE"], ["#27ae60", "#e74c3c", "#e67e22"]):
        s = cv_df.sort_values(met, ascending=(met != "R²"))
        bars = ax.barh(s["Model"], s[met], color=col, alpha=0.8)
        # Add error bars
        std_col = f"{met}_std"
        ax.errorbar(s[met], range(len(s)), xerr=s[std_col], fmt="none", color="black", capsize=3)
        ax.set_title(f"10-Fold CV {met} (± Std)", fontweight="bold")
        ax.set_xlabel(met)
    plt.suptitle(f"Model Comparison — 10-Fold CV on {len(df)} Students", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "data", "cv_comparison.png"), dpi=150)
    plt.close()

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 6: Hyperparameter Tuning (on full dataset with 10-fold CV)
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)

    # Tune Gradient Boosting
    print("Tuning Gradient Boosting (60 iters × 10-fold CV)...")
    gb_params = {
        "m__n_estimators":     [100, 200, 300, 500],
        "m__learning_rate":    [0.01, 0.05, 0.08, 0.1, 0.15],
        "m__max_depth":        [3, 4, 5, 6],
        "m__subsample":        [0.7, 0.8, 0.9, 1.0],
        "m__min_samples_split":[2, 5, 10],
        "m__max_features":     ["sqrt", "log2", None],
        "m__min_samples_leaf": [1, 2, 4],
    }
    gb_search = RandomizedSearchCV(
        make_pipe(GradientBoostingRegressor(random_state=42)),
        gb_params, n_iter=60, cv=10,
        scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=1
    )
    gb_search.fit(X, y)
    print(f"\n✅ Best GB CV RMSE: {-gb_search.best_score_:.4f}")

    # Tune XGBoost
    if XGB_OK:
        print("\nTuning XGBoost...")
        xgb_params = {
            "m__n_estimators":     [100, 200, 300, 500],
            "m__learning_rate":    [0.01, 0.05, 0.1],
            "m__max_depth":        [3, 4, 5, 6, 7],
            "m__subsample":        [0.7, 0.8, 0.9],
            "m__colsample_bytree": [0.7, 0.8, 1.0],
            "m__reg_alpha":        [0, 0.1, 0.5, 1.0],
            "m__reg_lambda":       [1, 1.5, 2, 3],
            "m__min_child_weight": [1, 3, 5],
        }
        xgb_search = RandomizedSearchCV(
            make_pipe(xgb.XGBRegressor(random_state=42, verbosity=0)),
            xgb_params, n_iter=60, cv=10,
            scoring="neg_root_mean_squared_error",
            random_state=42, n_jobs=-1, verbose=1
        )
        xgb_search.fit(X, y)
        print(f"✅ Best XGB CV RMSE: {-xgb_search.best_score_:.4f}")

    if LGB_OK:
        print("\nTuning LightGBM...")
        lgb_params = {
            "m__n_estimators":     [100, 200, 300, 500],
            "m__learning_rate":    [0.01, 0.05, 0.1],
            "m__num_leaves":       [20, 31, 50, 70],
            "m__max_depth":        [-1, 5, 7, 10],
            "m__subsample":        [0.7, 0.8, 0.9],
            "m__colsample_bytree": [0.7, 0.8, 1.0],
            "m__reg_alpha":        [0, 0.1, 0.5],
            "m__min_child_samples":[10, 20, 30],
        }
        lgb_search = RandomizedSearchCV(
            make_pipe(lgb.LGBMRegressor(random_state=42, verbose=-1)),
            lgb_params, n_iter=60, cv=10,
            scoring="neg_root_mean_squared_error",
            random_state=42, n_jobs=-1, verbose=1
        )
        lgb_search.fit(X, y)
        print(f"✅ Best LGB CV RMSE: {-lgb_search.best_score_:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 7: Stacking Ensemble + Final Evaluation
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("BUILDING STACKING ENSEMBLE")
    print("=" * 70)

    # Extract tuned params
    gb_params_inner = {k.replace("m__", ""): v for k, v in gb_search.best_params_.items()}

    stack_estimators = [
        ("rf",    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ("et",    ExtraTreesRegressor(n_estimators=200, random_state=1, n_jobs=-1)),
        ("gb",    GradientBoostingRegressor(**gb_params_inner, random_state=42)),
        ("ridge", Ridge(alpha=1.0)),
        ("knn",   KNeighborsRegressor(n_neighbors=7)),
    ]
    if XGB_OK:
        xgb_params_inner = {k.replace("m__", ""): v for k, v in xgb_search.best_params_.items()}
        stack_estimators.append(("xgb", xgb.XGBRegressor(**xgb_params_inner, random_state=42, verbosity=0)))
    if LGB_OK:
        lgb_params_inner = {k.replace("m__", ""): v for k, v in lgb_search.best_params_.items()}
        stack_estimators.append(("lgb", lgb.LGBMRegressor(**lgb_params_inner, random_state=42, verbose=-1)))

    stack_pipe = Pipeline([
        ("pre", preprocessor),
        ("stack", StackingRegressor(
            estimators=stack_estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5, passthrough=False, n_jobs=-1
        ))
    ])

    # 10-Fold CV on Stacking Ensemble
    print("Running 10-Fold CV on Stacking Ensemble...")
    stack_mae  = -cross_val_score(stack_pipe, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
    stack_rmse = np.sqrt(-cross_val_score(stack_pipe, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1))
    stack_r2   = cross_val_score(stack_pipe, X, y, cv=kf, scoring="r2", n_jobs=-1)

    print(f"\n✅ Stacking Ensemble — 10-Fold CV Results:")
    print(f"   MAE:  {stack_mae.mean():.4f} ± {stack_mae.std():.4f}")
    print(f"   RMSE: {stack_rmse.mean():.4f} ± {stack_rmse.std():.4f}")
    print(f"   R²:   {stack_r2.mean():.4f} ± {stack_r2.std():.4f}")

    # Plot per-fold performance
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, scores, met, col in zip(axes,
        [stack_r2, stack_mae, stack_rmse],
        ["R²", "MAE", "RMSE"],
        ["#27ae60", "#e74c3c", "#e67e22"]):
        ax.bar(range(1, 11), scores, color=col, alpha=0.8)
        ax.axhline(scores.mean(), color="red", lw=2, ls="--", label=f"Mean={scores.mean():.3f}")
        ax.set_xlabel("Fold"); ax.set_ylabel(met)
        ax.set_title(f"Stacking Ensemble — {met} per Fold", fontweight="bold")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "data", "stacking_cv_folds.png"), dpi=150)
    plt.close()

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 8: Train Final Model + Holdout Evaluation
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL MODEL TRAINING + HOLDOUT EVALUATION")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    stack_pipe.fit(X_train, y_train)
    preds = np.clip(stack_pipe.predict(X_test), 0, 10)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    a05  = np.mean(np.abs(preds - y_test) <= 0.5) * 100
    a10  = np.mean(np.abs(preds - y_test) <= 1.0) * 100

    print(f"\n📊 Holdout Results (20% test set):")
    print(f"   MAE:      {mae:.4f}")
    print(f"   RMSE:     {rmse:.4f}")
    print(f"   R²:       {r2:.4f}")
    print(f"   ±0.5 acc: {a05:.1f}%")
    print(f"   ±1.0 acc: {a10:.1f}%")

    # Actual vs Predicted + Residuals
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.scatter(y_test, preds, alpha=0.5, s=30, color="steelblue", edgecolors="none")
    mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect")
    ax.set_xlabel("Actual CGPA"); ax.set_ylabel("Predicted CGPA")
    ax.set_title(f"Actual vs Predicted\nStacking Ensemble (n={len(y_test)})", fontweight="bold")
    ax.legend()

    ax2 = axes[1]
    res = y_test.values - preds
    ax2.hist(res, bins=30, color="#e74c3c", edgecolor="white", alpha=0.85)
    ax2.axvline(0, color="black", lw=1.5)
    ax2.set_xlabel("Residual (Actual − Predicted)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residuals Distribution", fontweight="bold")
    ax2.text(0.02, 0.95, f"Mean: {res.mean():.3f}\nStd: {res.std():.3f}",
             transform=ax2.transAxes, va="top")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "data", "final_evaluation.png"), dpi=150)
    plt.close()

    # Feature importance
    try:
        fi = gb_search.best_estimator_.named_steps["m"].feature_importances_
        fi_df = pd.DataFrame({"Feature": FEATURES, "Importance": fi}).sort_values("Importance", ascending=False)
        plt.figure(figsize=(10, 7))
        sns.barplot(data=fi_df, y="Feature", x="Importance", palette="Blues_r")
        plt.title("Feature Importance — Gradient Boosting (Tuned)\nIncludes intro_grade & hw_grade", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(BASE, "data", "feature_importance_v2.png"), dpi=150)
        plt.close()
        print("\nFeature Importance:")
        print(fi_df.to_string(index=False))
    except Exception as e:
        print(f"Feature importance unavailable: {e}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 9: Save Final Model
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    # Re-fit on ALL data for production
    stack_pipe.fit(X, y)
    model_path = os.path.join(BASE, "best_cgpa_model_v2.pkl")
    joblib.dump(stack_pipe, model_path, compress=3)
    print(f"✅ Saved: {model_path} ({os.path.getsize(model_path) / 1e6:.1f} MB)")

    # Save metadata
    meta = {
        "model": "StackingRegressor",
        "features": FEATURES,
        "n_features": len(FEATURES),
        "n_samples": len(df),
        "n_imputed_targets": int(n_imputed),
        "cv_folds": 10,
        "cv_r2": round(stack_r2.mean(), 4),
        "cv_rmse": round(stack_rmse.mean(), 4),
        "holdout_r2": round(r2, 4),
        "holdout_rmse": round(rmse, 4),
        "holdout_a05": round(a05, 1),
        "holdout_a10": round(a10, 1),
        "new_features": ["intro_grade", "hw_grade"],
    }
    with open(os.path.join(BASE, "model_meta_v2.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("✅ Saved model_meta_v2.json")

    print("\n" + "=" * 70)
    print("🎉 DONE — Enhanced Pipeline Complete!")
    print("=" * 70)
    print(f"""
    Summary of Improvements:
      ✅ Rows: {n_original} → {len(df)} (imputed {n_imputed} missing targets)
      ✅ Features: 18 → {len(FEATURES)} (added intro_grade + hw_grade)
      ✅ Model selection: 10-Fold CV (was single 80/20 split)
      ✅ Holdout R²: {r2:.4f}  |  ±1.0 acc: {a10:.1f}%
    """)

if __name__ == "__main__":
    main()
