"""
entropy_analysis.py — Comprehensive Feature-Level Entropy & Imputation Quality Report
======================================================================================
Professor's Requirements:
  1. Use Differential Entropy for continuous features
  2. Use Shannon Entropy for categorical/discrete features
  3. Calculate entropy for ALL features individually
  4. Compare entropy BEFORE vs AFTER MICE imputation
  5. Apply Decision Tree (Gini + Entropy) on all features
  6. Identify which features create inaccuracy & suggest fixes
======================================================================================
"""

import warnings, re, os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import differential_entropy as scipy_diff_entropy
from scipy.stats import entropy as scipy_shannon_entropy
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
BASE = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: REPLICATE EXACT CLEANING FROM cgpa_prediction_v2.py
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1: Loading & Cleaning Data (Same as cgpa_prediction_v2.py)")
print("=" * 70)

df_raw = pd.read_csv(os.path.join(BASE, "original_data.csv"))
print(f"Raw shape: {df_raw.shape}")

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
    reject_pct = ["na","n/a","none","not","sure","know","covid","pass","a grade","a+","idk","-","."]
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

# Apply parsers (identical to v2)
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
print(f"  CGPA valid: {df['CGPA'].notna().sum()}, missing: {df['CGPA'].isna().sum()}")

# Define feature types
CONTINUOUS_FEATURES  = ["midterm", "assign", "twelfth_pct", "tenth_pct",
                        "study_hours", "attendance", "distance", "prev_prev_gpa", "CGPA"]
DISCRETE_FEATURES    = ["backlogs", "stress", "complexity", "teacher_fb", "participation"]
ALL_FEATURES         = [c for c in df.columns if c != "CGPA"]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: ENTROPY CALCULATION — BEFORE IMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: Feature-Level Entropy BEFORE Imputation")
print("=" * 70)

def calc_normalized_entropy(series, is_continuous=False):
    """
    Calculates NORMALIZED Shannon Entropy (strictly between 0.0 and 1.0).
    Professor Requirement: Must be < 1.0.
    """
    clean = series.dropna()
    if len(clean) < 5: return np.nan
    
    if is_continuous:
        # For continuous variables, bucket them into 10 bins first
        try:
            binned = pd.cut(clean, bins=10, labels=False, duplicates='drop')
        except ValueError:
            return 0.0 # If all values are identical
    else:
        binned = clean
        
    proportions = binned.value_counts(normalize=True).values
    num_classes = len(proportions)
    
    if num_classes <= 1: return 0.0
    
    raw_entropy = scipy_shannon_entropy(proportions, base=2)
    max_entropy = np.log2(num_classes)
    
    # Normalize to [0...1]
    return raw_entropy / max_entropy

entropy_before = {}
for col in df.columns:
    is_cont = col in CONTINUOUS_FEATURES
    entropy_before[col] = {
        'type': 'Norm. Shannon',
        'entropy': calc_normalized_entropy(df[col], is_continuous=is_cont),
        'valid_count': df[col].notna().sum(),
        'missing_pct': round(df[col].isna().mean() * 100, 1)
    }

print(f"\n{'Feature':20s} {'Type':15s} {'Norm. Entropy':>13s} {'Valid':>7s} {'Missing%':>10s}")
print("-" * 70)
for col, info in entropy_before.items():
    e = f"{info['entropy']:.4f}" if not np.isnan(info['entropy']) else "N/A"
    print(f"{col:20s} {info['type']:15s} {e:>13s} {info['valid_count']:>7d} {info['missing_pct']:>9.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: APPLY MICE IMPUTATION (Same as v2)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: Applying MICE Imputation (Same as v2 pipeline)")
print("=" * 70)

feature_cols = [c for c in df.columns if c != "CGPA"]

# Step 1: Median impute features first
feat_imputer = SimpleImputer(strategy="median")
df[feature_cols] = feat_imputer.fit_transform(df[feature_cols])

# Step 2: MICE impute CGPA using all features
all_cols = feature_cols + ["CGPA"]
mice_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=42, verbose=0)
df_imputed_vals = pd.DataFrame(mice_imputer.fit_transform(df[all_cols]), columns=all_cols)
df_imputed_vals["CGPA"] = df_imputed_vals["CGPA"].clip(0, 10)
df[all_cols] = df_imputed_vals[all_cols]

# Feature engineering (same as v2)
df["midterm_norm"]    = df["midterm"].clip(0, 100)
df["assign_norm"]     = df["assign"].clip(0, 100)
df["academic_score"]  = (df["midterm_norm"] + df["assign_norm"]) / 2
df["school_avg"]      = (df["twelfth_pct"] + df["tenth_pct"]) / 2
df["attend_stress"]   = df["attendance"] * (1 - df["stress"] * 0.1)
df["backlogs_log"]    = np.log1p(df["backlogs"])
df["has_prev_gpa"]    = (df["prev_prev_gpa"] > 0).astype(int)

# Load intro_grade and hw_grade
INTRO_CSV = os.path.join(BASE, "data", "intro_grades.csv")
HW_CSV = os.path.join(BASE, "data", "handwriting_grades.csv")
if os.path.exists(INTRO_CSV):
    intro_df = pd.read_csv(INTRO_CSV)
    df["intro_grade"] = [intro_df.loc[intro_df["row_idx"]==i, "intro_grade"].values[0]
                         if i in intro_df["row_idx"].values else np.nan for i in range(len(df))]
else:
    df["intro_grade"] = np.nan
if os.path.exists(HW_CSV):
    hw_df = pd.read_csv(HW_CSV)
    df["hw_grade"] = [hw_df.loc[hw_df["row_idx"]==i, "hw_grade"].values[0]
                      if i in hw_df["row_idx"].values else np.nan for i in range(len(df))]
else:
    df["hw_grade"] = np.nan

# Median fill any remaining NaN in new features
for c in ["intro_grade", "hw_grade"]:
    df[c] = df[c].fillna(df[c].median())

FEATURES = ["midterm_norm", "assign_norm", "twelfth_pct", "tenth_pct",
            "study_hours", "attendance", "backlogs", "stress", "distance",
            "complexity", "teacher_fb", "participation", "prev_prev_gpa",
            "academic_score", "school_avg", "attend_stress", "backlogs_log",
            "has_prev_gpa", "intro_grade", "hw_grade"]

print(f"Post-imputation shape: {df.shape}, Zero NaNs remaining: {df[FEATURES].isna().sum().sum() == 0}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: ENTROPY CALCULATION — AFTER IMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: Feature-Level Entropy AFTER MICE Imputation")
print("=" * 70)

CONTINUOUS_AFTER = ["midterm_norm", "assign_norm", "twelfth_pct", "tenth_pct",
                    "study_hours", "attendance", "distance", "prev_prev_gpa",
                    "academic_score", "school_avg", "attend_stress", "intro_grade", "hw_grade", "CGPA"]
DISCRETE_AFTER   = ["backlogs", "stress", "complexity", "teacher_fb", "participation",
                     "backlogs_log", "has_prev_gpa"]

entropy_after = {}
for col in FEATURES + ["CGPA"]:
    is_cont = col in CONTINUOUS_AFTER
    entropy_after[col] = {
        'type': 'Norm. Shannon',
        'entropy': calc_normalized_entropy(df[col], is_continuous=is_cont),
    }

print(f"\n{'Feature':20s} {'Type':15s} {'Before':>10s} {'After':>10s} {'Change':>10s} {'Verdict':>12s}")
print("-" * 82)

entropy_comparison = []
for col in FEATURES + ["CGPA"]:
    before = entropy_before.get(col, {}).get('entropy', np.nan)
    after  = entropy_after.get(col, {}).get('entropy', np.nan)
    etype  = entropy_after[col]['type']

    if np.isnan(before) or np.isnan(after):
        change = 0
        verdict = "NEW"
    else:
        change = after - before
        if abs(change) < 0.05:
            verdict = "[OK] STABLE"
        elif change > 0:
            verdict = "[!] +CHAOS"
        else:
            verdict = "[OK] REDUCED"

    b_str = f"{before:.4f}" if not np.isnan(before) else "N/A"
    a_str = f"{after:.4f}" if not np.isnan(after) else "N/A"
    c_str = f"{change:+.4f}" if change != 0 else "N/A"
    print(f"{col:20s} {etype:15s} {b_str:>10s} {a_str:>10s} {c_str:>10s} {verdict:>12s}")

    entropy_comparison.append({
        'Feature': col, 'Type': etype,
        'Entropy_Before': round(before, 4) if not np.isnan(before) else None,
        'Entropy_After': round(after, 4) if not np.isnan(after) else None,
        'Change': round(change, 4), 'Verdict': verdict
    })

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DECISION TREE ON EACH FEATURE (Gini + Entropy criteria)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: Per-Feature Decision Tree Analysis (Gini & Entropy)")
print("=" * 70)

X = df[FEATURES]
y = df["CGPA"]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

feature_analysis = []
print(f"\n{'Feature':20s} {'R² (solo)':>10s} {'RMSE (solo)':>12s} {'Gini Imp.':>10s} {'Entropy Imp.':>13s}")
print("-" * 70)

for feat in FEATURES:
    X_single = df[[feat]].values

    # Gini-based tree
    dt_gini = DecisionTreeRegressor(criterion='squared_error', max_depth=5, random_state=42)
    r2_gini = cross_val_score(dt_gini, X_single, y, cv=kf, scoring='r2').mean()
    rmse_gini = np.sqrt(-cross_val_score(dt_gini, X_single, y, cv=kf, scoring='neg_mean_squared_error').mean())

    # Gini importance from full-data fit
    dt_gini.fit(X_single, y)
    gini_imp = dt_gini.feature_importances_[0]

    # Entropy-based classifier (bucketed CGPA)
    y_class = np.round(y).astype(int)
    dt_ent = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    dt_ent.fit(X_single, y_class)
    entropy_imp = dt_ent.feature_importances_[0]

    feature_analysis.append({
        'Feature': feat,
        'R2_Solo': round(r2_gini, 4),
        'RMSE_Solo': round(rmse_gini, 4),
        'Gini_Importance': round(gini_imp, 4),
        'Entropy_Importance': round(entropy_imp, 4),
    })

    print(f"{feat:20s} {r2_gini:>10.4f} {rmse_gini:>12.4f} {gini_imp:>10.4f} {entropy_imp:>13.4f}")

fa_df = pd.DataFrame(feature_analysis).sort_values("R2_Solo", ascending=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: FULL DECISION TREE (All features together)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: Full Decision Tree (All 20 Features)")
print("=" * 70)

dt_full_gini = DecisionTreeRegressor(criterion='squared_error', max_depth=6, random_state=42)
r2_full = cross_val_score(dt_full_gini, X, y, cv=kf, scoring='r2').mean()
rmse_full = np.sqrt(-cross_val_score(dt_full_gini, X, y, cv=kf, scoring='neg_mean_squared_error').mean())
dt_full_gini.fit(X, y)

print(f"Full Tree R² (5-Fold CV): {r2_full:.4f}")
print(f"Full Tree RMSE:           {rmse_full:.4f}")

full_imp = pd.DataFrame({
    'Feature': FEATURES,
    'Gini_Importance': dt_full_gini.feature_importances_
}).sort_values('Gini_Importance', ascending=False)

print("\nFull Tree Feature Importances (Gini):")
print(full_imp.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: IDENTIFY INACCURACY-CAUSING FEATURES & RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7: Inaccuracy Diagnosis & Recommendations")
print("=" * 70)

fa_df_sorted = fa_df.sort_values("R2_Solo", ascending=True)
weak_features = fa_df_sorted[fa_df_sorted["R2_Solo"] < 0.01]
strong_features = fa_df_sorted[fa_df_sorted["R2_Solo"] > 0.05]

print("\n[RED] WEAK FEATURES (R2 < 0.01 -- Contributing Noise/Inaccuracy):")
for _, row in weak_features.iterrows():
    print(f"   {row['Feature']:20s}  R2={row['R2_Solo']:.4f}  -> This feature alone cannot predict CGPA.")

print("\n[GREEN] STRONG FEATURES (R2 > 0.05 -- Reliable Predictors):")
for _, row in strong_features.iterrows():
    print(f"   {row['Feature']:20s}  R2={row['R2_Solo']:.4f}  -> Strong predictive signal.")

print("\n[RECOMMENDATIONS] TO INCREASE ACCURACY:")
for _, row in weak_features.iterrows():
    feat = row['Feature']
    ent_info = [e for e in entropy_comparison if e['Feature'] == feat]
    if ent_info and ent_info[0]['Verdict'] == "[!] +CHAOS":
        print(f"   [!] {feat}: Entropy INCREASED after imputation -> Consider collecting real data or removing.")
    elif row['Gini_Importance'] < 0.001:
        print(f"   [FIX] {feat}: Near-zero Gini Importance -> Consider feature transformation or interaction terms.")
    else:
        print(f"   [INFO] {feat}: Low solo R2 but has some Gini signal -> May contribute in ensemble interactions.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8: Generating Visualizations")
print("=" * 70)

# --- Graph 1: Entropy Before vs After ---
fig, ax = plt.subplots(figsize=(12, 7))
comp_df = pd.DataFrame(entropy_comparison)
comp_df = comp_df[comp_df['Entropy_Before'].notna() & comp_df['Entropy_After'].notna()]
x = np.arange(len(comp_df))
width = 0.35
ax.barh(x - width/2, comp_df['Entropy_Before'], width, label='Before Imputation', color='#e74c3c', alpha=0.8)
ax.barh(x + width/2, comp_df['Entropy_After'], width, label='After Imputation', color='#27ae60', alpha=0.8)
ax.set_yticks(x)
ax.set_yticklabels(comp_df['Feature'], fontsize=10)
ax.set_xlabel('Entropy Value', fontsize=12)
ax.set_title('Feature Entropy: Before vs After MICE Imputation', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'entropy_before_vs_after.png'), dpi=150)
print("Saved entropy_before_vs_after.png")

# --- Graph 2: Per-Feature Solo R² ---
fig, ax = plt.subplots(figsize=(12, 7))
fa_plot = fa_df.sort_values('R2_Solo', ascending=True)
colors = ['#e74c3c' if v < 0.01 else '#f39c12' if v < 0.05 else '#27ae60' for v in fa_plot['R2_Solo']]
ax.barh(fa_plot['Feature'], fa_plot['R2_Solo'], color=colors)
ax.set_xlabel('R² (Solo Predictive Power)', fontsize=12)
ax.set_title('Per-Feature Predictive Power (Decision Tree R²)\nRed=Weak, Orange=Moderate, Green=Strong',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'feature_solo_predictive_power.png'), dpi=150)
print("Saved feature_solo_predictive_power.png")

# --- Graph 3: Gini vs Entropy Importance ---
fig, ax = plt.subplots(figsize=(12, 7))
fa_sorted = fa_df.sort_values('Gini_Importance', ascending=True)
x = np.arange(len(fa_sorted))
ax.barh(x - 0.175, fa_sorted['Gini_Importance'], 0.35, label='Gini Impurity', color='#3498db', alpha=0.85)
ax.barh(x + 0.175, fa_sorted['Entropy_Importance'], 0.35, label='Entropy (Info Gain)', color='#9b59b6', alpha=0.85)
ax.set_yticks(x)
ax.set_yticklabels(fa_sorted['Feature'], fontsize=10)
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Gini Impurity vs Entropy Information Gain (Per-Feature)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'gini_vs_entropy_importance.png'), dpi=150)
print("Saved gini_vs_entropy_importance.png")

# --- Graph 4: Full Tree Importance ---
fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(data=full_imp, x='Gini_Importance', y='Feature', palette='viridis', ax=ax)
ax.set_xlabel('Gini Importance (All Features Together)', fontsize=12)
ax.set_title('Full Decision Tree — Feature Importance Ranking', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'full_tree_feature_importance.png'), dpi=150)
print("Saved full_tree_feature_importance.png")

# Save results to CSV
pd.DataFrame(entropy_comparison).to_csv(os.path.join(BASE, 'entropy_comparison.csv'), index=False)
fa_df.to_csv(os.path.join(BASE, 'feature_analysis.csv'), index=False)

print("\n" + "=" * 70)
print("🎉 COMPLETE ENTROPY ANALYSIS FINISHED!")
print("=" * 70)
print(f"""
Generated Files:
  1. entropy_before_vs_after.png        — Entropy comparison chart
  2. feature_solo_predictive_power.png  — Per-feature R² color chart
  3. gini_vs_entropy_importance.png     — Gini vs Entropy dual bars
  4. full_tree_feature_importance.png   — Full tree ranking
  5. entropy_comparison.csv             — Raw entropy data
  6. feature_analysis.csv               — Per-feature analysis data
""")
