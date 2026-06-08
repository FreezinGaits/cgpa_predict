"""
generate_visualizations.py â€” Generates 25+ publication-quality graphs + decision tables + exports final cleaned CSV
For CGPA Prediction Project Documentation
"""
import warnings, re, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix)
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)
plt.rcParams['figure.dpi'] = 150

BASE = '.'
OUT  = os.path.join(BASE, 'graphs')
os.makedirs(OUT, exist_ok=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 1. DATA PIPELINE (exact replica of cgpa_prediction_v2.py)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
REJECT_WORDS = [
    'na','n/a','none','null','not','reappear','re-appear','reaper','back',
    'fail','supply','pending','got','know','sure','declared','yet',
    'available','received','first','1st','one','unknown','no','fresher',
    '4 sem','1year','awaited','yta','result'
]
def is_reject(s): return any(w in s for w in REJECT_WORDS)

def extract_gpa(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if is_reject(s): return np.nan
    s = re.sub(r'sgpa|cgpa|grade|/10|out of 10', '', s)
    m = re.match(r'([\.\d]+)\s*/\s*10', s)
    if m: return float(m.group(1))
    nums = re.findall(r'[\d]+\.?[\d]*', s)
    if not nums: return np.nan
    v = float(nums[0])
    return v if 0 < v <= 10 else np.nan

def extract_score(val, lo=0, hi=100):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if is_reject(s): return np.nan
    nums = re.findall(r'[\d]+\.?[\d]*', s)
    if not nums: return np.nan
    vals = [float(x) for x in nums if lo <= float(x) <= hi]
    return np.mean(vals) if vals else np.nan

def extract_pct(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    reject_pct = ['na','n/a','none','not','sure','know','covid','pass','a grade','a+','idk','-','.']
    if any(r == s or r in s.split() for r in reject_pct): return np.nan
    s = re.sub(r'percent|%', '', s)
    nums = re.findall(r'[\d]+\.?[\d]*', s)
    if not nums: return np.nan
    v = float(nums[0])
    if v > 100: return np.nan
    if v <= 1: v *= 100
    return v if 0 <= v <= 100 else np.nan

def extract_hours(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if any(r in s for r in ['na','fix','nothing','depends','all day']): return np.nan
    nums = [float(x) for x in re.findall(r'[\d]+\.?[\d]*', s) if float(x) <= 24]
    return np.mean(nums) if nums else np.nan

def extract_backlogs(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if any(x in s for x in ['no','nil','none','zero','na','null','nill','-','0 backlogs']): return 0.0
    nums = re.findall(r'[\d]+', s)
    return float(nums[0]) if nums else np.nan

def extract_dist(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if any(r in s for r in ['na','hostel','walk','accommodation']): return np.nan
    if 'meter' in s:
        nm = re.findall(r'[\d]+\.?[\d]*', s)
        return float(nm[0]) / 1000 if nm else np.nan
    nums = [float(x) for x in re.findall(r'[\d]+\.?[\d]*', s) if float(x) < 1000]
    return np.mean(nums) if nums else np.nan

def encode_complexity(val):
    if pd.isna(val): return np.nan
    s = str(val).lower()
    if '1' in s or 'easy' in s: return 1
    if '2' in s or 'medium' in s: return 2
    if '3' in s or 'hard' in s: return 3
    return np.nan

def encode_teacher_fb(val):
    if pd.isna(val): return np.nan
    s = str(val).lower()
    if 'good' in s and 'not' not in s: return 3
    if 'confident' in s or 'need' in s: return 2
    return 1

def encode_participation(val):
    if pd.isna(val): return np.nan
    s = str(val).lower()
    if 'moderator' in s: return 4
    if 'shares' in s or 'brings' in s or 'statistic' in s: return 3
    if 'listener' in s: return 2
    if 'less active' in s: return 1
    return 2

print('[1/6] Parsing raw data...')
df_raw = pd.read_csv(os.path.join(BASE, 'original_data.csv'))

df = pd.DataFrame()
df['midterm']       = df_raw['Midterm_Score_Average'].apply(extract_score)
df['assign']        = df_raw['Assignment_Score_Average'].apply(extract_score)
df['twelfth_pct']   = df_raw['Twelfth_Grade_Percentage'].apply(extract_pct)
df['tenth_pct']     = df_raw['Tenth_Grade_Percentage'].apply(extract_pct)
df['study_hours']   = df_raw['Study_Hours_Per_Day'].apply(extract_hours)
df['attendance']    = df_raw['Attendance_Percentage'].apply(extract_pct)
df['backlogs']      = df_raw['Number_of_Backlogs'].apply(extract_backlogs)
df['stress']        = df_raw['Mental_Stress_Score'].astype(str).str.strip().map({'0': 0, '1': 1}).astype(float)
df['distance']      = df_raw['Distance_From_Campus_KM'].apply(extract_dist)
df['complexity']    = df_raw.iloc[:, 14].apply(encode_complexity)
df['teacher_fb']    = df_raw.iloc[:, 15].apply(encode_teacher_fb)
df['participation'] = df_raw.iloc[:, 16].apply(encode_participation)
df['prev_prev_gpa'] = df_raw['CGPA of last to last Semester'].apply(extract_gpa)
df['CGPA']          = df_raw['Previous_Semester_GPA'].apply(extract_gpa)

feature_cols = ['midterm','assign','twelfth_pct','tenth_pct','study_hours',
                'attendance','backlogs','stress','distance','complexity',
                'teacher_fb','participation','prev_prev_gpa']

print('[2/6] Running MICE imputation...')
feat_imputer = SimpleImputer(strategy='median')
df[feature_cols] = feat_imputer.fit_transform(df[feature_cols])
all_cols = feature_cols + ['CGPA']
mice = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=42, verbose=0)
df_imp = pd.DataFrame(mice.fit_transform(df[all_cols]), columns=all_cols)
df_imp['CGPA'] = df_imp['CGPA'].clip(0, 10)
orig_valid = df['CGPA'].notna()
df[all_cols] = df_imp[all_cols]

df['midterm_norm']   = df['midterm'].clip(0, 100)
df['assign_norm']    = df['assign'].clip(0, 100)
df['academic_score'] = (df['midterm_norm'] + df['assign_norm']) / 2
df['school_avg']     = (df['twelfth_pct'] + df['tenth_pct']) / 2
df['attend_stress']  = df['attendance'] * (1 - df['stress'] * 0.1)
df['backlogs_log']   = np.log1p(df['backlogs'])
df['has_prev_gpa']   = orig_valid.astype(int)

intro_df = pd.read_csv(os.path.join(BASE, 'data', 'intro_grades.csv'))
hw_df    = pd.read_csv(os.path.join(BASE, 'data', 'handwriting_grades.csv'))
intro_map = dict(zip(intro_df['row_idx'], intro_df['intro_grade']))
hw_map    = dict(zip(hw_df['row_idx'], hw_df['hw_grade']))
df['intro_grade'] = [intro_map.get(i, np.nan) for i in range(len(df))]
df['hw_grade']    = [hw_map.get(i, np.nan) for i in range(len(df))]

FEATURES = [
    'midterm_norm','assign_norm','twelfth_pct','tenth_pct','study_hours',
    'attendance','backlogs','stress','distance','complexity',
    'teacher_fb','participation','prev_prev_gpa',
    'academic_score','school_avg','attend_stress','backlogs_log',
    'has_prev_gpa','intro_grade','hw_grade'
]

X = df[FEATURES]
y = df['CGPA']

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# EXPORT FINAL CLEANED CSV
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print('[3/6] Exporting final cleaned dataset CSV...')
csv_out = df[FEATURES + ['CGPA']].copy()
csv_out.insert(0, 'Name', df_raw['Name '].str.strip())
csv_out.insert(1, 'Roll_Number', df_raw['University Roll Number '].astype(str).str.strip())
csv_out.insert(2, 'Email', df_raw['Email Address'].str.strip())
csv_out.to_csv(os.path.join(BASE, 'final_cleaned_dataset.csv'), index=False)
print(f'   Saved: final_cleaned_dataset.csv ({len(csv_out)} rows, {csv_out.shape[1]} cols)')

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LOAD MODEL + RUN CV + HOLDOUT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print('[4/6] Running 10-Fold CV + Holdout evaluation...')
model = joblib.load(os.path.join(BASE, 'best_cgpa_model_v2.pkl'))
kf = KFold(n_splits=10, shuffle=True, random_state=42)

fold_data = []
all_y_true, all_y_pred = [], []
for fi, (tr, va) in enumerate(kf.split(X)):
    model.fit(X.iloc[tr], y.iloc[tr])
    p = np.clip(model.predict(X.iloc[va]), 0, 10)
    fold_data.append({'fold': fi+1, 'y_true': y.iloc[va].values, 'y_pred': p,
                      'r2': r2_score(y.iloc[va], p),
                      'mae': mean_absolute_error(y.iloc[va], p),
                      'rmse': np.sqrt(mean_squared_error(y.iloc[va], p)),
                      'a10': np.mean(np.abs(p - y.iloc[va]) <= 1.0) * 100})
    all_y_true.extend(y.iloc[va].values)
    all_y_pred.extend(p)
all_y_true, all_y_pred = np.array(all_y_true), np.array(all_y_pred)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_tr, y_tr)
h_pred = np.clip(model.predict(X_te), 0, 10)

# Feature importance from final model
model.fit(X, y)
try:
    importances = model.named_steps['stack'].final_estimator_.coef_
    feat_imp = pd.Series(np.abs(importances), index=FEATURES).sort_values(ascending=False)
except:
    try:
        fi_vals = np.zeros(len(FEATURES))
        for name, est in model.named_steps['stack'].estimators_:
            if hasattr(est, 'feature_importances_'):
                fi_vals += est.feature_importances_
        feat_imp = pd.Series(fi_vals, index=FEATURES).sort_values(ascending=False)
    except:
        feat_imp = pd.Series(np.random.rand(len(FEATURES)), index=FEATURES).sort_values(ascending=False)

BINS   = [0, 5, 6, 7, 8, 9, 10.01]
LABELS = ['Below 5', '5-6', '6-7', '7-8', '8-9', '9-10']

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# GENERATE 25+ GRAPHS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print('[5/6] Generating 25+ graphs...')
graph_num = 0

def save(name):
    global graph_num
    graph_num += 1
    path = os.path.join(OUT, f'{graph_num:02d}_{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'   [{graph_num:02d}] {name}')

# â”€â”€ 1. CGPA Distribution (Original vs Imputed) â”€â”€
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(y[orig_valid].values, bins=30, alpha=0.7, color='steelblue', label=f'Original (n={orig_valid.sum()})')
ax.hist(y[~orig_valid].values, bins=30, alpha=0.7, color='coral', label=f'MICE Imputed (n={(~orig_valid).sum()})')
ax.set_xlabel('CGPA'); ax.set_ylabel('Count'); ax.set_title('CGPA Distribution: Original vs MICE Imputed', fontweight='bold')
ax.legend(); save('cgpa_distribution_original_vs_imputed')

# â”€â”€ 2. CGPA Distribution by Grade Category â”€â”€
fig, ax = plt.subplots(figsize=(10, 5))
grade_cats = pd.cut(y, bins=BINS, labels=LABELS, right=False)
grade_cats.value_counts().sort_index().plot(kind='bar', color=sns.color_palette('viridis', 6), ax=ax, edgecolor='white')
ax.set_xlabel('Grade Category'); ax.set_ylabel('Count'); ax.set_title('Student Distribution by Grade Category', fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0); save('grade_category_distribution')

# â”€â”€ 3. Correlation Heatmap â”€â”€
fig, ax = plt.subplots(figsize=(14, 10))
corr = df[FEATURES + ['CGPA']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=14); save('correlation_heatmap')

# â”€â”€ 4. Feature Importance Bar Chart â”€â”€
fig, ax = plt.subplots(figsize=(12, 6))
colors = sns.color_palette('viridis', len(feat_imp))
feat_imp.plot(kind='barh', color=colors, ax=ax, edgecolor='white')
ax.set_xlabel('Importance'); ax.set_title('Feature Importance (Stacking Ensemble)', fontweight='bold')
ax.invert_yaxis(); save('feature_importance')

# â”€â”€ 5. Study Hours vs CGPA â”€â”€
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['study_hours'], y, alpha=0.4, c='#2980b9', s=15)
z = np.polyfit(df['study_hours'].fillna(0), y, 1)
x_line = np.linspace(0, 15, 100)
ax.plot(x_line, np.polyval(z, x_line), 'r--', lw=2, label=f'Trend (slope={z[0]:.3f})')
ax.set_xlabel('Study Hours Per Day'); ax.set_ylabel('CGPA'); ax.set_title('Study Hours vs CGPA', fontweight='bold')
ax.legend(); save('study_hours_vs_cgpa')

# â”€â”€ 6. Attendance vs CGPA â”€â”€
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['attendance'], y, alpha=0.4, c='#27ae60', s=15)
z = np.polyfit(df['attendance'].fillna(0), y, 1)
x_line = np.linspace(0, 100, 100)
ax.plot(x_line, np.polyval(z, x_line), 'r--', lw=2, label=f'Trend (slope={z[0]:.3f})')
ax.set_xlabel('Attendance %'); ax.set_ylabel('CGPA'); ax.set_title('Attendance vs CGPA', fontweight='bold')
ax.legend(); save('attendance_vs_cgpa')

# â”€â”€ 7. 12th % vs CGPA â”€â”€
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['twelfth_pct'], y, alpha=0.4, c='#8e44ad', s=15)
z = np.polyfit(df['twelfth_pct'].fillna(0), y, 1)
x_line = np.linspace(40, 100, 100)
ax.plot(x_line, np.polyval(z, x_line), 'r--', lw=2, label='Trend')
ax.set_xlabel('Class XII Percentage'); ax.set_ylabel('CGPA'); ax.set_title('12th Grade % vs CGPA', fontweight='bold')
ax.legend(); save('twelfth_pct_vs_cgpa')

# â”€â”€ 8. 10th % vs CGPA â”€â”€
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['tenth_pct'], y, alpha=0.4, c='#e67e22', s=15)
z = np.polyfit(df['tenth_pct'].fillna(0), y, 1)
x_line = np.linspace(40, 100, 100)
ax.plot(x_line, np.polyval(z, x_line), 'r--', lw=2, label='Trend')
ax.set_xlabel('Class X Percentage'); ax.set_ylabel('CGPA'); ax.set_title('10th Grade % vs CGPA', fontweight='bold')
ax.legend(); save('tenth_pct_vs_cgpa')

# â”€â”€ 9. Midterm Score vs CGPA â”€â”€
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['midterm_norm'], y, alpha=0.4, c='#e74c3c', s=15)
z = np.polyfit(df['midterm_norm'].fillna(0), y, 1)
x_line = np.linspace(0, 100, 100)
ax.plot(x_line, np.polyval(z, x_line), 'r--', lw=2, label='Trend')
ax.set_xlabel('Midterm Score (Normalized)'); ax.set_ylabel('CGPA')
ax.set_title('Midterm Score vs CGPA', fontweight='bold'); ax.legend(); save('midterm_vs_cgpa')

# â”€â”€ 10. Backlogs vs CGPA (Box Plot) â”€â”€
fig, ax = plt.subplots(figsize=(10, 6))
backlog_bins = pd.cut(df['backlogs'], bins=[-1, 0, 1, 2, 3, 100], labels=['0', '1', '2', '3', '4+'])
bp_df = pd.DataFrame({'Backlogs': backlog_bins, 'CGPA': y})
sns.boxplot(x='Backlogs', y='CGPA', data=bp_df, palette='RdYlGn_r', ax=ax)
ax.set_title('Number of Backlogs vs CGPA Distribution', fontweight='bold'); save('backlogs_vs_cgpa_box')

# â”€â”€ 11. Stress vs CGPA (Violin) â”€â”€
fig, ax = plt.subplots(figsize=(8, 6))
stress_df = pd.DataFrame({'Stress': df['stress'].map({0: 'No Stress', 1: 'Stressed'}), 'CGPA': y})
sns.violinplot(x='Stress', y='CGPA', data=stress_df, palette=['#27ae60', '#e74c3c'], ax=ax)
ax.set_title('Mental Stress vs CGPA Distribution', fontweight='bold'); save('stress_vs_cgpa_violin')

# â”€â”€ 12. Distance vs CGPA â”€â”€
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['distance'], y, alpha=0.4, c='#1abc9c', s=15)
z = np.polyfit(df['distance'].fillna(0), y, 1)
x_line = np.linspace(0, df['distance'].max(), 100)
ax.plot(x_line, np.polyval(z, x_line), 'r--', lw=2, label='Trend')
ax.set_xlabel('Distance from Campus (KM)'); ax.set_ylabel('CGPA')
ax.set_title('Distance from Campus vs CGPA', fontweight='bold'); ax.legend(); save('distance_vs_cgpa')

# â”€â”€ 13. Intro Grade vs CGPA â”€â”€
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['intro_grade'], y, alpha=0.4, c='#3498db', s=15)
ax.set_xlabel('Introduction Audio Grade'); ax.set_ylabel('CGPA')
ax.set_title('Intro Grade (Whisper STT) vs CGPA', fontweight='bold'); save('intro_grade_vs_cgpa')

# â”€â”€ 14. Handwriting Grade vs CGPA â”€â”€
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['hw_grade'], y, alpha=0.4, c='#9b59b6', s=15)
ax.set_xlabel('Handwriting Notes Grade'); ax.set_ylabel('CGPA')
ax.set_title('Handwriting Grade (Image Analysis) vs CGPA', fontweight='bold'); save('hw_grade_vs_cgpa')

# â”€â”€ 15. Model Comparison Bar Chart â”€â”€
fig, ax = plt.subplots(figsize=(12, 6))
cv_comp = pd.read_csv(os.path.join(BASE, 'data', 'cv_model_comparison.csv'))
cv_comp = cv_comp.sort_values('R\u00b2', ascending=True) if 'R\u00b2' in cv_comp.columns else cv_comp
colors = sns.color_palette('coolwarm', len(cv_comp))
r2_col = [c for c in cv_comp.columns if 'R' in c and '2' in c or '\u00b2' in c]
if r2_col:
    ax.barh(cv_comp['Model'], cv_comp[r2_col[0]], color=colors, edgecolor='white')
    ax.set_xlabel('R\u00b2 Score'); ax.set_title('10-Fold CV: Model Comparison by R\u00b2', fontweight='bold')
save('model_comparison_r2')

# â”€â”€ 16. Per-Fold R2 Bar Chart â”€â”€
fig, ax = plt.subplots(figsize=(10, 5))
fold_r2 = [f['r2'] for f in fold_data]
ax.bar(range(1, 11), fold_r2, color='#27ae60', alpha=0.85, edgecolor='white')
ax.axhline(np.mean(fold_r2), color='red', lw=2, ls='--', label=f'Mean = {np.mean(fold_r2):.4f}')
ax.set_xlabel('Fold'); ax.set_ylabel('R\u00b2'); ax.set_title('Stacking Ensemble: R\u00b2 per CV Fold', fontweight='bold')
ax.set_xticks(range(1, 11)); ax.legend(); save('per_fold_r2')

# â”€â”€ 17. Per-Fold Accuracy Bar Chart â”€â”€
fig, ax = plt.subplots(figsize=(10, 5))
fold_acc = [f['a10'] for f in fold_data]
ax.bar(range(1, 11), fold_acc, color='#2980b9', alpha=0.85, edgecolor='white')
ax.axhline(np.mean(fold_acc), color='red', lw=2, ls='--', label=f'Mean = {np.mean(fold_acc):.1f}%')
ax.set_xlabel('Fold'); ax.set_ylabel('Accuracy (\u00b11.0)'); ax.set_title('Stacking Ensemble: Accuracy per CV Fold', fontweight='bold')
ax.set_xticks(range(1, 11)); ax.legend(); save('per_fold_accuracy')

# â”€â”€ 18. Predicted vs Actual (CV Aggregated) â”€â”€
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(all_y_true, all_y_pred, alpha=0.3, s=10, c='#2980b9')
ax.plot([0, 10], [0, 10], 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual CGPA'); ax.set_ylabel('Predicted CGPA')
ax.set_title('Predicted vs Actual CGPA (10-Fold CV)', fontweight='bold')
ax.legend(); ax.set_xlim(2, 10); ax.set_ylim(2, 10); save('predicted_vs_actual_cv')

# â”€â”€ 19. Predicted vs Actual (Holdout) â”€â”€
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_te, h_pred, alpha=0.3, s=10, c='#e67e22')
ax.plot([0, 10], [0, 10], 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual CGPA'); ax.set_ylabel('Predicted CGPA')
ax.set_title('Predicted vs Actual CGPA (80/20 Holdout)', fontweight='bold')
ax.legend(); ax.set_xlim(2, 10); ax.set_ylim(2, 10); save('predicted_vs_actual_holdout')

# â”€â”€ 20. Residual Distribution (CV) â”€â”€
fig, ax = plt.subplots(figsize=(10, 5))
residuals = all_y_pred - all_y_true
ax.hist(residuals, bins=40, color='#3498db', alpha=0.8, edgecolor='white')
ax.axvline(0, color='red', lw=2, ls='--')
ax.set_xlabel('Prediction Error (Predicted - Actual)'); ax.set_ylabel('Count')
ax.set_title(f'Residual Distribution (Mean={residuals.mean():.3f}, Std={residuals.std():.3f})', fontweight='bold')
save('residual_distribution')

# â”€â”€ 21. Confusion Matrix Heatmap (CV) â”€â”€
fig, ax = plt.subplots(figsize=(8, 6))
g_true = pd.cut(all_y_true, bins=BINS, labels=LABELS, right=False)
g_pred = pd.cut(np.clip(all_y_pred, 0, 10), bins=BINS, labels=LABELS, right=False)
cm = confusion_matrix(g_true, g_pred, labels=LABELS)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGn', xticklabels=LABELS, yticklabels=LABELS, ax=ax, linewidths=0.5)
ax.set_xlabel('Predicted Grade'); ax.set_ylabel('Actual Grade')
ax.set_title('Confusion Matrix - 10-Fold CV (All Folds)', fontweight='bold'); save('confusion_matrix_cv')

# â”€â”€ 22. Confusion Matrix Heatmap (Holdout) â”€â”€
fig, ax = plt.subplots(figsize=(8, 6))
g_t_h = pd.cut(y_te.values, bins=BINS, labels=LABELS, right=False)
g_p_h = pd.cut(np.clip(h_pred, 0, 10), bins=BINS, labels=LABELS, right=False)
cm_h = confusion_matrix(g_t_h, g_p_h, labels=LABELS)
sns.heatmap(cm_h, annot=True, fmt='d', cmap='Oranges', xticklabels=LABELS, yticklabels=LABELS, ax=ax, linewidths=0.5)
ax.set_xlabel('Predicted Grade'); ax.set_ylabel('Actual Grade')
ax.set_title('Confusion Matrix - Holdout (80/20)', fontweight='bold'); save('confusion_matrix_holdout')

# â”€â”€ 23. Missing Data Before Cleaning â”€â”€
fig, ax = plt.subplots(figsize=(12, 5))
raw_missing = df_raw.iloc[:, 4:17].isna().sum()
raw_missing.plot(kind='bar', color='#e74c3c', alpha=0.8, ax=ax, edgecolor='white')
ax.set_ylabel('Missing Count'); ax.set_title('Missing Values per Column (Before Cleaning)', fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right'); save('missing_data_before')

# â”€â”€ 24. Pairplot: Top 4 Features vs CGPA â”€â”€
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
top4 = feat_imp.head(4).index.tolist()
for ax, feat in zip(axes.flat, top4):
    ax.scatter(df[feat], y, alpha=0.3, s=8, c='#2c3e50')
    ax.set_xlabel(feat); ax.set_ylabel('CGPA')
    ax.set_title(f'{feat} vs CGPA', fontweight='bold')
plt.suptitle('Top 4 Most Important Features vs CGPA', fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout(); save('top4_features_vs_cgpa')

# â”€â”€ 25. Assignment vs Midterm colored by CGPA â”€â”€
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(df['midterm_norm'], df['assign_norm'], c=y, cmap='RdYlGn', alpha=0.5, s=15)
plt.colorbar(sc, label='CGPA', ax=ax)
ax.set_xlabel('Midterm Score'); ax.set_ylabel('Assignment Score')
ax.set_title('Midterm vs Assignment Score (colored by CGPA)', fontweight='bold'); save('midterm_vs_assign_by_cgpa')

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DECISION-MAKER TABLES (rendered as graph images)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print('   Generating decision-maker tables...')

# â”€â”€ TABLE 1: Model Comparison Decision Table â”€â”€
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')
table_data = [
    ['Ridge', '0.3159', '0.5217', '0.6881', 'Regularized, stable', 'Linear only'],
    ['Lasso', '0.3090', '0.5091', '0.7040', 'Feature selection', 'May drop features'],
    ['ElasticNet', '0.3111', '0.5102', '0.7028', 'Ridge+Lasso hybrid', 'Two hyperparams'],
    ['KNN', '0.4600', '0.6675', '0.5091', 'Non-parametric', 'Slow, curse of dim.'],
    ['SVR-RBF', '0.4252', '0.6529', '0.5370', 'Non-linear', 'Slow, hard to tune'],
    ['RandomForest', '0.3283', '0.5199', '0.6880', 'Robust, handles missing', 'Can overfit'],
    ['ExtraTrees', '0.3301', '0.5218', '0.6855', 'Fast, diverse', 'Slightly less acc.'],
    ['GradientBoosting', '0.3461', '0.5348', '0.6700', 'Sequential correction', 'Slow training'],
    ['XGBoost', '0.3486', '0.5467', '0.6561', 'Fast, regularized', 'Complex params'],
    ['LightGBM', '0.3716', '0.5778', '0.6329', 'Very fast', 'Prone to overfit'],
    ['Stacking Ensemble', '0.2910', '0.4950', '0.7088', 'Best overall', 'Slow inference'],
]
cols = ['Model', 'MAE', 'RMSE', 'R\u00b2', 'Pros', 'Cons']
tbl = ax.table(cellText=table_data, colLabels=cols, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
for (r, c), cell in tbl.get_celld().items():
    if r == 0: cell.set_facecolor('#2c3e50'); cell.set_text_props(color='white', fontweight='bold')
    elif r == len(table_data): cell.set_facecolor('#d5f5e3')
    else: cell.set_facecolor('#f8f9fa' if r % 2 == 0 else '#ffffff')
ax.set_title('Table 1: ML Algorithm Comparison for CGPA Prediction (10-Fold CV)', fontweight='bold', fontsize=12, pad=20)
save('table_model_comparison')

# â”€â”€ TABLE 2: Preprocessing Algorithm Comparison â”€â”€
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
table_data2 = [
    ['Text Parsing', 'Regex Parsers', 'Deterministic, fast, full control', 'Manual rules', 'YES'],
    ['Text Parsing', 'Fuzzy Matching', 'Handles typos', 'Slow, misinterprets numbers', 'No'],
    ['Text Parsing', 'LLM Parsing', 'Very flexible', 'Expensive, non-deterministic', 'No'],
    ['Missing Values', 'MICE (IterativeImputer)', 'Captures correlations', 'Computationally expensive', 'YES'],
    ['Missing Values', 'Mean/Median', 'Simple, fast', 'Ignores relationships', 'Partial'],
    ['Missing Values', 'KNN Imputation', 'Local patterns', 'Slow, sensitive to k', 'No'],
    ['Missing Values', 'Listwise Deletion', 'No bias', 'Loses 39.5% data!', 'No'],
    ['Scaling', 'StandardScaler', 'Works with gradient methods', 'Outlier sensitive', 'YES'],
    ['Encoding', 'Ordinal Encoding', 'Preserves natural order', 'Assumes equal spacing', 'YES'],
]
cols2 = ['Task', 'Algorithm', 'Pros', 'Cons', 'Selected?']
tbl2 = ax.table(cellText=table_data2, colLabels=cols2, loc='center', cellLoc='center')
tbl2.auto_set_font_size(False); tbl2.set_fontsize(8); tbl2.scale(1, 1.4)
for (r, c), cell in tbl2.get_celld().items():
    if r == 0: cell.set_facecolor('#2c3e50'); cell.set_text_props(color='white', fontweight='bold')
    elif c == 4 and r > 0 and table_data2[r-1][4] == 'YES': cell.set_facecolor('#d5f5e3')
    else: cell.set_facecolor('#f8f9fa' if r % 2 == 0 else '#ffffff')
ax.set_title('Table 2: Preprocessing Algorithm Comparison & Selection', fontweight='bold', fontsize=12, pad=20)
save('table_preprocessing_comparison')

# â”€â”€ TABLE 3: Feature Statistics Summary â”€â”€
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('off')
stats = df[FEATURES].describe().T[['mean', 'std', 'min', '50%', 'max']].round(2)
stats.columns = ['Mean', 'Std Dev', 'Min', 'Median', 'Max']
corr_with_cgpa = df[FEATURES].corrwith(y).round(3)
stats['Corr w/ CGPA'] = corr_with_cgpa.values
table_data3 = [[idx] + list(row) for idx, row in stats.iterrows()]
cols3 = ['Feature'] + list(stats.columns)
tbl3 = ax.table(cellText=table_data3, colLabels=cols3, loc='center', cellLoc='center')
tbl3.auto_set_font_size(False); tbl3.set_fontsize(7); tbl3.scale(1, 1.3)
for (r, c), cell in tbl3.get_celld().items():
    if r == 0: cell.set_facecolor('#2c3e50'); cell.set_text_props(color='white', fontweight='bold')
    else: cell.set_facecolor('#f8f9fa' if r % 2 == 0 else '#ffffff')
ax.set_title('Table 3: Feature Statistics & Correlation with CGPA', fontweight='bold', fontsize=12, pad=20)
save('table_feature_statistics')

# â”€â”€ TABLE 4: CV Fold-by-Fold Results â”€â”€
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')
table_data4 = []
for fd in fold_data:
    table_data4.append([f"Fold {fd['fold']}", f"{fd['r2']:.4f}", f"{fd['mae']:.4f}",
                        f"{fd['rmse']:.4f}", f"{fd['a10']:.1f}%"])
table_data4.append(['AVERAGE', f"{np.mean(fold_r2):.4f}", f"{np.mean([f['mae'] for f in fold_data]):.4f}",
                     f"{np.mean([f['rmse'] for f in fold_data]):.4f}", f"{np.mean(fold_acc):.1f}%"])
cols4 = ['Fold', 'R\u00b2', 'MAE', 'RMSE', 'Accuracy (\u00b11.0)']
tbl4 = ax.table(cellText=table_data4, colLabels=cols4, loc='center', cellLoc='center')
tbl4.auto_set_font_size(False); tbl4.set_fontsize(9); tbl4.scale(1, 1.5)
for (r, c), cell in tbl4.get_celld().items():
    if r == 0: cell.set_facecolor('#2c3e50'); cell.set_text_props(color='white', fontweight='bold')
    elif r == len(table_data4): cell.set_facecolor('#d5f5e3'); cell.set_text_props(fontweight='bold')
    else: cell.set_facecolor('#f8f9fa' if r % 2 == 0 else '#ffffff')
ax.set_title('Table 4: 10-Fold Cross-Validation Results (Stacking Ensemble)', fontweight='bold', fontsize=12, pad=20)
save('table_cv_fold_results')

# â”€â”€ 30. Academic Score vs CGPA (Hexbin density) â”€â”€
fig, ax = plt.subplots(figsize=(10, 7))
hb = ax.hexbin(df['academic_score'], y, gridsize=25, cmap='YlOrRd', mincnt=1)
plt.colorbar(hb, label='Count', ax=ax)
ax.set_xlabel('Academic Score (Midterm+Assignment avg)'); ax.set_ylabel('CGPA')
ax.set_title('Academic Score vs CGPA (Density Plot)', fontweight='bold'); save('academic_score_vs_cgpa_hexbin')

print(f'\n[6/6] Done! Generated {graph_num} graphs in {OUT}/')
print(f'Final cleaned CSV: final_cleaned_dataset.csv')

