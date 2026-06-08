import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

BASE = os.path.dirname(os.path.abspath(__file__))

print("Loading final cleaned dataset...")
df = pd.read_csv(os.path.join(BASE, 'final_cleaned_dataset.csv'))

# The 20 exact numeric features your model expects
FEATURES = [
    'midterm_norm','assign_norm','twelfth_pct','tenth_pct','study_hours',
    'attendance','backlogs','stress','distance','complexity',
    'teacher_fb','participation','prev_prev_gpa',
    'academic_score','school_avg','attend_stress','backlogs_log',
    'has_prev_gpa','intro_grade','hw_grade'
]

X = df[FEATURES]
y = df['CGPA']

# Perfectly replicate the 80/20 Holdout Test (random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("Training a fast, clean Stacking Ensemble ONCE...")
# We use only 100 trees and 3 CV internal folds for massive speed with zero MemoryError
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)),
    ('et', ExtraTreesRegressor(n_estimators=100, random_state=1, n_jobs=1)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('ridge', Ridge()),
    ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=1, verbosity=0))
]
stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv=3, n_jobs=1)

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler()),
    ('stack', stack)
])

# Train just once! (Takes <5 seconds)
pipe.fit(X_train, y_train)

print("Running exact holdout predictions...")
preds = np.clip(pipe.predict(X_test), 0, 10)
res = y_test.values - preds

mean_error = res.mean()
std_error = res.std()

print("Plotting the true data with beautiful markings...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LEFT: True Actual vs Predicted
ax = axes[0]
ax.scatter(y_test, preds, alpha=0.5, s=30, color="steelblue", edgecolors="none")
mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect Target")
ax.set_xlabel("Actual CGPA"); ax.set_ylabel("Predicted CGPA")
ax.set_title(f"Actual vs Predicted\nStacking Ensemble (Real Data)", fontweight="bold")
ax.legend()

# RIGHT: Real Residuals geometrically labeled!
ax2 = axes[1]
sns.histplot(res, bins=30, kde=True, color="#e74c3c", edgecolor="white", alpha=0.85, ax=ax2)
max_y = ax2.get_ylim()[1]

# Layer 1: Mean
ax2.axvline(mean_error, color="black", lw=3)
ax2.text(mean_error + 0.04, max_y * 0.95, f"Mean Error:\n {mean_error:.3f}", color="black", fontweight="bold", fontsize=10)

# Layer 2: ±1 Std
ax2.axvline(mean_error + std_error, color="#2980b9", lw=2, ls="--")
ax2.axvline(mean_error - std_error, color="#2980b9", lw=2, ls="--")
ax2.text(mean_error + std_error + 0.04, max_y * 0.85, f"+1 Std Dev\n(+{std_error:.3f})", color="#2980b9", fontweight="bold", fontsize=9)
ax2.text(mean_error - std_error - 0.28, max_y * 0.85, f"-1 Std Dev\n(-{std_error:.3f})", color="#2980b9", fontweight="bold", fontsize=9)
ax2.axvspan(mean_error - std_error, mean_error + std_error, color='#2980b9', alpha=0.15)

# Layer 3: ±2 Std
ax2.axvline(mean_error + 2*std_error, color="#27ae60", lw=2, ls=":")
ax2.axvline(mean_error - 2*std_error, color="#27ae60", lw=2, ls=":")
ax2.text(mean_error + 2*std_error + 0.04, max_y * 0.70, f"+2 Std Dev\n(+{2*std_error:.3f})", color="#27ae60", fontweight="bold", fontsize=9)
ax2.text(mean_error - 2*std_error - 0.28, max_y * 0.70, f"-2 Std Dev\n(-{2*std_error:.3f})", color="#27ae60", fontweight="bold", fontsize=9)

ax2.set_xlabel("Prediction Error (Actual \u2212 Predicted)")
ax2.set_ylabel("Count of Students")
ax2.set_title("True Residuals bounded by 68-95-99.7 Rule", fontweight="bold")

plt.tight_layout()
out_path = os.path.join(BASE, 'data', 'final_evaluation_instant.png')
plt.savefig(out_path, dpi=150)
print(f"DONE! Super fast image generated at: {out_path}")
