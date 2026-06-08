"""
Quick script: Generate the REAL Residuals Distribution graph 
with Standard Deviation markings using the saved model.
"""
import warnings, os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

BASE = os.path.dirname(os.path.abspath(__file__))

# 1. Load the saved model
print("Loading model...")
stack_pipe = joblib.load(os.path.join(BASE, "best_cgpa_model_v2.pkl"))

# 2. Load the final cleaned dataset
print("Loading data...")
df = pd.read_csv(os.path.join(BASE, "final_cleaned_dataset.csv"))

# 3. Get the exact same features the model was trained on
meta = joblib.load(os.path.join(BASE, "model_meta_v2.json")) if os.path.exists(os.path.join(BASE, "model_meta_v2.json")) else None

# Read feature names from the preprocessor
preprocessor = stack_pipe.named_steps['pre']
raw_feature_names = preprocessor.get_feature_names_out()
clean_features = [name.replace('num__', '').replace('cat__', '') for name in raw_feature_names]

print(f"Features expected by model: {clean_features}")

# Find which columns exist in the CSV
available = [f for f in clean_features if f in df.columns]
print(f"Available in CSV: {available}")

X = df[available]
y = df["CGPA"]

# 4. Do the EXACT same 80/20 split with random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# 5. Get predictions
preds = np.clip(stack_pipe.predict(X_test), 0, 10)
residuals = y_test.values - preds

mean_error = residuals.mean()
std_error = residuals.std()

print(f"Mean Error: {mean_error:.4f}")
print(f"Std Dev:    {std_error:.4f}")

# 6. Generate the beautiful marked graph
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# LEFT: Actual vs Predicted
ax = axes[0]
ax.scatter(y_test, preds, alpha=0.5, s=30, color="steelblue", edgecolors="none")
mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect")
ax.set_xlabel("Actual CGPA"); ax.set_ylabel("Predicted CGPA")
ax.set_title(f"Actual vs Predicted\nStacking Ensemble (n={len(y_test)})", fontweight="bold")
ax.legend()

# RIGHT: Residuals with Standard Deviation markings
ax2 = axes[1]
sns.histplot(residuals, bins=30, kde=True, color="#e74c3c", edgecolor="white", alpha=0.85, ax=ax2)
max_y = ax2.get_ylim()[1]

# LAYER 1: Mean
ax2.axvline(mean_error, color="black", lw=3)
ax2.text(mean_error + 0.04, max_y * 0.95, f"Mean Error:\n {mean_error:.3f}", color="black", fontweight="bold", fontsize=10)

# LAYER 2: ±1 Std Dev (68%)
ax2.axvline(mean_error + std_error, color="#2980b9", lw=2, ls="--")
ax2.axvline(mean_error - std_error, color="#2980b9", lw=2, ls="--")
ax2.text(mean_error + std_error + 0.04, max_y * 0.85, f"+1σ\n(+{std_error:.3f})", color="#2980b9", fontweight="bold", fontsize=10)
ax2.text(mean_error - std_error - 0.25, max_y * 0.85, f"-1σ\n(-{std_error:.3f})", color="#2980b9", fontweight="bold", fontsize=10)
ax2.axvspan(mean_error - std_error, mean_error + std_error, color='#2980b9', alpha=0.15, label="68% Confidence")

# LAYER 3: ±2 Std Dev (95%)
ax2.axvline(mean_error + (2 * std_error), color="#27ae60", lw=2, ls=":")
ax2.axvline(mean_error - (2 * std_error), color="#27ae60", lw=2, ls=":")
ax2.text(mean_error + (2 * std_error) + 0.04, max_y * 0.70, f"+2σ\n(+{(2*std_error):.3f})", color="#27ae60", fontweight="bold", fontsize=10)
ax2.text(mean_error - (2 * std_error) - 0.25, max_y * 0.70, f"-2σ\n(-{(2*std_error):.3f})", color="#27ae60", fontweight="bold", fontsize=10)

ax2.set_xlabel("Prediction Error (Actual − Predicted)")
ax2.set_ylabel("Count of Students")
ax2.set_title("Residuals Distribution with Standard Deviations", fontweight="bold")
ax2.legend(loc="upper right")

plt.suptitle("Stacking Ensemble — Final Holdout Evaluation with 68-95-99.7 Rule", fontsize=15, fontweight="bold")
plt.tight_layout()

out_path = os.path.join(BASE, "data", "final_evaluation_marked.png")
plt.savefig(out_path, dpi=150)
print(f"\n✅ Saved to: {out_path}")
plt.show()
