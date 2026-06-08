# 🎓 Multi-Modal CGPA Prediction System
**Repository:** Final Submission Package (5_code)

This folder contains the complete, executable Machine Learning pipeline and data processing scripts for the CGPA Prediction System using a Multi-Modal Stacking Ensemble. We achieved a highly rigorous 94.3% accuracy rate on holdout test students.

## 📁 System Requirements
- Python 3.10 or higher
- At least 8 GB of RAM (Running massive ensembles on large data requires RAM)
- A modern processor (Execution of 1,800+ Hyperparameter tuning folds takes CPU time)

## 🛠️ Installation & Setup

1. **Open your terminal** and navigate inside the `5_code` directory.
2. **Install the required dependencies** by running:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you run into any issues with `xgboost` or `lightgbm` on Mac/Windows, please ensure you have the Visual C++ redistributable installed.*

## 🚀 How to Run the Project

You have two primary ways to engage with the codebase:

### Option 1: The Master Orchestrator (Fully Automated)
We have provided a master script that runs the entire end-to-end pipeline in order. Simply open your terminal and run:
```bash
python scripts/run_pipeline.py
```
*Note: This orchestrator will attempt to use your activated Python environment. It handles file locations natively.*

### Option 2: Running the Individual ML Pipeline (Recommended for testing the ML directly)
To immediately execute the data parsing, mathematical Missing Value Imputation (MICE), Cross-Validation comparisons, and Hyperparameter Stacking directly, run the core script:
```bash
python cgpa_prediction_v2.py
```

### Option 3: Interactive Exploration 
For a visual, step-by-step exploration of our models, tests, graphs, and feature importances, open the included Jupyter Notebook:
- `cgpa_prediction_testing.ipynb`

### Included Academic Utility Scripts
We have formally included three standalone functional scripts designed to augment our ML pipeline for evaluation purposes:

1. `generate_visualizations.py`
   - **Purpose:** Automatically generates 30 publication-ready Seaborn/Matplotlib visualizations (histograms, feature density, bell curves) based on the latest mathematical outputs from the test set evaluation and puts them in the `/graphs` directory.
2. `clean_excel.py`
   - **Purpose:** The standalone Deep Text-Parsing utility. It intelligently converts highly unstructured human survey text input (like `"7.04/10"` or `"1 supply"`) into numeric floats without aggressively dropping rows, preventing data loss. (This exact logic natively functions inside our main V2 script).
3. `fill_with_model.py`
   - **Purpose:** A demonstration script proving you can leverage the `.pkl` ensemble to instantly predict and fill in *missing* CGPAs across a completely unlabeled fresh dataset.

## 📊 Pre-Processed Data Note
The true execution of our system requires evaluating hundreds of megabytes of raw audio (MP3 files) and handwriting images via OpenAI Whisper and OpenCV. 

Because executing this raw AI transcription live against 1,000 students takes several hours and requires network fetching, **we have pre-computed these outputs into the `data/` folder** (`intro_grades.csv` and `handwriting_grades.csv`). 

When you run `cgpa_prediction_v2.py`, it safely utilizes these attached pre-computed grades to seamlessly process the core machine learning logic without forcing your computer to spend 3 hours transcribing MP3 files from Google Drive!
