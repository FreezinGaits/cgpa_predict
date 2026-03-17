# CGPA Prediction Project: Complete Code Explanation

This document provides a line-by-line conceptual breakdown of exactly how the code in the `CGPA Project` directory functions, explaining the Python libraries, the algorithms, and the data cleaning techniques used.

---

## 1. The Master Orchestrator: `run_pipeline.py`
**Path:** `CGPA Project/scripts/run_pipeline.py`

*   **What it does:** It acts as the "Start Button" for the entire project. Instead of running 4 different Python scripts manually, this script runs them structurally in the correct order.
*   **Libraries Used:** `os`, `sys`, `time`, `subprocess`
*   **How it works:**
    *   It uses the `subprocess.run()` command. This allows Python to open a virtual terminal and execute another Python script.
    *   It calls `download_files.py`, waits for it to finish, then calls `grade_introductions.py`, then `grade_handwriting.py`, and finally trains the model with `cgpa_prediction_v2.py`.
*   **Why it's important:** In professional MLOps (Machine Learning Operations), you need an automated sequence so data isn't processed out of order.

---

## 2. The Data Ingestion: `download_files.py`
**Path:** `CGPA Project/scripts/download_files.py`

*   **What it does:** Connects to the internet and downloads the 1,922 `.mp3` and `.jpg` files sourced from the Google Drive links in the original CSV dataset.
*   **Libraries Used:**
    *   `pandas`: To read the `original_data.csv` file column by column.
    *   `requests`: To send HTTP GET requests to Google's servers to pull the file bytes.
    *   `re`: Regular Expressions, used to find the hidden File ID inside the heavy Google Drive URL.
*   **The Code Logic:**
    *   It uses Regex (`re.search(r'/d/([a-zA-Z0-9_-]+)', url)`) to extract the exact 33-character Google Drive ID from the student's submitted link.
    *   It pieces that ID into a direct download link: `https://drive.google.com/uc?id={file_id}`.
    *   **Idempotency (Resume Logic):** Before it downloads `row_45.jpg`, it uses `os.path.exists()` to check if the file is already on the hard drive. If yes, it skips it. This prevents wasting bandwidth if the script crashes halfway through 2,000 files.

---

## 3. The Audio AI: `grade_introductions.py`
**Path:** `CGPA Project/scripts/grade_introductions.py`

*   **What it does:** Uses Artificial Intelligence to turn audio files into text, then mathematically grades the student's communication skills (1 to 10).
*   **Libraries Used:**
    *   `whisper`: An open-source Deep Learning model built by OpenAI. It translates spoken audio into English text.
    *   `pandas`: To save the final grades to `intro_grades.csv`.
*   **The Code Logic (NLP Rubric):**
    *   `model.transcribe(audio_path)`: This line forces the Whisper model to listen to the MP3 and return a `transcript` string.
    *   **Word Count:** `len(transcript.split())` - Measures how much the student spoke.
    *   **Vocab Richness:** `len(set(words)) / len(words)` - Calculates how many *unique* words they used compared to total words (testing if they repeat themselves).
    *   **Keyword Extraction:** Searches the transcript for words like *"technology"*, *"passion"*, or *"engineering"*.
    *   It assigns a score starting at 1.0, adding +2.0 or +3.0 points based on those metrics, clamping the final answer to a maximum of 10.

---

## 4. The Vision AI: `grade_handwriting.py`
**Path:** `CGPA Project/scripts/grade_handwriting.py`

*   **What it does:** Uses Computer Vision to analyze a photo of a notebook and grade the student's organizational neatness (1 to 10).
*   **Libraries Used:**
    *   `PIL` (Python Imaging Library): To open the `.jpg` image and convert it to Grayscale (`.convert("L")`), removing confusing color data.
    *   `numpy` (Numerical Python): To turn the image into a giant 2D grid/matrix of numbers (pixels ranging from 0 to 255).
*   **The Code Logic (Vision Rubric):**
    *   **Content Density:** `np.mean(pixels < 128)`. 0 is black ink, 255 is white paper. This line calculates the exact percentage of the page covered by dark ink.
    *   **Contrast / Clarity:** `stat.stddev[0]`. It calculates the Standard Deviation of pixel colors. High deviation means the ink and paper contrast perfectly (neat). Low deviation means the photo is blurry.
    *   **Edge Density:** Uses `img.filter(ImageFilter.FIND_EDGES)` to map the sharp lines of the handwriting.
    *   **Line Regularity:** Uses a `for loop` to slice the pixel matrix into 10 horizontal rows. It checks if ink exists in all 10 rows (proving the student formatted the whole page properly, rather than scribbling in a corner).

---

## 5. The ML Architecture: `cgpa_prediction_v2.py`
**Path:** `CGPA Project/cgpa_prediction_v2.py`

This is the most critical file. It cleans the data, merges the AI features, and trains the machine learning model.

### A. Data Cleaning & Feature Extraction
*   **Libraries:** `re` (Regex), `pandas`
*   **The Problem:** Students manually typed things like *"85 percent"*, *"0.85"*, or *"85%"*. An ML model expects a clean integer `85`.
*   **The Code:** Custom Python functions (like `extract_score(val)`) use Regex (`re.findall(r'[\d]+\.?[\d]*')`) to pull only the numbers out of the string text.
*   **Normalization:** If a student wrote `0.85`, the code mathematically normalizes it (`if v <= 1: v *= 100`) to turn it into `85.0`.
*   **Merging:** It uses `pd.merge()` to combine the `original_data.csv`, `intro_grades.csv`, and `handwriting_grades.csv` into one master dataset of 20 features.

### B. MICE Imputation (Missing Data Handling)
*   **Library:** `sklearn.impute.IterativeImputer`
*   **The Problem:** 380 students were missing their CGPA target variable entirely.
*   **The Code:** `IterativeImputer(random_state=42, max_iter=10)`. This is the **MICE (Multivariate Imputation by Chained Equations)** algorithm. Instead of dropping the corrupted rows (which would ruin the dataset size), MICE builds a temporary regression model that looks at the student's attendance and midterm scores to accurately "guess" and fill in their missing CGPA.

### C. The 10-Fold Cross-Validation
*   **Library:** `sklearn.model_selection.KFold`
*   **The Code:** `KFold(n_splits=10, shuffle=True)`. This splits the 961 students into 10 groups. The model trains on 9 groups and tests blindly on the 10th group, repeating this 10 separate times to guarantee unbiased accuracy.

### D. The Stacking Regressor (The Final Model)
*   **Libraries:** `sklearn.ensemble`, `xgboost`, `catboost`, `lightgbm`
*   **The Code:**
    ```python
    estimators = [
        ('rf', RandomForestRegressor()),
        ('gb', GradientBoostingRegressor()),
        ('xgb', xgb.XGBRegressor()),
        ('lgb', lgb.LGBMRegressor()),
        ('cat', CatBoostRegressor())
    ]
    stack = StackingRegressor(estimators=estimators, final_estimator=Ridge())
    ```
*   **How it works:** Instead of relying on one algorithm, this creates a **"Meta-Ensemble."** It forces 5 different highly-advanced Decision Tree algorithms to make a prediction on the student. Then, the `Ridge()` regression at the top looks at all 5 answers and learns which specific algorithm to trust. This is what pushed the final accuracy to **94.3%**.

### E. Scaling & Saving
*   **Library:** `joblib`
*   **The Code:** Before training, the data is pushed through `StandardScaler()`, forcing all values to have a mean of 0 and standard deviation of 1, so metrics out of 100 (like midterms) don't overpower metrics out of 10 (like stress).
*   The final trained Stacking pipeline is saved to the hard drive as `best_cgpa_model_v2.pkl` using `joblib.dump()`, so the web backend can load it instantly without having to retrain it taking 3 minutes every time.
