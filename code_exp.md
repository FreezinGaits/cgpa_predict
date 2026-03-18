# CGPA Prediction Project: Complete Code & Architecture Explanation

This document provides a comprehensive, line-by-line conceptual breakdown of **every single script** in the CGPA Project. It explains the Python libraries used, the Machine Learning algorithms deployed, the specific data cleaning techniques implemented, and the API architecture that serves the model to the frontend.

---

## PART 1: The Data Pipeline & Orchestration

### 1. The Master Orchestrator (`run_pipeline.py`)
**Path:** `CGPA Project/scripts/run_pipeline.py`

*   **What it does:** It acts as the "Start Button" for the entire project. Instead of running 4 different Python scripts manually, this script runs them structurally in the correct order to form an automated MLOps pipeline.
*   **Libraries Used:** `os`, `sys`, `time`, `subprocess`
*   **The Code Logic:**
    *   It uses the `subprocess.run(..., capture_output=False)` command. This allows Python to spawn a sub-process and execute another Python script, piping the output directly to the main terminal.
    *   It guarantees order: Download → Grade Audio → Grade Images → Train Model.
*   **Why it's important:** In professional Machine Learning environments, you need an automated sequence so data isn't processed out of order, and so the entire pipeline can be re-run with a single command if the dataset changes.

### 2. The Data Ingestion (`download_files.py`)
**Path:** `CGPA Project/scripts/download_files.py`

*   **What it does:** Connects to the internet and downloads the 1,922 `.mp3` and `.jpg` files sourced from the Google Drive links in the original CSV dataset.
*   **Libraries Used:** `pandas`, `requests`, `re` (Regular Expressions), `os`
*   **The Code Logic:**
    *   **Regex Extraction:** The Google Drive URLs in the CSV look like `drive.google.com/file/d/XYZ123/view`. The script uses `re.search(r'/d/([a-zA-Z0-9_-]+)', url)` to extract just the `XYZ123` ID.
    *   **Direct Download:** It pieces that ID into a direct download link: `https://drive.google.com/uc?id={file_id}` and uses `requests.get()` to pull the raw bytes.
    *   **Idempotency (Resume Logic):** Before downloading a file (e.g., `row_45.jpg`), it uses `os.path.exists(path)` to check if the file is already on the hard drive. If yes, it skips the HTTP request. This prevents wasting bandwidth and time if the script crashes halfway through 2,000 files.

---

## PART 2: The Multi-Modal AI Extraction

### 3. The Audio AI (`grade_introductions.py`)
**Path:** `CGPA Project/scripts/grade_introductions.py`

*   **What it does:** Uses Artificial Intelligence to turn audio files into text, then mathematically grades the student's communication skills (1 to 10).
*   **Libraries Used:** `whisper` (by OpenAI), `pandas`, `re`, `shutil`
*   **The Code Logic (NLP Rubric):**
    *   `model.transcribe(audio_path)`: Forces the Whisper Deep Learning model to listen to the MP3 and return a `transcript` string.
    *   **Word Count:** `len(transcript.split())` - Measures total words spoken. High word count indicates effort (+3 points).
    *   **Sentence Count:** Splits text by punctuation `re.split(r'[.!?]+', text)`. More sentences = well-structured thought (+2 points).
    *   **Vocab Richness:** `len(set(words)) / len(words)` - Calculates unique words compared to total words. This proves the student isn't just repeating words (+2 points).
    *   **Keyword Hits:** Searches for academic words (*"technology"*, *"passion"*, *"engineering"*). Proves academic focus (+2 points).

### 4. The Vision AI (`grade_handwriting.py`)
**Path:** `CGPA Project/scripts/grade_handwriting.py`

*   **What it does:** Uses Computer Vision to analyze a photo of a notebook and grade the student's organizational neatness (1 to 10).
*   **Libraries Used:** `PIL` (Python Imaging Library), `numpy`
*   **The Code Logic (Vision Rubric):**
    *   `Image.open(path).convert("L")`: Loads the image and converts it entirely to Grayscale, removing confusing color data.
    *   `np.array(img)`: Converts the image into a 2D mathematical matrix of pixels (0 = black, 255 = white).
    *   **Content Density:** `np.mean(pixels < 128)`. Calculates the exact percentage of the page covered by dark ink (+3 points).
    *   **Contrast / Clarity:** `ImageStat.Stat(img).stddev[0]`. Calculates the Standard Deviation of pixel colors. High deviation means ink and paper contrast perfectly (neat and clear) (+2 points).
    *   **Edge Density:** Uses `img.filter(ImageFilter.FIND_EDGES)` to map sharp lines. Scribbles lack sharp edges (+2 points).
    *   **Line Regularity:** Slices the numpy matrix into 10 horizontal strips (rows) using a `for` loop. Checks if ink exists in *all* 10 rows, proving the student formatted the whole page properly instead of scribbling in a corner (+2 points).

---

## PART 3: The Machine Learning Architecture

### 5. The Model Trainer (`cgpa_prediction_v2.py`)
**Path:** `CGPA Project/cgpa_prediction_v2.py`

This is the most critical file. It cleans the survey data, merges the AI features, and trains the machine learning model.

#### A. Custom Data Cleaning Extractors
The raw dataset was filled with messy, human-typed text (e.g., typing "85 percent" instead of `85`).
*   **`extract_score(val)`:** Uses regex `re.findall(r'[\d]+\.?[\d]*')` to pull numbers out of text. It mathematically normalizes decimals (if the student typed `0.85`, it runs `if v <= 1: v *= 100` to turn it into `85.0`).
*   **`extract_backlogs(val)`:** Converts words like "none", "nil", and "zero" explicitly to `0.0`.
*   **`extract_dist(val)`:** Uses regex to find text like "500 meters" and mathematically converts it: `float(nm[0]) / 1000`, turning it into `0.5` kilometers.

#### B. Feature Engineering (Interaction Terms)
Instead of just using standard columns, the code mathematically mixes columns to create stronger signals for the model:
*   `academic_score`: `(df['midterm'] + df['assign']) / 2`
*   `school_avg`: `(df['tenth_pct'] + df['twelfth_pct']) / 2`
*   `backlogs_log`: `np.log1p(df['backlogs'])`. Applies a logarithmic penalty so that having 1 backlog is bad, but having 5 backlogs doesn't simply multiply the penalty by 5 (it creates a curve).
*   `attend_stress`: `df['attendance'] / (df['stress'] + 1)`. An interaction term checking if a student maintains high attendance *despite* high stress.

#### C. MICE Imputation (Missing Target Recovery)
*   **Library:** `sklearn.impute.IterativeImputer`
*   **The Code Logic:** 380 rows were missing their CGPA target variable. Using `IterativeImputer(estimator=BayesianRidge(), max_iter=20)`, the code uses **Multivariate Imputation by Chained Equations (MICE)**. It essentially builds a mini-regression model to look at the other 19 features of a student (like attendance and midterms) to mathematically guess and fill in their missing CGPA, expanding our training data from 581 rows to a robust 961 rows.

#### D. The Stacking Regressor (Meta-Ensemble)
*   **Libraries:** `sklearn.ensemble`, `xgboost`, `catboost`, `lightgbm`
*   **The Code Logic:**
    1.  **Level 0 Estimators (The Crowd):** The code trains Random Forest, Gradient Boosting, XGBoost, LightGBM, and CatBoost simultaneously.
    2.  **Level 1 Estimator (The Judge):** It passes all 5 predictions to a `Ridge()` regression layer. The Ridge model learns which tree algorithm to trust in specific edge cases.
    3.  **10-Fold CV:** It wraps this using `KFold(n_splits=10, shuffle=True)` so the model trains on 90% of the data and blindly tests on 10%, repeating 10 times to guarantee an unbiased, true accuracy score (achieving ±1.0 accuracy of 94.3% with an RMSE of 0.514).

#### E. Saving the Model
*   **Library:** `joblib`
*   **The Code Logic:** It pushes data through a `StandardScaler()` inside a SciKit-Learn `Pipeline`, and saves the final output using `joblib.dump(..., 'best_cgpa_model_v2.pkl')`.

---

## PART 4: The Production Deployment (Backend API)

To make the ML model usable to humans (or a React JS frontend), it had to be deployed via a REST API.

### 6. The API Data Schemas (`schemas.py`)
**Path:** `api/schemas.py`
*   **Libraries:** `pydantic.BaseModel`
*   **What it does:** It acts as a strict "bouncer" for incoming web requests.
*   **The Code Logic:** It defines `StudentInput`. If a web user tries to send an attendance value of `150` (which is impossible), Pydantic catches it because of validation rules: `Field(..., ge=0, le=100)`. It ensures the API never crashes due to bad data.

### 7. The ML Singleton Class (`predictor.py`)
**Path:** `api/predictor.py`
*   **Libraries:** `joblib`, `pandas`, `numpy`
*   **What it does:** It loads the heavy `best_cgpa_model_v2.pkl` into RAM exactly one time when the server starts (Singleton pattern), preventing a 3-second lag on every web request.
*   **The Code Logic:**
    *   **Prediction:** The `predict()` method takes a Python dictionary from the web, identically reconstructs the Engineered Interaction features (like `academic_score` and `attend_stress`), processes the `intro_grade` and `hw_grade`, and passes the 20 variables directly to the `joblib` model object to generate a `predicted_cgpa` float.
    *   **Confidence Interval:** Calculates `{cgpa - CV_RMSE}` to `{cgpa + CV_RMSE}` to give the user a realistic prediction margin.
    *   **Dynamic Insights:** Uses simple `if/elif` blocks to generate English sentences based on the exact numbers the user sent (e.g., `if data['attendance'] < 60: return "Critical attendance."`).

### 8. The API Server (`main.py`)
**Path:** `api/main.py`
*   **Libraries:** `fastapi`, `tempfile`, `whisper`
*   **What it does:** The actual web server that listens for HTTP requests.
*   **The Code Logic:**
    *   `/predict` (POST): Accepts the JSON from the frontend, uses the `CGPAPredictor` singleton, and returns the JSON result.
    *   `/grade-intro` and `/grade-handwriting` (POST): Accepts live `.mp3` or `.jpg` file uploads directly from the React UI via `UploadFile = File(...)`.
    *   It saves the uploaded file to a hidden Windows `tempfile`, lazy-loads the Whisper/OpenCV models (only loading them into RAM if explicitly requested), runs the exact same grading rubric scripts mentioned in Part 2 over the uploaded file, deletes the temp file to save storage, and returns the live AI score (1-10) directly to the web user in under 2 seconds.
