# Multi-Modal CGPA Prediction System: Technical Documentation

## 1. Project Overview & Objective

Traditional CGPA prediction systems rely entirely on structured tabular data, such as midterm scores, attendance percentages, and historical grades. While these factors are important, they fail to capture the qualitative skills that often separate an average student from a top-tier one: communication ability and organizational skills.

The objective of this project was to build a comprehensive, multi-modal Machine Learning pipeline. The goal was to accurately predict a student's semester CGPA by combining:
1. **Quantitative Survey Data** (Tabular: Hours studied, distance, attendance, backlogs)
2. **Communication Skills** (Audio: Spoken self-introductions)
3. **Organizational/Note-taking Skills** (Vision: Photos of handwritten notes)

By bridging standard predictive modeling with modern AI extraction techniques (Whisper AI for Speech-to-Text, and Computer Vision for Image Analysis), the project successfully increased predictive accuracy to **94.3% (±1.0 GPA margin)**.

---

## 2. Infrastructure & Data Challenges

The raw dataset originally contained records for 961 students. However, achieving raw numerical data was only the first hurdle.

### Challenge 1: The Missing Target Variable Problem
**The Problem:** Out of 961 students, 380 rows were missing their actual CGPA (the target variable). Standard data science practice is to drop rows with missing targets. Doing so would have reduced our dataset to just 581 samples—a severe 40% loss of valuable feature data, leading to a weaker, overfit model.
**The Solution:** Instead of dropping the rows, I implemented **MICE (Multivariate Imputation by Chained Equations)** using Scikit-Learn’s `IterativeImputer`. MICE treats the missing CGPA as an unknown variable and predicts it using the other 17 available features (like attendance and midterms) in an iterative, round-robin fashion until the values stabilize. This allowed the final model to train on a full, robust dataset of 961 students.

### Challenge 2: Handling Large-Scale Media Downloads 
**The Problem:** The raw audio (.mp3) and image (.jpg) files were hosted on Google Drive, totaling nearly 2 GB of data. 
**The Solution:** I wrote a custom Python ingestion script using `requests` to automate the downloading of the 1,922 media files. I built in resume-capability so that if a download failed due to network errors, the script could pick up exactly where it left off without starting from scratch. 

---

## 3. Feature Extraction: The "Multi-Modal" Advantage

To feed audio and visual data into a regression model, it had to be quantified. I engineered two entirely new features representing the student's qualitative abilities.

### A. Introduction Grade (`intro_grade`)
*   **Methodology:** I utilized OpenAI’s **Whisper** (Base model) to loop through the 961 student `.mp3` files locally and transcribe their spoken introductions into text.
*   **The Heuristic Algorithm:** Once transcribed, I passed the text through a custom Natural Language Processing (NLP) rubric. The script calculated:
    *   **Word Count & Sentence Count:** To measure thoroughness.
    *   **Vocabulary Richness:** The ratio of unique words to total words.
    *   **Keyword Extraction:** Searching for academic/career-oriented keywords (e.g., "engineering", "goals", "passion").
*   **Result:** A normalized score from 1 to 10.
*   **Challenge Responded:** Transcribing nearly a thousand audio files on a local CPU takes hours and is prone to crashing on corrupted files. I wrapped the transcription step in a `try-except` block to catch corrupted media (assigning a default median grade) and forced the script to save progress to a CSV every 25 files to prevent catastrophic data loss.

### B. Handwriting Grade (`hw_grade`)
*   **Methodology:** I used **Python Imaging Library (PIL)** and basic **Computer Vision (OpenCV)** principles to analyze the student's notebook photos.
*   **The Heuristic Algorithm:** The images were converted to grayscale, and I calculated:
    *   **Content Density:** Percentage of dark edge pixels to total area (how much writing is on the page).
    *   **Contrast Density:** The standard deviation of pixel intensities (ink clarity).
    *   **Line Regularity:** Breaking the page into horizontal strips to ensure the writing was evenly distributed, not just scribbled in a corner.
*   **Result:** A normalized score from 1 to 10 representing structural neatness.

### C. The Full 20-Feature Array
After combining the standard quantitative data, the engineered features, and the multi-modal AI features, the final dataset consisted of exactly 20 features fed into the Machine Learning model:

**Standard Survey Features (10):**
1. `midterm_norm` (Midterm Score)
2. `assign_norm` (Assignment Score)
3. `twelfth_pct` (12th Grade Percentage)
4. `tenth_pct` (10th Grade Percentage)
5. `study_hours` (Study Hours/Day)
6. `attendance` (Attendance Percentage)
7. `backlogs` (Number of Backlogs)
8. `stress` (Mental Stress Level 0-10)
9. `distance` (Distance from Campus in km)
10. `complexity` (Content Complexity 1-3)
11. `teacher_fb` (Teacher Feedback 1-3)
12. `participation` (Discussion Participation 1-4)
13. `prev_prev_gpa` (Historical GPA)

**Engineered Interaction Features (5):**
14. `academic_score` (Average of Midterm and Assignment)
15. `school_avg` (Average of 10th and 12th relative performance)
16. `backlogs_log` (Log-transformed penalty for having multiple backlogs)
17. `attend_stress` (Interaction term tracking high attendance vs low stress)
18. `has_prev_gpa` (Boolean flag indicating if historical GPA data is present)

**Multi-Modal AI Features (2):**
19. `intro_grade` (Audio analysis of communication skills)
20. `hw_grade` (Computer Vision analysis of organizational skills)

---

## 4. Machine Learning Architecture & Algorithms

With 20 extracted features ready, the focus shifted to finding the algorithm that could best map these inputs to a CGPA.

### Algorithm Selection & Testing
I set up a rigorous testing pipeline evaluating 11 different algorithms using **10-Fold Cross-Validation**. The models tested included:
*   *Linear Models:* Ridge, Lasso, ElasticNet (Performed adequately, but failed on complex data interactions).
*   *Distance/Support Vector:* KNN, SVR-RBF (Struggled with mixed feature scales and dimensionality).
*   *Tree-Based Ensembles:* Random Forest, ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost.

**Finding:** The Boosted Tree models (specifically XGBoost, CatBoost, and GradientBoosting) consistently outperformed the linear models. They easily handled the non-linear relationship between variables like `stress` and `attendance`.

### The Final Model: Stacking Regressor (Meta-Ensemble)
Instead of picking just one model, I opted for a "Wisdom of the Crowd" approach to squeeze out the highest possible accuracy by building a **Stacking Regressor**.

1.  **Level 0 (Base Estimators):** I combined Ridge Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, and CatBoost. Each of these models views the data in slightly different ways.
2.  **Level 1 (Meta-Estimator):** I capped the stack with a final **Ridge Regressor**. This meta-model takes the *predictions* of the six Level 0 models as its inputs, learning which model's prediction to trust most under specific conditions.

---

## 5. Model Evaluation & Results

The addition of the multi-modal features and the Stacking Ensemble architecture yielded highly robust metrics during out-of-sample data validation.

| Metric | Final Result | Interpretation |
| :--- | :--- | :--- |
| **Accuracy (± 1.0 margin)** | **94.3%** | For 94 out of 100 students, the prediction is within 1 grade point. |
| **Accuracy (± 0.5 margin)** | **78.8%** | Extremely high precision for the majority of the dataset. |
| **R² Score** | **0.763** | The model explains 76.3% of the variance in student CGPAs. |
| **RMSE** | **0.514** | On average, the prediction is off by only 0.514 GPA points. |

### Feature Importance Analysis
Upon extracting the feature importance from the tree-based estimators, the traditional metrics (Historical GPA, 12th Grade %, Distance to Campus) understandably dominated the broader trend.

However, the newly engineered qualitative features (`intro_grade` and `hw_grade`) proved their worth. While they did not have the massive raw correlation of a midterm exam, **they acted as crucial tie-breakers**. If two students had identical exam scores, the model used the audio and handwriting grades to successfully differentiate them, which ultimately lowered the residual error and pushed the final ±1.0 accuracy past 94%.

---

## 6. Conclusion 

By refusing to drop missing data points (leveraging MICE imputation) and refusing to ignore unstructured data (leveraging AI audio transcription and vision filters), the model was fed a wider, more holistic profile of the student. 

The successful implementation of this pipeline proves that academic prediction models can be significantly improved by factoring in qualitative expressions of effort, organization, and communication—mirroring the actual multifaceted nature of academic success.
