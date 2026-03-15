"""
grade_introductions.py — Transcribe intro MP3s with Whisper and grade them.
Run from project root:  .venv/Scripts/python.exe "CGPA Project/scripts/grade_introductions.py"

Creates:
    CGPA Project/data/intro_grades.csv  — columns: row_idx, transcript, word_count,
                                           sentence_count, vocab_richness, intro_grade
"""
import os, sys, warnings, re
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTRO_DIR = os.path.join(BASE, "data", "intros")
OUT_CSV   = os.path.join(BASE, "data", "intro_grades.csv")


def grade_transcript(text: str) -> dict:
    """Grade a student introduction transcript on multiple dimensions."""
    if not text or len(text.strip()) < 10:
        return {"word_count": 0, "sentence_count": 0, "vocab_richness": 0.0, "intro_grade": 1}

    words = text.split()
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]
    sentence_count = max(len(sentences), 1)
    unique_words = len(set(w.lower() for w in words))
    vocab_richness = unique_words / max(word_count, 1)

    # ── Scoring rubric (1–10) ──
    score = 0.0

    # 1. Length (0–3 pts): Did they speak enough? 5–8 sentences asked.
    if word_count >= 80:
        score += 3.0
    elif word_count >= 50:
        score += 2.0
    elif word_count >= 25:
        score += 1.0

    # 2. Sentence structure (0–2 pts): Multiple coherent sentences
    if sentence_count >= 5:
        score += 2.0
    elif sentence_count >= 3:
        score += 1.5
    elif sentence_count >= 2:
        score += 1.0

    # 3. Vocabulary richness (0–2 pts): Diverse word usage
    if vocab_richness >= 0.7:
        score += 2.0
    elif vocab_richness >= 0.5:
        score += 1.5
    elif vocab_richness >= 0.3:
        score += 1.0

    # 4. Content keywords (0–2 pts): Mentions education, goals, skills
    text_lower = text.lower()
    content_keywords = [
        "study", "learn", "university", "college", "semester", "engineering",
        "computer", "science", "goal", "interest", "project", "skill",
        "experience", "future", "career", "passion", "hobby", "technology",
        "develop", "coding", "program", "degree", "education", "work"
    ]
    keyword_hits = sum(1 for kw in content_keywords if kw in text_lower)
    if keyword_hits >= 5:
        score += 2.0
    elif keyword_hits >= 3:
        score += 1.5
    elif keyword_hits >= 1:
        score += 1.0

    # 5. Base score (everyone gets at least 1 for trying)
    score += 1.0

    final_grade = int(min(10, max(1, round(score))))
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "vocab_richness": round(vocab_richness, 3),
        "intro_grade": final_grade,
    }


def main():
    import whisper

    print("Loading Whisper model (base)...")
    model = whisper.load_model("base")
    print("✅ Whisper loaded\n")

    # Check which files exist
    files = sorted([f for f in os.listdir(INTRO_DIR) if f.endswith(".mp3")])
    print(f"Found {len(files)} MP3 files in {INTRO_DIR}")

    # Load existing results if resuming
    if os.path.exists(OUT_CSV):
        existing = pd.read_csv(OUT_CSV)
        done_indices = set(existing["row_idx"].tolist())
        print(f"Resuming: {len(done_indices)} already processed")
    else:
        existing = pd.DataFrame()
        done_indices = set()

    results = existing.to_dict("records") if len(existing) > 0 else []

    for i, fname in enumerate(files):
        idx = int(fname.replace("row_", "").replace(".mp3", ""))
        if idx in done_indices:
            continue

        fpath = os.path.join(INTRO_DIR, fname)
        try:
            result = model.transcribe(fpath, language="en")
            transcript = result["text"].strip()
        except Exception as e:
            transcript = ""
            print(f"  ⚠️ Whisper failed for {fname}: {e}")

        grades = grade_transcript(transcript)
        grades["row_idx"] = idx
        grades["transcript"] = transcript
        results.append(grades)

        if (i + 1) % 25 == 0 or i == len(files) - 1:
            print(f"  [{i+1:4d}/{len(files)}]  idx={idx}  words={grades['word_count']}  grade={grades['intro_grade']}/10")
            # Save progress periodically
            pd.DataFrame(results).to_csv(OUT_CSV, index=False)

    # Final save
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Saved {len(df_out)} intro grades → {OUT_CSV}")
    print(f"   Mean grade: {df_out['intro_grade'].mean():.2f}/10")
    print(f"   Mean words: {df_out['word_count'].mean():.0f}")


if __name__ == "__main__":
    main()
