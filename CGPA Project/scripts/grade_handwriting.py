"""
grade_handwriting.py — Grade handwritten notes images using image analysis.
Run from project root:  .venv/Scripts/python.exe "CGPA Project/scripts/grade_handwriting.py"

Creates:
    CGPA Project/data/handwriting_grades.csv — columns: row_idx, content_density,
                                                contrast, edge_density, hw_grade
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from PIL import Image, ImageStat, ImageFilter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES_DIR = os.path.join(BASE, "data", "notes")
OUT_CSV   = os.path.join(BASE, "data", "handwriting_grades.csv")


def grade_handwriting(img_path: str) -> dict:
    """
    Grade handwriting quality from an image using multiple visual metrics.

    Metrics:
      - content_density: ratio of dark pixels (more writing = more content)
      - contrast:        std dev of pixel intensities (clear ink vs background)
      - edge_density:    ratio of edge pixels (more edges = more detail/structure)
      - line_regularity: how evenly the writing covers the page vertically

    Returns dict with metrics + hw_grade (1–10).
    """
    try:
        img = Image.open(img_path).convert("L")  # grayscale
    except Exception as e:
        return {"content_density": 0, "contrast": 0, "edge_density": 0,
                "line_regularity": 0, "hw_grade": 1}

    pixels = np.array(img)
    stat = ImageStat.Stat(img)

    # 1. Content density — how much of the page has writing
    #    Dark pixels (< 128) relative to total
    content_density = float(np.mean(pixels < 128))

    # 2. Contrast — std deviation of pixel values
    #    High contrast → clear handwriting, low → faded/blurry
    contrast = float(stat.stddev[0])

    # 3. Edge density — apply edge filter, count strong edges
    edges = np.array(img.filter(ImageFilter.FIND_EDGES))
    edge_density = float(np.mean(edges > 30))

    # 4. Line regularity — divide page into horizontal strips,
    #    check if writing is evenly distributed (not just a small corner)
    n_strips = 10
    h = pixels.shape[0]
    strip_h = h // n_strips
    strip_densities = []
    for i in range(n_strips):
        strip = pixels[i * strip_h : (i + 1) * strip_h, :]
        strip_densities.append(np.mean(strip < 128))
    # More strips with content = better distribution (max 1.0)
    non_empty_strips = sum(1 for d in strip_densities if d > 0.02)
    line_regularity = non_empty_strips / n_strips

    # ── Scoring rubric (1–10) ──────────────────────────────────────────────
    score = 0.0

    # Content density (0–3 pts)
    if content_density >= 0.25:
        score += 3.0
    elif content_density >= 0.15:
        score += 2.0
    elif content_density >= 0.08:
        score += 1.0

    # Contrast / Neatness (0–2 pts)
    if contrast >= 50:
        score += 2.0
    elif contrast >= 35:
        score += 1.5
    elif contrast >= 20:
        score += 1.0

    # Edge density / Detail (0–2 pts)
    if edge_density >= 0.20:
        score += 2.0
    elif edge_density >= 0.12:
        score += 1.5
    elif edge_density >= 0.05:
        score += 1.0

    # Line regularity / Coverage (0–2 pts)
    if line_regularity >= 0.8:
        score += 2.0
    elif line_regularity >= 0.5:
        score += 1.5
    elif line_regularity >= 0.3:
        score += 1.0

    # Base score
    score += 1.0

    final_grade = int(min(10, max(1, round(score))))

    return {
        "content_density": round(content_density, 4),
        "contrast": round(contrast, 1),
        "edge_density": round(edge_density, 4),
        "line_regularity": round(line_regularity, 2),
        "hw_grade": final_grade,
    }


def main():
    files = sorted([f for f in os.listdir(NOTES_DIR) if f.endswith(".jpg")])
    print(f"Found {len(files)} note images in {NOTES_DIR}")

    # Resume support
    if os.path.exists(OUT_CSV):
        existing = pd.read_csv(OUT_CSV)
        done_indices = set(existing["row_idx"].tolist())
        print(f"Resuming: {len(done_indices)} already processed")
    else:
        existing = pd.DataFrame()
        done_indices = set()

    results = existing.to_dict("records") if len(existing) > 0 else []

    for i, fname in enumerate(files):
        idx = int(fname.replace("row_", "").replace(".jpg", ""))
        if idx in done_indices:
            continue

        fpath = os.path.join(NOTES_DIR, fname)
        grades = grade_handwriting(fpath)
        grades["row_idx"] = idx
        results.append(grades)

        if (i + 1) % 50 == 0 or i == len(files) - 1:
            print(f"  [{i+1:4d}/{len(files)}]  idx={idx}  density={grades['content_density']:.3f}  grade={grades['hw_grade']}/10")
            pd.DataFrame(results).to_csv(OUT_CSV, index=False)

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Saved {len(df_out)} handwriting grades → {OUT_CSV}")
    print(f"   Mean grade: {df_out['hw_grade'].mean():.2f}/10")


if __name__ == "__main__":
    main()
