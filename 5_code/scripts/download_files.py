"""
download_files.py - Download intro audio and handwritten notes from Google Drive.
"""
import os, sys, time, re, requests
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV  = os.path.join(BASE, "original_data.csv")
INTRO_DIR = os.path.join(BASE, "data", "intros")
NOTES_DIR = os.path.join(BASE, "data", "notes")
os.makedirs(INTRO_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)

SESSION = requests.Session()

def extract_file_id(url):
    if pd.isna(url): return None
    m = re.search(r'(?:id=|/d/)([a-zA-Z0-9_-]{20,})', str(url))
    return m.group(1) if m else None

def download_one(file_id, dest, retries=3):
    """Download using requests (bypasses gdown rate limits)."""
    for attempt in range(retries):
        try:
            # First try direct download
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            r = SESSION.get(url, timeout=30, allow_redirects=True)
            
            # Check if we got the virus scan warning page (large files)
            if b"confirm=" in r.content and len(r.content) < 10000:
                # Extract confirm token
                import re as _re
                token = _re.search(r'confirm=([0-9A-Za-z_-]+)', r.text)
                if token:
                    url = f"https://drive.google.com/uc?export=download&confirm={token.group(1)}&id={file_id}"
                    r = SESSION.get(url, timeout=30, allow_redirects=True)
            
            if r.status_code == 200 and len(r.content) > 100:
                with open(dest, 'wb') as f:
                    f.write(r.content)
                return True
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(2)
    return False

def main():
    df = pd.read_csv(CSV)
    intro_col = df.columns[18]
    notes_col = df.columns[17]
    total = len(df)
    print(f"Total rows: {total}")

    intro_ok = intro_fail = notes_ok = notes_fail = 0

    for idx in range(total):
        tag = f"row_{idx:03d}"

        # Download intro audio
        intro_path = os.path.join(INTRO_DIR, f"{tag}.mp3")
        if not os.path.exists(intro_path):
            fid = extract_file_id(df.iloc[idx][intro_col])
            if fid and download_one(fid, intro_path):
                intro_ok += 1
            else:
                intro_fail += 1
        else:
            intro_ok += 1

        # Download notes image
        notes_path = os.path.join(NOTES_DIR, f"{tag}.jpg")
        if not os.path.exists(notes_path):
            fid = extract_file_id(df.iloc[idx][notes_col])
            if fid and download_one(fid, notes_path):
                notes_ok += 1
            else:
                notes_fail += 1
        else:
            notes_ok += 1

        if (idx + 1) % 25 == 0 or idx == total - 1:
            print(f"[{idx+1:4d}/{total}]  intros: {intro_ok} ok {intro_fail} fail  |  notes: {notes_ok} ok {notes_fail} fail")

        # Small delay every 100 to avoid throttling
        if (idx + 1) % 100 == 0:
            time.sleep(3)

    print(f"\nDone! Intros: {intro_ok}/{total}, Notes: {notes_ok}/{total}")

if __name__ == "__main__":
    main()
