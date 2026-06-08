"""
run_pipeline.py — Master orchestrator for the enhanced CGPA prediction pipeline.
==================================================================================
Run from project root:
    .venv/Scripts/python.exe "scripts/run_pipeline.py"

This script runs all 4 steps in order:
  Step 1: Download intro audio + handwritten notes from Google Drive
  Step 2: Grade introductions using Whisper speech-to-text
  Step 3: Grade handwriting using image analysis
  Step 4: Run the enhanced ML pipeline (cgpa_prediction_v2.py)
"""
import os, sys, subprocess, time

# Fix Windows encoding issues
os.environ["PYTHONIOENCODING"] = "utf-8"

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 5_code root
SCRIPTS = os.path.join(BASE, "scripts")
PYTHON = sys.executable  # Dynamically adapt to user's Python environment


def run_step(step_num, desc, script_path):
    """Run a step and show output."""
    print(f"\n{'=' * 70}")
    print(f"STEP {step_num}: {desc}")
    print(f"{'=' * 70}\n")
    
    result = subprocess.run(
        [PYTHON, script_path],
        cwd=BASE,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\n[!] Step {step_num} had issues (exit code {result.returncode})")
        print("    Continuing anyway...\n")
    else:
        print(f"\n[OK] Step {step_num} complete\n")
    
    return result.returncode


def main():
    print("=" * 70)
    print("ENHANCED CGPA PREDICTION PIPELINE")
    print("=" * 70)
    print(f"Python: {PYTHON}")
    print(f"Base:   {BASE}")
    
    t0 = time.time()
    
    # Step 1: Download files
    run_step(1, "Download intro audio + notes images from Google Drive",
             os.path.join(SCRIPTS, "download_files.py"))
    
    # Step 2: Grade introductions (Whisper)
    run_step(2, "Transcribe intros with Whisper + grade them",
             os.path.join(SCRIPTS, "grade_introductions.py"))
    
    # Step 3: Grade handwriting (image analysis)
    run_step(3, "Analyze handwritten notes + grade them",
             os.path.join(SCRIPTS, "grade_handwriting.py"))
    
    # Step 4: Run the ML pipeline
    run_step(4, "Run enhanced ML pipeline (MICE imputation + CV + Feature Engineering)",
             os.path.join(BASE, "cgpa_prediction_v2.py"))
    
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL STEPS COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
