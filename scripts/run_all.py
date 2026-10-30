"""Re-run the whole study from the raw training log, in order.

    python scripts/run_all.py

Each step reads only files written by the steps before it, so this
rebuilds data/processed/ and reports/ from data/raw/liftoff_workout_data.csv.

The sleep test needs data/raw/sleep_data.csv, which is not public. Without
it, that step is skipped and reports/sleep_results.csv stays as committed.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLEEP_DATA = ROOT / "data" / "raw" / "sleep_data.csv"

STEPS = [
    ["src/clean_raw.py"],
    ["scripts/run_pipeline.py", "data/processed/liftoff_clean.csv",
     "--athlete-id", "me", "--weight-unit", "lb", "--output-dir", "data/processed"],
    ["src/select_exercises.py"],
    ["src/baseline.py"],
    ["src/ridge_model.py"],
    ["src/ablation.py"],
    ["src/random_forest.py"],
    ["src/robustness.py"],
    ["src/sleep.py"],
    ["src/data_summary.py"],
    ["src/figures.py"],
]

for script, *args in STEPS:
    if script == "src/sleep.py" and not SLEEP_DATA.exists():
        print(f"\n=== {script}: skipped (data/raw/sleep_data.csv is not public)")
        continue
    print(f"\n=== {script}", flush=True)
    subprocess.run([sys.executable, script, *args], cwd=ROOT, check=True)

print("\nAll steps finished.")
