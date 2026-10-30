"""Entry 1 cleaning step: remove the two known logging errors from the raw
LiftOff export and write a cleaned copy to data/processed/.

The raw file in data/raw/ is never modified.

Removed:
  1. Incline Dumbbell Bench Press, 2025-01-31 -- logged as 6.6 lb (2.99 kg)
     for 9 reps. Typo: sessions either side are ~35-40 kg e1RM.
  2. 2025-07-22 session -- sets logged with Weight = 0 and Reps = 0
     (aborted/empty entries). The Bench Press sets from that day have
     real values and are KEPT.

Why this is done on the raw sets and not on supervised_dataset.csv:
a bad session also contaminates its neighbours -- it becomes the previous
row's `target`, the next row's `prev_score`, and feeds the rolling means.
Deleting only the bad row would leave those contaminated values behind.
So we clean the raw log and rebuild the dataset with the pipeline.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "liftoff_workout_data.csv"
OUT = ROOT / "data" / "processed" / "liftoff_clean.csv"

raw = pd.read_csv(RAW)
date = raw["Date"].astype(str)

typo = date.str.startswith("2025-01-31") & (
    raw["Exercise Name"] == "Incline Dumbbell Bench Press"
)
empty = date.str.startswith("2025-07-22") & (raw["Weight"] == 0) & (raw["Reps"] == 0)

removed = raw[typo | empty]
clean = raw[~(typo | empty)]

print(f"Raw rows:     {len(raw)}")
print(f"Removed rows: {len(removed)}  (typo: {typo.sum()}, empty: {empty.sum()})")
print(f"Clean rows:   {len(clean)}")
print()
print(removed[["Date", "Exercise Name", "Set Order", "Weight", "Reps"]].to_string())

OUT.parent.mkdir(parents=True, exist_ok=True)
clean.to_csv(OUT, index=False)
removed.to_csv(ROOT / "data" / "processed" / "removed_rows.csv", index=False)
print(f"\nSaved: {OUT}")
