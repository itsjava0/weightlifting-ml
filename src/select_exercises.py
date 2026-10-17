"""Day 1: choose which exercises to model and save the modelling dataset.

Rule: keep exercises with 30+ sessions before the Feb 2026 cutoff.
A "session" here is one row of supervised_dataset.csv (one session of
that exercise that has a known next-session target).

All rows of the selected exercises are kept, including those after the
cutoff -- the cutoff only decides *which* exercises qualify.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "supervised_dataset.csv"
OUT = ROOT / "data" / "processed" / "model_dataset.csv"

CUTOFF = pd.Timestamp("2026-02-01")
MIN_SESSIONS = 30

df = pd.read_csv(SRC, parse_dates=["timestamp"])

before = df[df["timestamp"] < CUTOFF]
counts = before["exercise"].value_counts()
chosen = counts[counts >= MIN_SESSIONS].index.tolist()

print(f"Sessions per exercise before {CUTOFF.date()} (showing 20+):")
print(counts[counts >= 20].to_string())
print(f"\nSelected ({len(chosen)} exercises with {MIN_SESSIONS}+ sessions):")

model_df = df[df["exercise"].isin(chosen)].sort_values(["exercise", "timestamp"])
for ex in chosen:
    e = model_df[model_df["exercise"] == ex]
    print(f"  {ex}: {counts[ex]} before cutoff, {len(e)} total, "
          f"{e['timestamp'].min().date()} to {e['timestamp'].max().date()}")

model_df.to_csv(OUT, index=False)
print(f"\nSaved {len(model_df)} rows to {OUT}")
