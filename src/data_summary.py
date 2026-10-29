"""Descriptive numbers quoted in the report, so each one comes from code.

  - how often the score stays exactly the same between sessions
  - how many changes are smaller than one 2.5 kg plate
  - how biased the three-session mean baseline is
  - how much training history the walk-forward models had
  - how often each exercise is done for more than 10 reps

Outputs (in reports/): data_summary.csv
"""
from pathlib import Path

import pandas as pd

from evaluate import MIN_TRAIN

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
LB_TO_KG = 0.45359237

df = pd.read_csv(ROOT / "data" / "processed" / "model_dataset.csv")
change = df["target"] - df["performance_score"]

preds = pd.read_csv(REPORTS / "baseline_predictions.csv")
roll = preds[preds["method"] == "rolling_mean_3"]
roll_err = roll["predicted"] - roll["actual"]

rows_per_ex = df.groupby("exercise").size()

sets = pd.read_csv(ROOT / "data" / "processed" / "liftoff_clean.csv")
sets = sets[sets["Exercise Name"].isin(rows_per_ex.index) & (sets["Reps"] > 0)]
over_10 = (sets.assign(over=sets["Reps"] > 10)
           .groupby("Exercise Name")["over"].mean() * 100).round(1)

summary = {
    "rows": len(df),
    "no_change_%": round(100 * (change == 0).mean(), 1),
    "change_under_2.5kg_%": round(100 * (change.abs() < 2.5).mean(), 1),
    "rolling_mean_too_low_%": round(100 * (roll_err < 0).mean(), 1),
    "rolling_mean_median_error_kg": round(roll_err.median(), 2),
    "min_training_rows": MIN_TRAIN,
    "max_training_rows": int(rows_per_ex.max() - 1),
}
pd.Series(summary).to_csv(REPORTS / "data_summary.csv", header=["value"])

print("Summary numbers used in the report:\n")
for k, v in summary.items():
    print(f"  {k:30s} {v}")
print("\nShare of sets with more than 10 reps, by exercise (%):\n")
print(over_10.sort_values(ascending=False).to_string())
