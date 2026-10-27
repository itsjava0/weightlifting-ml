"""Day 5b: does sleep improve the prediction?

Sleep source: Apple Health export (data/raw/sleep_data.csv), Apple Watch
records only, from late June 2025. Other sources in the export are left
out: "Pillow" overlaps the Watch (would double-count the same nights), and
the older iPhone records ("j", 2023-2025) only log time in bed, not sleep.

Nightly sleep = total hours of Core + Deep + REM + Unspecified sleep.
A sleep segment belongs to the night of the date 12 hours before it
started, so a night that starts at 23:30 on 3 May and ends at 07:00 on
4 May counts as the night of 3 May. Nights under 2 hours are treated as
missing (the watch was off or only recorded part of the night).

Timing: a sleep feature describes the night(s) BEFORE THE SESSION BEING
PREDICTED. That isn't known right after today's workout, so these
predictions are framed as "made on the morning of the next workout,
before lifting". No feature uses the target session's lifts.

Features added:
  sleep_last_night  hours asleep the night before the predicted session
  sleep_avg_3       mean hours over the 3 nights before it

Fair comparison: every method is scored only on predictions whose target
session has sleep data, and walk-forward history also comes only from
rows with sleep data, so all methods see the same sessions.

Outputs (in reports/): sleep_results.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import walk_forward_evaluate

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MIN_NIGHT_HOURS = 2

BASE_FEATURES = [
    "days_since_prev_session", "above_recent_avg", "rolling_std_score_3",
    "total_volume_kg", "sessions_trained_before", "best_set_reps",
]
SLEEP_FEATURES = ["sleep_last_night", "sleep_avg_3"]

# --- nightly sleep from the Apple Watch ---------------------------------
raw = pd.read_csv(ROOT / "data" / "raw" / "sleep_data.csv")
raw = raw[raw["sourceName"].str.contains("Watch") & raw["value"].str.contains("Asleep")]
start = pd.to_datetime(raw["startDate"].str[:19])
end = pd.to_datetime(raw["endDate"].str[:19])
night = (start - pd.Timedelta(hours=12)).dt.normalize()
hours = ((end - start).dt.total_seconds() / 3600).groupby(night).sum()
hours = hours[hours >= MIN_NIGHT_HOURS]
nights = hours.reindex(pd.date_range(hours.index.min(), hours.index.max()))  # gaps -> NaN
print(f"Sleep: {nights.notna().sum()} usable nights, "
      f"{nights.index.min().date()} to {nights.index.max().date()}, "
      f"mean {nights.mean():.2f} h")

# --- attach sleep to each row via the date of the PREDICTED session -----
df = pd.read_csv(ROOT / "data" / "processed" / "model_dataset.csv", parse_dates=["timestamp"])
sessions = pd.read_csv(ROOT / "data" / "processed" / "exercise_sessions.csv", parse_dates=["timestamp"])
sessions = sessions.sort_values(["exercise", "timestamp"])
sessions["next_timestamp"] = sessions.groupby("exercise")["timestamp"].shift(-1)
df = df.merge(sessions[["exercise", "timestamp", "next_timestamp"]], on=["exercise", "timestamp"])

df["above_recent_avg"] = df["performance_score"] - df["rolling_mean_score_3"]
df["change"] = df["target"] - df["performance_score"]

prev_night = df["next_timestamp"].dt.normalize() - pd.Timedelta(days=1)
df["sleep_last_night"] = prev_night.map(nights)
df["sleep_avg_3"] = [
    nights.reindex(pd.date_range(d - pd.Timedelta(days=2), d)).mean() for d in prev_night
]
sub = df.dropna(subset=SLEEP_FEATURES).sort_values(["exercise", "timestamp"])
print(f"Rows with sleep for the predicted session: {len(sub)} of {len(df)}")


# --- methods ---------------------------------------------------------------
def last_value(train, row):
    return row["performance_score"].iloc[0]


def ridge_with(features):
    def predict(train, row):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
        m.fit(train[features], train["change"])
        return row["performance_score"].iloc[0] + m.predict(row[features])[0]
    return predict


methods = {
    "baseline (last_value)": last_value,
    "ridge, sleep only": ridge_with(SLEEP_FEATURES),
    "ridge, 6 features": ridge_with(BASE_FEATURES),
    "ridge, 6 features + sleep": ridge_with(BASE_FEATURES + SLEEP_FEATURES),
}
preds = {name: walk_forward_evaluate(sub, fn) for name, fn in methods.items()}
errs = {name: (p["predicted"] - p["actual"]).abs() for name, p in preds.items()}

base_err = errs["baseline (last_value)"]
rows = []
for name, e in errs.items():
    rows.append({
        "method": name,
        "n": len(e),
        "MAE": e.mean(),
        "vs_baseline": e.mean() - base_err.mean(),
        "p_vs_baseline": np.nan if name.startswith("baseline") else wilcoxon(e, base_err).pvalue,
    })
res = pd.DataFrame(rows).round(3)
res.to_csv(REPORTS / "sleep_results.csv", index=False)

with_sleep = errs["ridge, 6 features + sleep"]
without = errs["ridge, 6 features"]
print("\nWalk-forward MAE (kg), only sessions with sleep data:\n")
print(res.to_string(index=False))
print(f"\nAdding sleep to the 6-feature model changes MAE by "
      f"{with_sleep.mean() - without.mean():+.3f} kg "
      f"(paired Wilcoxon p = {wilcoxon(with_sleep, without).pvalue:.3f})")
print("Correlation of sleep_last_night with next-session change: "
      f"{sub['sleep_last_night'].corr(sub['change']):+.2f}")
