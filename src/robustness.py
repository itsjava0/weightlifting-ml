"""Day 5: robustness checks. Does the main finding hold when the set-up changes?

Main finding (Days 2-4): nothing beats "same as last session".

Each check rebuilds the dataset from a set log, then re-runs BOTH the
baseline and Ridge on that version, so they are always compared on the
same data:

  1. reference        Epley, all reps, original log (should reproduce Days 2-3)
  2. max 10 reps      only sets with <= 10 reps count toward the top set
                      (inside the range where 1RM formulas are reliable)
  3. Brzycki          Brzycki formula instead of Epley (sets <= 10 reps,
                      since Brzycki breaks down above that)

All checks use the original LiftOff log (with the two Day 1 errors removed).

The dataset builder follows the pipeline's definitions:
  - session = one timestamp; score = best e1RM over the session's sets
  - rows start once 3 earlier sessions exist; target = next session's score
  - rolling features use only sessions BEFORE the current one
Check 1 confirms the builder matches the pipeline.

Outputs (in reports/): robustness_results.csv
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
LB_TO_KG = 0.45359237

EXERCISES = [
    "Lying Leg Curl", "Dumbbell Lateral Raise", "Lat Pulldown", "Cable Lateral Raise",
    "Incline Dumbbell Bench Press", "Machine Reverse Fly", "Squat", "Tricep Rope Pushdown",
]
FEATURES = [
    "days_since_prev_session", "above_recent_avg", "rolling_std_score_3",
    "total_volume_kg", "sessions_trained_before", "best_set_reps",
]


def epley(w, r):
    return w * (1 + r / 30)


def brzycki(w, r):
    return w / (1.0278 - 0.0278 * r)


def build_dataset(log_path, unit="lb", formula=epley, max_reps=None):
    raw = pd.read_csv(log_path)
    sets = pd.DataFrame({
        "timestamp": pd.to_datetime(raw["Date"]),
        "exercise": raw["Exercise Name"],
        "weight_kg": raw["Weight"] * (LB_TO_KG if unit == "lb" else 1.0),
        "reps": raw["Reps"],
    })
    sets = sets[sets["exercise"].isin(EXERCISES) & (sets["weight_kg"] > 0) & (sets["reps"] > 0)]
    sets["volume"] = sets["weight_kg"] * sets["reps"]
    eligible = sets if max_reps is None else sets[sets["reps"] <= max_reps]
    eligible = eligible.assign(score=formula(eligible["weight_kg"], eligible["reps"]))

    top = (eligible.sort_values("score")
           .groupby(["exercise", "timestamp"]).tail(1)
           [["exercise", "timestamp", "score"]]
           .rename(columns={"score": "performance_score"}))
    # Like the pipeline: best_set_reps = most reps in any set of the session.
    per_session = sets.groupby(["exercise", "timestamp"]).agg(
        total_volume_kg=("volume", "sum"), best_set_reps=("reps", "max")).reset_index()
    s = top.merge(per_session, on=["exercise", "timestamp"])
    s = s.sort_values(["exercise", "timestamp"]).reset_index(drop=True)

    g = s.groupby("exercise")
    s["sessions_trained_before"] = g.cumcount()
    s["days_since_prev_session"] = g["timestamp"].diff().dt.total_seconds() / 86400
    prev = g["performance_score"].shift()
    s["rolling_mean_score_3"] = prev.groupby(s["exercise"]).transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    s["rolling_std_score_3"] = prev.groupby(s["exercise"]).transform(
        lambda x: x.rolling(3, min_periods=2).std())
    s["target"] = g["performance_score"].shift(-1)

    s = s[(s["sessions_trained_before"] >= 3) & s["target"].notna()].copy()
    s["above_recent_avg"] = s["performance_score"] - s["rolling_mean_score_3"]
    s["change"] = s["target"] - s["performance_score"]
    return s.dropna(subset=FEATURES)


def last_value(train, row):
    return row["performance_score"].iloc[0]


def ridge(train, row):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
    m.fit(train[FEATURES], train["change"])
    return row["performance_score"].iloc[0] + m.predict(row[FEATURES])[0]


def run_check(name, df):
    b = walk_forward_evaluate(df, last_value)
    r = walk_forward_evaluate(df, ridge)
    eb = (b["predicted"] - b["actual"]).abs()
    er = (r["predicted"] - r["actual"]).abs()
    per_ex = pd.DataFrame({"exercise": b["exercise"], "eb": eb, "er": er}).groupby("exercise").mean()
    return {
        "check": name,
        "rows": len(df),
        "n_predictions": len(b),
        "baseline_MAE": eb.mean(),
        "ridge_MAE": er.mean(),
        "diff": er.mean() - eb.mean(),
        "baseline_MAE_%": 100 * eb.mean() / b["actual"].mean(),
        "ridge_wins_exercises": int((per_ex["er"] < per_ex["eb"]).sum()),
        "wilcoxon_p": wilcoxon(er, eb).pvalue,
    }


original = ROOT / "data" / "processed" / "liftoff_clean.csv"

checks = [
    ("1. reference (Epley, all reps)", build_dataset(original, "lb")),
    ("2. top set max 10 reps", build_dataset(original, "lb", max_reps=10)),
    ("3. Brzycki, max 10 reps", build_dataset(original, "lb", brzycki, max_reps=10)),
]

rows = []
for name, df in checks:
    print(f"Running {name} ...")
    rows.append(run_check(name, df))

res = pd.DataFrame(rows).round({"baseline_MAE": 2, "ridge_MAE": 2, "diff": 2,
                                "baseline_MAE_%": 1, "wilcoxon_p": 3})
res.to_csv(REPORTS / "robustness_results.csv", index=False)

pd.set_option("display.width", 220)
print("\nWalk-forward MAE (kg). diff > 0 means the baseline wins.\n")
print(res.to_string(index=False))
print("\nGreig et al. (2023) load-velocity benchmark: SEE ~ 9.8% of 1RM")
