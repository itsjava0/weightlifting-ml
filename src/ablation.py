"""Day 4: which features carry signal?

Three tests, all using the same walk-forward evaluator, the same 244
predictions, and the same Ridge set-up as ridge_model.py:

1. Drop one: remove one feature from the full model, re-run.
   If error goes UP, that feature was helping.
   If error goes DOWN, that feature was adding noise.

2. Only one: a model with a single feature, compared with the baseline.
   Answers: does any variable beat "same as last session" on its own?

3. No features: the model only adds the average past change to today's
   score. Separates "the drift" from "the features".

Outputs (in reports/):
  ablation_results.csv       -- overall MAE for every run
  ablation_per_exercise.csv  -- per-exercise MAE for every run
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import walk_forward_evaluate

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "model_dataset.csv"
REPORTS = ROOT / "reports"

FEATURES = [
    "days_since_prev_session",
    "above_recent_avg",
    "rolling_std_score_3",
    "total_volume_kg",
    "sessions_trained_before",
    "best_set_reps",
]

df = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values(["exercise", "timestamp"])
df["above_recent_avg"] = df["performance_score"] - df["rolling_mean_score_3"]
df["change"] = df["target"] - df["performance_score"]


def make_ridge(features):
    def predict(train, test_row):
        if not features:  # no features: average past change only
            return test_row["performance_score"].iloc[0] + train["change"].mean()
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
        model.fit(train[features], train["change"])
        return test_row["performance_score"].iloc[0] + model.predict(test_row[features])[0]
    return predict


runs = {"full model (all 6)": FEATURES, "no features (drift only)": []}
for f in FEATURES:
    runs[f"drop {f}"] = [x for x in FEATURES if x != f]
for f in FEATURES:
    runs[f"only {f}"] = [f]

base = pd.read_csv(REPORTS / "baseline_predictions.csv")
base = base[base["method"] == "last_value"]
base_mae = (base["predicted"] - base["actual"]).abs().mean()
base_ex = base.assign(err=(base.predicted - base.actual).abs()).groupby("exercise")["err"].mean()

overall, per_ex = [], {"baseline (last_value)": base_ex}
for i, (name, feats) in enumerate(runs.items(), 1):
    print(f"[{i}/{len(runs)}] {name}")
    p = walk_forward_evaluate(df, make_ridge(feats))
    p["err"] = (p["predicted"] - p["actual"]).abs()
    overall.append({"run": name, "MAE": p["err"].mean()})
    per_ex[name] = p.groupby("exercise")["err"].mean()

res = pd.DataFrame(overall)
full = res.loc[res.run == "full model (all 6)", "MAE"].iloc[0]
res["vs_full"] = res["MAE"] - full
res["vs_baseline"] = res["MAE"] - base_mae
res = res.round(3)
res.to_csv(REPORTS / "ablation_results.csv", index=False)
pd.DataFrame(per_ex).round(2).to_csv(REPORTS / "ablation_per_exercise.csv")

pd.set_option("display.width", 200)
print(f"\nBaseline (last_value) MAE: {base_mae:.3f} kg")
print("vs_full:     drop-one runs -- positive = feature was helping, negative = it added noise")
print("vs_baseline: positive = worse than 'same as last session'\n")
print(res.to_string(index=False))
