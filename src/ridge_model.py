"""Entry 3: Ridge regression vs the last_value baseline, walk-forward.

Design choices (see journal entry 3):

1. Predict the CHANGE, not the level.
   The model predicts  change = next score - this session's score,
   then adds it back:  prediction = this session's score + predicted change.
   Ridge shrinks the feature coefficients toward zero. The intercept is
   not penalised, so a model whose features carry no information falls
   back to "change = average past change" (the no-feature model in
   ablation.py), which is close to, but not the same as, the last_value
   baseline. (Corrected 2026-10-28: first described as falling back to
   "change = 0".)

2. One model per exercise, refitted at every walk-forward step on that
   exercise's earlier sessions only.

3. Features are standardised (StandardScaler) inside each fit. Ridge
   penalises coefficient size, so unscaled features (volume in the
   thousands, reps up to 20) would be penalised unequally. The scaler is
   fitted on the training rows only, so no test information leaks in.

4. Ridge strength (alpha) is chosen by RidgeCV on the training rows only.

Leakage check -- every feature is known by the end of the current session:
  days_since_prev_session  gap between previous and current session
  above_recent_avg         current score - mean of the 3 sessions before it
  rolling_std_score_3      spread of the 3 sessions before the current one
  total_volume_kg          current session's total volume
  sessions_trained_before  count of earlier sessions
  best_set_reps            most reps in any set of the current session
None of them use the target or any later session.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import walk_forward_evaluate, score

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


def ridge_predict(train, test_row):
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
    model.fit(train[FEATURES], train["change"])
    predicted_change = model.predict(test_row[FEATURES])[0]
    return test_row["performance_score"].iloc[0] + predicted_change


ridge = walk_forward_evaluate(df, ridge_predict)
ridge["method"] = "ridge"
ridge.to_csv(REPORTS / "ridge_predictions.csv", index=False)

base = pd.read_csv(REPORTS / "baseline_predictions.csv", parse_dates=["timestamp"])
base = base[base["method"] == "last_value"]

# Same sessions, same order -> paired comparison.
paired = ridge.merge(base, on=["exercise", "timestamp", "actual"],
                     suffixes=("_ridge", "_base"))
assert len(paired) == len(ridge) == len(base), "prediction sets do not line up"
paired["err_ridge"] = (paired["predicted_ridge"] - paired["actual"]).abs()
paired["err_base"] = (paired["predicted_base"] - paired["actual"]).abs()

rows = []
for ex, g in list(paired.groupby("exercise")) + [("ALL", paired)]:
    rows.append({
        "exercise": ex,
        "n": len(g),
        "baseline_MAE": g["err_base"].mean(),
        "ridge_MAE": g["err_ridge"].mean(),
        "diff": g["err_ridge"].mean() - g["err_base"].mean(),
        "ridge_better_%": 100 * (g["err_ridge"] < g["err_base"]).mean(),
    })
table = pd.DataFrame(rows).round(2)
table["result"] = np.where(table["diff"] < 0, "ridge wins", "baseline wins")
table.to_csv(REPORTS / "ridge_vs_baseline.csv", index=False)

print("Walk-forward MAE (kg). diff < 0 means Ridge is better.\n")
print(table.to_string(index=False))

wins = (table.loc[table.exercise != "ALL", "diff"] < 0).sum()
stat, p = wilcoxon(paired["err_ridge"], paired["err_base"])
print(f"\nRidge beats the baseline on {wins} of 8 exercises.")
print(f"Paired Wilcoxon test on all {len(paired)} absolute errors: p = {p:.3f}")
print("(p < 0.05 would mean the overall difference is unlikely to be chance)")
