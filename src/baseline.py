"""Score the two naive baselines with walk-forward evaluation.

These results are the bar every model is compared against. Re-run this
script whenever the data or the performance score changes (e.g. the
robustness checks), so a model is always compared with a baseline built
from the same data.

Baseline 1 -- last_value: next session = this session's score.
Baseline 2 -- rolling_mean_3: next session = mean of the last 3 scores,
              including this session.

Note on the pipeline's columns: `prev_score` and `rolling_mean_score_3` are
lagged by one session -- they describe sessions BEFORE this row, not this
row. The latest score known at prediction time is `performance_score`
(this session), so the baselines use that. Using `prev_score` would skip
the most recent session and make the baseline look weaker than it is.

Outputs (in reports/):
  baseline_results.csv      -- MAE / RMSE / MAE_% per exercise and overall
  baseline_predictions.csv  -- every individual prediction (used to compare models)
"""
from pathlib import Path

import pandas as pd

from evaluate import walk_forward_evaluate, score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "model_dataset.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["timestamp"])
df = df.sort_values(["exercise", "timestamp"])

# Rolling mean of the last 3 scores, including the current session.
df["roll3_incl"] = (
    df.groupby("exercise")["performance_score"]
      .transform(lambda s: s.rolling(3, min_periods=1).mean())
)


def last_value(train, test_row):
    return test_row["performance_score"].iloc[0]


def rolling_mean_3(train, test_row):
    return test_row["roll3_incl"].iloc[0]


baselines = {"last_value": last_value, "rolling_mean_3": rolling_mean_3}

all_preds, tables = [], []
for name, fn in baselines.items():
    preds = walk_forward_evaluate(df, fn)
    preds["method"] = name
    all_preds.append(preds)
    t = score(preds)
    t.insert(0, "method", name)
    tables.append(t)

results = pd.concat(tables).reset_index().rename(columns={"index": "exercise"})
results.to_csv(REPORTS / "baseline_results.csv", index=False)
pd.concat(all_preds).to_csv(REPORTS / "baseline_predictions.csv", index=False)

wide = results.pivot(index="exercise", columns="method", values="MAE")
wide["n"] = results[results.method == "last_value"].set_index("exercise")["n"]
order = [e for e in wide.index if e != "ALL"] + ["ALL"]
print("Walk-forward MAE (kg), predictions start after 10 sessions of history\n")
print(wide.loc[order, ["n", "last_value", "rolling_mean_3"]].to_string())
print(f"\nSaved: {REPORTS / 'baseline_results.csv'}")
print(f"Saved: {REPORTS / 'baseline_predictions.csv'}")
