"""Entry 4 (optional): does a non-linear model help?

RandomForest (as used by Baek et al., 2025) with the same features, the
same "predict the change" set-up, and the same walk-forward evaluation as
ridge_model.py. Reported as a check; it did not beat Ridge.
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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


def rf_predict(train, test_row):
    # min_samples_leaf=5 stops the trees memorising single sessions,
    # which matters with only 10-49 training sessions per exercise.
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                  random_state=0, n_jobs=-1)
    model.fit(train[FEATURES], train["change"])
    return test_row["performance_score"].iloc[0] + model.predict(test_row[FEATURES])[0]


preds = walk_forward_evaluate(df, rf_predict)
preds["method"] = "random_forest"
preds.to_csv(REPORTS / "rf_predictions.csv", index=False)

rf = score(preds)[["n", "MAE"]].rename(columns={"MAE": "RF_MAE"})
ridge = pd.read_csv(REPORTS / "ridge_vs_baseline.csv").set_index("exercise")
table = rf.join(ridge[["baseline_MAE", "ridge_MAE"]])[["n", "baseline_MAE", "ridge_MAE", "RF_MAE"]]
table.to_csv(REPORTS / "rf_vs_ridge.csv")

print("Walk-forward MAE (kg)\n")
print(table.to_string())

# For any exercise where RF beats the baseline, test whether it's chance.
from scipy.stats import wilcoxon

base = pd.read_csv(REPORTS / "baseline_predictions.csv", parse_dates=["timestamp"])
base = base[base["method"] == "last_value"]
paired = preds.merge(base, on=["exercise", "timestamp", "actual"], suffixes=("_rf", "_base"))
for ex, g in paired.groupby("exercise"):
    err_rf = (g["predicted_rf"] - g["actual"]).abs()
    err_base = (g["predicted_base"] - g["actual"]).abs()
    if err_rf.mean() < err_base.mean():
        p = wilcoxon(err_rf, err_base).pvalue
        print(f"\nRF beats the baseline on {ex} (n={len(g)}): paired Wilcoxon p = {p:.2f}")
