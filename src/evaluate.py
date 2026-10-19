"""Walk-forward evaluation, shared by every baseline and model in this project.

For each exercise, sessions are sorted by time. To predict session T we use
only sessions before T: train on sessions 0..T-1, predict T, then slide
forward one session and repeat.

Every method is scored on exactly the same sessions (from MIN_TRAIN onward),
so baselines and models are always compared like for like.

What one row means: a row describes one session of one exercise. Everything
in it is known by the end of that session. `target` is the score of the
NEXT session of the same exercise, so the prediction is made "after today's
workout, for the next one".
"""
import numpy as np
import pandas as pd

MIN_TRAIN = 10  # sessions of history required before the first prediction


def walk_forward_evaluate(df, predict_fn, min_train=MIN_TRAIN):
    """Run walk-forward evaluation for every exercise in df.

    predict_fn(train, test_row) -> float
        train    : DataFrame of all earlier sessions of this exercise
        test_row : one-row DataFrame, the session we predict from
    Returns a DataFrame of predictions: exercise, timestamp, actual, predicted.
    """
    rows = []
    for exercise, ex in df.groupby("exercise"):
        ex = ex.sort_values("timestamp").reset_index(drop=True)
        for t in range(min_train, len(ex)):
            train = ex.iloc[:t]
            test_row = ex.iloc[[t]]
            rows.append({
                "exercise": exercise,
                "timestamp": test_row["timestamp"].iloc[0],
                "actual": test_row["target"].iloc[0],
                "predicted": float(predict_fn(train, test_row)),
            })
    return pd.DataFrame(rows)


def score(preds):
    """MAE, RMSE and MAE as % of the mean actual, per exercise plus overall."""
    def metrics(p):
        err = p["predicted"] - p["actual"]
        mae = err.abs().mean()
        return pd.Series({
            "n": len(p),
            "MAE": mae,
            "RMSE": np.sqrt((err ** 2).mean()),
            "MAE_%": 100 * mae / p["actual"].mean(),
        })

    per_ex = preds.groupby("exercise").apply(metrics, include_groups=False)
    per_ex.loc["ALL"] = metrics(preds)
    per_ex["n"] = per_ex["n"].astype(int)
    return per_ex.round(2)
