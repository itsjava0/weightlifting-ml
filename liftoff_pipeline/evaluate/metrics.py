"""Evaluation: metrics + a predicted-vs-actual plot.

Kept model-agnostic - pass in any {name: predictions} dict (baselines,
the trained model, both) and get back a comparable metrics table.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> dict:
    mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true, y_pred = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0}
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
        "n": int(mask.sum()),
    }


def compare_predictions(y_true, predictions: dict) -> pd.DataFrame:
    """predictions: {model_name: array-like of predictions, same length/
    index as y_true}. Returns one row of metrics per model, sorted best
    (lowest MAE) first."""
    rows = {name: compute_metrics(y_true, preds) for name, preds in predictions.items()}
    return pd.DataFrame(rows).T.sort_values("mae")


def plot_predicted_vs_actual(y_true, y_pred, title: str, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(y_true, y_pred, alpha=0.5, s=15)
    lo, hi = np.nanmin(y_true), np.nanmax(y_true)
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
