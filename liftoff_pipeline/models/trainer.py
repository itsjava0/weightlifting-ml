"""Model training utilities.

Deliberately thin: a RandomForest as the default "real" model, purely as
a reasonable placeholder. Swapping it for whatever the actual research
phase decides on (gradient boosting, a mixed-effects model, etc.) means
passing a different `model` into train_model - nothing else changes.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DEFAULT_FEATURE_COLUMNS = [
    "sessions_trained_before",
    "days_since_prev_session",
    "prev_score",
    "score_delta_prev",
    "rolling_mean_score_3",
    "rolling_std_score_3",
    "rolling_mean_score_5",
    "rolling_std_score_5",
    "best_score_to_date",
]


@dataclass
class TrainedModel:
    model: object
    feature_columns: list
    train_rows: int
    test_rows: int


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Global chronological split (by session timestamp), not random.
    Shuffling would leak future sessions into training, which for a
    "predict future performance" problem is exactly the mistake to avoid.
    """
    cutoff = df["timestamp"].quantile(1 - test_frac)
    train = df[df["timestamp"] <= cutoff].copy()
    test = df[df["timestamp"] > cutoff].copy()
    return train, test


def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns=None,
    model=None,
) -> TrainedModel:
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    model = model or RandomForestRegressor(
        n_estimators=200, max_depth=6, random_state=0
    )

    train_clean = train_df.dropna(subset=feature_columns + ["target"])
    model.fit(train_clean[feature_columns], train_clean["target"])

    return TrainedModel(
        model=model,
        feature_columns=feature_columns,
        train_rows=len(train_clean),
        test_rows=len(test_df),
    )
