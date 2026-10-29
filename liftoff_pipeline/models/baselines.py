"""Dead-simple baselines. A real model that can't beat these on held-out
data isn't earning its complexity - always compute these alongside
whatever model comes later.
"""

import pandas as pd


def naive_baselines(df: pd.DataFrame) -> dict:
    """Given the supervised table (needs prev_score, rolling_mean_score_3
    columns), return {baseline_name: predictions} aligned to df's index.
    """
    return {
        "last_value": df["prev_score"],
        "rolling_mean_3": df["rolling_mean_score_3"],
    }
