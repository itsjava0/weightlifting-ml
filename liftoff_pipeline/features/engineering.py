"""Turn cleaned sets into a model-ready table.

Three steps, each usable on its own:

1. add_set_metrics       - per-set e1RM + volume
2. build_exercise_session_table - one row per (athlete, exercise, session):
                                   the "how did this exercise go that day"
                                   table, still just descriptive.
3. build_supervised_dataset      - adds historical/rolling features computed
                                   ONLY from prior sessions (no leakage) and
                                   a future target column, i.e. the actual
                                   (X, y) table a model can train on.

Performance metric note: e1RM (Epley) only makes sense when there's an
external load. For bodyweight moves (pull-ups, push-ups, ...) we use the
best single-set rep count instead. `performance_score` is metric_type-
aware so downstream code can stay agnostic; `metric_type` tells you which
one it is for any given exercise.
"""

import numpy as np
import pandas as pd


def add_set_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-set e1RM (Epley formula) and volume. NaN e1RM for
    bodyweight/cardio sets - there's no external load to extrapolate a 1RM
    from."""
    df = df.copy()
    loaded = ~df["is_bodyweight"] & ~df["is_cardio"]
    df["e1rm_kg"] = np.where(
        loaded, df["weight_kg"] * (1 + df["reps"] / 30), np.nan
    )
    df["volume_kg"] = df["weight_kg"] * df["reps"]
    return df


def build_exercise_session_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse sets -> one row per (athlete_id, session_id, exercise)."""
    df = add_set_metrics(df) if "e1rm_kg" not in df.columns else df

    grouped = df.groupby(["athlete_id", "session_id", "exercise"], as_index=False)
    table = grouped.agg(
        timestamp=("timestamp", "first"),
        n_sets=("set_number", "count"),
        total_reps=("reps", "sum"),
        best_weight_kg=("weight_kg", "max"),
        best_e1rm_kg=("e1rm_kg", "max"),
        best_set_reps=("reps", "max"),
        total_volume_kg=("volume_kg", "sum"),
        is_bodyweight=("is_bodyweight", "any"),
        is_cardio=("is_cardio", "any"),
    )

    table["metric_type"] = np.where(
        table["is_cardio"], "cardio",
        np.where(table["is_bodyweight"], "reps", "e1rm"),
    )
    table["performance_score"] = np.where(
        table["metric_type"] == "e1rm",
        table["best_e1rm_kg"],
        np.where(table["metric_type"] == "reps", table["best_set_reps"], np.nan),
    )
    return table.sort_values(["athlete_id", "exercise", "timestamp"]).reset_index(
        drop=True
    )


def build_supervised_dataset(
    exercise_session_table: pd.DataFrame,
    rolling_windows=(3, 5),
    horizon: int = 1,
    min_sessions_per_exercise: int = 4,
) -> pd.DataFrame:
    """Add leakage-safe historical features + a future target, and filter
    down to rows that are actually usable for training.

    Every feature is built from .shift()/.diff()/.rolling()/.expanding()
    on the *prior* value within each (athlete, exercise) group, so a given
    row only ever sees sessions strictly BEFORE it - required for a
    time-series problem like this, or the model would be trivially
    "cheating" using data it wouldn't have had at prediction time.

    target = performance_score `horizon` sessions in the future, for that
    same (athlete, exercise).
    """
    table = (
        exercise_session_table[exercise_session_table["metric_type"] != "cardio"]
        .sort_values(["athlete_id", "exercise", "timestamp"])
        .reset_index(drop=True)
    )

    by_ex = table.groupby(["athlete_id", "exercise"])

    table["sessions_trained_before"] = by_ex.cumcount()
    table["days_since_prev_session"] = (
        by_ex["timestamp"].diff().dt.total_seconds() / 86400
    )
    table["prev_score"] = by_ex["performance_score"].shift(1)
    prev_score_2 = by_ex["performance_score"].shift(2)
    table["score_delta_prev"] = table["prev_score"] - prev_score_2

    # Rolling stats operate on the already-shifted prev_score column, grouped
    # again by (athlete, exercise) - keeps each exercise's history separate.
    by_ex_prev = table.groupby(["athlete_id", "exercise"])["prev_score"]
    for w in rolling_windows:
        table[f"rolling_mean_score_{w}"] = by_ex_prev.transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        table[f"rolling_std_score_{w}"] = by_ex_prev.transform(
            lambda s: s.rolling(w, min_periods=2).std()
        )
    table["best_score_to_date"] = by_ex_prev.transform(lambda s: s.expanding().max())

    table["target"] = by_ex["performance_score"].shift(-horizon)

    enough_history = table["sessions_trained_before"] >= (min_sessions_per_exercise - 1)
    has_target = table["target"].notna()

    return table[enough_history & has_target].reset_index(drop=True)
