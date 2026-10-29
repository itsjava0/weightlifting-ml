"""Cleaning steps for the canonical "sets" table.

Input/output here is always the canonical schema from ingest/base.py, plus
a few derived columns added along the way (session_id, is_bodyweight,
is_cardio, set_number). Nothing app-specific belongs in this file - that's
what the loaders are for.
"""

import pandas as pd


def _assign_session_id(df: pd.DataFrame) -> pd.DataFrame:
    """One session = one (athlete_id, timestamp) pair. Number sessions
    per athlete in chronological order so they're easy to reason about
    and to join on later."""
    df = df.sort_values(["athlete_id", "timestamp", "exercise", "set_index"])
    session_keys = (
        df[["athlete_id", "timestamp"]]
        .drop_duplicates()
        .sort_values(["athlete_id", "timestamp"])
        .assign(session_id=lambda d: d.groupby("athlete_id").cumcount())
    )
    return df.merge(session_keys, on=["athlete_id", "timestamp"], how="left")


def _flag_set_types(df: pd.DataFrame) -> pd.DataFrame:
    """Distinguish cardio entries (tracked by distance/time, not load) from
    strength sets, and flag bodyweight strength sets (weight_kg == 0 but
    reps are still meaningful, e.g. pull-ups)."""
    df = df.copy()
    df["is_cardio"] = (df["distance_km"] > 0) | (
        (df["duration_s"] > 0) & (df["reps"] == 0)
    )
    df["is_bodyweight"] = (df["weight_kg"] == 0) & ~df["is_cardio"]
    return df


def _basic_sanity_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that can't be real sets. Defensive - none of these are
    expected to fire on a clean LiftOff export, but a second athlete's
    data (or a future export) may not be as tidy."""
    before = len(df)
    df = df[(df["weight_kg"] >= 0) & (df["reps"] >= 0)]
    df = df.drop_duplicates(
        subset=["athlete_id", "timestamp", "exercise", "set_index"]
    )
    dropped = before - len(df)
    if dropped:
        print(f"[clean] dropped {dropped} invalid/duplicate row(s)")
    return df


def clean_sets(df: pd.DataFrame) -> pd.DataFrame:
    df = _basic_sanity_filter(df)
    df = _assign_session_id(df)
    df = _flag_set_types(df)
    df["weight_kg"] = df["weight_kg"].round(2)
    df["set_number"] = df["set_index"] + 1  # human-readable, 1-indexed
    return df.reset_index(drop=True)
