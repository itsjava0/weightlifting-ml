"""Loader for LiftOff app CSV exports.

Observed raw format (one row per working set):

    Date,Duration,Workout Name,Exercise Name,Set Order,Weight,Reps,Distance,Seconds,RPE,Notes
    2024-10-07 19:28:19,01 hours 25 minutes 51 seconds,,Smith Machine Incline Bench Press,0,132.3,12,0,0,,

Quirks handled here (found via EDA on a real export, 2645 rows / 257
sessions / 112 exercises):

- "Duration" is a free-text string like "01 hours 25 minutes 51 seconds",
  not a number - needs regex parsing.
- "Workout Name" is blank for most sessions, and LiftOff sometimes writes a
  bare "." as a placeholder instead of leaving it empty - both are treated
  as missing.
- "Weight" is 0 for bodyweight moves (pull-ups, push-ups, dips, ...) where
  Reps is still meaningful - these are kept, not dropped, and flagged
  downstream rather than here (loader stays dumb/structural).
- "Distance" / "Seconds" are almost always 0 and only populated for a
  handful of cardio entries (Walking, Spinning) where Weight/Reps are 0.
- "RPE" and "Notes" were empty for every row in the sample export but are
  kept as optional columns since LiftOff's schema supports them.
- A small number of rows are entirely empty aborted-workout artifacts
  (no Exercise Name, no Duration, no Set Order, everything else 0) - these
  aren't real sets and are dropped here before type coercion.
- Weight is exported in whatever unit the user's app is set to (lb by
  default here) - converted to kg immediately so every downstream stage
  only ever deals in kg, regardless of which athlete/app it came from.
"""

import re
import pandas as pd

from .base import BaseLoader, CANONICAL_COLUMNS

LB_TO_KG = 0.45359237
_DURATION_RE = re.compile(
    r"(?P<h>\d+)\s*hours?\s*(?P<m>\d+)\s*minutes?\s*(?P<s>\d+)\s*seconds?"
)


def _parse_duration_minutes(text) -> float:
    """'01 hours 25 minutes 51 seconds' -> 85.85 (minutes). NaN-safe."""
    if pd.isna(text):
        return float("nan")
    match = _DURATION_RE.search(str(text))
    if not match:
        return float("nan")
    h, m, s = (int(match.group(g)) for g in ("h", "m", "s"))
    return h * 60 + m + s / 60


class LiftOffLoader(BaseLoader):
    RAW_COLUMN_MAP = {
        "Date": "timestamp",
        "Duration": "_raw_duration",
        "Workout Name": "workout_label",
        "Exercise Name": "exercise",
        "Set Order": "set_index",
        "Weight": "_raw_weight",
        "Reps": "reps",
        "Distance": "distance_km",
        "Seconds": "duration_s",
        "RPE": "rpe",
        "Notes": "notes",
    }

    def load(self, path: str) -> pd.DataFrame:
        raw = pd.read_csv(path)
        missing = set(self.RAW_COLUMN_MAP) - set(raw.columns)
        if missing:
            raise ValueError(
                f"LiftOff export is missing expected columns: {missing}"
            )

        df = raw.rename(columns=self.RAW_COLUMN_MAP)

        # Drop empty/aborted-workout artifact rows (no exercise logged at all)
        n_before = len(df)
        df = df.dropna(subset=["exercise", "set_index"])
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"[ingest] dropped {n_dropped} empty/aborted-workout row(s)")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["session_duration_min"] = df["_raw_duration"].map(_parse_duration_minutes)

        # LiftOff uses a bare "." as a placeholder for "no workout name given"
        df["workout_label"] = df["workout_label"].replace({".": pd.NA})

        df["set_index"] = df["set_index"].astype(int)
        df["reps"] = df["reps"].astype(int)
        df["distance_km"] = df["distance_km"].astype(float)
        df["duration_s"] = df["duration_s"].astype(float)

        weight = df["_raw_weight"].astype(float)
        df["weight_kg"] = weight * LB_TO_KG if self.weight_unit == "lb" else weight

        df["athlete_id"] = self.athlete_id

        return self._validate(df)
