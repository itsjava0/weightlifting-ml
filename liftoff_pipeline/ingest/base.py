"""Abstract loader interface + canonical schema.

Every loader (LiftOff today, potentially a second app's export later for
your friend's data) must convert its raw export into a DataFrame with
exactly these columns. Everything downstream (clean/features/models) only
ever talks to this canonical schema, so adding a new data source later
means writing one new loader class - nothing else changes.

One row = one working set.
"""

from abc import ABC, abstractmethod
import pandas as pd

CANONICAL_COLUMNS = [
    "athlete_id",       # str   - who this set belongs to
    "timestamp",        # datetime64 - session start time
    "session_duration_min",  # float - total workout duration, minutes
    "workout_label",    # str/NaN - user-given workout name, if any
    "exercise",         # str   - exercise name
    "set_index",        # int   - 0-indexed order within (session, exercise)
    "weight_kg",        # float - external load, kg (0 for pure bodyweight)
    "reps",             # int
    "distance_km",      # float - for cardio-style entries, else 0
    "duration_s",        # float - timed-set/cardio duration in seconds, else 0
    "rpe",               # float/NaN
    "notes",             # str/NaN
]


class BaseLoader(ABC):
    """Subclass this for each new raw export format."""

    def __init__(self, athlete_id: str, weight_unit: str = "lb"):
        self.athlete_id = athlete_id
        self.weight_unit = weight_unit

    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """Return a DataFrame with exactly CANONICAL_COLUMNS."""
        raise NotImplementedError

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = set(CANONICAL_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Loader output missing columns: {missing}")
        return df[CANONICAL_COLUMNS]
