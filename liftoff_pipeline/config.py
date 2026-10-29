"""Central configuration for a pipeline run.

Keeping every path/knob in one dataclass means the same pipeline code can be
re-run for a different athlete's export, a different weight unit, or a
different output directory just by constructing a new PipelineConfig -
nothing else in the codebase needs to change.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

WeightUnit = Literal["lb", "kg"]


@dataclass
class PipelineConfig:
    # --- input ---
    raw_csv_path: str
    athlete_id: str = "athlete_1"
    source_app: str = "liftoff"          # which loader to use, see ingest/
    weight_unit: WeightUnit = "lb"       # unit the RAW export is in

    # --- feature engineering ---
    rolling_windows: tuple = (3, 5)      # sessions, for rolling e1RM features
    min_sessions_per_exercise: int = 4   # drop exercises with too little history
    prediction_horizon: int = 1          # predict N sessions ahead

    # --- output ---
    output_dir: str = "outputs"

    def output_path(self, *parts: str) -> Path:
        p = Path(self.output_dir, *parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
