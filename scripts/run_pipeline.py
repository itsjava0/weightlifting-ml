#!/usr/bin/env python3
"""CLI entry point.

Usage:
    python scripts/run_pipeline.py path/to/export.csv \
        --athlete-id me --weight-unit lb --output-dir outputs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liftoff_pipeline import PipelineConfig
from liftoff_pipeline.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to a raw LiftOff export CSV")
    parser.add_argument("--athlete-id", default="athlete_1")
    parser.add_argument("--source-app", default="liftoff")
    parser.add_argument("--weight-unit", default="lb", choices=["lb", "kg"])
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--min-sessions-per-exercise", type=int, default=4)
    parser.add_argument("--prediction-horizon", type=int, default=1)
    args = parser.parse_args()

    config = PipelineConfig(
        raw_csv_path=args.csv_path,
        athlete_id=args.athlete_id,
        source_app=args.source_app,
        weight_unit=args.weight_unit,
        output_dir=args.output_dir,
        min_sessions_per_exercise=args.min_sessions_per_exercise,
        prediction_horizon=args.prediction_horizon,
    )

    results = run_pipeline(config)

    print(f"\nSets after cleaning:        {len(results['sets']):>6}")
    print(f"Exercise-session rows:      {len(results['exercise_sessions']):>6}")
    print(f"Supervised (X, y) rows:     {len(results.get('supervised', [])):>6}")
    if "metrics" in results:
        print("\nModel comparison on held-out sessions (sorted by MAE):")
        print(results["metrics"].round(2).to_string())
    print(f"\nArtifacts written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
