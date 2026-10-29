"""Unit tests using a small synthetic CSV, mirroring the real LiftOff export
format (including its quirks: '.' placeholder workout names, an
empty/aborted-workout row, and a bodyweight exercise) - not the real 2600+
row file, so tests stay fast and don't depend on any particular athlete's
data.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liftoff_pipeline.ingest.liftoff_loader import LiftOffLoader, _parse_duration_minutes
from liftoff_pipeline.clean import clean_sets
from liftoff_pipeline.features import (
    add_set_metrics,
    build_exercise_session_table,
    build_supervised_dataset,
)

FIXTURE_CSV = """Date,Duration,Workout Name,Exercise Name,Set Order,Weight,Reps,Distance,Seconds,RPE,Notes
2024-01-01 08:00:00,00 hours 30 minutes 00 seconds,,Bench Press,0,100,10,0,0,,
2024-01-01 08:00:00,00 hours 30 minutes 00 seconds,,Bench Press,1,110,8,0,0,,
2024-01-01 08:00:00,00 hours 30 minutes 00 seconds,,Pull Ups,0,0,10,0,0,,
2024-01-08 08:00:00,00 hours 00 minutes 00 seconds,,,,0,0,0,0,,
2024-01-08 08:05:00,00 hours 31 minutes 00 seconds,.,Bench Press,0,102,10,0,0,,
2024-01-08 08:05:00,00 hours 31 minutes 00 seconds,.,Bench Press,1,112,8,0,0,,
2024-01-15 08:00:00,00 hours 29 minutes 00 seconds,,Bench Press,0,104,10,0,0,,
2024-01-15 08:00:00,00 hours 29 minutes 00 seconds,,Bench Press,1,114,8,0,0,,
2024-01-22 08:00:00,00 hours 32 minutes 00 seconds,,Bench Press,0,106,10,0,0,,
2024-01-22 08:00:00,00 hours 32 minutes 00 seconds,,Bench Press,1,116,8,0,0,,
2024-01-29 08:00:00,00 hours 30 minutes 00 seconds,,Bench Press,0,108,10,0,0,,
2024-01-29 08:00:00,00 hours 30 minutes 00 seconds,,Bench Press,1,118,8,0,0,,
"""


@pytest.fixture
def fixture_csv_path(tmp_path):
    p = tmp_path / "fixture.csv"
    p.write_text(FIXTURE_CSV)
    return str(p)


def test_duration_parsing():
    assert _parse_duration_minutes("01 hours 25 minutes 51 seconds") == pytest.approx(
        85.85
    )
    assert _parse_duration_minutes("00 hours 00 minutes 00 seconds") == 0
    import math

    assert math.isnan(_parse_duration_minutes(None))


def test_loader_drops_empty_rows_and_converts_units(fixture_csv_path):
    loader = LiftOffLoader(athlete_id="test_athlete", weight_unit="lb")
    df = loader.load(fixture_csv_path)
    # the fully-empty aborted-workout row should be gone
    assert len(df) == 11
    assert (df["athlete_id"] == "test_athlete").all()
    # 100 lb -> ~45.36 kg
    bench_first_set = df[(df["exercise"] == "Bench Press") & (df["weight_kg"] < 46)]
    assert len(bench_first_set) >= 1
    assert bench_first_set["weight_kg"].iloc[0] == pytest.approx(45.36, abs=0.01)


def test_liftoff_placeholder_workout_name_becomes_missing(fixture_csv_path):
    loader = LiftOffLoader(athlete_id="a", weight_unit="lb")
    df = loader.load(fixture_csv_path)
    assert df["workout_label"].isna().all()  # both NaN and "." collapse to missing


def test_clean_sets_flags_bodyweight(fixture_csv_path):
    loader = LiftOffLoader(athlete_id="a", weight_unit="lb")
    sets = clean_sets(loader.load(fixture_csv_path))
    pull_ups = sets[sets["exercise"] == "Pull Ups"]
    assert pull_ups["is_bodyweight"].all()
    bench = sets[sets["exercise"] == "Bench Press"]
    assert not bench["is_bodyweight"].any()


def test_session_ids_increase_chronologically(fixture_csv_path):
    loader = LiftOffLoader(athlete_id="a", weight_unit="lb")
    sets = clean_sets(loader.load(fixture_csv_path))
    by_time = sets.sort_values("timestamp")
    assert by_time["session_id"].is_monotonic_increasing


def test_exercise_session_table_uses_reps_metric_for_bodyweight(fixture_csv_path):
    loader = LiftOffLoader(athlete_id="a", weight_unit="lb")
    sets = add_set_metrics(clean_sets(loader.load(fixture_csv_path)))
    table = build_exercise_session_table(sets)
    pull_ups = table[table["exercise"] == "Pull Ups"].iloc[0]
    assert pull_ups["metric_type"] == "reps"
    assert pull_ups["performance_score"] == 10
    bench = table[table["exercise"] == "Bench Press"].iloc[0]
    assert bench["metric_type"] == "e1rm"
    assert bench["performance_score"] > 0


def test_supervised_dataset_has_no_leakage(fixture_csv_path):
    loader = LiftOffLoader(athlete_id="a", weight_unit="lb")
    sets = add_set_metrics(clean_sets(loader.load(fixture_csv_path)))
    table = build_exercise_session_table(sets)
    supervised = build_supervised_dataset(
        table, rolling_windows=(2,), horizon=1, min_sessions_per_exercise=2
    )
    bench_rows = supervised[supervised["exercise"] == "Bench Press"].sort_values(
        "timestamp"
    )
    assert len(bench_rows) > 0
    # target for row i must equal performance_score of the (i+1)th chronological
    # Bench Press session, never the current or an earlier one
    full_bench = table[table["exercise"] == "Bench Press"].sort_values("timestamp")
    full_scores = full_bench["performance_score"].reset_index(drop=True)
    full_times = full_bench["timestamp"].reset_index(drop=True)
    for _, row in bench_rows.iterrows():
        idx = full_times[full_times == row["timestamp"]].index[0]
        assert row["target"] == pytest.approx(full_scores.iloc[idx + 1])
        # prev_score must come from strictly before this session
        if idx > 0:
            assert row["prev_score"] == pytest.approx(full_scores.iloc[idx - 1])


def test_min_sessions_per_exercise_filters_short_histories(fixture_csv_path):
    loader = LiftOffLoader(athlete_id="a", weight_unit="lb")
    sets = add_set_metrics(clean_sets(loader.load(fixture_csv_path)))
    table = build_exercise_session_table(sets)
    # Pull Ups only has 1 session -> can never have enough history
    supervised = build_supervised_dataset(table, min_sessions_per_exercise=4)
    assert "Pull Ups" not in supervised["exercise"].values
