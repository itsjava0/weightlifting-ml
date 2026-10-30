# liftoff_pipeline

Turns a raw LiftOff export into a table with one row per exercise session:
predictors built from earlier sessions, plus the next session's score as
the target. In this project it is step 2 of `scripts/run_all.py`:

```bash
python scripts/run_pipeline.py data/processed/liftoff_clean.csv \
    --athlete-id me --weight-unit lb --output-dir data/processed
```

## Stages

```
ingest/     reads an export into a standard set-level table, drops empty
            rows and converts lb to kg
clean/      assigns session IDs, flags bodyweight and cardio entries,
            drops invalid or duplicate rows
features/   e1RM and volume per set -> one row per exercise session ->
            predictors from earlier sessions only, plus the next
            session's score as the target
models/     simple baselines and a random forest on a chronological split
evaluate/   MAE, RMSE, R² and a predicted-versus-actual plot
```

Only the loader in `ingest/` knows about LiftOff. Every later stage works
on the standard schema defined in `ingest/base.py`.

## Outputs

Written to the `--output-dir` folder:

- `sets_clean.csv`: one row per set, weights in kg
- `exercise_sessions.csv`: one row per exercise session (number of sets,
  best e1RM, most reps in a set, total volume, performance score)
- `supervised_dataset.csv`: predictors and next-session target for every
  session that has a next session (847 rows for my log)
- `metrics.csv` and `pred_vs_actual.png`: the pipeline's built-in check

The numbers in `metrics.csv` are not the study's results. They come from
a single chronological train/test split over all exercises, and the
pipeline's `last_value` baseline uses `prev_score`, the session before the
current one, instead of the current session's score (see journal entry 2).
The study's evaluation is in `src/` and its results are in `reports/`.

## Performance score

For loaded exercises, the Epley e1RM of the best set in the session:
weight × (1 + reps / 30). For bodyweight exercises, the most reps in one
set. Cardio entries, logged by distance or time, are excluded. All eight
exercises in the study are loaded exercises.

## Avoiding leakage

Every history feature (`prev_score`, `rolling_mean_score_3`,
`best_score_to_date`, and so on) is built with `.shift()`, `.rolling()` or
`.expanding()`, so a row only sees earlier sessions. Train/test splits
are chronological, never random. `tests/test_pipeline.py` checks this.

## Adding another athlete

1. Write a loader in `ingest/` that returns the columns in
   `CANONICAL_COLUMNS` (`ingest/base.py`). `ingest/liftoff_loader.py` is
   an example.
2. Register it in the `LOADERS` dictionary in `ingest/__init__.py`.
3. Run the pipeline once per athlete with a different `--athlete-id`.
   The ID is carried through every stage, so the outputs can be combined
   with `pd.concat`.

## Tests

```bash
python -m pytest tests
```

The tests use a small synthetic export, not my data.
