# liftoff_pipeline

A reusable pipeline that turns raw set-by-set workout logs into a
model-ready dataset for predicting weightlifting performance, plus a
baseline modeling + evaluation harness. This is the "program" - the
plumbing that the actual research analysis will sit on top of.

## Why it's structured this way

The point of building this *before* the research analysis is that it
should keep working when:
- you have 2 more months of your own data (Nov 2026),
- your friend starts contributing a second dataset from his own tracker,
- you want to swap the placeholder model for whatever the research
  actually calls for.

So each stage is a separate, independently testable module, and nothing
downstream of `ingest/` knows or cares which app the data came from - it
only ever sees the canonical schema defined in `ingest/base.py`.

```
raw export (CSV)
   │  ingest/          -> one loader class per source app; all output
   ▼                      the same canonical "sets" schema
canonical sets table
   │  clean/           -> session IDs, bodyweight/cardio flags, sanity
   ▼                      filtering, unit conversion already done in ingest
cleaned sets table
   │  features/        -> per-set e1RM & volume -> one row per
   ▼                      (athlete, exercise, session) -> leakage-safe
                          historical features + future target
supervised (X, y) table
   │  models/          -> naive baselines + a placeholder RandomForest,
   ▼                      chronological (not random) train/test split
trained model + predictions
   │  evaluate/        -> MAE / RMSE / R², predicted-vs-actual plot
   ▼
metrics + plot
```

## Quickstart

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py path/to/your_export.csv \
    --athlete-id me --weight-unit lb --output-dir outputs
```

This writes to `outputs/`:
- `sets_clean.csv` - one row per set, cleaned + unit-normalized to kg
- `exercise_sessions.csv` - one row per (session, exercise): best e1RM/reps,
  volume, set count, etc.
- `supervised_dataset.csv` - the (X, y) table: historical features +
  future performance target, ready for modeling
- `metrics.csv` - MAE/RMSE/R² for the naive baselines vs. the placeholder model
- `pred_vs_actual.png` - scatter plot of predicted vs. actual on held-out sessions

## What "performance" means here

Two different metrics depending on the exercise, unified into one
`performance_score` column so the rest of the pipeline doesn't need to
special-case anything:

- **Loaded exercises** (bench press, squat, ...): estimated 1-rep max via
  the Epley formula (`weight * (1 + reps/30)`), taken from the best set in
  that session.
- **Bodyweight exercises** (pull-ups, push-ups, ...): best single-set rep
  count, since there's no external load to extrapolate a 1RM from.

`metric_type` on each row tells you which one applies. Pure cardio entries
(walking, spinning - tracked by distance/time, not load or reps) are
excluded from the supervised dataset entirely; they were noise for this
question, not signal.

## Avoiding leakage

Every historical feature (`prev_score`, `rolling_mean_score_3`,
`best_score_to_date`, ...) is built with `.shift()` / `.rolling()` /
`.expanding()` **before** grouping the target, so a row only ever sees
sessions strictly before it. `models/trainer.time_based_split` splits
chronologically rather than randomly for the same reason - random
splitting would let the model peek at future training sessions.
`tests/test_pipeline.py` has an explicit test asserting this holds.

## Adding a second athlete's data (e.g. your friend's tracker)

1. Write a new loader class in `ingest/` that outputs the same
   `CANONICAL_COLUMNS` schema (see `ingest/base.py` for the exact spec and
   `ingest/liftoff_loader.py` for a worked example).
2. Register it in `ingest/__init__.py`'s `LOADERS` dict under a new
   `source_app` key.
3. Run the pipeline once per athlete (different `athlete_id`/`source_app`
   in `PipelineConfig`), then `pd.concat` the two `sets_clean.csv` outputs
   before re-running `features`/`models`/`evaluate` on the combined data -
   `athlete_id` is threaded through every stage precisely so pooling later
   is just a concat, not a rewrite.

## Extending the model

`models/trainer.train_model` takes a `model=` argument - anything with a
scikit-learn-style `.fit(X, y)` / `.predict(X)` interface. Swap in
whatever the research phase actually calls for without touching feature
engineering or evaluation:

```python
from sklearn.ensemble import GradientBoostingRegressor
trained = train_model(train_df, test_df, model=GradientBoostingRegressor())
```

## Current result on the sample data (for reference, not a claim about the model)

On this athlete's real export (2645 raw rows → 2641 sets after dropping
empty rows → 852 usable supervised rows), the "last value" baseline
currently beats the placeholder RandomForest on held-out sessions (MAE
6.45 vs. 8.90). That's a completely normal thing to see at this stage with
generic historical features and no exercise-specific tuning - it's exactly
why the baselines are computed alongside any real model, and it's a
natural starting point for the actual research phase rather than a bug in
the pipeline.

## Tests

```bash
pytest tests/ -v
```

Uses a small synthetic fixture (not the real export) so tests stay fast
and don't depend on any one athlete's data.
