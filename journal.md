# Research Journal

## Entry 1 — 2026-10-16 — Setup and data audit

### Research question
For my most-trained exercises, can I predict next-session e1RM more
accurately than the "same as last session" baseline using training-load,
frequency, and recovery features?

### Data source
- LiftOff app export, 2,645 raw set rows (Aug 2024 – May 2026)
- Single athlete (me); self-logged
- Raw files stored untouched in data/raw/ (read-only)
- Weights were entered in kg; LiftOff exports them in lb. The pipeline
  converts back to kg (--weight-unit lb), so all results are in kg.

### Definitions
- Performance score = Epley e1RM, weight × (1 + reps/30), of the
  heaviest set in a session (the "top set")
- Target = the next session's performance score for the same exercise
  (verified: target equals the next row's performance_score exactly)

### Removed rows (14 sets, listed in data/processed/removed_rows.csv)
1. Incline Dumbbell Bench Press, 2025-01-31: logged as 3 kg × 9
   (stored as 6.6 lb in the export), 2 sets.
   Why it's a typo: the sessions before and after were around 35–40 kg
   e1RM; I have never lifted 3 kg on this exercise.
2. 2025-07-22: 12 sets logged as 0 kg × 0 reps (Incline Bench Press,
   Machine Shoulder Press, Lying Tricep Extension, Dumbbell Lateral Raise).
   Why the Bench Press sets were kept: they had real weights and reps
   logged. Only the empty sets were removed, not the whole session.

Cleaning was done on the raw set log, and the dataset was then rebuilt
with the pipeline. Why deleting only the bad row would not be enough:
a bad session also becomes the previous row's target, the next row's
prev_score, and part of the rolling averages, so one bad entry corrupts
about five rows.

Effect: 2,641 → 2,627 sets; 1,187 → 1,182 exercise-sessions;
852 → 847 supervised rows.
Baseline (last_value) MAE improved 6.45 → 6.31 kg after cleaning.

### Exercise selection
Rule: 30+ sessions before the 2026-02-01 cutoff. 8 exercises, 324 rows
(saved as data/processed/model_dataset.csv):

| Exercise | Sessions before cutoff | Total |
|---|---|---|
| Lying Leg Curl | 47 | 47 |
| Dumbbell Lateral Raise | 46 | 50 |
| Lat Pulldown | 39 | 43 |
| Cable Lateral Raise | 38 | 40 |
| Incline Dumbbell Bench Press | 38 | 40 |
| Machine Reverse Fly | 34 | 34 |
| Squat | 34 | 38 |
| Tricep Rope Pushdown | 30 | 32 |

Why 30 sessions is a reasonable minimum: walk-forward evaluation and the
rolling features need earlier sessions to learn from, so with fewer than
~30 there are too few predictions left to give a stable MAE. 30 is a
practical cutoff, not an exact threshold.

Notes:
- Tricep Rope Pushdown qualifies at exactly 30 — the thinnest series.
- Lying Leg Curl and Machine Reverse Fly have no sessions after the cutoff.


## Entry 2 — 2026-10-19 — Baselines and walk-forward evaluation

### Evaluation method
Walk-forward, per exercise: to predict session T, use only sessions
before T, then slide forward one session. Predictions start after 10
sessions of history. Every method is scored on the same 244 predictions.
Code: src/evaluate.py, src/baseline.py

### Correction found
The pipeline's prev_score and rolling_mean_score_3 are lagged by one
session: they describe the session BEFORE the current row. The latest
score known at prediction time is performance_score (the current
session), so the baselines use that. The old 6.45 kg baseline used
prev_score and understated the baseline.

### Results (MAE, kg)
| Exercise | n | last_value | rolling_mean_3 |
|---|---|---|---|
| Cable Lateral Raise | 30 | 1.52 | 2.14 |
| Dumbbell Lateral Raise | 40 | 1.70 | 2.17 |
| Incline DB Bench Press | 30 | 3.16 | 3.33 |
| Lat Pulldown | 33 | 3.65 | 5.21 |
| Lying Leg Curl | 37 | 5.79 | 8.78 |
| Machine Reverse Fly | 24 | 6.01 | 6.52 |
| Squat | 28 | 5.64 | 8.17 |
| Tricep Rope Pushdown | 22 | 3.28 | 4.29 |
| ALL | 244 | 3.76 | 5.03 |

### Updated success criterion
Beat last_value (3.76 kg overall, walk-forward), replacing the old
6.45 kg target. The bar is now harder.

### Why last_value beat the rolling mean on every exercise
The rolling mean lags behind because it includes older, weaker sessions:
it predicted too low 62% of the time, about 1.15 kg under on average.
Since I'm still slowly progressing, and my weight often stays exactly
the same between sessions, the most recent session is a better guide
than an average of the last three.

