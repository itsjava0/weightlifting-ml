# Research Journal

## Entry 1 — 2026-10-16 — Setup and data audit

### Research question
For my most-trained exercises, can I predict next-session e1RM more
accurately than the "same as last session" baseline using training-load,
frequency, and recovery features?

### Data source
- LiftOff app export, 2,645 raw set rows (Jul 2024 – May 2026)
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


## Entry 3 — 2026-10-24 — First model: Ridge regression

### Features (all known by the end of the current session)
- days_since_prev_session: gap before this session
- above_recent_avg: this score minus the mean of the 3 sessions before it
- rolling_std_score_3: spread of the 3 sessions before this one
- total_volume_kg: this session's total volume
- sessions_trained_before: number of earlier sessions
- best_set_reps: most reps in any set of this session (corrected on
  2026-10-25; first described as reps in the top set)
Leakage check: none of them use the target or any later session.

### Design
- The model predicts the change (next score - this score), then adds it
  to this session's score. If the features are useless, Ridge shrinks
  the feature weights to zero and predicts the average past change (the
  "no features" model in entry 4), which is close to the last_value
  baseline. (Corrected on 2026-10-28; first described as shrinking
  toward "change = 0". The intercept is not penalised.)
- One model per exercise, refitted at every walk-forward step.
- Features standardised; scaler and Ridge strength (RidgeCV) fitted on
  training rows only.
Code: src/ridge_model.py

### Results (MAE, kg)
| Exercise | n | Baseline | Ridge | Diff |
|---|---|---|---|---|
| Cable Lateral Raise | 30 | 1.52 | 1.84 | +0.32 |
| Dumbbell Lateral Raise | 40 | 1.70 | 2.10 | +0.40 |
| Incline DB Bench Press | 30 | 3.16 | 3.35 | +0.19 |
| Lat Pulldown | 33 | 3.65 | 3.93 | +0.28 |
| Lying Leg Curl | 37 | 5.79 | 6.25 | +0.46 |
| Machine Reverse Fly | 24 | 6.01 | 6.67 | +0.66 |
| Squat | 28 | 5.64 | 6.32 | +0.68 |
| Tricep Rope Pushdown | 22 | 3.28 | 3.32 | +0.03 |
| ALL | 244 | 3.76 | 4.14 | +0.38 |

Ridge beats the baseline on 0 of 8 exercises. Paired Wilcoxon test on
all 244 errors: p < 0.001, so the baseline's advantage is not chance.

### Extra check
A model with no features, which just adds my average past change to
today's score, also loses: MAE 3.97 kg. So even the upward drift does
not help, and the features add a further 0.17 kg of error.

### Answer
No. At this data size, Ridge with training-load, frequency and recovery
features does not beat "same as last session".

### Why Ridge couldn't beat "same as last session"
Strength changes slowly, so the last session's weight and reps are
already a good estimate (25% of sessions don't change at all). The
remaining changes depend on things I didn't log, like sleep and effort,
and with only 10–49 earlier sessions per exercise the model mostly learned noise.

## Entry 4 — 2026-10-25 — Feature test (ablation) and RandomForest

### Method
Same walk-forward evaluator, same 244 predictions, same Ridge set-up as
entry 3. Fourteen runs:
- Drop one: remove one feature from the full model. If error rises, the
  feature was helping; if it falls, it was adding noise.
- Only one: a model with a single feature, compared with the baseline.
- No features: only adds the average past change to today's score.
Code: src/ablation.py, src/random_forest.py

### Results (MAE, kg; baseline = 3.76)
| Run | MAE |
|---|---|
| No features (average change only) | 3.97 |
| Only best_set_reps | 3.99 |
| Only sessions_trained_before | 3.99 |
| Only days_since_prev_session | 4.08 |
| Only rolling_std_score_3 | 4.11 |
| Only total_volume_kg | 4.11 |
| Full model (all 6) | 4.14 |
| Only above_recent_avg | 4.21 |

Drop one (change vs full model, 4.14): dropping days_since_prev_session
improved the model (-0.03); dropping any other feature made it worse
(+0.01 to +0.13).

### What it shows
1. No feature carries signal: every single-feature model is worse than
   the no-feature model.
2. The drop-one results look like most features "help", but the full
   model is already worse than using no features. They only help by
   partly cancelling each other's noise, not by predicting anything.
3. Differences between features are 0.01-0.15 kg on a noisy target, so
   their ranking is not meaningful. The robust result is the overall one.

### RandomForest (optional check)
| Exercise | Baseline | Ridge | RandomForest |
|---|---|---|---|
| ALL (244) | 3.76 | 4.14 | 4.25 |

RandomForest (the model used by Baek et al.) did worse than Ridge, so a
non-linear model does not help. It beat the baseline on one exercise,
Tricep Rope Pushdown (3.10 vs 3.28 kg, n = 22), but a paired Wilcoxon
test gives p = 0.59, so this is very likely chance. Not kept as a main
result.

### What surprised me most
Rest days made predictions worse. I expected rest to matter for
strength, but the model can't tell whether a gap was planned rest or
illness, so it can't tell which way to adjust. This matches Objection 3
from Phase 1.

## Entry 5 — 2026-10-25 — Robustness checks and threats to validity

### Method
The dataset is rebuilt from the original log in three ways, and both the
baseline and Ridge are re-run on each version, so they are always
compared on the same data. Code: src/robustness.py

### Results (walk-forward MAE, kg)
| Check | Predictions | Baseline | Ridge | Ridge wins | p |
|---|---|---|---|---|---|
| 1. Reference (Epley, all reps) | 244 | 3.76 | 4.15 | 0 of 8 | <0.001 |
| 2. Top sets <=10 reps only | 115 | 5.44 | 7.26 | 1 of 8 | 0.001 |
| 3. Brzycki, <=10 reps | 115 | 5.58 | 7.56 | 1 of 8 | <0.001 |

1. The reference rebuild reproduces Days 2-3 (3.76 exactly; Ridge 4.15
   vs 4.14 from rounding differences), so the rebuild matches the pipeline.
2. Keeping only sets within Brzycki's reliable range (<=10 reps), Ridge
   loses by even more. This check costs a lot of data: lateral raises
   are mostly done at 12+ reps, so Dumbbell Lateral Raise drops from 50
   to 9 sessions and the number of predictions halves.
3. Brzycki instead of Epley changes almost nothing (5.58 vs 5.44), so
   the choice of formula does not drive the result.

Conclusion: the main finding holds in every check.

### Comparison with Greig et al. (2023)
The baseline's error is 5.4% of the lifted weight, below Greig et al.'s
~9.8% for load-velocity 1RM prediction. They measured a different task
(a one-off 1RM test, not next-session prediction), so this is context,
not a direct comparison.

### Gym change
From Oct 2024 to Oct 2025 I trained at almost always the same gym. From
Oct 2025 I rotated between 3-4 gyms per month. I expected this to make
predictions worse, but the baseline's error was actually lower after the
change (3.10 kg, 4.3%, n = 102) than before it (4.23 kg, 7.5%, n = 142).
A likely reason is that the earlier period also had bigger swings in
progress, so the gym change is confounded with training stage and its
effect can't be separated.
(Dropped on 2026-10-29: because the effect can't be separated from
training stage, this comparison is not in the final report, and its
code was removed from src/robustness.py and src/figures.py. The
October 2025 change is still marked in Figure 1 and listed under
threats to validity.)

### Threats to validity
- Single athlete: results describe me, not lifters in general.
- Small samples: 22-40 predictions per exercise, so per-exercise results
  are fragile.
- Self-logged data: weights and reps typed by hand; two typos were found
  and removed, others may remain.
- Missing causes: no sleep, effort (RPE) or bodyweight logged.
- Gym and equipment changes: from Oct 2025 I rotated between 3-4 gyms,
  and the same exercise on a different machine can give a different
  weight.
- The <=10-rep check loses much of the data, especially lateral raises.
- Ambiguous gaps: days_since_prev_session can't tell rest from illness.

## Entry 6 — 2026-10-27 — Sleep test

### Why
Sleep was one of my original predictors, and missing sleep data was one
of my explanations for why the models fail. I have Apple Watch sleep
data from late June 2025, so I tested it.

### Data
- Source: Apple Health export (data/raw/sleep_data.csv), Apple Watch
  records only. Left out: "Pillow" (records the same nights as the watch,
  would double-count) and older iPhone records (time in bed, not sleep).
- Nightly sleep = hours of Core + Deep + REM + Unspecified sleep.
  Nights under 2 hours treated as missing (watch off for part of night).
- 396 usable nights, 2025-06-23 to 2026-09-18, mean 6.93 h.
- Features: sleep_last_night (night before the predicted session) and
  sleep_avg_3 (mean of the 3 nights before it). Since last night's sleep
  is only known on the day of the next workout, these predictions are
  framed as "made on the morning of the next workout, before lifting".
Code: src/sleep.py

### Results (walk-forward MAE, kg; only sessions with sleep data)
| Method | n | MAE |
|---|---|---|
| Baseline (same as last session) | 82 | 3.70 |
| Ridge, sleep only | 82 | 4.37 |
| Ridge, 6 features | 82 | 4.64 |
| Ridge, 6 features + sleep | 82 | 4.91 |

Adding sleep to the 6-feature model: +0.27 kg (paired Wilcoxon
p = 0.64, not significant). Correlation of last night's sleep with the
next-session change: -0.07.

### Conclusion
No sign that sleep improves the prediction in this data. This is a weak
test: 82 predictions over about 11 months, with models trained on little
history. So the claim is "no effect detected", not "sleep doesn't affect
strength".

### Note for publishing
Sleep data is personal health data. Plan: keep sleep_data.csv out of the
public repo and say in the README that it's available on request.
