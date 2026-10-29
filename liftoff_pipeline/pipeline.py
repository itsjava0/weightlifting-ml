"""End-to-end orchestrator: raw export -> cleaned data -> features ->
model -> evaluation, saving intermediate artifacts along the way so each
stage can be inspected independently.
"""

from .config import PipelineConfig
from .ingest import LOADERS
from .clean import clean_sets
from .features import add_set_metrics, build_exercise_session_table, build_supervised_dataset
from .models import DEFAULT_FEATURE_COLUMNS, naive_baselines, time_based_split, train_model
from .evaluate import compare_predictions, plot_predicted_vs_actual


def run_pipeline(config: PipelineConfig) -> dict:
    # 1. ingest
    loader_cls = LOADERS[config.source_app]
    loader = loader_cls(athlete_id=config.athlete_id, weight_unit=config.weight_unit)
    raw_sets = loader.load(config.raw_csv_path)

    # 2. clean
    sets = clean_sets(raw_sets)
    sets = add_set_metrics(sets)
    sets.to_csv(config.output_path("sets_clean.csv"), index=False)

    # 3. features
    exercise_sessions = build_exercise_session_table(sets)
    exercise_sessions.to_csv(config.output_path("exercise_sessions.csv"), index=False)

    supervised = build_supervised_dataset(
        exercise_sessions,
        rolling_windows=config.rolling_windows,
        horizon=config.prediction_horizon,
        min_sessions_per_exercise=config.min_sessions_per_exercise,
    )
    supervised.to_csv(config.output_path("supervised_dataset.csv"), index=False)

    if supervised.empty:
        print(
            "[pipeline] Supervised dataset is empty - not enough per-exercise "
            "history yet given min_sessions_per_exercise="
            f"{config.min_sessions_per_exercise}. Ingestion/cleaning/feature "
            "outputs were still saved; skipping model training."
        )
        return {
            "sets": sets,
            "exercise_sessions": exercise_sessions,
            "supervised": supervised,
        }

    # 4. model
    train_df, test_df = time_based_split(supervised)
    trained = train_model(train_df, test_df, feature_columns=DEFAULT_FEATURE_COLUMNS)

    test_clean = test_df.dropna(subset=trained.feature_columns + ["target"])

    results = {
        "sets": sets,
        "exercise_sessions": exercise_sessions,
        "supervised": supervised,
        "train_df": train_df,
        "test_df": test_df,
        "trained_model": trained,
    }

    if test_clean.empty:
        print("[pipeline] No usable test rows after dropping missing features - "
              "skipping evaluation. Try a longer history or smaller test_frac.")
        return results

    # 5. evaluate
    model_preds = trained.model.predict(test_clean[trained.feature_columns])
    predictions = {**naive_baselines(test_clean), "random_forest": model_preds}

    metrics_table = compare_predictions(test_clean["target"], predictions)
    metrics_table.to_csv(config.output_path("metrics.csv"))

    plot_predicted_vs_actual(
        test_clean["target"],
        model_preds,
        title="Random Forest: predicted vs actual performance_score",
        path=str(config.output_path("pred_vs_actual.png")),
    )

    results["metrics"] = metrics_table
    return results
