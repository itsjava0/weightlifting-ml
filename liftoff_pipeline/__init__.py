"""
liftoff_pipeline
=================

A reusable pipeline for turning raw set-by-set workout logs (starting with
LiftOff exports) into a model-ready dataset for predicting weightlifting
performance, plus a light modeling/evaluation harness.

Stages (see pipeline.py for the orchestrator):

    ingest    -> raw export file(s)      -> canonical long-format "sets" table
    clean     -> canonical sets table    -> cleaned sets table
    features  -> cleaned sets table      -> session/exercise-level features
                                             + a supervised (X, y) table
    models    -> supervised table        -> trained baseline/ML model
    evaluate  -> model + holdout data    -> metrics + plots

Each stage is a separate module so any piece can be reused or swapped later
(e.g. plugging in a second athlete's data from a different app, or swapping
the baseline model for a real one) without rewriting the rest.
"""

from .config import PipelineConfig

__all__ = ["PipelineConfig"]
__version__ = "0.1.0"
