from .base import CANONICAL_COLUMNS
from .liftoff_loader import LiftOffLoader

# Registry so the pipeline can pick a loader by name (config.source_app).
# Add a new entry here when a second app's export format is supported
# (e.g. your friend's tracker) - it just needs to return the same
# canonical schema.
LOADERS = {
    "liftoff": LiftOffLoader,
}

__all__ = ["CANONICAL_COLUMNS", "LiftOffLoader", "LOADERS"]
