"""Scoring, calibration and audit tools for perturbation-response predictions.

Modules
-------
metrics    pseudobulk responses, PDC (Eq. 1) and the 2,000-HVG panel
score      predicted-NN-distance (Definition 1) and the oracle distance
calibrate  isotonic rule (Definition 2), Learn-then-Test, threshold transfer
stats      paired, family-clustered and simultaneous bootstrap intervals
audit      keep-k projections, direction patching and the ridge read-out
"""
from . import audit, calibrate, metrics, score, stats
from .calibrate import learn_then_test, isotonic_threshold, rank_k_baseline
from .metrics import Responses, pdc, pseudobulk_response
from .score import oracle_nn_distance, predicted_nn_distance

__version__ = "0.1.0"

__all__ = [
    "audit",
    "calibrate",
    "metrics",
    "score",
    "stats",
    "Responses",
    "pdc",
    "pseudobulk_response",
    "predicted_nn_distance",
    "oracle_nn_distance",
    "rank_k_baseline",
    "isotonic_threshold",
    "learn_then_test",
    "__version__",
]
