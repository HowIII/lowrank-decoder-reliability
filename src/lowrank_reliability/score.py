"""Predicted-NN-distance (Definition 1) and related nearest-neighbour scores.

All distances are cosine distances, ``1 - a.b / (|a| |b|)``, between response
vectors on the same gene panel. Norms are floored at 1e-12, so a zero vector
is at distance 1 from every other vector. Computations run in float64 and in
row chunks, so memory stays at ``chunk_size x n_train`` floats.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .metrics import CONTROL, Responses, pseudobulk

__all__ = [
    "cosine_distances",
    "nearest_neighbor",
    "predicted_nn_distance",
    "oracle_nn_distance",
    "bottom_k_mean_distance",
    "bottom_percent_k",
    "training_table",
]

_EPS = 1e-12


def _unit_rows(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(norms, _EPS)


def _as_matrix(a, name: str) -> np.ndarray:
    if isinstance(a, Responses):
        a = a.values
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {a.shape}")
    return a


def _training_matrix(table) -> np.ndarray:
    t = _as_matrix(table, "training table")
    if t.shape[0] == 0:
        raise ValueError("training table is empty")
    if not np.isfinite(t).all():
        raise ValueError("training table contains non-finite values")
    return t


def _chunks(n: int, chunk_size: int):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


def cosine_distances(A, B, chunk_size: int = 1024) -> np.ndarray:
    """Cosine distance between every row of ``A`` and every row of ``B``."""
    a = _as_matrix(A, "A")
    b = _unit_rows(_as_matrix(B, "B"))
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"gene dimension mismatch {a.shape[1]} vs {b.shape[1]}")
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.float64)
    for start, stop in _chunks(a.shape[0], chunk_size):
        out[start:stop] = 1.0 - _unit_rows(a[start:stop]) @ b.T
    return out


def nearest_neighbor(queries, table, chunk_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Distance to, and index of, the nearest row of ``table`` for each query.

    A query row with a non-finite entry gets distance NaN and index -1.
    """
    q = _as_matrix(queries, "queries")
    t = _unit_rows(_training_matrix(table))
    if q.shape[1] != t.shape[1]:
        raise ValueError(f"gene dimension mismatch {q.shape[1]} vs {t.shape[1]}")
    dist = np.full(q.shape[0], np.nan)
    index = np.full(q.shape[0], -1, dtype=np.int64)
    finite = np.isfinite(q).all(axis=1)
    rows = np.flatnonzero(finite)
    for start, stop in _chunks(rows.size, chunk_size):
        sel = rows[start:stop]
        d = 1.0 - _unit_rows(q[sel]) @ t.T
        j = d.argmin(axis=1)
        index[sel] = j
        dist[sel] = d[np.arange(sel.size), j]
    return dist, index


def predicted_nn_distance(
    predicted,
    training,
    return_index: bool = False,
    chunk_size: int = 1024,
):
    """Predicted-NN-distance of each predicted response (Definition 1, Eq. 5).

    ``predicted`` holds predicted responses (one row per test perturbation) and
    ``training`` the training response table, one row per training
    perturbation, on the same genes. The score needs only the model output and
    the cached table.
    """
    dist, index = nearest_neighbor(predicted, training, chunk_size)
    return (dist, index) if return_index else dist


def oracle_nn_distance(
    observed,
    training,
    return_index: bool = False,
    chunk_size: int = 1024,
):
    """Oracle distance: from each observed response to the training table (Eq. 6)."""
    dist, index = nearest_neighbor(observed, training, chunk_size)
    return (dist, index) if return_index else dist


def bottom_k_mean_distance(queries, table, k: int, chunk_size: int = 1024) -> np.ndarray:
    """Mean of the ``k`` smallest distances from each query to the table.

    ``k=1`` gives the predicted-NN-distance. A query row with a non-finite
    entry gets NaN.
    """
    q = _as_matrix(queries, "queries")
    t = _unit_rows(_training_matrix(table))
    if not 1 <= k <= t.shape[0]:
        raise ValueError(f"k must be in [1, {t.shape[0]}], got {k}")
    out = np.full(q.shape[0], np.nan)
    rows = np.flatnonzero(np.isfinite(q).all(axis=1))
    for start, stop in _chunks(rows.size, chunk_size):
        sel = rows[start:stop]
        d = 1.0 - _unit_rows(q[sel]) @ t.T
        if k < d.shape[1]:
            d = np.partition(d, k - 1, axis=1)[:, :k]
        out[sel] = np.sort(d, axis=1)[:, :k].mean(axis=1)
    return out


def bottom_percent_k(percent: float, n_train: int) -> int:
    """Number of neighbours in the bottom ``percent`` of a table of ``n_train`` rows."""
    return max(1, int(np.ceil(percent / 100.0 * n_train)))


def training_table(
    X,
    labels: Iterable,
    train_names: Iterable[str],
    control: str = CONTROL,
    control_mean: np.ndarray | None = None,
) -> Responses:
    """Training response table: mean expression of each training perturbation minus the control mean.

    ``X`` holds observed expression (cells by genes) for the cell line,
    ``labels`` the perturbation of each cell and ``train_names`` the training
    perturbations. The control mean defaults to the mean of all cells labelled
    ``control`` in ``X``. Training perturbations without cells are dropped.
    """
    means = pseudobulk(X, labels)
    index = means.index()
    if control_mean is None:
        if control not in index:
            raise ValueError(f"no rows labelled {control!r}")
        reference = means.row(control)
    else:
        reference = np.asarray(control_mean, dtype=np.float64).ravel()
        if reference.shape[0] != means.values.shape[1]:
            raise ValueError("control_mean length does not match the number of genes")
    names = sorted({str(n) for n in train_names} - {control})
    present = [n for n in names if n in index]
    return Responses(tuple(present), means.subset(present).values - reference[None, :])
