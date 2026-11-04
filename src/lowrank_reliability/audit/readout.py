"""Ridge read-out of the decoder's conditioning embeddings (Eq. 2).

One row per (perturbation p, batch g) pair with at least ``min_cells`` cells:

    x_{p,g} = [ basal_mean_ctrl(g) ; pert_enc(p) ; batch_enc(g) ]

where ``basal_mean_ctrl(g)`` is the mean encoder embedding of the control
cells of batch g, ``pert_enc(p)`` the decoder's perturbation-embedding row and
``batch_enc(g)`` its batch-embedding row. The target is the mean expression of
the pair's cells. A closed-form ridge regression (unpenalized intercept) is
fit on training rows only, with the penalty chosen by grouped cross-validation
on held-out PDC over training perturbations. Predictions are averaged over a
perturbation's batches with weights equal to cell counts and scored with PDC
against the control mean.

Arithmetic follows the dtype of the inputs; ``readout_rows`` returns float32
features and targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..metrics import CONTROL

__all__ = [
    "RIDGE_GRID",
    "ReadoutRows",
    "embedding_tables",
    "readout_rows",
    "ridge_fit",
    "ridge_predict",
    "grouped_folds",
    "aggregate_pdc",
    "select_penalty",
    "RidgeReadout",
    "fit_ridge_readout",
]

#: Ridge penalties searched by grouped cross-validation.
RIDGE_GRID = (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


@dataclass(frozen=True)
class ReadoutRows:
    """Features and targets of (perturbation, batch) rows."""

    X: np.ndarray
    Y: np.ndarray
    perturbation: tuple[str, ...]
    batch: tuple[str, ...]
    n_cells: np.ndarray

    def __len__(self) -> int:
        return len(self.perturbation)

    def take(self, mask: np.ndarray) -> "ReadoutRows":
        idx = np.flatnonzero(mask)
        return ReadoutRows(
            self.X[idx],
            self.Y[idx],
            tuple(self.perturbation[i] for i in idx),
            tuple(self.batch[i] for i in idx),
            self.n_cells[idx],
        )


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def embedding_tables(
    state_dict: Mapping,
    pert_layer: str = "pert_encoder.0",
    batch_key: str = "batch_encoder.weight",
) -> tuple[np.ndarray, np.ndarray]:
    """Perturbation and batch embedding tables of an ST checkpoint.

    The perturbation encoder maps a one-hot vector through ``pert_layer``, so
    the row of perturbation j is ``W[:, j] + b``. The batch table is the batch
    embedding matrix. Values may be NumPy arrays or torch tensors.
    """
    weight = _to_numpy(state_dict[f"{pert_layer}.weight"])
    bias = _to_numpy(state_dict[f"{pert_layer}.bias"])
    pert_table = weight.T + bias[None, :]
    batch_table = _to_numpy(state_dict[batch_key])
    return pert_table, batch_table


def readout_rows(
    embeddings,
    expression,
    perturbations: Iterable,
    batches: Iterable,
    pert_table: np.ndarray,
    pert_index: Mapping[str, int],
    batch_table: np.ndarray,
    batch_index: Mapping[str, int],
    test_perturbations: Iterable[str],
    control: str = CONTROL,
    min_cells: int = 3,
    min_control_cells: int = 5,
) -> tuple[ReadoutRows, ReadoutRows, np.ndarray]:
    """Build training and test rows for the ridge read-out.

    ``embeddings`` (cells x d) are the frozen encoder embeddings, ``expression``
    (cells x genes) the observed expression on the scored genes, and
    ``perturbations`` and ``batches`` the labels of each cell. Rows of
    ``test_perturbations`` form the test set; every other pair, the control
    included, is a training row. A pair is kept when it has at least
    ``min_cells`` cells, its batch has at least ``min_control_cells`` control
    cells, and both labels appear in the embedding indices. Returns
    ``(train, test, control_mean)`` where ``control_mean`` is the mean
    expression of all control cells.
    """
    basal = np.asarray(embeddings, dtype=np.float32)
    y_all = np.asarray(expression, dtype=np.float32)
    pert = np.asarray(perturbations).astype(str)
    batch = np.asarray(batches).astype(str)
    if not (basal.shape[0] == y_all.shape[0] == pert.size == batch.size):
        raise ValueError("embeddings, expression and labels must have one row per cell")
    test = {str(p) for p in test_perturbations}
    is_control = pert == control
    if not is_control.any():
        raise ValueError(f"no cells labelled {control!r}")
    control_mean = y_all[is_control].mean(axis=0)

    basal_mean = {}
    for g in np.unique(batch):
        members = is_control & (batch == g)
        if members.sum() >= min_control_cells:
            basal_mean[g] = basal[members].mean(axis=0)

    pairs: dict[tuple[str, str], list[int]] = {}
    for i, key in enumerate(zip(pert.tolist(), batch.tolist())):
        pairs.setdefault(key, []).append(i)

    parts = {"train": ([], [], [], [], []), "test": ([], [], [], [], [])}
    for (p, g), idx in pairs.items():
        if len(idx) < min_cells:
            continue
        if g not in basal_mean or p not in pert_index or g not in batch_index:
            continue
        x = np.concatenate(
            [basal_mean[g], pert_table[pert_index[p]], batch_table[batch_index[g]]],
            dtype=np.float32,
        )
        y = y_all[idx].mean(axis=0).astype(np.float32)
        xs, ys, ps, gs, ns = parts["test" if p in test else "train"]
        xs.append(x)
        ys.append(y)
        ps.append(p)
        gs.append(g)
        ns.append(len(idx))

    def build(name: str) -> ReadoutRows:
        xs, ys, ps, gs, ns = parts[name]
        if not xs:
            raise ValueError(f"no {name} rows")
        return ReadoutRows(np.stack(xs), np.stack(ys), tuple(ps), tuple(gs),
                           np.asarray(ns, dtype=np.int64))

    return build("train"), build("test"), control_mean


def ridge_fit(X: np.ndarray, Y: np.ndarray, penalty: float) -> np.ndarray:
    """Closed-form ridge weights with an unpenalized intercept (last row of the result)."""
    n, d = X.shape
    x1 = np.hstack([X, np.ones((n, 1), dtype=X.dtype)])
    reg = penalty * np.eye(d + 1, dtype=X.dtype)
    reg[-1, -1] = 0.0
    return np.linalg.solve(x1.T @ x1 + reg, x1.T @ Y)


def ridge_predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Predictions of ``ridge_fit`` weights."""
    x1 = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    return x1 @ W


def grouped_folds(groups: Sequence[str], n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Fold index of each row, keeping all rows of a group in one fold.

    Groups are sorted, permuted with ``numpy.random.default_rng(seed)`` and
    dealt to folds in turn.
    """
    names = sorted(set(groups))
    order = np.random.default_rng(seed).permutation(len(names))
    fold_of = {names[order[i]]: i % n_folds for i in range(len(names))}
    return np.array([fold_of[g] for g in groups], dtype=np.int64)


def aggregate_pdc(
    predicted: np.ndarray,
    rows: ReadoutRows,
    control_mean: np.ndarray,
) -> dict[str, float]:
    """PDC per perturbation after cell-weighted averaging over its batches.

    ``predicted`` holds one predicted expression row per row of ``rows``; the
    observed side uses ``rows.Y``. Perturbations appear in first-row order.
    """
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(rows.perturbation):
        groups.setdefault(p, []).append(i)
    out = {}
    for p, idx in groups.items():
        w = rows.n_cells[idx].astype(np.float32)
        pred = np.average(predicted[idx], axis=0, weights=w)
        true = np.average(rows.Y[idx], axis=0, weights=w)
        pv = pred - control_mean
        tv = true - control_mean
        pv = pv - pv.mean()
        tv = tv - tv.mean()
        denom = np.linalg.norm(pv) * np.linalg.norm(tv)
        out[p] = float((pv @ tv) / denom) if denom > 1e-12 else float("nan")
    return out


def _mean_finite(values: Iterable[float]) -> float:
    vals = [v for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def select_penalty(
    train: ReadoutRows,
    control_mean: np.ndarray,
    grid: Sequence[float] = RIDGE_GRID,
    n_folds: int = 5,
    seed: int = 0,
) -> tuple[float, dict[float, float]]:
    """Penalty with the highest mean held-out PDC under grouped cross-validation.

    Folds group rows by perturbation (``grouped_folds``). For each penalty the
    score is the mean over folds of the mean held-out PDC. Ties go to the
    earlier grid value.
    """
    folds = grouped_folds(train.perturbation, n_folds, seed)
    scores: dict[float, float] = {}
    for penalty in grid:
        per_fold = []
        for f in range(n_folds):
            fit, held = folds != f, folds == f
            if not held.any():
                continue
            w = ridge_fit(train.X[fit], train.Y[fit], penalty)
            held_rows = train.take(held)
            pdcs = aggregate_pdc(ridge_predict(w, held_rows.X), held_rows, control_mean)
            score = _mean_finite(pdcs.values())
            if np.isfinite(score):
                per_fold.append(score)
        scores[float(penalty)] = float(np.mean(per_fold)) if per_fold else float("nan")
    finite = [p for p in scores if np.isfinite(scores[p])]
    if not finite:
        raise ValueError("no penalty produced a finite cross-validation score")
    best = max(finite, key=lambda p: scores[p])
    return best, scores


@dataclass(frozen=True)
class RidgeReadout:
    """Fitted ridge read-out and its test PDC per perturbation."""

    penalty: float
    cv_scores: dict
    weights: np.ndarray
    test_pdc: dict

    @property
    def mean_pdc(self) -> float:
        return _mean_finite(self.test_pdc.values())


def fit_ridge_readout(
    train: ReadoutRows,
    test: ReadoutRows,
    control_mean: np.ndarray,
    grid: Sequence[float] = RIDGE_GRID,
    n_folds: int = 5,
    seed: int = 0,
) -> RidgeReadout:
    """Select the penalty on training rows, refit on all of them and score the test rows."""
    penalty, scores = select_penalty(train, control_mean, grid, n_folds, seed)
    w = ridge_fit(train.X, train.Y, penalty)
    test_pdc = aggregate_pdc(ridge_predict(w, test.X), test, control_mean)
    return RidgeReadout(penalty, scores, w, test_pdc)
