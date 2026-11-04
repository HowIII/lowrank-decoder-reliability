"""Pseudobulk responses, the Pearson delta correlation (PDC) and the HVG panel.

A pseudobulk response is the mean expression of the cells of one perturbation
minus a control mean. PDC (Eq. 1) is the Pearson correlation across genes
between a predicted and an observed response.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from scipy import sparse

CONTROL = "non-targeting"
N_HVG = 2000

__all__ = [
    "CONTROL",
    "N_HVG",
    "Responses",
    "pseudobulk",
    "control_mean",
    "pseudobulk_response",
    "pdc",
    "pdc_table",
    "hvg_mask",
    "hvg_expression",
    "log1p_cp10k",
]


@dataclass(frozen=True)
class Responses:
    """A matrix with one named row per perturbation.

    ``values[i]`` belongs to ``names[i]``. Names are kept in the order given;
    the constructors in this module sort them lexicographically.
    """

    names: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        names = tuple(str(n) for n in self.names)
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise ValueError(f"values must be 2-D, got shape {values.shape}")
        if values.shape[0] != len(names):
            raise ValueError(
                f"{len(names)} names for {values.shape[0]} rows"
            )
        if len(set(names)) != len(names):
            raise ValueError("names must be unique")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "values", values)

    def __len__(self) -> int:
        return len(self.names)

    def index(self) -> dict[str, int]:
        """Map each name to its row index."""
        return {name: i for i, name in enumerate(self.names)}

    def row(self, name: str) -> np.ndarray:
        """Return the row of one perturbation."""
        return self.values[self.index()[name]]

    def subset(self, names: Iterable[str]) -> "Responses":
        """Rows for ``names``, in that order. Raises KeyError for a missing name."""
        names = [str(n) for n in names]
        idx = self.index()
        rows = [idx[n] for n in names]
        return Responses(tuple(names), self.values[rows])

    def drop(self, names: Iterable[str]) -> "Responses":
        """All rows except those in ``names``."""
        excluded = {str(n) for n in names}
        keep = [n for n in self.names if n not in excluded]
        return self.subset(keep)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {name: self.values[i] for i, name in enumerate(self.names)}


def _as_labels(labels: Iterable, n_rows: int) -> np.ndarray:
    labels = np.asarray(labels).astype(str)
    if labels.ndim != 1 or labels.shape[0] != n_rows:
        raise ValueError(f"expected {n_rows} labels, got shape {labels.shape}")
    return labels


def pseudobulk(X, labels: Iterable) -> Responses:
    """Mean of the rows of ``X`` (cells by genes) for each label.

    ``X`` may be a dense array or a SciPy sparse matrix. Means are computed in
    float64 and returned with labels in sorted order.
    """
    n_rows = X.shape[0]
    labels = _as_labels(labels, n_rows)
    names, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse, minlength=names.size).astype(np.float64)
    indicator = sparse.csr_matrix(
        (np.ones(n_rows, dtype=np.float64), (inverse, np.arange(n_rows))),
        shape=(names.size, n_rows),
    )
    if sparse.issparse(X):
        sums = (indicator @ sparse.csr_matrix(X, dtype=np.float64)).toarray()
    else:
        sums = indicator @ np.asarray(X, dtype=np.float64)
    return Responses(tuple(names.tolist()), sums / counts[:, None])


def control_mean(X, labels: Iterable, control: str = CONTROL) -> np.ndarray:
    """Mean expression of the cells labelled ``control``."""
    labels = _as_labels(labels, X.shape[0])
    mask = labels == control
    if not mask.any():
        raise ValueError(f"no rows labelled {control!r}")
    rows = X[np.flatnonzero(mask)]
    if sparse.issparse(rows):
        return np.asarray(rows.astype(np.float64).mean(axis=0)).ravel()
    return np.asarray(rows, dtype=np.float64).mean(axis=0)


def pseudobulk_response(
    X,
    labels: Iterable,
    control: str = CONTROL,
    control_mean: np.ndarray | None = None,
) -> Responses:
    """Per-perturbation mean expression minus the control mean.

    With ``control_mean=None`` the control mean is the mean of the rows of
    ``X`` labelled ``control``. For predictions this is the mean prediction for
    the control cells, which is how the paper's evaluation scores the model's
    output; pass the cell line's observed control mean to use one vector for
    both sides of Eq. 1. The control row is dropped from the result.
    """
    means = pseudobulk(X, labels)
    if control_mean is None:
        if control not in means.index():
            raise ValueError(f"no rows labelled {control!r}")
        reference = means.row(control)
    else:
        reference = np.asarray(control_mean, dtype=np.float64).ravel()
        if reference.shape[0] != means.values.shape[1]:
            raise ValueError("control_mean length does not match the number of genes")
    kept = [n for n in means.names if n != control]
    return Responses(tuple(kept), means.subset(kept).values - reference[None, :])


def pdc(predicted, observed):
    """Pearson delta correlation between predicted and observed responses.

    Both inputs are responses (expression minus control mean). For 1-D inputs
    the result is a float; for 2-D inputs it is one correlation per row,
    computed across columns (genes). A row with zero variance gives NaN.
    """
    a = np.asarray(predicted, dtype=np.float64)
    b = np.asarray(observed, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    one_dim = a.ndim == 1
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.full(a.shape[0], np.nan)
    ok = np.isfinite(denom) & (denom >= 1e-12)
    out[ok] = (a[ok] * b[ok]).sum(axis=1) / denom[ok]
    return float(out[0]) if one_dim else out


def pdc_table(
    predicted: Responses,
    observed: Responses,
    control: str = CONTROL,
) -> dict[str, float]:
    """PDC for every perturbation present in both tables, in sorted name order."""
    common = sorted((set(predicted.names) & set(observed.names)) - {control})
    values = pdc(predicted.subset(common).values, observed.subset(common).values)
    return {name: float(v) for name, v in zip(common, values)}


def hvg_mask(var, key: str = "highly_variable", n_genes: int | None = N_HVG) -> np.ndarray:
    """Boolean mask of the highly variable genes in a gene table.

    ``var`` is a table indexed by gene with a boolean column ``key`` (for
    example ``adata.var``). When ``n_genes`` is given the mask must select
    exactly that many genes.
    """
    mask = np.asarray(var[key]).astype(bool)
    if n_genes is not None and int(mask.sum()) != n_genes:
        raise ValueError(f"{key!r} selects {int(mask.sum())} genes, expected {n_genes}")
    return mask


def hvg_expression(
    adata,
    obsm_key: str = "X_hvg",
    var_key: str = "highly_variable",
    n_genes: int | None = N_HVG,
) -> np.ndarray:
    """Expression of every cell on the HVG panel as a dense float32 array.

    Uses ``adata.obsm[obsm_key]`` when present and otherwise the HVG columns of
    ``adata.X``. In the Replogle-Nadig files used by the paper the two are
    identical: log(1 + x) of counts per 10^4, normalized over the full library
    and then restricted to the 2,000 HVGs.
    """
    obsm: Mapping = getattr(adata, "obsm", {}) or {}
    if obsm_key in obsm:
        values = obsm[obsm_key]
    else:
        values = adata.X[:, hvg_mask(adata.var, var_key, n_genes)]
    if sparse.issparse(values):
        values = values.toarray()
    values = np.asarray(values, dtype=np.float32)
    if n_genes is not None and values.shape[1] != n_genes:
        raise ValueError(f"HVG matrix has {values.shape[1]} genes, expected {n_genes}")
    return values


def log1p_cp10k(counts, target_sum: float = 1e4):
    """log(1 + x) of counts scaled to ``target_sum`` per cell.

    Each cell is scaled by its total count over all genes, so the panel should
    be selected after this step. Dense input returns a float64 array; sparse
    input returns a CSR matrix.
    """
    if sparse.issparse(counts):
        m = sparse.csr_matrix(counts, dtype=np.float64)
        totals = np.asarray(m.sum(axis=1)).ravel()
        scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
        m = sparse.diags(scale) @ m
        m.data = np.log1p(m.data)
        return sparse.csr_matrix(m)
    x = np.asarray(counts, dtype=np.float64)
    totals = x.sum(axis=1, keepdims=True)
    scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
    return np.log1p(x * scale)
