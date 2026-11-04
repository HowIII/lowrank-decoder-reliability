"""Percentile bootstrap intervals: per perturbation, clustered by gene family, and simultaneous.

Paired comparisons take per-perturbation differences on the perturbations
scored by both conditions. The resampling unit is the perturbation, or the
gene family when perturbations of one family are correlated.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Hashable, Mapping, Sequence

import numpy as np

__all__ = [
    "Interval",
    "ClusteredInterval",
    "SimultaneousIntervals",
    "bootstrap_means",
    "bootstrap_mean_ci",
    "paired_differences",
    "paired_bootstrap",
    "FAMILIES",
    "gene_family",
    "clustered_bootstrap",
    "paired_clustered_bootstrap",
    "simultaneous_bootstrap",
]


@dataclass(frozen=True)
class Interval:
    """Point estimate with a percentile-bootstrap interval."""

    estimate: float
    low: float
    high: float
    n: int
    n_boot: int
    level: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def within(self, margin: float) -> bool:
        """True when the whole interval lies in [-margin, +margin]."""
        return self.low >= -margin and self.high <= margin


@dataclass(frozen=True)
class ClusteredInterval(Interval):
    """Interval from resampling clusters; ``estimate`` is the mean of cluster means."""

    n_clusters: int = 0
    cluster_means: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SimultaneousIntervals:
    """Intervals with joint coverage over several contrasts.

    Every contrast gets ``estimate +/- radius``, where ``radius`` is the
    ``level`` quantile of the largest absolute bootstrap deviation across
    contrasts.
    """

    estimates: np.ndarray
    low: np.ndarray
    high: np.ndarray
    radius: float
    n: int
    n_boot: int
    level: float


def bootstrap_means(values, n_boot: int, seed: int, chunk_rows: int = 512) -> np.ndarray:
    """Means of ``n_boot`` resamples (with replacement) of ``values``.

    Resample indices come from ``numpy.random.default_rng(seed).integers`` in
    blocks of ``chunk_rows`` resamples, which yields the same draws as a
    single call of shape ``(n_boot, n)``.
    """
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, chunk_rows):
        size = min(chunk_rows, n_boot - start)
        idx = rng.integers(0, values.size, size=(size, values.size))
        out[start:start + size] = values[idx].mean(axis=1)
    return out


def _percentiles(boot: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    return float(np.quantile(boot, tail)), float(np.quantile(boot, 1.0 - tail))


def bootstrap_mean_ci(values, n_boot: int = 5000, seed: int = 42, level: float = 0.95) -> Interval:
    """Percentile-bootstrap interval for the mean of the finite entries of ``values``."""
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, n_boot, level)
    low, high = _percentiles(bootstrap_means(values, n_boot, seed), level)
    return Interval(float(values.mean()), low, high, int(values.size), n_boot, level)


def paired_differences(a, b) -> tuple[list, np.ndarray]:
    """Per-perturbation differences ``a - b`` on the shared finite support.

    ``a`` and ``b`` are mappings from perturbation name to value (the shared
    names are used in sorted order) or equal-length arrays.
    """
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        names = sorted(set(a) & set(b))
        x = np.array([a[k] for k in names], dtype=np.float64)
        y = np.array([b[k] for k in names], dtype=np.float64)
    else:
        x = np.asarray(a, dtype=np.float64).ravel()
        y = np.asarray(b, dtype=np.float64).ravel()
        if x.shape != y.shape:
            raise ValueError(f"length mismatch {x.shape} vs {y.shape}")
        names = list(range(x.size))
    keep = np.isfinite(x) & np.isfinite(y)
    return [n for n, k in zip(names, keep) if k], x[keep] - y[keep]


def paired_bootstrap(a, b, n_boot: int = 5000, seed: int = 42, level: float = 0.95) -> Interval:
    """Mean of ``a - b`` with a percentile interval from resampling perturbations."""
    _, diff = paired_differences(a, b)
    return bootstrap_mean_ci(diff, n_boot=n_boot, seed=seed, level=level)


#: The nine gene-family buckets used as clusters (Appendix, HGNC-prefix scheme).
FAMILIES = (
    "cell_cycle",
    "mito_translation",
    "mitoribosomal",
    "other",
    "oxphos_mito",
    "ribosomal_60S_40S",
    "rna_pol",
    "splicing",
    "translation_init",
)

_MITO_TRANSLATION = {
    "GFM1", "GFM2", "TSFM", "TFAM", "TFB1M", "TUFM", "OPA1", "LONP1", "LONP2",
    "MRPP1", "MRPP2", "MRPP3", "PNPT1", "FASTKD5", "MRPP4",
}
_OXPHOS = ("NDUF", "COX", "ATP5", "ATP6", "UQCR", "SDHA", "SDHB", "SDHC", "SDHD")
_AARS = (
    "AARS", "DARS", "FARS", "YARS", "IARS", "MARS", "NARS", "RARS", "TARS", "VARS",
    "WARS", "GARS", "KARS", "LARS", "HARS", "EARS", "SARS", "EPRS", "QARS", "PARS",
    "CARS", "DTYM",
)
_CELL_CYCLE = (
    "CDC", "CDK", "CCN", "MCM", "CENP", "BUB", "AURK", "KIF", "PLK", "CHK", "RAD",
    "BRCA", "TOP", "RFC", "CHEK", "ORC", "WEE", "POL",
)


def gene_family(name: str) -> str:
    """Gene-family bucket of a perturbed gene from its HGNC symbol prefix."""
    p = str(name).upper()
    if p.startswith("MT-"):
        return "oxphos_mito"
    if p in _MITO_TRANSLATION:
        return "mito_translation"
    if p.startswith(("MRPL", "MRPS")):
        return "mitoribosomal"
    if p.startswith(("RPL", "RPS", "RPLP", "RACK")):
        return "ribosomal_60S_40S"
    if p.startswith(_OXPHOS):
        return "oxphos_mito"
    if p.startswith(("SNRP", "SF3", "U2AF", "PRPF", "RBM")):
        return "splicing"
    if p.startswith("EIF") or p.startswith(_AARS):
        return "translation_init"
    if p.startswith(_CELL_CYCLE):
        return "rna_pol" if p.startswith("POLR") else "cell_cycle"
    return "other"


def clustered_bootstrap(
    values,
    clusters: Sequence[Hashable],
    n_boot: int = 5000,
    seed: int = 42,
    level: float = 0.95,
) -> ClusteredInterval:
    """Mean of cluster means with an interval from resampling whole clusters.

    The estimate averages the values within each cluster and then averages
    the cluster means. Each bootstrap draw resamples clusters (sorted by
    label) with replacement and averages their means.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    clusters = list(clusters)
    if len(clusters) != values.size:
        raise ValueError("one cluster label per value is required")
    groups: dict = defaultdict(list)
    for c, v in zip(clusters, values):
        if np.isfinite(v):
            groups[c].append(v)
    if not groups:
        return ClusteredInterval(float("nan"), float("nan"), float("nan"), 0, n_boot, level)
    labels = sorted(groups)
    means = np.array([np.mean(groups[c]) for c in labels])
    rng = np.random.default_rng(seed)
    boot = means[rng.integers(0, means.size, size=(n_boot, means.size))].mean(axis=1)
    low, high = _percentiles(boot, level)
    return ClusteredInterval(
        estimate=float(means.mean()), low=low, high=high,
        n=int(sum(len(v) for v in groups.values())), n_boot=n_boot, level=level,
        n_clusters=int(means.size),
        cluster_means={c: float(m) for c, m in zip(labels, means)},
    )


def paired_clustered_bootstrap(
    a: Mapping[str, float],
    b: Mapping[str, float],
    cluster_of: Callable[[str], Hashable] = gene_family,
    n_boot: int = 5000,
    seed: int = 42,
    level: float = 0.95,
) -> ClusteredInterval:
    """Family-clustered interval for the mean of ``a - b`` over shared perturbations."""
    names, diff = paired_differences(a, b)
    return clustered_bootstrap(diff, [cluster_of(n) for n in names], n_boot, seed, level)


def simultaneous_bootstrap(
    differences,
    n_boot: int = 10000,
    seed: int = 0,
    level: float = 0.95,
    chunk_rows: int = 250,
) -> SimultaneousIntervals:
    """Simultaneous percentile intervals for several paired contrasts on one support.

    ``differences`` has one row per contrast and one column per perturbation.
    Each draw resamples perturbations (the same columns for every contrast);
    the radius is the ``level`` quantile of the largest absolute deviation of
    the resampled means from the observed means across contrasts.
    """
    diff = np.asarray(differences, dtype=np.float64)
    if diff.ndim != 2:
        raise ValueError("differences must be contrasts x perturbations")
    m, n = diff.shape
    observed = diff.mean(axis=1)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, m), dtype=np.float64)
    for start in range(0, n_boot, chunk_rows):
        stop = min(start + chunk_rows, n_boot)
        idx = rng.integers(0, n, size=(stop - start, n))
        boot[start:stop] = diff[:, idx].mean(axis=2).T
    radius = float(np.quantile(np.max(np.abs(boot - observed[None, :]), axis=1), level))
    return SimultaneousIntervals(observed, observed - radius, observed + radius, radius, n, n_boot, level)
