"""Acceptance thresholds for the predicted-NN-distance.

A prediction is accepted when its predicted-NN-distance is at most a
threshold tau (Definition 2). This module sets tau in three ways:

* ``isotonic_threshold``: the smallest distance at which an isotonic
  (decreasing) fit of PDC on distance reaches a target, usually the rank-1 SVD
  baseline plus 0.02 (Definition 2 and the appendix calibration rule).
* ``learn_then_test``: the largest threshold on a fixed grid whose one-sided
  Hoeffding bound, Bonferroni-corrected over the grid, certifies a target mean
  PDC among accepted predictions at confidence 1 - delta.
* ``pooled_threshold``: one threshold shared by several configurations.

It also provides the evaluation helpers used for threshold transfer between
configurations and for repeated calibration/deployment splits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression

from .metrics import Responses, pdc as _pdc

__all__ = [
    "LTT_GRID",
    "rank_k_baseline",
    "isotonic_threshold",
    "bounded_loss",
    "hoeffding_ucb",
    "LTTResult",
    "learn_then_test",
    "TrustPrefix",
    "trust_prefix",
    "decile_prefix_threshold",
    "evaluate_threshold",
    "SplitSummary",
    "random_split_study",
    "ltt_rule",
    "decile_rule",
    "isotonic_rule",
    "ConfigScores",
    "accepted_gain",
    "transfer_matrix",
    "PooledThreshold",
    "pooled_threshold",
    "pooled_threshold_bootstrap",
]

#: The fixed Learn-then-Test grid: 201 cosine distances from 0 to 2 in steps of 0.01.
LTT_GRID = np.linspace(0.0, 2.0, 201)


def _finite_pair(distance, pdc) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(distance, dtype=np.float64).ravel()
    p = np.asarray(pdc, dtype=np.float64).ravel()
    if d.shape != p.shape:
        raise ValueError(f"distance and pdc lengths differ: {d.shape} vs {p.shape}")
    keep = np.isfinite(d) & np.isfinite(p)
    return d[keep], p[keep]


# ---------------------------------------------------------------------------
# Reference level
# ---------------------------------------------------------------------------
def rank_k_baseline(responses, k: int = 1) -> float:
    """Mean PDC between each response and its rank-k SVD reconstruction.

    ``responses`` is a perturbations-by-genes response matrix. The paper's
    acceptance target is this value at ``k=1`` plus 0.02, computed on the
    held-out responses of the cell line.
    """
    m = responses.values if isinstance(responses, Responses) else responses
    m = np.asarray(m, dtype=np.float64)
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    recon = (u[:, :k] * s[:k]) @ vt[:k]
    return float(np.nanmean(_pdc(recon, m)))


# ---------------------------------------------------------------------------
# Isotonic rule (Definition 2)
# ---------------------------------------------------------------------------
def isotonic_threshold(distance, pdc, target: float, grid_size: int = 5000) -> float:
    """Smallest distance at which the isotonic fit of PDC on distance reaches ``target``.

    The fit is decreasing in distance. It is evaluated on ``grid_size`` evenly
    spaced distances between the smallest and largest observed distance, and
    the crossing is interpolated linearly between the last grid point above
    ``target`` and the first at or below it. If the fit stays above
    ``target`` everywhere the largest distance is returned (every prediction
    is accepted); if it is at or below ``target`` everywhere the smallest
    distance is returned. Fewer than five finite pairs give NaN.
    """
    x, y = _finite_pair(distance, pdc)
    if x.size < 5:
        return float("nan")
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(x, y)
    xs = np.linspace(x.min(), x.max(), grid_size)
    ys = iso.predict(xs)
    below = ys <= target
    if not below.any():
        return float(xs.max())
    if below.all():
        return float(xs.min())
    i = int(np.argmax(below))
    if i == 0:
        return float(xs[0])
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


# ---------------------------------------------------------------------------
# Learn-then-Test
# ---------------------------------------------------------------------------
def bounded_loss(pdc) -> np.ndarray:
    """Loss ``(1 - PDC) / 2`` in [0, 1]; mean loss <= (1 - alpha)/2 iff mean PDC >= alpha."""
    return (1.0 - np.clip(np.asarray(pdc, dtype=np.float64), -1.0, 1.0)) / 2.0


def hoeffding_ucb(losses, delta: float) -> float:
    """One-sided (1 - delta) Hoeffding upper bound on the mean of losses in [0, 1]."""
    losses = np.asarray(losses, dtype=np.float64)
    n = losses.size
    if n == 0:
        return 1.0
    return float(losses.mean() + np.sqrt(np.log(1.0 / delta) / (2.0 * n)))


@dataclass(frozen=True)
class LTTResult:
    """Outcome of Learn-then-Test on one calibration set.

    ``threshold`` is None when no grid point is certified.
    """

    threshold: float | None
    n_accepted: int
    upper_bound: float | None
    alpha: float
    delta: float

    @property
    def certified(self) -> bool:
        return self.threshold is not None


def learn_then_test(
    distance,
    pdc,
    alpha: float = 0.30,
    delta: float = 0.10,
    grid: Sequence[float] = LTT_GRID,
    min_accepted: int = 100,
) -> LTTResult:
    """Largest grid threshold that certifies mean PDC >= ``alpha`` among accepted predictions.

    For each threshold on ``grid`` the calibration perturbations with
    distance <= threshold are accepted, and the one-sided Hoeffding bound on
    their mean loss ``(1 - PDC)/2`` is computed at level ``delta / len(grid)``
    (Bonferroni over the grid). A threshold is certified when the bound is at
    most ``(1 - alpha)/2``. Thresholds that accept fewer than
    ``min_accepted`` calibration perturbations are skipped; the paper uses
    100. The grid does not depend on the calibration data, so the returned
    threshold satisfies E[PDC | distance <= threshold] >= ``alpha`` with
    probability at least 1 - ``delta`` when calibration and deployment
    perturbations are drawn i.i.d. from the same population.
    """
    d, p = _finite_pair(distance, pdc)
    grid = np.sort(np.asarray(grid, dtype=np.float64).ravel())
    target = (1.0 - alpha) / 2.0
    loss = bounded_loss(p)
    per_threshold = delta / grid.size
    best_tau, best_n, best_ucb = None, 0, None
    for tau in grid:
        accepted = d <= tau
        n = int(accepted.sum())
        if n < min_accepted:
            continue
        ucb = hoeffding_ucb(loss[accepted], per_threshold)
        if ucb <= target:
            best_tau, best_n, best_ucb = float(tau), n, ucb
    return LTTResult(best_tau, best_n, best_ucb, alpha, delta)


# ---------------------------------------------------------------------------
# Decile rules
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrustPrefix:
    """Trust prefix D and the decile table it was read from."""

    D: int
    threshold: float
    deciles: list[dict] = field(default_factory=list)


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int, level: float):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    tail = (1.0 - level) / 2.0
    return float(values.mean()), float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def trust_prefix(
    distance,
    pdc,
    target: float,
    n_bins: int = 10,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> TrustPrefix:
    """Trust prefix D (appendix definition).

    Perturbations are split into ``n_bins`` equal-count bins by distance
    (bins of the average ranks, as ``pandas.qcut``), nearest first. Each bin
    gets the mean PDC and a percentile-bootstrap interval at ``level`` from
    ``n_boot`` resamples (bin ``j`` uses seed ``seed + j``). D is the number of
    leading bins whose lower bound exceeds ``target``; ``threshold`` is the
    largest distance in bin D (NaN when D = 0).
    """
    x, y = _finite_pair(distance, pdc)
    if x.size < n_bins:
        return TrustPrefix(0, float("nan"), [])
    ranks = rankdata(x, method="average")
    edges = np.unique(np.quantile(ranks, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:  # all distances tied: pandas.qcut assigns no bin
        return TrustPrefix(0, float("nan"), [])
    labels = np.searchsorted(edges, ranks, side="left") - 1
    labels[ranks == edges[0]] = 0
    rows = []
    for j in range(n_bins):
        members = labels == j
        if not members.any():
            continue
        mean, low, high = _bootstrap_ci(y[members], n_boot, seed + j, level)
        rows.append({
            "bin": j,
            "n": int(members.sum()),
            "min_distance": float(x[members].min()),
            "max_distance": float(x[members].max()),
            "mean_pdc": mean,
            "low": low,
            "high": high,
        })
    D = 0
    for row in rows:
        if row["low"] > target:
            D += 1
        else:
            break
    threshold = rows[D - 1]["max_distance"] if D > 0 else float("nan")
    return TrustPrefix(D, threshold, rows)


def decile_prefix_threshold(
    distance,
    pdc,
    target: float,
    n_bins: int = 10,
    n_boot: int = 1000,
    lower_quantile: float = 0.05,
    seed: int = 0,
) -> tuple[float | None, int]:
    """Threshold at the upper edge of the leading deciles whose bootstrap lower bound reaches ``target``.

    Perturbations are sorted by distance and cut into ``n_bins`` equal-count
    bins. Walking from the nearest bin, a bin passes when the
    ``lower_quantile`` quantile of ``n_boot`` bootstrap means is at least
    ``target``; the walk stops at the first failure. Returns the largest
    distance in the last passing bin and the number of perturbations at or
    below it, or ``(None, 0)`` when the first bin fails. This is the
    comparison rule of the appendix Learn-then-Test table.
    """
    x, y = _finite_pair(distance, pdc)
    n = x.size
    if n < n_bins:
        return None, 0
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    rng = np.random.default_rng(seed)
    passed = 0
    for j in range(n_bins):
        values = ys[edges[j]:edges[j + 1]]
        if values.size < 2:
            break
        idx = rng.integers(0, values.size, size=(n_boot, values.size))
        lower = float(np.quantile(values[idx].mean(axis=1), lower_quantile))
        if lower >= target:
            passed = j + 1
        else:
            break
    if passed == 0:
        return None, 0
    tau = float(xs[edges[passed] - 1])
    return tau, int((x <= tau).sum())


# ---------------------------------------------------------------------------
# Calibration / deployment splits
# ---------------------------------------------------------------------------
def evaluate_threshold(threshold: float | None, distance, pdc, alpha: float) -> dict:
    """Accepted share and mean PDC of accepted predictions at ``threshold``.

    ``below_target`` is True when the accepted mean PDC is below ``alpha``.
    A threshold of None accepts nothing and has ``certified=False``.
    """
    d, p = _finite_pair(distance, pdc)
    if threshold is None:
        return {"certified": False, "accepted": 0.0, "n_accepted": 0,
                "mean_pdc": float("nan"), "below_target": False}
    keep = d <= threshold
    n = int(keep.sum())
    if n == 0:
        return {"certified": True, "accepted": 0.0, "n_accepted": 0,
                "mean_pdc": float("nan"), "below_target": False}
    mean = float(p[keep].mean())
    return {"certified": True, "accepted": n / d.size, "n_accepted": n,
            "mean_pdc": mean, "below_target": bool(mean < alpha)}


@dataclass(frozen=True)
class SplitSummary:
    """Summary of one threshold rule over repeated calibration/deployment splits.

    ``n_certified`` counts the splits where the rule returned a threshold and
    ``n_evaluated`` those of them that accepted at least one deployment
    perturbation. ``accepted`` averages the deployment share accepted over all
    splits (0 for a split without a threshold) and
    ``accepted_when_certified`` over the certified splits. ``mean_pdc`` (the
    mean PDC of accepted deployment predictions) and ``below_target_rate``
    (the share with that mean below ``alpha``) average over the evaluated
    splits.
    """

    n_splits: int
    n_certified: int
    n_evaluated: int
    certified_fraction: float
    accepted: float
    accepted_when_certified: float
    mean_pdc: float
    below_target_rate: float


def _summarize(rows: list[dict]) -> SplitSummary:
    certified = [r for r in rows if r["certified"]]
    used = [r for r in certified if r["n_accepted"] > 0]
    n = len(rows)
    return SplitSummary(
        n_splits=n,
        n_certified=len(certified),
        n_evaluated=len(used),
        certified_fraction=len(certified) / n if n else float("nan"),
        accepted=float(np.sum([r["accepted"] for r in used]) / n) if n else float("nan"),
        accepted_when_certified=(
            float(np.mean([r["accepted"] for r in certified])) if certified else float("nan")
        ),
        mean_pdc=float(np.mean([r["mean_pdc"] for r in used])) if used else float("nan"),
        below_target_rate=float(np.mean([r["below_target"] for r in used])) if used else float("nan"),
    )


Rule = Callable[[np.ndarray, np.ndarray, int], "float | None"]


def ltt_rule(
    alpha: float = 0.30,
    delta: float = 0.10,
    grid: Sequence[float] = LTT_GRID,
    min_accepted: int = 100,
) -> Rule:
    """Learn-then-Test as a rule for ``random_split_study``."""
    def rule(d: np.ndarray, p: np.ndarray, split: int) -> float | None:
        return learn_then_test(d, p, alpha, delta, grid, min_accepted).threshold
    return rule


def decile_rule(
    target: float = 0.30,
    n_bins: int = 10,
    n_boot: int = 1000,
    lower_quantile: float = 0.05,
) -> Rule:
    """``decile_prefix_threshold`` as a rule; split ``r`` uses bootstrap seed ``r``."""
    def rule(d: np.ndarray, p: np.ndarray, split: int) -> float | None:
        return decile_prefix_threshold(d, p, target, n_bins, n_boot, lower_quantile, seed=split)[0]
    return rule


def isotonic_rule(target: float, grid_size: int = 5000) -> Rule:
    """``isotonic_threshold`` as a rule."""
    def rule(d: np.ndarray, p: np.ndarray, split: int) -> float | None:
        tau = isotonic_threshold(d, p, target, grid_size)
        return tau if np.isfinite(tau) else None
    return rule


def random_split_study(
    distance,
    pdc,
    rules: Mapping[str, Rule],
    alpha: float = 0.30,
    n_splits: int = 200,
    seed: int = 0,
) -> dict[str, SplitSummary]:
    """Fit each rule on random calibration halves and score it on the other halves.

    Split ``r`` permutes the perturbations with ``numpy.random.default_rng(seed)``,
    takes the first ``n // 2`` as calibration and the rest as deployment, and
    calls ``rule(distance_cal, pdc_cal, r)``. Every rule sees the same splits.
    """
    d, p = _finite_pair(distance, pdc)
    n = d.size
    rng = np.random.default_rng(seed)
    rows: dict[str, list[dict]] = {name: [] for name in rules}
    for r in range(n_splits):
        perm = rng.permutation(n)
        cal, dep = perm[: n // 2], perm[n // 2:]
        for name, rule in rules.items():
            tau = rule(d[cal], p[cal], r)
            rows[name].append(evaluate_threshold(tau, d[dep], p[dep], alpha))
    return {name: _summarize(v) for name, v in rows.items()}


# ---------------------------------------------------------------------------
# Thresholds across configurations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfigScores:
    """Per-perturbation scores of one encoder-cell configuration.

    ``baseline`` is the reference PDC of the configuration (the rank-1 SVD
    baseline in the paper); accepted-set quality is reported relative to it.
    """

    distance: np.ndarray
    pdc: np.ndarray
    baseline: float

    def finite(self) -> tuple[np.ndarray, np.ndarray]:
        return _finite_pair(self.distance, self.pdc)


def accepted_gain(threshold: float, scores: ConfigScores, min_accepted: int = 5) -> float:
    """Mean PDC of predictions accepted at ``threshold`` minus the configuration baseline.

    NaN when fewer than ``min_accepted`` predictions are accepted.
    """
    d, p = scores.finite()
    keep = d <= threshold
    if int(keep.sum()) < min_accepted:
        return float("nan")
    return float(p[keep].mean()) - float(scores.baseline)


def transfer_matrix(
    configs: Mapping[str, ConfigScores],
    thresholds: Mapping[str, float],
    min_accepted: int = 5,
) -> tuple[list[str], np.ndarray]:
    """Gain of each target configuration under each source configuration's threshold.

    Entry ``[i, j]`` applies the threshold fitted on configuration ``i`` to
    configuration ``j`` and returns ``accepted_gain`` on ``j``.
    """
    keys = list(configs)
    out = np.full((len(keys), len(keys)), np.nan)
    for i, source in enumerate(keys):
        for j, target in enumerate(keys):
            out[i, j] = accepted_gain(thresholds[source], configs[target], min_accepted)
    return keys, out


@dataclass(frozen=True)
class PooledThreshold:
    """A threshold shared by several configurations.

    ``all_configs`` is True when every configuration accepted at least
    ``min_accepted`` predictions at the chosen threshold; otherwise the
    threshold maximizes the mean over the configurations that did.
    """

    threshold: float
    mean_gain: float
    all_configs: bool
    grid: np.ndarray
    gains: np.ndarray


def _config_list(configs) -> list[ConfigScores]:
    return list(configs.values()) if isinstance(configs, Mapping) else list(configs)


def pooled_threshold(
    configs: Mapping[str, ConfigScores] | Sequence[ConfigScores],
    n_grid: int = 50,
    quantile_range: tuple[float, float] = (0.02, 0.98),
    min_accepted: int = 5,
    grid: Sequence[float] | None = None,
) -> PooledThreshold:
    """One threshold for all configurations.

    Candidate thresholds are ``n_grid`` evenly spaced values between the
    ``quantile_range`` quantiles of all finite distances pooled across
    configurations (or ``grid`` when given). Each candidate is scored by the
    mean over configurations of ``accepted_gain``; the best candidate among
    those where every configuration accepts at least ``min_accepted``
    predictions is returned, with ties broken toward the smaller threshold.
    """
    items = _config_list(configs)
    if grid is None:
        pooled = np.concatenate([
            np.asarray(c.distance, dtype=np.float64)[np.isfinite(np.asarray(c.distance, dtype=np.float64))]
            for c in items
        ])
        if pooled.size == 0:
            empty = np.zeros(0)
            return PooledThreshold(float("nan"), float("nan"), False, empty, np.zeros((0, len(items))))
        lo, hi = np.quantile(pooled, quantile_range[0]), np.quantile(pooled, quantile_range[1])
        grid = np.linspace(float(lo), float(hi), n_grid)
    grid = np.asarray(grid, dtype=np.float64)
    finite = [c.finite() for c in items]
    gains = np.full((grid.size, len(items)), np.nan)
    for j, ((d, p), c) in enumerate(zip(finite, items)):
        for i, tau in enumerate(grid):
            keep = d <= tau
            if int(keep.sum()) >= min_accepted:
                gains[i, j] = float(p[keep].mean()) - float(c.baseline)
    complete = np.isfinite(gains).all(axis=1)
    strict = np.full(grid.size, np.nan)
    strict[complete] = gains[complete].mean(axis=1)
    lax = np.array([row[np.isfinite(row)].mean() if np.isfinite(row).any() else np.nan
                    for row in gains])
    for score, all_configs in ((strict, True), (lax, False)):
        ok = np.isfinite(score)
        if ok.any():
            best = np.max(score[ok])
            tied = np.flatnonzero(ok & (np.abs(score - best) < 1e-12))
            pick = tied[np.argmin(grid[tied])]
            return PooledThreshold(float(grid[pick]), float(score[pick]), all_configs, grid, gains)
    return PooledThreshold(float("nan"), float("nan"), False, grid, gains)


def pooled_threshold_bootstrap(
    configs: Mapping[str, ConfigScores] | Sequence[ConfigScores],
    n_boot: int = 5000,
    seed: int = 0,
    level: float = 0.95,
    n_grid: int = 50,
    quantile_range: tuple[float, float] = (0.02, 0.98),
    min_accepted: int = 5,
) -> tuple[np.ndarray, float, float]:
    """Bootstrap distribution of ``pooled_threshold``.

    Each draw resamples configurations with replacement and, within each
    drawn configuration, its perturbations with replacement; the candidate
    grid is rebuilt from the resampled distances. Returns
    ``(thresholds, low, high)`` with the percentile interval at ``level``.
    """
    items = _config_list(configs)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        drawn = rng.integers(0, len(items), size=len(items))
        sample = []
        for i in drawn:
            c = items[i]
            d = np.asarray(c.distance, dtype=np.float64)
            p = np.asarray(c.pdc, dtype=np.float64)
            idx = rng.integers(0, d.size, size=d.size)
            sample.append(ConfigScores(d[idx], p[idx], c.baseline))
        out[b] = pooled_threshold(sample, n_grid, quantile_range, min_accepted).threshold
    finite = out[np.isfinite(out)]
    if finite.size == 0:
        return out, float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    return out, float(np.quantile(finite, tail)), float(np.quantile(finite, 1.0 - tail))
