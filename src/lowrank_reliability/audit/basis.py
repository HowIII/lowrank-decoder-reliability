"""Subspaces of the first attention output X_p and their spectra (NumPy only).

``second_moment`` and ``fixed_basis`` build the frozen training basis of the
fixed-basis arm; ``random_orthonormal_basis`` builds its rank-matched random
control; ``donor_directions`` caches the per-perturbation right singular
vectors used for direction patching.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

__all__ = [
    "second_moment",
    "fixed_basis",
    "energy_fraction",
    "canonicalize_signs",
    "random_orthonormal_basis",
    "rank1_fraction",
    "participation_ratio",
    "donor_directions",
    "ragged_activations",
]


def _activation_blocks(activations) -> list[np.ndarray]:
    if isinstance(activations, Mapping):
        blocks = [activations[k] for k in sorted(activations)]
    else:
        blocks = list(activations)
    if not blocks:
        raise ValueError("no activation blocks given")
    out = []
    for block in blocks:
        b = np.asarray(block)
        out.append(b.reshape(-1, b.shape[-1]))
    return out


def second_moment(activations: Mapping[str, np.ndarray] | Iterable[np.ndarray]) -> np.ndarray:
    """Uncentered second moment of training activations, each perturbation weighted equally.

    ``activations`` maps each training perturbation to its activation rows
    (any shape ending in the hidden dimension H). The result is
    ``(1/P) sum_p (1/n_p) H_p^T H_p`` in float64, symmetrized.
    """
    blocks = _activation_blocks(activations)
    dim = blocks[0].shape[1]
    total = np.zeros((dim, dim), dtype=np.float64)
    for block in blocks:
        if block.shape[1] != dim:
            raise ValueError("all blocks must share the hidden dimension")
        if block.shape[0] == 0:
            raise ValueError("every perturbation needs at least one activation row")
        x = block.astype(np.float64)
        if not np.isfinite(x).all():
            raise ValueError("activations contain non-finite values")
        total += (x.T @ x) / float(block.shape[0])
    moment = total / float(len(blocks))
    return 0.5 * (moment + moment.T)


def canonicalize_signs(matrix: np.ndarray) -> np.ndarray:
    """Flip each column so that its entry of largest magnitude is positive."""
    out = np.array(matrix, dtype=np.float64, copy=True)
    for j in range(out.shape[1]):
        anchor = int(np.argmax(np.abs(out[:, j])))
        if out[anchor, j] < 0:
            out[:, j] *= -1.0
    return out


def fixed_basis(moment: np.ndarray, k: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` eigenvectors of a second-moment matrix.

    Returns ``(basis, eigenvalues)``: ``basis`` is H x k with orthonormal
    columns in decreasing eigenvalue order (signs canonicalized), and
    ``eigenvalues`` holds all eigenvalues in decreasing order with negative
    round-off clipped to 0. The first ``j`` columns form the rank-``j`` fixed
    basis for every ``j <= k``.
    """
    moment = np.asarray(moment, dtype=np.float64)
    values, vectors = np.linalg.eigh(moment)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    basis = canonicalize_signs(vectors[:, order[:k]])
    return basis, values


def energy_fraction(eigenvalues: np.ndarray, k: int) -> float:
    """Share of the second-moment trace in the top ``k`` eigenvalues."""
    values = np.sort(np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0))[::-1]
    return float(values[:k].sum() / values.sum())


def random_orthonormal_basis(dim: int, k: int = 32, seed: int = 0) -> np.ndarray:
    """Haar-random ``dim x k`` orthonormal basis from a seeded Gaussian matrix.

    The basis is the reduced QR factor of a ``numpy.random.default_rng(seed)``
    Gaussian matrix with column signs fixed by the diagonal of R and then
    canonicalized. Its first ``j`` columns give the rank-``j`` random control.
    """
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((dim, k))
    q, r = np.linalg.qr(gaussian, mode="reduced")
    q = q * np.where(np.diag(r) < 0, -1.0, 1.0)[None, :]
    return canonicalize_signs(q)


def rank1_fraction(x: np.ndarray) -> float:
    """Share of squared singular values in the top singular value: sigma_1^2 / sum sigma_j^2."""
    s = np.linalg.svd(np.asarray(x, dtype=np.float64), compute_uv=False)
    return float(s[0] ** 2 / np.sum(s ** 2))


def participation_ratio(singular_values: np.ndarray) -> float:
    """(sum sigma_j^2)^2 / sum sigma_j^4 for the singular values of a matrix."""
    s2 = np.asarray(singular_values, dtype=np.float64) ** 2
    return float(s2.sum() ** 2 / np.sum(s2 ** 2))


def donor_directions(
    activations: Mapping[str, np.ndarray],
    n_components: int = 3,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Top right singular vectors and singular values of each donor activation.

    ``activations`` maps a perturbation to its S x H activation from a cached
    donor forward pass. Returns ``(directions, singular_values)`` with
    ``directions[p]`` of shape ``(n_components, H)`` (rows v_1, v_2, ...) and
    ``singular_values[p]`` of shape ``(n_components,)``, both float32. Missing
    components (S < n_components) are zero.
    """
    directions: dict[str, np.ndarray] = {}
    sigmas: dict[str, np.ndarray] = {}
    for name, x in activations.items():
        x = np.asarray(x, dtype=np.float64)
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        v = np.zeros((n_components, x.shape[1]), dtype=np.float32)
        sig = np.zeros(n_components, dtype=np.float32)
        m = min(n_components, vt.shape[0])
        v[:m] = vt[:m]
        sig[:m] = s[:m]
        directions[str(name)] = v
        sigmas[str(name)] = sig
    return directions, sigmas


def ragged_activations(
    rows: np.ndarray,
    offsets: np.ndarray,
    names: Iterable[str],
) -> dict[str, np.ndarray]:
    """Split concatenated activation rows into one block per perturbation.

    Rows of ``names[i]`` are ``rows[offsets[i]:offsets[i + 1]]``.
    """
    names = [str(n) for n in names]
    offsets = np.asarray(offsets, dtype=np.int64)
    if offsets.size != len(names) + 1:
        raise ValueError("offsets must have one more entry than names")
    return {n: rows[offsets[i]:offsets[i + 1]] for i, n in enumerate(names)}
