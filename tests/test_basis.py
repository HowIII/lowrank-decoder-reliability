import numpy as np
import pytest

from lowrank_reliability.audit import basis


def test_second_moment_weights_each_perturbation_equally():
    rng = np.random.default_rng(0)
    h1, h2 = rng.normal(size=(3, 5)), rng.normal(size=(30, 5))
    expected = 0.5 * (h1.T @ h1 / 3 + h2.T @ h2 / 30)
    np.testing.assert_allclose(basis.second_moment({"b": h2, "a": h1}), expected, atol=1e-12)
    np.testing.assert_allclose(basis.second_moment([h1, h2]), expected, atol=1e-12)


def test_fixed_basis_orthonormal_sorted_and_canonical():
    rng = np.random.default_rng(1)
    m = basis.second_moment([rng.normal(size=(50, 8)) * np.arange(1, 9)])
    v, ev = basis.fixed_basis(m, k=4)
    np.testing.assert_allclose(v.T @ v, np.eye(4), atol=1e-10)
    assert np.all(np.diff(ev) <= 0) and np.all(ev >= 0)
    for j in range(4):
        np.testing.assert_allclose(m @ v[:, j], ev[j] * v[:, j], atol=1e-8)
        assert v[np.argmax(np.abs(v[:, j])), j] > 0


def test_energy_fraction():
    ev = np.array([4.0, 3.0, 2.0, 1.0])
    assert basis.energy_fraction(ev, 1) == pytest.approx(0.4)
    assert basis.energy_fraction(ev, 4) == pytest.approx(1.0)


def test_random_orthonormal_basis_is_seeded_and_nested():
    a = basis.random_orthonormal_basis(20, 6, seed=3)
    b = basis.random_orthonormal_basis(20, 6, seed=3)
    c = basis.random_orthonormal_basis(20, 6, seed=4)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    np.testing.assert_allclose(a.T @ a, np.eye(6), atol=1e-12)


def test_spectral_summaries():
    rng = np.random.default_rng(2)
    rank1 = np.outer(rng.normal(size=10), rng.normal(size=6))
    assert basis.rank1_fraction(rank1) == pytest.approx(1.0)
    assert basis.participation_ratio(np.ones(5)) == pytest.approx(5.0)
    assert basis.participation_ratio(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_donor_directions_match_svd():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(12, 7))
    v, s = basis.donor_directions({"p": x}, n_components=3)
    _, s_ref, vt = np.linalg.svd(x, full_matrices=False)
    np.testing.assert_allclose(np.abs(v["p"]), np.abs(vt[:3]), atol=1e-6)
    np.testing.assert_allclose(s["p"], s_ref[:3], rtol=1e-6)
    small, _ = basis.donor_directions({"q": x[:2]}, n_components=3)
    assert np.all(small["q"][2] == 0)


def test_ragged_activations():
    rows = np.arange(12.0).reshape(6, 2)
    blocks = basis.ragged_activations(rows, [0, 2, 6], ["a", "b"])
    np.testing.assert_array_equal(blocks["a"], rows[:2])
    np.testing.assert_array_equal(blocks["b"], rows[2:])
    with pytest.raises(ValueError):
        basis.ragged_activations(rows, [0, 6], ["a", "b"])
