from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from lowrank_reliability import metrics


def test_pdc_identical_responses_is_one():
    rng = np.random.default_rng(0)
    delta = rng.normal(size=(5, 200))
    np.testing.assert_allclose(metrics.pdc(delta, delta), 1.0, atol=1e-12)
    assert metrics.pdc(delta[0], delta[0]) == pytest.approx(1.0)


def test_pdc_sign_and_affine_invariance():
    rng = np.random.default_rng(1)
    obs = rng.normal(size=300)
    assert metrics.pdc(-obs, obs) == pytest.approx(-1.0)
    assert metrics.pdc(3.0 * obs + 2.0, obs) == pytest.approx(1.0)


def test_pdc_matches_numpy_corrcoef():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(4, 50)), rng.normal(size=(4, 50))
    expected = [np.corrcoef(x, y)[0, 1] for x, y in zip(a, b)]
    np.testing.assert_allclose(metrics.pdc(a, b), expected, atol=1e-12)


def test_pdc_constant_row_is_nan():
    a = np.ones((2, 10))
    b = np.arange(20.0).reshape(2, 10)
    assert np.isnan(metrics.pdc(a, b)).all()


def test_pseudobulk_means_dense_and_sparse():
    X = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0], [0.0, 1.0]])
    labels = ["b", "a", "b", "a"]
    dense = metrics.pseudobulk(X, labels)
    assert dense.names == ("a", "b")
    np.testing.assert_allclose(dense.values, [[1.5, 1.5], [3.0, 2.0]])
    sp = metrics.pseudobulk(sparse.csr_matrix(X), labels)
    np.testing.assert_allclose(sp.values, dense.values)


def test_pseudobulk_response_subtracts_control_and_drops_it():
    X = np.array([[1.0, 1.0], [3.0, 1.0], [5.0, 3.0], [7.0, 5.0]])
    labels = ["non-targeting", "non-targeting", "g1", "g1"]
    resp = metrics.pseudobulk_response(X, labels)
    assert resp.names == ("g1",)
    np.testing.assert_allclose(resp.values, [[4.0, 3.0]])
    fixed = metrics.pseudobulk_response(X, labels, control_mean=np.zeros(2))
    np.testing.assert_allclose(fixed.row("g1"), [6.0, 4.0])


def test_pseudobulk_response_needs_control():
    with pytest.raises(ValueError):
        metrics.pseudobulk_response(np.ones((2, 3)), ["a", "b"])


def test_pdc_table_aligns_names():
    pred = metrics.Responses(("a", "b", "c"), np.array([[1.0, 2.0, 3.0], [1.0, 0.0, 1.0], [3.0, 2.0, 1.0]]))
    obs = metrics.Responses(("c", "a"), np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]))
    table = metrics.pdc_table(pred, obs)
    assert list(table) == ["a", "c"]
    assert table["a"] == pytest.approx(1.0)
    assert table["c"] == pytest.approx(-1.0)


def test_responses_subset_and_drop():
    r = metrics.Responses(("x", "y", "z"), np.arange(6.0).reshape(3, 2))
    assert r.subset(["z", "x"]).names == ("z", "x")
    np.testing.assert_allclose(r.subset(["z"]).values, [[4.0, 5.0]])
    assert r.drop(["y"]).names == ("x", "z")
    with pytest.raises(KeyError):
        r.subset(["missing"])
    with pytest.raises(ValueError):
        metrics.Responses(("a", "a"), np.zeros((2, 2)))


def test_log1p_cp10k_dense_and_sparse():
    counts = np.array([[1.0, 3.0, 0.0], [0.0, 0.0, 0.0], [2.0, 2.0, 4.0]])
    expected = np.log1p(np.array([[2500.0, 7500.0, 0.0], [0.0, 0.0, 0.0], [2500.0, 2500.0, 5000.0]]))
    np.testing.assert_allclose(metrics.log1p_cp10k(counts), expected)
    np.testing.assert_allclose(metrics.log1p_cp10k(sparse.csr_matrix(counts)).toarray(), expected)


def test_hvg_expression_prefers_obsm_and_falls_back_to_mask():
    X = np.arange(12.0).reshape(3, 4)
    var = {"highly_variable": np.array([True, False, True, False])}
    adata = SimpleNamespace(X=X, var=var, obsm={})
    np.testing.assert_allclose(metrics.hvg_expression(adata, n_genes=2), X[:, [0, 2]])
    adata.obsm = {"X_hvg": X[:, :2] * 10}
    np.testing.assert_allclose(metrics.hvg_expression(adata, n_genes=2), X[:, :2] * 10)
    with pytest.raises(ValueError):
        metrics.hvg_mask(var, n_genes=3)
