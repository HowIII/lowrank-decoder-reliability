import numpy as np
import pytest

from lowrank_reliability.audit import readout


def test_ridge_recovers_a_linear_map():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    W = rng.normal(size=(5, 3))
    Y = X @ W + 2.0
    fit = readout.ridge_fit(X, Y, 1e-8)
    np.testing.assert_allclose(fit[:-1], W, atol=1e-6)
    np.testing.assert_allclose(fit[-1], 2.0, atol=1e-6)
    np.testing.assert_allclose(readout.ridge_predict(fit, X), Y, atol=1e-6)


def test_intercept_is_not_penalized():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 4))
    Y = rng.normal(size=(100, 2)) + 5.0
    W = readout.ridge_fit(X, Y, 1e12)
    np.testing.assert_allclose(readout.ridge_predict(W, X), np.tile(Y.mean(axis=0), (100, 1)), atol=1e-4)


def test_grouped_folds_keep_groups_together():
    groups = [f"p{i % 23}" for i in range(200)]
    folds = readout.grouped_folds(groups, n_folds=5, seed=0)
    for g in set(groups):
        assert len(set(folds[[i for i, x in enumerate(groups) if x == g]])) == 1
    per_fold = [len({g for g, f in zip(groups, folds) if f == k}) for k in range(5)]
    assert max(per_fold) - min(per_fold) <= 1
    np.testing.assert_array_equal(folds, readout.grouped_folds(groups, n_folds=5, seed=0))


def test_embedding_tables_from_a_state_dict():
    W = np.arange(6.0).reshape(2, 3)  # Linear(3 -> 2)
    b = np.array([10.0, 20.0])
    batch = np.eye(4)
    pert_table, batch_table = readout.embedding_tables(
        {"pert_encoder.0.weight": W, "pert_encoder.0.bias": b, "batch_encoder.weight": batch}
    )
    np.testing.assert_allclose(pert_table, W.T + b)
    np.testing.assert_allclose(batch_table, batch)


def _toy_cells():
    # Two batches; control cells, two training perturbations and one test perturbation.
    pert = ["non-targeting"] * 12 + ["g1"] * 6 + ["g2"] * 6 + ["t1"] * 6 + ["rare"] * 2
    batch = (["b0"] * 6 + ["b1"] * 6) + ["b0", "b1"] * 3 + ["b0", "b1"] * 3 + ["b0", "b1"] * 3 + ["b0", "b0"]
    rng = np.random.default_rng(2)
    emb = rng.normal(size=(len(pert), 3)).astype(np.float32)
    expr = rng.normal(size=(len(pert), 4)).astype(np.float32)
    pert_index = {"non-targeting": 0, "g1": 1, "g2": 2, "t1": 3, "rare": 4}
    batch_index = {"b0": 0, "b1": 1}
    pert_table = np.arange(10.0, dtype=np.float32).reshape(5, 2)
    batch_table = np.array([[100.0], [200.0]], dtype=np.float32)
    return pert, batch, emb, expr, pert_index, batch_index, pert_table, batch_table


def test_readout_rows_by_hand():
    pert, batch, emb, expr, pi, bi, pt, bt = _toy_cells()
    train, test, ctrl = readout.readout_rows(emb, expr, pert, batch, pt, pi, bt, bi, {"t1"}, min_cells=3)
    assert set(zip(train.perturbation, train.batch)) == {
        ("non-targeting", "b0"), ("non-targeting", "b1"), ("g1", "b0"), ("g1", "b1"), ("g2", "b0"), ("g2", "b1"),
    }
    assert set(zip(test.perturbation, test.batch)) == {("t1", "b0"), ("t1", "b1")}
    assert "rare" not in train.perturbation  # two cells < min_cells
    p = np.array(pert)
    b = np.array(batch)
    np.testing.assert_allclose(ctrl, expr[p == "non-targeting"].mean(axis=0), rtol=1e-6)
    row = [i for i, (q, g) in enumerate(zip(train.perturbation, train.batch)) if (q, g) == ("g1", "b1")][0]
    basal_b1 = emb[(p == "non-targeting") & (b == "b1")].mean(axis=0)
    np.testing.assert_allclose(train.X[row], np.r_[basal_b1, pt[1], bt[1]], rtol=1e-6)
    np.testing.assert_allclose(train.Y[row], expr[(p == "g1") & (b == "b1")].mean(axis=0), rtol=1e-6)
    assert train.n_cells[row] == 3


def test_aggregate_pdc_weights_batches_by_cells():
    rows = readout.ReadoutRows(
        X=np.zeros((2, 1)), Y=np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
        perturbation=("p", "p"), batch=("b0", "b1"), n_cells=np.array([3, 1]),
    )
    pred = np.array([[1.0, 2.0, 4.0], [0.0, 0.0, 0.0]])
    out = readout.aggregate_pdc(pred, rows, control_mean=np.zeros(3))
    true = (3 * rows.Y[0] + rows.Y[1]) / 4
    agg = 3 * pred[0] / 4
    assert out["p"] == pytest.approx(np.corrcoef(agg, true)[0, 1])


def test_fit_ridge_readout_selects_a_penalty_and_scores_test_rows():
    rng = np.random.default_rng(3)
    n_pert, n_batch, d, g = 40, 3, 6, 30
    W = rng.normal(size=(d, g)).astype(np.float32)
    X, perts, batches = [], [], []
    for p in range(n_pert):
        z = rng.normal(size=d)
        for b in range(n_batch):
            X.append(z + 0.1 * rng.normal(size=d))
            perts.append(f"p{p}")
            batches.append(f"b{b}")
    X = np.asarray(X, dtype=np.float32)
    Y = (X @ W + 0.05 * rng.normal(size=(X.shape[0], g))).astype(np.float32)
    rows = readout.ReadoutRows(X, Y, tuple(perts), tuple(batches), np.full(X.shape[0], 10))
    is_test = np.array([int(p[1:]) >= 30 for p in perts])
    fit = readout.fit_ridge_readout(rows.take(~is_test), rows.take(is_test), np.zeros(g, dtype=np.float32),
                                    grid=(1e-2, 1e0, 1e2, 1e6))
    assert fit.penalty in (1e-2, 1e0, 1e2)
    assert set(fit.cv_scores) == {1e-2, 1e0, 1e2, 1e6}
    assert len(fit.test_pdc) == 10 and fit.mean_pdc > 0.95
