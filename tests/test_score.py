import numpy as np
import pytest
from scipy.spatial.distance import cdist

from lowrank_reliability import metrics, score

TABLE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])


def test_cosine_distances_match_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(7, 20)), rng.normal(size=(5, 20))
    np.testing.assert_allclose(score.cosine_distances(a, b), cdist(a, b, metric="cosine"), atol=1e-12)


def test_hand_built_nearest_neighbours():
    queries = np.array([
        [2.0, 0.0, 0.0],    # same direction as row 0
        [1.0, 0.1, 0.0],    # close to row 0
        [0.0, 0.0, 1.0],    # orthogonal to every row
        [-1.0, -1.0, 0.0],  # opposite to row 2
        [3.0, 3.0, 0.0],    # same direction as row 2
    ])
    dist, idx = score.predicted_nn_distance(queries, TABLE, return_index=True)
    expected = [0.0, 1.0 - 1.0 / np.sqrt(1.01), 1.0, 1.0 + 1.0 / np.sqrt(2.0), 0.0]
    np.testing.assert_allclose(dist, expected, atol=1e-12)
    np.testing.assert_array_equal(idx, [0, 0, 0, 0, 2])


def test_zero_and_nan_queries():
    dist, idx = score.nearest_neighbor(np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 0.0]]), TABLE)
    assert dist[0] == pytest.approx(1.0)
    assert np.isnan(dist[1]) and idx[1] == -1


def test_training_table_must_be_finite():
    bad = TABLE.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        score.predicted_nn_distance(TABLE, bad)


def test_chunk_size_does_not_change_results():
    rng = np.random.default_rng(1)
    q, t = rng.normal(size=(50, 30)), rng.normal(size=(40, 30))
    d1, i1 = score.nearest_neighbor(q, t, chunk_size=1)
    d2, i2 = score.nearest_neighbor(q, t, chunk_size=1000)
    np.testing.assert_allclose(d1, d2, atol=1e-14)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_allclose(score.cosine_distances(q, t, chunk_size=3), score.cosine_distances(q, t), atol=1e-14)


def test_oracle_and_predicted_use_the_same_distance():
    rng = np.random.default_rng(2)
    q, t = rng.normal(size=(10, 8)), rng.normal(size=(6, 8))
    np.testing.assert_allclose(score.oracle_nn_distance(q, t), score.predicted_nn_distance(q, t))
    np.testing.assert_allclose(score.predicted_nn_distance(q, t), cdist(q, t, metric="cosine").min(axis=1), atol=1e-12)


def test_bottom_k_mean():
    q = np.array([[1.0, 0.1, 0.0]])
    d = score.cosine_distances(q, TABLE)[0]
    np.testing.assert_allclose(score.bottom_k_mean_distance(q, TABLE, 1), d.min())
    np.testing.assert_allclose(score.bottom_k_mean_distance(q, TABLE, 2), np.sort(d)[:2].mean())
    np.testing.assert_allclose(score.bottom_k_mean_distance(q, TABLE, 3), d.mean())
    with pytest.raises(ValueError):
        score.bottom_k_mean_distance(q, TABLE, 4)
    assert score.bottom_percent_k(1.0, 395) == 4
    assert score.bottom_percent_k(0.1, 100) == 1


def test_training_table_from_cells():
    X = np.array([[1.0, 1.0], [1.0, 3.0], [4.0, 2.0], [6.0, 2.0], [0.0, 0.0]])
    labels = ["non-targeting", "non-targeting", "g1", "g1", "test1"]
    table = score.training_table(X, labels, ["g1", "missing"])
    assert table.names == ("g1",)
    np.testing.assert_allclose(table.values, [[4.0, 0.0]])
    resp = metrics.Responses(("q",), np.array([[2.0, 0.0]]))
    assert score.predicted_nn_distance(resp, table)[0] == pytest.approx(0.0)


def test_training_table_checks_the_control_mean_length():
    with pytest.raises(ValueError):
        score.training_table(np.array([[5.0, 8.0]]), ["p"], ["p"], control_mean=np.array([2.0]))
