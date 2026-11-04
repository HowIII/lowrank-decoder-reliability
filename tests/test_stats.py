import numpy as np
import pytest

from lowrank_reliability import stats


def test_paired_bootstrap_of_identical_inputs_is_zero():
    a = {f"g{i}": float(i) for i in range(30)}
    r = stats.paired_bootstrap(a, dict(a))
    assert r.estimate == 0.0 and r.low == 0.0 and r.high == 0.0 and r.n == 30


def test_bootstrap_of_constant_values():
    r = stats.bootstrap_mean_ci(np.full(40, 0.3))
    assert r.estimate == pytest.approx(0.3) and r.low == pytest.approx(0.3) and r.high == pytest.approx(0.3)


def test_paired_bootstrap_uses_shared_finite_support():
    a = {"x": 1.0, "y": 2.0, "z": np.nan, "only_a": 5.0}
    b = {"x": 0.5, "y": 1.0, "z": 0.0}
    names, diff = stats.paired_differences(a, b)
    assert names == ["x", "y"]
    np.testing.assert_allclose(diff, [0.5, 1.0])
    assert stats.paired_bootstrap(a, b).n == 2


def test_bootstrap_is_seeded_and_covers_the_mean():
    x = np.random.default_rng(5).normal(loc=1.0, scale=1.0, size=400)
    r1 = stats.bootstrap_mean_ci(x, n_boot=2000, seed=3)
    r2 = stats.bootstrap_mean_ci(x, n_boot=2000, seed=3)
    assert r1 == r2
    assert r1.low < x.mean() < r1.high
    assert r1.high - r1.low == pytest.approx(2 * 1.96 * x.std() / np.sqrt(x.size), rel=0.15)


def test_chunked_draws_equal_a_single_draw():
    x = np.random.default_rng(1).normal(size=101)
    single = x[np.random.default_rng(7).integers(0, x.size, size=(1000, x.size))].mean(axis=1)
    np.testing.assert_array_equal(stats.bootstrap_means(x, 1000, seed=7, chunk_rows=64), single)


def test_gene_family_buckets():
    cases = {
        "MRPL12": "mitoribosomal", "MRPS5": "mitoribosomal", "RPL5": "ribosomal_60S_40S",
        "RPS6": "ribosomal_60S_40S", "NDUFA1": "oxphos_mito", "MT-ND1": "oxphos_mito",
        "TUFM": "mito_translation", "POLR2A": "rna_pol", "POLA1": "cell_cycle",
        "CDK1": "cell_cycle", "EIF3A": "translation_init", "AARS": "translation_init",
        "SF3B1": "splicing", "GAPDH": "other",
    }
    for gene, family in cases.items():
        assert stats.gene_family(gene) == family
    assert set(cases.values()) <= set(stats.FAMILIES)
    assert len(stats.FAMILIES) == 9


def test_clustered_bootstrap_estimate_is_mean_of_cluster_means():
    values = np.array([1.0, 1.0, 1.0, 1.0, 3.0])
    clusters = ["a", "a", "a", "a", "b"]
    r = stats.clustered_bootstrap(values, clusters, n_boot=500)
    assert r.estimate == pytest.approx(2.0)
    assert r.n_clusters == 2 and r.n == 5
    assert r.cluster_means == {"a": 1.0, "b": 3.0}
    assert 1.0 <= r.low <= r.high <= 3.0


def test_paired_clustered_bootstrap_groups_by_family():
    a = {"RPL5": 1.0, "RPS6": 1.0, "CDK1": 0.0, "GAPDH": 0.5}
    b = {k: 0.0 for k in a}
    r = stats.paired_clustered_bootstrap(a, b)
    assert r.cluster_means == {"cell_cycle": 0.0, "other": 0.5, "ribosomal_60S_40S": 1.0}
    assert r.estimate == pytest.approx(0.5)


def test_simultaneous_intervals_contain_pointwise_ones():
    rng = np.random.default_rng(2)
    diff = rng.normal(size=(4, 300)) * np.array([[0.5], [1.0], [1.5], [2.0]])
    sim = stats.simultaneous_bootstrap(diff, n_boot=2000, seed=1)
    np.testing.assert_allclose(sim.estimates, diff.mean(axis=1))
    for row, lo, hi in zip(diff, sim.low, sim.high):
        point = stats.bootstrap_mean_ci(row, n_boot=2000, seed=1)
        assert lo <= point.low + 1e-3 and point.high - 1e-3 <= hi
    one = stats.simultaneous_bootstrap(diff[:1], n_boot=2000, seed=1)
    assert one.radius <= sim.radius


def test_interval_helpers():
    r = stats.Interval(0.01, -0.005, 0.015, 10, 100, 0.95)
    assert r.within(0.02) and not r.within(0.01) and not r.excludes_zero
