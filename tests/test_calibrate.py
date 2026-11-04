import numpy as np
import pytest

from lowrank_reliability import calibrate as C

LOG_TERM_201 = np.log(201 / 0.10)  # Bonferroni level delta / |grid| with delta = 0.10


def test_bounded_loss_and_hoeffding():
    np.testing.assert_allclose(C.bounded_loss([-1.0, 0.0, 1.0, 2.0]), [1.0, 0.5, 0.0, 0.0])
    assert C.hoeffding_ucb([0.0, 1.0], 0.1) == pytest.approx(0.5 + np.sqrt(np.log(10) / 4))
    assert C.hoeffding_ucb([], 0.1) == 1.0


def test_ltt_grid():
    assert C.LTT_GRID.size == 201
    assert C.LTT_GRID[0] == 0.0 and C.LTT_GRID[-1] == 2.0
    np.testing.assert_allclose(np.diff(C.LTT_GRID), 0.01)


def test_ltt_step_case_by_hand():
    # 400 perfect predictions near training, 600 anti-correlated far away.
    d = np.r_[np.full(400, 0.155), np.full(600, 1.005)]
    p = np.r_[np.ones(400), -np.ones(600)]
    res = C.learn_then_test(d, p, alpha=0.30, delta=0.10)
    # Thresholds 0.16 ... 1.00 accept the 400 good points: bound sqrt(log(2010)/800) = 0.0975 <= 0.35.
    # Thresholds >= 1.01 accept all 1000 points with mean loss 0.6 > 0.35.
    assert res.certified
    assert res.threshold == pytest.approx(1.00)
    assert res.n_accepted == 400
    assert res.upper_bound == pytest.approx(np.sqrt(LOG_TERM_201 / 800))


def test_ltt_returns_none_when_target_is_out_of_reach():
    rng = np.random.default_rng(0)
    res = C.learn_then_test(rng.uniform(0, 1, 500), np.zeros(500), alpha=0.30)
    assert not res.certified and res.threshold is None and res.n_accepted == 0


def test_ltt_bonferroni_over_the_grid():
    # Mean loss 0.27 on 400 accepted points.
    d = np.full(400, 0.505)
    p = np.full(400, 1.0 - 2 * 0.27)
    # Over the 201-point grid: 0.27 + sqrt(log(2010)/800) = 0.3675 > 0.35.
    assert not C.learn_then_test(d, p).certified
    # With a single threshold the bound is 0.27 + sqrt(log(10)/800) = 0.3237 <= 0.35.
    single = C.learn_then_test(d, p, grid=[0.6])
    assert single.threshold == pytest.approx(0.6)
    assert single.upper_bound == pytest.approx(0.27 + np.sqrt(np.log(10) / 800))


def test_ltt_min_accepted():
    d = np.r_[np.full(50, 0.105), np.full(950, 1.505)]
    p = np.r_[np.ones(50), -np.ones(950)]
    assert not C.learn_then_test(d, p, min_accepted=100).certified
    res = C.learn_then_test(d, p, min_accepted=0)
    # 50 perfect points: bound sqrt(log(2010)/100) = 0.276 <= 0.35; largest threshold below 1.505 is 1.50.
    assert res.threshold == pytest.approx(1.50)
    assert res.n_accepted == 50


def test_ltt_drops_non_finite_pairs():
    d = np.r_[np.full(400, 0.155), np.nan, 0.2]
    p = np.r_[np.ones(400), 1.0, np.nan]
    assert C.learn_then_test(d, p).n_accepted == 400


def test_isotonic_threshold_linear_case():
    d = np.linspace(0.0, 1.0, 101)
    assert C.isotonic_threshold(d, 1.0 - d, target=0.6) == pytest.approx(0.4, abs=1e-9)
    assert C.isotonic_threshold(d, 1.0 - d, target=-1.0) == pytest.approx(1.0)
    assert C.isotonic_threshold(d, 1.0 - d, target=2.0) == pytest.approx(0.0)
    assert np.isnan(C.isotonic_threshold(d[:4], d[:4], target=0.5))


def test_isotonic_threshold_pools_violators():
    d = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    p = np.array([0.9, 0.95, 0.5, 0.55, 0.1])
    # Decreasing fit: 0.925, 0.925, 0.525, 0.525, 0.1; crossing 0.7 between d = 1 and d = 2.
    assert C.isotonic_threshold(d, p, target=0.7) == pytest.approx(1.5625, abs=1e-9)


def test_rank_k_baseline():
    rng = np.random.default_rng(3)
    u, v, w = rng.normal(size=20), rng.normal(size=50), rng.normal(size=50)
    rank1 = np.outer(u, v)
    assert C.rank_k_baseline(rank1, k=1) == pytest.approx(1.0)
    rank2 = rank1 + np.outer(rng.normal(size=20), w)
    assert C.rank_k_baseline(rank2, k=2) == pytest.approx(1.0)
    assert C.rank_k_baseline(rank2, k=1) < 1.0


def _separated():
    d = np.linspace(0.01, 1.0, 100)
    p = np.r_[np.full(30, 0.9), np.zeros(70)]
    return d, p


def test_trust_prefix_by_hand():
    d, p = _separated()
    tp = C.trust_prefix(d, p, target=0.5)
    assert tp.D == 3
    assert tp.threshold == pytest.approx(0.30)
    assert [row["n"] for row in tp.deciles] == [10] * 10
    assert C.trust_prefix(d, p, target=0.95).D == 0


def test_decile_prefix_threshold_by_hand():
    d, p = _separated()
    tau, n = C.decile_prefix_threshold(d, p, target=0.5)
    assert tau == pytest.approx(0.30) and n == 30
    assert C.decile_prefix_threshold(d, p, target=0.95) == (None, 0)


def test_evaluate_threshold():
    d = np.array([0.1, 0.2, 0.3, 0.4])
    p = np.array([0.8, 0.6, 0.2, 0.0])
    out = C.evaluate_threshold(0.25, d, p, alpha=0.5)
    assert out["certified"] and out["n_accepted"] == 2
    assert out["accepted"] == pytest.approx(0.5) and out["mean_pdc"] == pytest.approx(0.7)
    assert not out["below_target"]
    assert C.evaluate_threshold(None, d, p, alpha=0.5)["accepted"] == 0.0


def test_random_split_study_on_an_easy_population():
    rng = np.random.default_rng(4)
    d = rng.uniform(0.0, 1.0, 600)
    p = np.full(600, 0.9)
    res = C.random_split_study(
        d, p, {"ltt": C.ltt_rule(), "decile": C.decile_rule(0.30), "iso": C.isotonic_rule(0.5)},
        alpha=0.30, n_splits=5, seed=0,
    )
    assert res["ltt"].n_splits == 5 and res["ltt"].n_certified == 5
    assert res["ltt"].accepted == pytest.approx(1.0)
    assert res["ltt"].mean_pdc == pytest.approx(0.9)
    assert res["ltt"].below_target_rate == 0.0
    assert res["decile"].accepted > 0.95
    assert res["iso"].accepted > 0.95


def _two_configs():
    a = C.ConfigScores(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                       np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]), 0.5)
    b = C.ConfigScores(np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65]),
                       np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]), 0.6)
    return {"A": a, "B": b}


def test_transfer_matrix_by_hand():
    keys, m = C.transfer_matrix(_two_configs(), {"A": 0.5, "B": 0.45})
    assert keys == ["A", "B"]
    assert m[0, 0] == pytest.approx(0.2) and m[0, 1] == pytest.approx(0.2)
    assert m[1, 1] == pytest.approx(0.2)
    assert np.isnan(m[1, 0])  # four accepted predictions, fewer than five


def test_pooled_threshold_by_hand():
    configs = _two_configs()
    best = C.pooled_threshold(configs, grid=[0.2, 0.5, 0.65], min_accepted=2)
    assert best.threshold == pytest.approx(0.2) and best.mean_gain == pytest.approx(0.35)
    assert best.all_configs
    best3 = C.pooled_threshold(configs, grid=[0.2, 0.5, 0.65], min_accepted=3)
    assert best3.threshold == pytest.approx(0.5) and best3.mean_gain == pytest.approx(0.2)
    # 0.21 and 0.2 accept the same predictions; the tie goes to the smaller threshold.
    tied = C.pooled_threshold(configs, grid=[0.21, 0.2], min_accepted=2)
    assert tied.threshold == pytest.approx(0.2)


def test_pooled_threshold_bootstrap_runs():
    taus, low, high = C.pooled_threshold_bootstrap(_two_configs(), n_boot=20, seed=0, min_accepted=2)
    assert taus.shape == (20,) and low <= high


def test_ltt_returns_the_largest_certified_threshold_for_any_grid_order():
    d = np.full(400, 0.1)
    p = np.ones(400)
    assert C.learn_then_test(d, p, grid=[1.0, 0.5]).threshold == pytest.approx(1.0)


def test_trust_prefix_with_tied_distances_has_no_bins():
    tp = C.trust_prefix(np.ones(20), np.ones(20), target=0.5)
    assert tp.D == 0 and np.isnan(tp.threshold) and tp.deciles == []


def test_decile_prefix_count_includes_ties_at_the_threshold():
    tau, n = C.decile_prefix_threshold(np.ones(4), [1.0, 1.0, 0.0, 0.0], 0.5, n_bins=2)
    assert tau == pytest.approx(1.0) and n == 4


def test_split_summary_separates_certified_and_evaluated_splits():
    res = C.random_split_study([0.1, 0.2], [0.9, 0.9], {"r": lambda d, p, s: 0.05}, n_splits=1)
    s = res["r"]
    assert s.n_certified == 1 and s.n_evaluated == 0
    assert s.accepted == 0.0 and np.isnan(s.mean_pdc)


def test_pooled_bootstrap_without_any_threshold():
    one = [C.ConfigScores(np.arange(4.0) / 10, np.ones(4), 0.5)]
    taus, low, high = C.pooled_threshold_bootstrap(one, n_boot=2, seed=0)
    assert np.isnan(taus).all() and np.isnan(low) and np.isnan(high)


def test_pooled_threshold_without_finite_distances():
    c = C.ConfigScores(np.array([np.nan, np.nan]), np.array([0.5, 0.5]), 0.0)
    assert np.isnan(C.pooled_threshold([c]).threshold)
    c2 = C.ConfigScores(np.array([0.1, np.nan]), np.array([0.5, np.nan]), 0.0)
    taus, low, high = C.pooled_threshold_bootstrap([c2], n_boot=3, seed=0, min_accepted=1)
    assert taus.shape == (3,)
