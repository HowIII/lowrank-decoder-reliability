"""Score predictions with predicted-NN-distance and calibrate an acceptance threshold.

Runs on synthetic responses in a few seconds:

    python examples/quickstart.py

Replace the synthetic arrays with your model's predicted responses, the
observed responses of labeled calibration perturbations and the training
response table of the same cell line.
"""
import numpy as np
from scipy.stats import spearmanr

from lowrank_reliability import calibrate, metrics, score, stats

rng = np.random.default_rng(0)
n_genes, n_train, n_test = 2000, 400, 1000

# Responses share one axis and differ in a perturbation-specific part.
axis = rng.normal(size=n_genes)
train = np.outer(rng.gamma(2.0, 0.5, n_train), axis) + rng.normal(scale=3.0, size=(n_train, n_genes))

# Half of the test perturbations resemble a training perturbation; the model
# reproduces those and returns only the shared axis for the others.
near = rng.random(n_test) < 0.5
source = rng.integers(0, n_train, n_test)
new = np.outer(rng.gamma(2.0, 0.5, n_test), 0.5 * axis) + rng.normal(scale=3.0, size=(n_test, n_genes))
observed = np.where(near[:, None], train[source] + rng.normal(size=(n_test, n_genes)), new)
predicted = np.where(
    near[:, None],
    observed + rng.normal(size=(n_test, n_genes)),
    np.outer(rng.gamma(2.0, 1.0, n_test), axis) + rng.normal(scale=0.5, size=(n_test, n_genes)),
)

pdc = metrics.pdc(predicted, observed)                        # Eq. 1, one value per perturbation
distance = score.predicted_nn_distance(predicted, train)      # Definition 1
oracle = score.oracle_nn_distance(observed, train)            # Eq. 6
print(f"mean PDC {pdc.mean():.3f}; Spearman with PDC: predicted-NN-distance "
      f"{spearmanr(distance, pdc)[0]:.3f}, oracle distance {spearmanr(oracle, pdc)[0]:.3f}")

# Calibrate on one half, deploy on the other. The paper uses alpha = 0.30 on
# its configurations; these synthetic responses call for a higher target.
alpha = 0.60
perm = rng.permutation(n_test)
cal, dep = perm[: n_test // 2], perm[n_test // 2:]
ltt = calibrate.learn_then_test(distance[cal], pdc[cal], alpha=alpha, delta=0.10)
print(f"Learn-then-Test threshold {ltt.threshold:.2f} ({ltt.n_accepted} of {cal.size} calibration predictions)")
deployed = calibrate.evaluate_threshold(ltt.threshold, distance[dep], pdc[dep], alpha=alpha)
print(f"deployment half: {deployed['accepted']:.2f} accepted, mean PDC of accepted {deployed['mean_pdc']:.3f}")

# Isotonic rule with the rank-1 SVD baseline plus 0.02 as the target (Definition 2).
target = calibrate.rank_k_baseline(observed[cal], k=1) + 0.02
tau = calibrate.isotonic_threshold(distance[cal], pdc[cal], target)
iso = calibrate.evaluate_threshold(tau, distance[dep], pdc[dep], alpha=target)
print(f"isotonic threshold {tau:.2f} for target {target:.3f}: {iso['accepted']:.2f} accepted, mean PDC {iso['mean_pdc']:.3f}")

# Accepted versus rejected predictions on the deployment half.
accepted = distance[dep] <= ltt.threshold
interval = stats.bootstrap_mean_ci(pdc[dep][accepted], seed=0)
print(f"accepted mean PDC {interval.estimate:.3f} [{interval.low:.3f}, {interval.high:.3f}], "
      f"rejected mean PDC {pdc[dep][~accepted].mean():.3f}")
