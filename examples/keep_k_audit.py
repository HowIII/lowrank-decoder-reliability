"""Keep-k arms at the first attention output of an arc-state ST checkpoint.

Every test sentence is loaded once and passed through the model once per arm,
so all arms see the same cells and the same control pairing. PDC is scored
against the observed test cells and each arm is compared with the unmodified
decoder by a paired bootstrap.

    python examples/keep_k_audit.py \
        --checkpoint-dir ST-SE-Replogle/fewshot/hepg2 \
        --toml State-Replogle-Filtered/hepg2.toml \
        --data-dir State-Replogle-Filtered/splits \
        --observed ST-SE-Replogle/fewshot/hepg2/eval_best.ckpt/adata_real.h5ad \
        --train-activations se_hepg2_L0.npz --k 1 8

``--train-activations`` is an .npz file with the training activations of the
same checkpoint (keys ``attn_out_L0``, ``pert_offsets``, ``pert_names``; one
block of rows per training perturbation). Without it the fixed-basis arms are
skipped. Needs torch, arc-state 0.10.4, cell-load, anndata and toml.
"""
import argparse

import anndata as ad
import numpy as np

from lowrank_reliability import metrics, stats
from lowrank_reliability.audit import basis, hooks, state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--toml", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--observed", required=True, help="h5ad with the observed test cells (obs['gene'] labels)")
    ap.add_argument("--train-activations", default=None)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-basis-seed", type=int, default=20260724)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()

    state.seed_everything(args.seed)
    model = state.load_model(args.checkpoint_dir, device=args.device)
    dm = state.load_datamodule(args.checkpoint_dir, args.toml, args.data_dir, num_workers=args.num_workers)
    dm.batch_size = 1
    dm.setup(stage="test")
    module = hooks.attention_output(model, args.layer)
    hidden = model.transformer_backbone.config.hidden_size

    arms = {"unmodified": None}
    for k in args.k:
        arms[f"svd_k{k}"] = hooks.KeepK(module, k, arm="svd")
        arms[f"random_k{k}"] = hooks.KeepK(module, k, arm="random", seed=args.seed)
        arms[f"random_complement_k{k}"] = hooks.KeepK(module, k, arm="random_complement", seed=args.seed)
    if args.train_activations:
        z = np.load(args.train_activations)
        blocks = basis.ragged_activations(z["attn_out_L0"], z["pert_offsets"], z["pert_names"])
        # Nested bases of 32 columns, as in the paper; rank k uses the first k columns.
        width = max(32, max(args.k))
        fixed, eigenvalues = basis.fixed_basis(basis.second_moment(blocks), k=width)
        control = basis.random_orthonormal_basis(hidden, width, seed=args.random_basis_seed)
        print(f"training basis from {len(blocks)} perturbations; top-4 energy {basis.energy_fraction(eigenvalues, 4):.3f}")
        for k in args.k:
            arms[f"fixed_k{k}"] = hooks.KeepK(module, k, arm="fixed", basis=fixed)
            arms[f"random_fixed_k{k}"] = hooks.KeepK(module, k, arm="fixed", basis=control)

    means = state.predict_means(model, dm.test_dataloader(), arms, device=args.device, max_batches=args.max_batches)
    real = ad.read_h5ad(args.observed)
    x = real.X.toarray() if hasattr(real.X, "toarray") else np.asarray(real.X)
    observed = metrics.pseudobulk_response(x, real.obs["gene"].astype(str).values)
    scores = {
        arm: metrics.pdc_table(metrics.pseudobulk_response(m.values, list(m.names)), observed)
        for arm, (m, _) in means.items()
    }
    reference = scores["unmodified"]
    print(f"{'arm':26s} {'mean PDC':>9s} {'minus unmodified [95% CI]':>32s}")
    for arm, values in scores.items():
        diff = stats.paired_bootstrap(values, reference)
        print(f"{arm:26s} {np.nanmean(list(values.values())):9.4f} "
              f"{diff.estimate:+9.4f} [{diff.low:+.4f}, {diff.high:+.4f}]")


if __name__ == "__main__":
    main()
