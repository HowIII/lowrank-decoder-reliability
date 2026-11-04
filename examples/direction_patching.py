"""Direction patching between two arc-state ST checkpoints (Eq. 4).

The donor checkpoint is run once over its test cells while its first
attention output is recorded; each perturbation's donor direction is the top
right singular vector of its recorded rows. The recipient is then run with
its leading component sigma_1 u_1 v_1^T replaced by sigma_1 u_1 v_1(donor)^T
and compared with the recipient without the intervention on the same batches.

    python examples/direction_patching.py \
        --donor-dir state_tahoe_hepg2 --donor-toml hepg2.toml --donor-data-dir splits_tahoe \
        --recipient-dir ST-SE-Replogle/fewshot/hepg2 --recipient-toml hepg2.toml \
        --recipient-data-dir splits --observed adata_real.h5ad

Needs torch, arc-state 0.10.4, cell-load, anndata and toml.
"""
import argparse

import anndata as ad
import numpy as np

from lowrank_reliability import metrics, stats
from lowrank_reliability.audit import basis, hooks, state


def loader(ckpt, toml, data_dir, seed, workers):
    state.seed_everything(seed)
    dm = state.load_datamodule(ckpt, toml, data_dir, num_workers=workers)
    dm.batch_size = 1
    dm.setup(stage="test")
    return dm.test_dataloader()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--donor-dir", required=True)
    ap.add_argument("--donor-toml", required=True)
    ap.add_argument("--donor-data-dir", default=None)
    ap.add_argument("--recipient-dir", required=True)
    ap.add_argument("--recipient-toml", required=True)
    ap.add_argument("--recipient-data-dir", default=None)
    ap.add_argument("--observed", required=True, help="h5ad with the recipient's observed test cells")
    ap.add_argument("--component", type=int, default=0, help="0-based singular component to patch")
    ap.add_argument("--magnitude", choices=["recipient", "donor", "unit"], default="recipient")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    donor = state.load_model(args.donor_dir, device=args.device)
    recorder = hooks.ActivationRecorder(hooks.attention_output(donor, args.layer))
    state.predict_means(donor, loader(args.donor_dir, args.donor_toml, args.donor_data_dir, args.seed, args.num_workers),
                        {"record": recorder}, device=args.device)
    recorded = {name: rows for name, rows in recorder.by_context().items() if name is not None}
    directions, singular_values = basis.donor_directions(recorded, n_components=args.component + 1)
    print(f"donor directions for {len(directions)} perturbations")

    recipient = state.load_model(args.recipient_dir, device=args.device)
    patch = hooks.DirectionPatch(
        hooks.attention_output(recipient, args.layer), directions,
        components=(args.component,), magnitude=args.magnitude, singular_values=singular_values,
    )
    means = state.predict_means(
        recipient, loader(args.recipient_dir, args.recipient_toml, args.recipient_data_dir, args.seed, args.num_workers),
        {"baseline": None, "patched": patch}, device=args.device,
    )
    real = ad.read_h5ad(args.observed)
    x = real.X.toarray() if hasattr(real.X, "toarray") else np.asarray(real.X)
    observed = metrics.pseudobulk_response(x, real.obs["gene"].astype(str).values)
    pdc = {arm: metrics.pdc_table(metrics.pseudobulk_response(m.values, list(m.names)), observed)
           for arm, (m, _) in means.items()}
    diff = stats.paired_bootstrap(pdc["patched"], pdc["baseline"])
    print(f"mean PDC without intervention {np.nanmean(list(pdc['baseline'].values())):.4f}; "
          f"patched minus baseline {diff.estimate:+.4f} [{diff.low:+.4f}, {diff.high:+.4f}] "
          f"({patch.patched} sentences patched, {patch.skipped} left unchanged)")


if __name__ == "__main__":
    main()
