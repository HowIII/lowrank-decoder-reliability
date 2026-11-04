"""Ridge read-out of an ST checkpoint's conditioning embeddings (Eq. 2).

    python examples/ridge_readout.py \
        --checkpoint-dir ST-SE-Replogle/fewshot/k562 \
        --cells State-Replogle-Filtered/splits/k562.h5ad --embedding-key X_state \
        --test-perturbations k562_test.txt

``--cells`` is the cell-line h5ad with the frozen encoder embeddings in
``obsm[--embedding-key]``, the HVG expression in ``obsm['X_hvg']`` and the
labels ``obs['gene']`` and ``obs['gem_group']``. ``--test-perturbations`` is a
text file with one held-out perturbation per line. Needs torch and anndata.
"""
import argparse

import anndata as ad
import numpy as np

from lowrank_reliability import metrics
from lowrank_reliability.audit import readout, state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--embedding-key", required=True)
    ap.add_argument("--test-perturbations", required=True)
    ap.add_argument("--batch-key", default="gem_group")
    args = ap.parse_args()

    with open(args.test_perturbations) as f:
        test = {line.strip() for line in f if line.strip()}
    pert_table, pert_index, batch_table, batch_index = state.load_embedding_tables(args.checkpoint_dir)
    adata = ad.read_h5ad(args.cells)
    train, held_out, control_mean = readout.readout_rows(
        adata.obsm[args.embedding_key],
        metrics.hvg_expression(adata),
        adata.obs["gene"].astype(str).values,
        adata.obs[args.batch_key].astype(str).values,
        pert_table, pert_index, batch_table, batch_index, test,
    )
    fit = readout.fit_ridge_readout(train, held_out, control_mean)
    print(f"{len(train)} training rows, {len(held_out)} test rows")
    for penalty, score in fit.cv_scores.items():
        print(f"  penalty {penalty:>9g}: held-out PDC {score:.4f}")
    print(f"selected penalty {fit.penalty:g}; test mean PDC {fit.mean_pdc:.4f} "
          f"over {np.isfinite(list(fit.test_pdc.values())).sum()} perturbations")


if __name__ == "__main__":
    main()
