"""Helpers for arc-state ST checkpoints (requires torch; model loading requires arc-state 0.10.4).

A checkpoint directory written by ``state tx train`` holds ``config.yaml``,
``var_dims.pkl``, ``pert_onehot_map.pt``, a batch map
(``batch_onehot_map.torch`` or ``.pkl``) and the Lightning checkpoint under
``checkpoints/``.
"""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from ..metrics import CONTROL, Responses
from .hooks import Intervention
from .readout import embedding_tables

__all__ = [
    "seed_everything",
    "find_checkpoint",
    "load_embedding_tables",
    "load_model",
    "load_datamodule",
    "predict_means",
]


def seed_everything(seed: int = 0) -> None:
    """Seed Python's, NumPy's and torch's global random number generators.

    The cell-load batch sampler shuffles cells with the global NumPy
    generator, and that order decides which control cell is paired with each
    perturbed cell. Seeding before ``setup`` makes the test batches, and so
    every arm's predictions, reproducible.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

_MODEL_CKPT_ORDER = (
    "checkpoints/best.ckpt",
    "checkpoints/final.ckpt",
    "checkpoints/last.ckpt",
    "eval_last.ckpt",
)
_TABLE_CKPT_ORDER = (
    "checkpoints/best.ckpt",
    "eval_best.ckpt",
    "best.ckpt",
    "version_0/checkpoints/best.ckpt",
)


def find_checkpoint(ckpt_dir, order: Iterable[str] = _MODEL_CKPT_ORDER) -> Path:
    """First existing checkpoint file under ``ckpt_dir`` in ``order``, else any ``*.ckpt`` file (``best`` first)."""
    root = Path(ckpt_dir)
    for rel in order:
        p = root / rel
        if p.is_file():
            return p
    found = sorted(p for p in root.glob("**/*.ckpt") if p.is_file())
    if not found:
        raise FileNotFoundError(f"no checkpoint file under {root}")
    best = [p for p in found if "best" in p.name]
    return best[0] if best else found[0]


def _load_map(path: Path) -> dict[str, int]:
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            raw = pickle.load(f)
    else:
        raw = torch.load(path, weights_only=False)
    return {
        str(name): int(torch.argmax(torch.as_tensor(vec)).item())
        for name, vec in raw.items()
    }


def load_embedding_tables(ckpt_dir) -> tuple[np.ndarray, dict, np.ndarray, dict]:
    """Perturbation and batch embedding tables of a checkpoint with their name indices.

    Returns ``(pert_table, pert_index, batch_table, batch_index)`` for
    ``readout.readout_rows``.
    """
    root = Path(ckpt_dir)
    ckpt = find_checkpoint(root, _TABLE_CKPT_ORDER)
    state_dict = torch.load(ckpt, weights_only=False, map_location="cpu")["state_dict"]
    pert_table, batch_table = embedding_tables(state_dict)
    pert_index = _load_map(root / "pert_onehot_map.pt")
    batch_path = root / "batch_onehot_map.torch"
    if not batch_path.exists():
        batch_path = root / "batch_onehot_map.pkl"
    batch_index = _load_map(batch_path)
    return pert_table, pert_index, batch_table, batch_index


def _read_config(root: Path) -> tuple[dict, dict]:
    import yaml

    with open(root / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(root / "var_dims.pkl", "rb") as f:
        var_dims = pickle.load(f)
    return cfg, var_dims


def load_model(ckpt_dir, device: str | torch.device = "cpu", checkpoint=None):
    """Load a ``StateTransitionPerturbationModel`` in eval mode from a checkpoint directory."""
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    root = Path(ckpt_dir)
    cfg, var_dims = _read_config(root)
    model_kwargs = cfg["model"]["kwargs"]
    data_kwargs = cfg["data"]["kwargs"]
    init_kwargs = {
        "input_dim": var_dims["input_dim"],
        "hidden_dim": model_kwargs["hidden_dim"],
        "gene_dim": var_dims["gene_dim"],
        "hvg_dim": var_dims.get("hvg_dim", var_dims["gene_dim"]),
        "output_dim": var_dims.get("output_dim", var_dims["gene_dim"]),
        "pert_dim": var_dims["pert_dim"],
        "batch_dim": var_dims.get("batch_dim", None),
        "cell_sentence_len": model_kwargs.get("cell_set_len", 64),
        "embed_key": data_kwargs.get("embed_key", None),
        "output_space": data_kwargs.get("output_space", "gene"),
        "batch_size": cfg["training"].get("batch_size", 64),
        "control_pert": data_kwargs.get("control_pert", CONTROL),
        "gene_names": var_dims.get("gene_names", None),
        "dropout": model_kwargs.get("dropout", 0.0),
        "lr": cfg["training"].get("lr", 1e-3),
        "loss_fn": cfg["training"].get("loss_fn", "mse"),
        **{k: v for k, v in model_kwargs.items() if k != "cell_set_len"},
    }
    path = Path(checkpoint) if checkpoint is not None else find_checkpoint(root)
    model = StateTransitionPerturbationModel.load_from_checkpoint(
        str(path), map_location=str(device), strict=False, **init_kwargs
    )
    return model.eval().to(device)


def load_datamodule(
    ckpt_dir,
    toml_path,
    data_dir=None,
    num_workers: int = 0,
    cell_sentence_len: int | None = None,
):
    """Build the checkpoint's ``PerturbationDataModule`` (requires cell-load and toml).

    The data settings come from ``config.yaml``; ``toml_path`` replaces the
    split file named there. With ``data_dir`` every dataset path in the TOML
    is replaced by that directory. ``cell_sentence_len`` is passed to the data
    module when given; the paper's keep-k runs use the cell-load default, so
    each test sentence holds all cells of one perturbation (at most 512).
    Call ``seed_everything`` first for reproducible batches.
    """
    import toml
    from cell_load.data_modules import PerturbationDataModule

    root = Path(ckpt_dir)
    cfg, _ = _read_config(root)
    state = dict(cfg["data"]["kwargs"])
    state["num_workers"] = num_workers
    state.pop("pin_memory", None)
    with open(toml_path) as f:
        toml_cfg = toml.load(f)
    if data_dir is not None:
        for name in toml_cfg.get("datasets", {}):
            toml_cfg["datasets"][name] = str(data_dir)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
    with tmp:
        toml.dump(toml_cfg, tmp)
    state["toml_config_path"] = tmp.name
    extra = {
        "map_controls": state.pop("map_controls", True),
        "cache_perturbation_control_pairs": state.pop("cache_perturbation_control_pairs", False),
        "perturbation_features_file": state.pop("perturbation_features_file", None),
        "exp_counts": state.pop("exp_counts", False),
        "normalize_counts": state.pop("normalize_counts", False),
        "store_raw_basal": state.pop("store_raw_basal", False),
        "barcode": state.pop("barcode", True),
        "use_consecutive_loading": state.pop("use_consecutive_loading", False),
        "h5_open_kwargs": state.pop("h5_open_kwargs", None),
    }
    if cell_sentence_len is not None:
        state["cell_sentence_len"] = int(cell_sentence_len)
    return PerturbationDataModule(**state, **extra)


def _batch_names(batch: Mapping, n_cells: int) -> list[str]:
    raw = batch.get("pert_name", ["unknown"])
    if isinstance(raw, str):
        return [raw] * n_cells
    if isinstance(raw, (list, tuple)):
        names = [str(x) for x in raw]
        return names if len(names) == n_cells else [names[0]] * n_cells
    return [str(raw)] * n_cells


def predict_means(
    model,
    loader: Iterable[Mapping],
    arms: Mapping[str, Intervention | None],
    device: str | torch.device = "cpu",
    max_batches: int | None = None,
) -> dict[str, tuple[Responses, dict[str, int]]]:
    """Mean predicted expression per perturbation under each arm.

    Every batch of ``loader`` (one cell sentence per batch, as produced by the
    data module's test loader with batch size 1) is moved to ``device`` once
    and passed through the model once per arm, with that arm's intervention
    active (``None`` runs the unmodified model). Before each pass the arm's
    context is set to the batch's perturbation. Predictions are the model
    output passed through its gene decoder when it has one. Returns, for each
    arm, the per-perturbation means (control included) and cell counts.
    """
    sums: dict[str, dict[str, np.ndarray]] = {name: {} for name in arms}
    counts: dict[str, int] = {}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            pert_emb = batch["pert_emb"]
            n_cells = pert_emb.shape[-2] if pert_emb.dim() == 3 else pert_emb.shape[0]
            names = _batch_names(batch, n_cells)
            for arm, intervention in arms.items():
                if intervention is None:
                    pred = _forward(model, moved)
                else:
                    intervention.set_context(names[0])
                    with intervention:
                        pred = _forward(model, moved)
                for name in set(names):
                    rows = [j for j, n in enumerate(names) if n == name]
                    block = pred[rows].astype(np.float64).sum(axis=0)
                    acc = sums[arm]
                    acc[name] = acc[name] + block if name in acc else block
            for name in names:
                counts[name] = counts.get(name, 0) + 1
    out = {}
    for arm, acc in sums.items():
        labels = sorted(acc)
        means = np.stack([acc[n] / counts[n] for n in labels]) if labels else np.zeros((0, 0))
        out[arm] = (Responses(tuple(labels), means), {n: counts[n] for n in labels})
    return out


def _forward(model, batch: Mapping) -> np.ndarray:
    output = model.forward(batch, padded=False)
    latent = output[0] if isinstance(output, tuple) else output
    decoder = getattr(model, "gene_decoder", None)
    pred = decoder(latent) if decoder is not None else latent
    pred = pred.detach().float().cpu().numpy()
    if pred.ndim == 3:
        pred = pred[0]
    return pred
