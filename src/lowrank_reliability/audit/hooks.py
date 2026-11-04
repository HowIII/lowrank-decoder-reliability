"""PyTorch forward hooks on the first attention output X_p of the ST decoder.

The hooked module is ``model.transformer_backbone.layers[layer].self_attn`` of
an arc-state ``StateTransitionPerturbationModel``: the self-attention sublayer
of block L_layer, before its residual connection. Its output (or the first
element of its output tuple) has shape (B, S, H) with one S x H matrix X_p per
cell sentence; every intervention acts on each X_p separately and leaves the
rest of the network unchanged.

Keep-k arms (``KeepK``) replace X_p by its projection onto a k-dimensional
subspace and rescale each projected row to its original norm:

* ``"svd"``: the top-k right singular vectors of X_p itself (Eq. 3);
* ``"fixed"``: the first k columns of a frozen basis, for example
  ``basis.fixed_basis`` of the training second moment or
  ``basis.random_orthonormal_basis`` for the rank-matched random control;
* ``"random"``: k orthonormal directions drawn uniformly for each input;
* ``"random_complement"``: k random orthonormal directions orthogonal to the
  top-32 right singular vectors of X_p.

``DirectionPatch`` replaces the recipient's component sigma_i u_i v_i^T by
sigma_i u_i v_i(donor)^T (Eq. 4). ``ActivationRecorder`` stores X_p.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

__all__ = [
    "CONTROL",
    "attention_output",
    "preserve_row_norms",
    "project",
    "top_k_basis",
    "random_frame",
    "patch_components",
    "Intervention",
    "KeepK",
    "DirectionPatch",
    "ActivationRecorder",
]

CONTROL = "non-targeting"
_EPS = 1e-12


def attention_output(model, layer: int = 0) -> torch.nn.Module:
    """The self-attention module of block ``layer`` in an arc-state ST model."""
    return model.transformer_backbone.layers[layer].self_attn


# ---------------------------------------------------------------------------
# Tensor operations
# ---------------------------------------------------------------------------
def preserve_row_norms(projected: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Rescale each row of ``projected`` to the norm of the matching row of ``original``.

    Rows whose projection has norm at most 1e-12 become zero.
    """
    original_norm = torch.linalg.vector_norm(original, dim=-1, keepdim=True)
    projected_norm = torch.linalg.vector_norm(projected, dim=-1, keepdim=True)
    scale = torch.where(
        projected_norm > _EPS,
        original_norm / projected_norm,
        torch.zeros_like(projected_norm),
    )
    return projected * scale


def project(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Orthogonal projection of the rows of ``x`` onto the columns of ``basis``: x B B^T."""
    return (x @ basis) @ basis.transpose(-1, -2)


def top_k_basis(x: torch.Tensor, k: int) -> torch.Tensor:
    """H x k matrix of the top-k right singular vectors of an S x H matrix."""
    _, _, vt = torch.linalg.svd(x, full_matrices=False)
    return vt[:k].transpose(0, 1).contiguous()


def random_frame(
    dim: int,
    k: int,
    generator: torch.Generator,
    device=None,
    dtype=None,
    exclude: torch.Tensor | None = None,
) -> torch.Tensor:
    """Random ``dim x k`` orthonormal frame from the QR factor of a Gaussian matrix.

    With ``exclude`` (``dim x m`` orthonormal columns) the Gaussian matrix is
    first projected onto the orthogonal complement of those columns, which
    needs ``k <= dim - m``.
    """
    if exclude is not None and k > dim - exclude.shape[1]:
        raise ValueError(f"cannot draw {k} directions orthogonal to {exclude.shape[1]} of {dim}")
    a = torch.randn(dim, k, device=device, dtype=dtype, generator=generator)
    if exclude is not None:
        a = a - exclude @ (exclude.transpose(0, 1) @ a)
    q, _ = torch.linalg.qr(a)
    return q


def patch_components(
    x: torch.Tensor,
    donor_v: torch.Tensor,
    components: Sequence[int],
    magnitude: str = "recipient",
    donor_sigma: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace singular components of an S x H matrix with donor directions (Eq. 4).

    For each index i in ``components`` the recipient term sigma_i u_i v_i^T is
    replaced by alpha u_i v_i(donor)^T, with the donor direction sign-aligned
    to the recipient's v_i. ``magnitude`` sets alpha: ``"recipient"`` uses the
    recipient sigma_i (Eq. 4), ``"donor"`` the donor sigma_i and ``"unit"`` 1.
    All components come from one SVD of the input. A component without a
    donor direction (index out of range or a zero row) is left unchanged.
    Computed in float32.
    """
    out = x.to(torch.float32)
    u, s, vt = torch.linalg.svd(out, full_matrices=False)
    for i in sorted(set(int(c) for c in components)):
        if i < 0 or i >= s.numel() or i >= donor_v.shape[0]:
            continue
        v_d = donor_v[i].to(device=out.device, dtype=out.dtype)
        if not bool(torch.any(v_d != 0)):
            continue
        sigma = s[i]
        v_r = vt[i]
        sign = torch.sign(torch.dot(v_d, v_r))
        if sign == 0:
            sign = torch.ones((), device=out.device, dtype=out.dtype)
        v_d = sign * v_d
        if magnitude == "recipient":
            alpha = sigma
        elif magnitude == "donor":
            if donor_sigma is None:
                raise ValueError("magnitude='donor' needs donor singular values")
            alpha = donor_sigma[i].to(device=out.device, dtype=out.dtype)
        elif magnitude == "unit":
            alpha = torch.ones((), device=out.device, dtype=out.dtype)
        else:
            raise ValueError(f"unknown magnitude {magnitude!r}")
        out = out - torch.outer(u[:, i], v_r) * sigma + torch.outer(u[:, i], v_d) * alpha
    return out.to(x.dtype)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
class Intervention:
    """Forward hook that rewrites the output of one module.

    Use as a context manager: the hook is registered on entry and removed on
    exit. ``calls`` counts forward passes through the module while the hook
    was active. ``set_context(name)`` records the perturbation of the next
    forward pass for interventions that need it.
    """

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.handle = None
        self.calls = 0
        self.context: str | None = None

    def set_context(self, name: str | None) -> None:
        self.context = None if name is None else str(name)

    def transform(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map the (B, S, H) output to its replacement."""
        raise NotImplementedError

    def _hook(self, module, args, output):
        self.calls += 1
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3:
            raise ValueError(f"expected a (B, S, H) output, got shape {tuple(hidden.shape)}")
        new = self.transform(hidden)
        if new is hidden:
            return output
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new

    def attach(self) -> "Intervention":
        if self.handle is None:
            self.handle = self.module.register_forward_hook(self._hook)
        return self

    def detach(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __enter__(self) -> "Intervention":
        return self.attach()

    def __exit__(self, *exc) -> bool:
        self.detach()
        return False


class KeepK(Intervention):
    """Keep-k projection of X_p (Eq. 3 and its control arms).

    Parameters
    ----------
    module : the attention module (see ``attention_output``).
    k : number of directions kept.
    arm : ``"svd"``, ``"fixed"``, ``"random"`` or ``"random_complement"``.
    basis : H x m array with orthonormal columns, m >= k (``"fixed"`` only);
        the first k columns are used.
    seed : base seed of the random arms. Input b of forward call c (counting
        from 1) uses a ``torch.Generator`` on the activation's device seeded
        with ``seed + 1_000_003 * c + b``.
    preserve_norm : rescale each projected row to its original norm.
    complement_rank : number of top singular directions excluded by
        ``"random_complement"``.
    """

    ARMS = ("svd", "fixed", "random", "random_complement")

    def __init__(
        self,
        module: torch.nn.Module,
        k: int,
        arm: str = "svd",
        basis=None,
        seed: int = 0,
        preserve_norm: bool = True,
        complement_rank: int = 32,
    ):
        super().__init__(module)
        if arm not in self.ARMS:
            raise ValueError(f"arm must be one of {self.ARMS}, got {arm!r}")
        if k < 1:
            raise ValueError("k must be positive")
        self.k = int(k)
        self.arm = arm
        self.seed = int(seed)
        self.preserve_norm = bool(preserve_norm)
        self.complement_rank = int(complement_rank)
        self._basis_cache: dict = {}
        if arm == "fixed":
            if basis is None:
                raise ValueError("arm='fixed' needs a basis")
            b = np.asarray(basis, dtype=np.float64)
            if b.ndim != 2 or b.shape[1] < self.k:
                raise ValueError(f"basis must be H x m with m >= k, got {b.shape}")
            b = b[:, : self.k]
            gram_error = np.abs(b.T @ b - np.eye(self.k)).max()
            if gram_error > 1e-5:
                raise ValueError(f"basis columns are not orthonormal (error {gram_error:.2e})")
            self.basis = b
        else:
            self.basis = None

    def _fixed(self, like: torch.Tensor) -> torch.Tensor:
        key = (like.device, like.dtype)
        if key not in self._basis_cache:
            self._basis_cache[key] = torch.as_tensor(
                self.basis.astype(np.float32), device=like.device
            ).to(like.dtype)
        return self._basis_cache[key]

    def _project_input(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        dim = x.shape[1]
        k = min(self.k, dim)
        if self.arm == "svd":
            return project(x, top_k_basis(x, k))
        if self.arm == "random":
            return project(x, random_frame(dim, k, generator, x.device, x.dtype))
        top = top_k_basis(x, self.complement_rank)
        return project(x, random_frame(dim, k, generator, x.device, x.dtype, exclude=top))

    def transform(self, hidden: torch.Tensor) -> torch.Tensor:
        # Half-precision activations are projected in float32 (outside any
        # autocast region) and cast back.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self._transform(hidden)

    def _transform(self, hidden: torch.Tensor) -> torch.Tensor:
        work = hidden.float() if hidden.dtype in (torch.float16, torch.bfloat16) else hidden
        if self.arm == "fixed":
            if work.shape[-1] != self.basis.shape[0]:
                raise ValueError(
                    f"basis has {self.basis.shape[0]} rows, activation has {work.shape[-1]} columns"
                )
            projected = project(work, self._fixed(work))
        else:
            projected = torch.empty_like(work)
            base = self.seed + 1_000_003 * self.calls
            for b in range(work.shape[0]):
                generator = torch.Generator(device=work.device)
                generator.manual_seed(base + b)
                projected[b] = self._project_input(work[b], generator)
        if self.preserve_norm:
            projected = preserve_row_norms(projected, work)
        return projected.to(hidden.dtype)


class DirectionPatch(Intervention):
    """Direction patching at X_p (Eq. 4).

    ``directions`` maps a perturbation name to the donor's top right singular
    vectors (rows v_1, v_2, ...), for example from ``basis.donor_directions``
    on a cached donor forward pass of the same perturbation. Before each
    forward pass call ``set_context(perturbation)``; the hook leaves X_p
    unchanged for the control perturbation and for perturbations without a
    donor direction. ``components`` are 0-based indices (0 = leading
    direction).
    """

    def __init__(
        self,
        module: torch.nn.Module,
        directions: Mapping[str, np.ndarray],
        components: Sequence[int] = (0,),
        magnitude: str = "recipient",
        singular_values: Mapping[str, np.ndarray] | None = None,
        control: str = CONTROL,
    ):
        super().__init__(module)
        if magnitude not in ("recipient", "donor", "unit"):
            raise ValueError(f"unknown magnitude {magnitude!r}")
        if magnitude == "donor" and singular_values is None:
            raise ValueError("magnitude='donor' needs singular_values")
        self.directions = {str(k): np.asarray(v, dtype=np.float32) for k, v in directions.items()}
        self.singular_values = (
            None if singular_values is None
            else {str(k): np.asarray(v, dtype=np.float32) for k, v in singular_values.items()}
        )
        self.components = tuple(sorted(set(int(c) for c in components)))
        if any(c < 0 for c in self.components):
            raise ValueError("components are 0-based non-negative indices")
        self.magnitude = magnitude
        self.control = control
        self.patched = 0
        self.skipped = 0

    def transform(self, hidden: torch.Tensor) -> torch.Tensor:
        name = self.context
        if name is None or name == self.control or name not in self.directions:
            self.skipped += 1
            return hidden
        donor_v = torch.as_tensor(self.directions[name], device=hidden.device)
        donor_sigma = None
        if self.singular_values is not None and name in self.singular_values:
            donor_sigma = torch.as_tensor(self.singular_values[name], device=hidden.device)
        out = hidden.clone()
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            for b in range(hidden.shape[0]):
                out[b] = patch_components(hidden[b], donor_v, self.components, self.magnitude, donor_sigma)
        self.patched += 1
        return out


class ActivationRecorder(Intervention):
    """Record X_p without changing it.

    Each forward pass appends ``(context, array)`` to ``records``, where the
    array is the (B, S, H) output as float32 NumPy.
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__(module)
        self.records: list[tuple[str | None, np.ndarray]] = []

    def transform(self, hidden: torch.Tensor) -> torch.Tensor:
        self.records.append(
            (self.context, hidden.detach().to(device="cpu", dtype=torch.float32).numpy().copy())
        )
        return hidden

    def by_context(self) -> dict[str | None, np.ndarray]:
        """Recorded rows grouped by context, each group stacked to rows x H."""
        out: dict[str | None, list[np.ndarray]] = {}
        for name, arr in self.records:
            out.setdefault(name, []).append(arr.reshape(-1, arr.shape[-1]))
        return {name: np.concatenate(blocks, axis=0) for name, blocks in out.items()}
