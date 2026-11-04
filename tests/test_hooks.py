import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lowrank_reliability.audit import basis, hooks  # noqa: E402

H = 12


class _Attention(torch.nn.Module):
    """Stands in for a LlamaAttention: returns (output, weights)."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(H, H)

    def forward(self, x):
        return self.proj(x), None


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()

    def forward(self, x):
        return x + self.self_attn(x)[0]


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(), _Layer()])


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_backbone = _Backbone()

    def forward(self, x):
        for layer in self.transformer_backbone.layers:
            x = layer(x)
        return x


@pytest.fixture()
def setup():
    torch.manual_seed(0)
    model = _Model().eval()
    x = torch.randn(3, 9, H)
    return model, hooks.attention_output(model, 0), x


def _captured(model, module, x, intervention):
    rec = hooks.ActivationRecorder(module)
    with intervention, rec:
        model(x)
    return rec.records[0][1]


def test_attention_output_locates_the_module(setup):
    model, attn, _ = setup
    assert attn is model.transformer_backbone.layers[0].self_attn


@pytest.mark.parametrize("arm", ["svd", "random", "random_complement"])
def test_every_arm_preserves_row_norms(setup, arm):
    model, attn, x = setup
    original = attn(x)[0].detach()
    out = hooks.KeepK(attn, k=2, arm=arm, seed=1, complement_rank=4).transform(original)
    np.testing.assert_allclose(
        torch.linalg.vector_norm(out, dim=-1).numpy(),
        torch.linalg.vector_norm(original, dim=-1).numpy(),
        rtol=1e-5,
    )


def test_fixed_arm_preserves_row_norms_and_stays_in_the_span(setup):
    _, attn, x = setup
    original = attn(x)[0].detach()
    b = basis.random_orthonormal_basis(H, 5, seed=2)
    out = hooks.KeepK(attn, k=3, arm="fixed", basis=b).transform(original)
    np.testing.assert_allclose(
        torch.linalg.vector_norm(out, dim=-1).numpy(),
        torch.linalg.vector_norm(original, dim=-1).numpy(),
        rtol=1e-5,
    )
    b3 = torch.as_tensor(b[:, :3], dtype=out.dtype)
    np.testing.assert_allclose(hooks.project(out, b3).numpy(), out.numpy(), atol=1e-5)


def test_full_rank_fixed_basis_returns_the_input(setup):
    model, attn, x = setup
    with torch.no_grad():
        ref = model(x)
        q = basis.random_orthonormal_basis(H, H, seed=0)
        with hooks.KeepK(attn, k=H, arm="fixed", basis=q):
            out = model(x)
        with hooks.KeepK(attn, k=H, arm="fixed", basis=np.eye(H)):
            out_eye = model(x)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=1e-5)
    np.testing.assert_allclose(out_eye.numpy(), ref.numpy(), atol=1e-6)


def test_svd_arm_with_all_directions_returns_the_input(setup):
    model, attn, x = setup
    with torch.no_grad():
        ref = model(x)
        with hooks.KeepK(attn, k=H, arm="svd"):
            out = model(x)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=1e-5)


def test_svd_k1_without_norm_is_the_rank1_reconstruction(setup):
    model, attn, x = setup
    with torch.no_grad():
        original = attn(x)[0]
        out = hooks.KeepK(attn, k=1, arm="svd", preserve_norm=False).transform(original)
    for b in range(x.shape[0]):
        u, s, vt = np.linalg.svd(original[b].numpy().astype(np.float64), full_matrices=False)
        np.testing.assert_allclose(out[b].numpy(), s[0] * np.outer(u[:, 0], vt[0]), atol=1e-5)
        assert np.linalg.matrix_rank(out[b].numpy(), tol=1e-4) == 1


def test_svd_k1_with_norm_puts_every_row_on_the_top_direction(setup):
    _, attn, x = setup
    with torch.no_grad():
        original = attn(x)[0]
        out = hooks.KeepK(attn, k=1, arm="svd").transform(original)
    for b in range(x.shape[0]):
        v1 = np.linalg.svd(original[b].numpy().astype(np.float64))[2][0]
        rows = out[b].numpy()
        expected = np.linalg.norm(original[b].numpy(), axis=1)[:, None] * np.sign(rows @ v1)[:, None] * v1[None, :]
        np.testing.assert_allclose(rows, expected, atol=1e-5)


def test_random_complement_is_orthogonal_to_the_top_directions(setup):
    _, attn, x = setup
    with torch.no_grad():
        original = attn(x)[0]
        out = hooks.KeepK(attn, k=2, arm="random_complement", complement_rank=4, seed=3).transform(original)
    for b in range(x.shape[0]):
        top = np.linalg.svd(original[b].numpy().astype(np.float64))[2][:4]
        assert np.abs(out[b].numpy() @ top.T).max() < 1e-4


def test_random_arm_is_seeded_per_call_and_input(setup):
    model, attn, x = setup
    with torch.no_grad():
        first = _captured(model, attn, x, hooks.KeepK(attn, k=2, arm="random", seed=7))
        again = _captured(model, attn, x, hooks.KeepK(attn, k=2, arm="random", seed=7))
        hook = hooks.KeepK(attn, k=2, arm="random", seed=7)
        rec = hooks.ActivationRecorder(attn)
        with hook, rec:
            model(x)
            model(x)
    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(rec.records[0][1], first)
    assert not np.allclose(rec.records[1][1], first)

    def span(m):
        return np.linalg.svd(m.astype(np.float64))[2][:2]

    # Inputs of one batch get different random subspaces.
    assert np.linalg.svd(span(first[0]) @ span(first[1]).T, compute_uv=False).min() < 0.999


def test_hook_is_removed_on_exit(setup):
    model, attn, x = setup
    with torch.no_grad():
        ref = model(x)
        hook = hooks.KeepK(attn, k=1, arm="svd")
        with hook:
            changed = model(x)
        after = model(x)
    assert hook.handle is None and hook.calls == 1
    assert not torch.allclose(changed, ref)
    np.testing.assert_array_equal(after.numpy(), ref.numpy())


def test_fixed_arm_rejects_a_non_orthonormal_basis(setup):
    _, attn, _ = setup
    with pytest.raises(ValueError):
        hooks.KeepK(attn, k=2, arm="fixed", basis=np.ones((H, 2)))
    with pytest.raises(ValueError):
        hooks.KeepK(attn, k=2, arm="fixed")
    with pytest.raises(ValueError):
        hooks.KeepK(attn, k=2, arm="unknown")


def test_patch_components_matches_eq4():
    torch.manual_seed(1)
    x = torch.randn(9, H, dtype=torch.float64)
    u, s, vt = torch.linalg.svd(x, full_matrices=False)
    donor = torch.linalg.qr(torch.randn(H, 3, dtype=torch.float64))[0].T
    out = hooks.patch_components(x, donor, components=(0,), magnitude="recipient")
    v_d = donor[0] * torch.sign(torch.dot(donor[0], vt[0]))
    expected = x - s[0] * torch.outer(u[:, 0], vt[0]) + s[0] * torch.outer(u[:, 0], v_d)
    np.testing.assert_allclose(out.numpy(), expected.numpy(), atol=1e-5)
    # A donor equal to the recipient's own directions leaves X unchanged.
    same = hooks.patch_components(x, -vt[:3], components=(0, 1, 2), magnitude="recipient")
    np.testing.assert_allclose(same.numpy(), x.numpy(), atol=1e-5)


def test_direction_patch_hook_uses_context(setup):
    model, attn, x = setup
    xs = x[:1]
    with torch.no_grad():
        ref = model(xs)
        rec = hooks.ActivationRecorder(attn)
        with rec:
            model(xs)
        own = rec.records[0][1][0]
        dirs, sig = basis.donor_directions({"GENE": np.random.default_rng(0).normal(size=(9, H))})
        patch = hooks.DirectionPatch(attn, dirs, components=(0,), magnitude="recipient")
        for context, should_change in (("non-targeting", False), ("OTHER", False), (None, False), ("GENE", True)):
            patch.set_context(context)
            with patch:
                out = model(xs)
            assert (not torch.allclose(out, ref)) == should_change
    assert patch.patched == 1 and patch.skipped == 3
    own_dirs, own_sig = basis.donor_directions({"GENE": own})
    self_patch = hooks.DirectionPatch(attn, own_dirs, components=(0,), magnitude="donor", singular_values=own_sig)
    self_patch.set_context("GENE")
    with torch.no_grad(), self_patch:
        out = model(xs)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=1e-4)


def test_activation_recorder(setup):
    model, attn, x = setup
    rec = hooks.ActivationRecorder(attn)
    with torch.no_grad(), rec:
        rec.set_context("a")
        model(x)
        rec.set_context("b")
        model(x[:1])
    grouped = rec.by_context()
    assert grouped["a"].shape == (3 * 9, H) and grouped["b"].shape == (9, H)


class _StateLike(torch.nn.Module):
    """Minimal stand-in for an arc-state ST model: forward(batch, padded) and a gene decoder."""

    def __init__(self):
        super().__init__()
        self.transformer_backbone = _Backbone()
        self.gene_decoder = torch.nn.Linear(H, 4)

    def forward(self, batch, padded=True):
        x = batch["pert_emb"].reshape(1, -1, H)
        for layer in self.transformer_backbone.layers:
            x = layer(x)
        return x


def test_predict_means_runs_every_arm_on_each_batch():
    from lowrank_reliability.audit import state

    torch.manual_seed(0)
    model = _StateLike().eval()
    batches = [
        {"pert_emb": torch.randn(5, H), "pert_name": ["a"] * 5},
        {"pert_emb": torch.randn(3, H), "pert_name": ["non-targeting"] * 3},
        {"pert_emb": torch.randn(4, H), "pert_name": ["a"] * 4},
    ]
    keep = hooks.KeepK(hooks.attention_output(model, 0), k=1)
    out = state.predict_means(model, batches, {"unmodified": None, "keep1": keep})
    means, counts = out["unmodified"]
    assert counts == {"a": 9, "non-targeting": 3}
    assert means.names == ("a", "non-targeting")
    with torch.no_grad():
        cells = torch.cat([model.gene_decoder(model(b))[0] for b in (batches[0], batches[2])])
    np.testing.assert_allclose(means.row("a"), cells.mean(dim=0).numpy(), atol=1e-6)
    assert keep.calls == 3 and keep.handle is None
    assert not np.allclose(out["keep1"][0].row("a"), means.row("a"))


def test_random_frame_needs_room_in_the_complement():
    g = torch.Generator().manual_seed(0)
    exclude = torch.eye(4)[:, :3]
    with pytest.raises(ValueError):
        hooks.random_frame(4, 2, g, exclude=exclude)
    q = hooks.random_frame(4, 1, g, exclude=exclude)
    assert torch.abs(exclude.T @ q).max() < 1e-6


def test_random_complement_keeps_row_norms_in_the_complement():
    torch.manual_seed(2)
    x = torch.randn(1, 40, 8)
    out = hooks.KeepK(torch.nn.Identity(), k=1, arm="random_complement", complement_rank=3).transform(x)
    top = np.linalg.svd(x[0].numpy().astype(np.float64))[2][:3]
    assert np.abs(out[0].numpy() @ top.T).max() < 1e-4
    np.testing.assert_allclose(np.linalg.norm(out[0].numpy(), axis=1), np.linalg.norm(x[0].numpy(), axis=1), rtol=1e-5)
    # k = 1 keeps one direction: every row is a signed multiple of it.
    assert np.linalg.matrix_rank(out[0].numpy(), tol=1e-4) == 1


def test_half_precision_activations():
    x = torch.randn(2, 6, 8).to(torch.float16)
    for arm, kwargs in (("svd", {}), ("random", {}), ("fixed", {"basis": np.eye(8)[:, :2]})):
        out = hooks.KeepK(torch.nn.Identity(), k=2, arm=arm, **kwargs).transform(x)
        assert out.dtype == torch.float16 and torch.isfinite(out).all()
    tiny = torch.tensor([[[1e-5, 1.0]]], dtype=torch.float16)
    out = hooks.KeepK(torch.nn.Identity(), k=1, arm="fixed", basis=np.array([[1.0], [0.0]])).transform(tiny)
    np.testing.assert_allclose(out.float().numpy(), [[[1.0, 0.0]]], atol=1e-3)


def test_direction_patch_rejects_negative_components():
    with pytest.raises(ValueError):
        hooks.DirectionPatch(torch.nn.Identity(), {"p": np.eye(3)}, components=(-1,))
    x = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
    out = hooks.patch_components(x, torch.tensor([[0.0, 1.0, 0.0]]), components=[-1])
    np.testing.assert_allclose(out.numpy(), x.numpy())


def test_patch_skips_a_missing_donor_component():
    x = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
    dirs, _ = basis.donor_directions({"p": np.array([[1.0, 0.0, 0.0]])}, n_components=2)
    out = hooks.patch_components(x, torch.as_tensor(dirs["p"]), components=(1,))
    np.testing.assert_allclose(out.numpy(), x.numpy(), atol=1e-6)


def test_projection_ignores_autocast():
    hook = hooks.KeepK(torch.nn.Identity(), 1, arm="fixed", basis=np.array([[1.0], [1e-8]]))
    x = torch.tensor([[[0.0, 1.0]]], dtype=torch.float16)
    plain = hook.transform(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = hook.transform(x)
    np.testing.assert_array_equal(plain.float().numpy(), mixed.float().numpy())
