"""Check 15 -- the ten-switch ladder between nanoGPT and the causal PT decoder.

``src/switches.py`` is analysis code that deliberately steps outside the factor-graph
discipline (see its module docstring). What has to be true of it is not "is it PT" -- it is
not, and it says so -- but three things that would silently void every number the ladder
produces:

1. **The transformer end really is the baseline.** If the all-transformer rung is not
   numerically the same model as ``src/gpt.py``, then rung 0 of the ladder is not the
   published GPT number (115.43 val ppl on PTB, ``REPORT_2026-08.md`` §1) and every "cost of
   switch k" is measured against a different origin. Tested by loading GPT's own state dict
   into ``SwitchModel`` and comparing logits.
2. **Every rung is causal.** The slot convention shifts the backbone output right by one, so
   an off-by-one in a readout is invisible in the shapes and shows up only as the model
   scoring a token it was allowed to see. That check has to cover every rung, because a
   switch changes the readout (rung 1) and the position encoding (rung 8), which is exactly
   where an off-by-one would enter.
3. **Every rung trains.** A switch that removes a matrix must not orphan the ones that stay:
   every parameter that exists has to receive gradient.
4. **Every switch does what its name says.** The three checks above are all invariants that
   hold for *any* rung, so a switch that silently did nothing -- or wired itself to the wrong
   tensor -- would pass all of them. A mutation run on 2026-09-08 confirmed the hole: with
   ``_constrain`` replaced by ``return h`` (switch 10 a no-op), with ``SwitchBlock.forward``
   forced onto the residual path (switch 3 a no-op), with the ``attn_query_proj=False`` query
   taken from the key instead of the state (switch 6 mis-wired), and with the tied value
   bound to the query instead of the key (switch 4 mis-wired), the suite still reported
   69 passed. That is the worst possible failure for this file: a no-op switch does not
   crash, it reports "this difference from a transformer costs nothing", and ``state`` and
   ``readout`` are precisely the two the experiment exists to price
   (``experiments/exp6_switches/spec_switches.py``). The last section of this file closes
   that hole by executing each switch against an independent reference.

All tensors here are tiny and CPU-only: vocab 40, ``n_embd`` 16, 2 heads, 2 layers,
``d_label`` 8, block 12, batch 3.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from src.gpt import GPT, GPTConfig
from src.switches import (
    PT_SWITCHES,
    SWITCH_NAMES,
    TRANSFORMER_SWITCHES,
    SwitchConfig,
    SwitchModel,
    SwitchSelfAttention,
    ladder,
)

BASE = dict(
    vocab_size=40, block_size=12, n_layer=2, n_head=2, n_embd=16, d_label=8, dropout=0.0
)
BATCH, BLOCK = 3, 12


def sw(seed: int = 0, **over) -> SwitchModel:
    cfg = dict(BASE)
    cfg.update(over)
    torch.manual_seed(seed)
    return SwitchModel(SwitchConfig(**cfg))


def gpt(seed: int = 0, **over) -> GPT:
    cfg = {k: v for k, v in BASE.items() if k != "d_label"}
    cfg.update(over)
    torch.manual_seed(seed)
    return GPT(GPTConfig(**cfg))


def sample(seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, BASE["vocab_size"], (BATCH, BLOCK))


def rungs() -> dict:
    """Every rung the tests sweep: both endpoints and each switch flipped alone."""
    out = {"all_transformer": dict(TRANSFORMER_SWITCHES), "all_pt": dict(PT_SWITCHES)}
    for name in SWITCH_NAMES:
        one = dict(TRANSFORMER_SWITCHES)
        one[name] = PT_SWITCHES[name]
        out[f"only_{name}"] = one
    return out


RUNGS = rungs()


# ------------------------------------------------- 1. the transformer end is the baseline --


def test_all_transformer_rung_loads_the_gpt_state_dict_and_matches_its_logits():
    """Rung 0 must be the published baseline, not a re-implementation that resembles it."""
    g, m = gpt(), sw()
    gs, ms = g.state_dict(), m.state_dict()
    assert set(gs) == set(ms), (
        f"state dict keys differ; only in GPT {sorted(set(gs) - set(ms))}, "
        f"only in SwitchModel {sorted(set(ms) - set(gs))}"
    )
    for k in gs:
        assert gs[k].shape == ms[k].shape, f"{k}: {gs[k].shape} vs {ms[k].shape}"

    m.load_state_dict(gs)  # strict=True by default
    g.eval(), m.eval()
    idx = sample()
    a, b = g(idx), m(idx)
    assert a.shape == b.shape == (BATCH, BLOCK, BASE["vocab_size"])
    assert torch.allclose(a, b, atol=1e-6, rtol=0), float((a - b).abs().max())
    assert torch.isclose(g.loss(idx), m.loss(idx), atol=1e-6)
    assert g.num_parameters() == m.num_parameters()


def test_all_transformer_rung_initialises_identically_from_the_same_seed():
    """Same module order, same init scheme: not just loadable, but the same draw."""
    g, m = gpt(seed=3), sw(seed=3)
    for (kg, vg), (km, vm) in zip(g.named_parameters(), m.named_parameters()):
        assert kg == km
        assert torch.equal(vg, vm), kg


def test_weight_sharing_rung_reproduces_the_looped_baseline():
    """Switch 9 alone is Experiment 2's Looped Transformer, so it must match it too."""
    g, m = gpt(shared_block=True), sw(weight_sharing=True)
    assert set(g.state_dict()) == set(m.state_dict())
    m.load_state_dict(g.state_dict())
    g.eval(), m.eval()
    idx = sample()
    assert torch.allclose(g(idx), m(idx), atol=1e-6, rtol=0)
    assert g.num_parameters() == m.num_parameters()
    # and it really is one block reused, not two initialised alike
    assert m.blocks is None and m.block is not None
    assert m.num_parameters()["non_embedding"] < sw().num_parameters()["non_embedding"]


# ------------------------------------------------------------- 2. every rung is causal --


@pytest.fixture
def single_threaded():
    """Pin the intra-op thread count for the duration of one test, then restore it.

    The causality check below asserts *bitwise* equality, which is the right assertion --
    masked positions carry softmax weight exactly zero, so a correct model returns exactly
    the same prefix -- but bitwise equality also picks up the one thing that is not a
    property of the model: the order in which a multi-threaded reduction accumulates. On
    2026-09-08 this test failed twice on the 56-core login node with ``all_pt: slot <= 6
    moved`` and could not be reproduced afterwards -- not in 40 consecutive runs of this
    file, not in 3000 repeats of the same forward pass (bit-identical every time), not in
    4400 base-versus-altered comparisons in a standalone process. A real causality leak is
    deterministic and would fail every run; a reduction-order difference is exactly this
    rare and this load-dependent. Pinning one thread removes the only candidate mechanism
    without weakening the assertion by a single bit, and matches what ``tests/conftest.py``
    already asks for: "the causality check in particular must run where the hardware is
    reproducible".
    """
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(n)


@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_slot_t_never_sees_token_t_or_later_on_any_rung(rung, single_threaded):
    """Changing token ``t`` must leave every logit at slot <= t bit-identical.

    Masked positions get softmax weight exactly zero, so the check is exact equality, not a
    tolerance; a tolerance would hide a leak of the right magnitude. The companion assertion
    -- that the *future* does move -- is what stops a rung from passing by ignoring its
    input altogether.

    The failure message carries the size of the movement, because that number is what
    separates the two ways this assertion can fire: a wiring or masking bug moves a logit by
    order 1e-3 or more at this scale, while float reduction noise moves it by order 1e-8.
    Without it a future failure is unreadable.
    """
    m = sw(**RUNGS[rung]).eval()
    idx = sample()
    base = m(idx)
    assert torch.isfinite(base).all()
    for t in range(1, BLOCK):
        alt = idx.clone()
        alt[:, t] = (idx[:, t] + 7) % BASE["vocab_size"]
        other = m(alt)
        assert torch.isfinite(other).all(), f"{rung}: non-finite logits after flipping {t}"
        moved = float((base[:, : t + 1] - other[:, : t + 1]).abs().max().detach())
        assert torch.equal(base[:, : t + 1], other[:, : t + 1]), (
            f"{rung}: slot <= {t} moved by {moved:.3e}"
        )
        if t + 1 < BLOCK:
            assert not torch.equal(
                base[:, t + 1 :], other[:, t + 1 :]
            ), f"{rung}: token {t} reaches no future slot"


@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_slot_zero_is_zeros_and_refused_by_the_loss_on_any_rung(rung):
    m = sw(**RUNGS[rung])
    idx = sample()
    assert torch.equal(m(idx)[:, 0], torch.zeros(BATCH, BASE["vocab_size"]))
    with pytest.raises(AssertionError, match="slot 0"):
        m.loss(idx, ignore_first=0)


# -------------------------------------------------------------- 3. every rung trains --


@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_every_rung_forwards_and_delivers_gradient_to_every_parameter(rung):
    """A switch may remove a matrix; it may not leave a surviving one unreachable."""
    m = sw(**RUNGS[rung])
    idx = sample()
    logits = m(idx)
    assert logits.shape == (BATCH, BLOCK, BASE["vocab_size"])
    assert torch.isfinite(logits).all()

    loss = m.loss(idx, ignore_first=1)
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{rung}: no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"{rung}: non-finite grad for {name}"
        assert float(p.grad.abs().max()) > 0.0, f"{rung}: all-zero grad for {name}"


@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_arc_regulariser_is_a_finite_scalar_zero_on_any_rung(rung):
    """The shared loop asks every model for it; no rung of this ladder has arc scores."""
    r = sw(**RUNGS[rung]).arc_regulariser()
    assert r.shape == () and float(r) == 0.0


# ------------------------------------------------------------------- the PT endpoint --


def test_all_pt_rung_runs_and_reports_its_parameter_count():
    """The far end of the ladder, with its budget printed for the run log."""
    pt_rung = sw(**PT_SWITCHES)
    tr_rung = sw()
    p_pt, p_tr = pt_rung.num_parameters(), tr_rung.num_parameters()
    print(f"\nall-transformer rung: {p_tr}")
    print(f"all-PT rung        : {p_pt}")

    idx = sample()
    logits = pt_rung(idx)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(pt_rung.loss(idx))

    # The PT rung drops W_Q, W_V, W_O, the MLP, both LayerNorms and the absolute position
    # table, and shares its one block, so its non-embedding budget must be far smaller.
    assert p_pt["non_embedding"] < p_tr["non_embedding"] / 4, (p_pt, p_tr)
    # It pays for that with a second word matrix S, because d_label != n_embd here.
    assert not pt_rung.tied()
    assert p_pt["embedding"] > 0


def test_mixture_readout_ties_S_to_the_input_embedding_when_the_widths_allow_it():
    """CLAUDE.md constraint 2: one word-label factor S mediates both directions.

    The tie is only expressible when the stream width and the label count agree, which is
    the setting in which the all-PT rung is coherent at all.
    """
    m = sw(readout="mixture", d_label=BASE["n_embd"])
    assert m.tied()
    assert m.lm_head.weight is m.wte.weight  # same object, not merely equal
    idx = sample()
    assert torch.isfinite(m(idx)).all()

    loose = sw(readout="mixture", d_label=BASE["n_embd"] // 2)
    assert not loose.tied()


def test_mixture_readout_is_chunk_invariant_over_the_vocabulary(monkeypatch):
    """Two implementations of the same reduction must agree: chunked LSE vs one-shot."""
    import src.switches as S

    m = sw(readout="mixture").eval()
    idx = sample()
    monkeypatch.setattr(S, "VOCAB_CHUNK", 10**6)
    whole = m(idx)
    monkeypatch.setattr(S, "VOCAB_CHUNK", 7)  # does not divide vocab 40
    chunked = m(idx)
    assert torch.allclose(whole, chunked, atol=1e-6, rtol=0), float(
        (whole - chunked).abs().max()
    )


def _numeric_rank(J: torch.Tensor, rtol: float = 1e-9) -> int:
    s = torch.linalg.svdvals(J)
    return int((s > rtol * s[0]).sum())


def test_mixture_readout_is_a_d_label_bottleneck_on_the_stream():
    """The point of switch 1, stated the way it is actually true.

    ``logits_w = LSE_a (log mu_a + S_{w,a})`` is *not* a linear map of the stream, so the
    matrix of logits over many contexts is not low rank -- that naive check fails, and it
    failed here first. The exact statement is about the derivative: the readout factors
    through ``log mu``, a point of the ``d_label``-simplex, so however wide the stream is,
    ``d(logits)/d(h)`` can have rank at most ``d_label - 1`` (one dimension is lost to the
    simplex's own shift invariance). Every direction of the stream beyond that is invisible
    to the output. That is the honest rank-``d`` softmax bottleneck of ``CLAUDE.md``: at
    ``n_embd`` 16 and ``d_label`` 8 exactly half the stream cannot reach a logit.

    The linear rung has no such bound: its Jacobian is ``W_lm``, of full rank
    ``min(n_embd, V)``.
    """
    d, C, V = BASE["d_label"], BASE["n_embd"], BASE["vocab_size"]
    torch.manual_seed(0)
    h = torch.randn(C, dtype=torch.float64)

    m = sw(readout="mixture").double().eval()
    J = torch.autograd.functional.jacobian(lambda x: m._mixture_logits(x.view(1, 1, -1)).view(-1), h)
    assert J.shape == (V, C)
    assert _numeric_rank(J) <= d - 1, f"mixture Jacobian rank {_numeric_rank(J)} > {d - 1}"

    lin = sw().double().eval()
    J2 = torch.autograd.functional.jacobian(lambda x: lin.lm_head(x.view(1, 1, -1)).view(-1), h)
    assert _numeric_rank(J2) == min(C, V), _numeric_rank(J2)


# ------------------------------------------------------------------ accounting, ladder --


@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_num_parameters_sums_to_the_actual_tensor_count(rung):
    m = sw(**RUNGS[rung])
    p = m.num_parameters()
    assert p["embedding"] + p["non_embedding"] == p["total"]
    assert p["total"] == sum(q.numel() for q in m.parameters())
    assert p["embedding"] > 0 and p["non_embedding"] > 0


def test_ladder_walks_one_switch_at_a_time_between_the_two_endpoints():
    order = list(SWITCH_NAMES)
    rows = list(ladder(order))
    assert len(rows) == len(order) + 1
    assert rows[0] == TRANSFORMER_SWITCHES
    assert rows[-1] == PT_SWITCHES
    for i in range(len(rows) - 1):
        diff = [k for k in SWITCH_NAMES if rows[i][k] != rows[i + 1][k]]
        assert diff == [order[i]], (i, diff)
    # every rung is a usable config, and reports how far along it is
    for i, row in enumerate(rows):
        cfg = SwitchConfig(**BASE, **row)
        assert cfg.switches() == row
        assert cfg.distance_from_transformer() == i

    back = list(ladder(order, reverse=True))
    assert back[0] == PT_SWITCHES and back[-1] == TRANSFORMER_SWITCHES

    part = list(ladder(["readout", "ffn"]))
    assert len(part) == 3
    assert part[-1]["readout"] == "mixture" and part[-1]["ffn"] is False
    assert part[-1]["norm"] == "layernorm"  # untouched names stay at the transformer end

    with pytest.raises(ValueError, match="unknown switch"):
        list(ladder(["not_a_switch"]))
    with pytest.raises(ValueError, match="repeated"):
        list(ladder(["ffn", "ffn"]))


def test_config_rejects_impossible_settings():
    with pytest.raises(ValueError, match="unknown readout"):
        SwitchConfig(vocab_size=10, readout="projection")
    with pytest.raises(ValueError, match="unknown state"):
        SwitchConfig(vocab_size=10, state="probability")
    with pytest.raises(ValueError, match="divisible"):
        SwitchConfig(vocab_size=10, n_embd=15, n_head=2)


# ------------------------------------- 4. every switch does what its name says --
#
# Everything above this line is an invariant that holds for *any* rung: it is causal, it
# trains, its parameters add up. None of it can see a switch that quietly does nothing, and
# a no-op switch is not a crash -- it is a published number saying that difference is free.
# The tests below execute each switch against a reference derived from what the switch is
# supposed to mean, with a negative control where the plausible mis-wiring is silent.


def _block_inputs(m: SwitchModel, idx: torch.Tensor) -> list:
    """Every tensor that enters a block on one forward pass, via a pre-hook.

    Reading the state between blocks is the only way to observe switch 10 at all: the
    simplex constraint is applied inside the backbone and nothing about the logits, the
    parameter count or the gradients reveals whether it was applied.
    """
    seen: list = []
    handles = [b.register_forward_pre_hook(lambda mod, args: seen.append(args[0].detach()))
               for b in m.blocks]
    try:
        m(idx)
    finally:
        for h in handles:
            h.remove()
    return seen


def test_switch_10_makes_the_state_a_point_of_the_simplex_at_every_block():
    """``state="simplex"``: PT's ``q`` is a distribution, at iteration 0 and after each update.

    This is what removes the need for LayerNorm and what bounds the query norm, so if the
    softmax were not actually applied the ladder would attribute both effects to the wrong
    switch. The transformer rung is read the same way as a control, so the measurement is
    known to be capable of failing.
    """
    idx = sample()
    simplex = _block_inputs(sw(state="simplex").eval(), idx)
    assert len(simplex) == BASE["n_layer"]
    for i, x in enumerate(simplex):
        assert torch.allclose(x.sum(-1), torch.ones_like(x.sum(-1)), atol=1e-6), i
        assert bool((x >= 0).all()), i

    free = _block_inputs(sw(state="unconstrained").eval(), idx)
    assert len(free) == BASE["n_layer"]
    assert not torch.allclose(
        free[0].sum(-1), torch.ones_like(free[0].sum(-1)), atol=1e-3
    ), "the transformer rung's state happens to be normalised; the control is vacuous"


@pytest.mark.parametrize("name", ["state", "residual"])
def test_a_switch_that_removes_no_parameter_still_changes_the_computation(name):
    """Switches 3 and 10 leave the parameter set untouched, so nothing else here sees them.

    Give the two rungs literally the same weights and the logits must still differ: that is
    the whole content of "this rung is a different model".
    """
    a = sw().eval()
    b = sw(**{name: PT_SWITCHES[name]}).eval()
    assert set(a.state_dict()) == set(b.state_dict()), name
    b.load_state_dict(a.state_dict())
    idx = sample()
    assert not torch.allclose(a(idx), b(idx), atol=1e-6, rtol=0), (
        f"switch {name!r} at its PT setting is a no-op on identical weights"
    )


def test_switch_3_replaces_the_stream_with_the_embedding_unary():
    """``residual=False`` is ``attn(ln_1(x)) + mlp(ln_2(x)) + emb``, both messages from ``x``.

    Two mis-implementations are silent everywhere else in this file: never taking the branch,
    and adding ``x`` back instead of ``emb`` -- which turns "replace" into "accumulate" and
    quietly restores the residual stream the switch exists to remove. PT re-adds the word
    unary ``S_w`` at every iteration, not the previous state, so ``emb`` is the whole point.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 5, BASE["n_embd"])
    emb = torch.randn(2, 5, BASE["n_embd"])

    blk = sw(residual=False).eval().blocks[0]
    out = blk(x, emb)
    assert torch.allclose(out, blk.attn(blk.ln_1(x)) + emb + blk.mlp(blk.ln_2(x)),
                          atol=1e-6, rtol=0)
    assert not torch.allclose(out, blk(x, x), atol=1e-6, rtol=0), (
        "the replace path is insensitive to emb, so it is still an accumulate path"
    )

    acc = sw(residual=True).eval().blocks[0]
    y = x + acc.attn(acc.ln_1(x))
    assert torch.allclose(acc(x, emb), y + acc.mlp(acc.ln_2(y)), atol=1e-6, rtol=0)
    assert torch.equal(acc(x, emb), acc(x, torch.zeros_like(emb))), (
        "the accumulate path must not read emb at all"
    )


def _reference_attention(q, k, v, n_head):
    """Masked multi-head attention written out, independent of ``src.switches``.

    Deliberately a second implementation rather than a call into the module: the thing being
    checked is *which tensor* plays each role, and only an outside expression can say that.
    """
    B, T, C = q.shape
    hd = C // n_head

    def heads(t):
        return t.view(B, T, n_head, hd).transpose(1, 2)

    att = (heads(q) @ heads(k).transpose(-2, -1)) / math.sqrt(hd)
    att = att.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
    return (F.softmax(att, dim=-1) @ heads(v)).transpose(1, 2).contiguous().view(B, T, C)


@pytest.mark.parametrize("value", ["separate", "tied"])
@pytest.mark.parametrize("query_proj", [True, False])
def test_switches_4_and_6_query_with_the_state_and_tie_the_value_to_the_key(query_proj, value):
    """The surviving projections must be packed into ``c_attn`` in the order ``q, k, v``.

    Switch 6 off means the query *is* the state -- PT's attention logit is an inner product
    with the label belief itself -- and switch 4 tied means the value is the *key*, because
    PT contracts the same arc score that produced the logit. Both mis-wirings (querying with
    the key; valuing with the query) leave the shapes, the parameter count, causality and the
    gradients untouched, so each is checked against the reference and against the mis-wiring
    it would be confused with.
    """
    C, H = BASE["n_embd"], BASE["n_head"]
    cfg = SwitchConfig(
        **dict(BASE, attn_query_proj=query_proj, attn_value=value, attn_out_proj=False)
    )
    torch.manual_seed(0)
    attn = SwitchSelfAttention(cfg).eval()
    assert attn.c_proj is None  # additive: switch 5 removed W_O, so `out` is the message sum
    n_proj = 1 + int(query_proj) + int(value == "separate")
    assert attn.c_attn.out_features == n_proj * C

    torch.manual_seed(2)
    x = torch.randn(2, 6, C)
    parts = list(attn.c_attn(x).split(C, dim=2))
    i = 0
    q = parts[i] if query_proj else x
    if query_proj:
        i += 1
    k = parts[i]
    i += 1
    v = parts[i] if value == "separate" else k

    out = attn(x)
    assert torch.allclose(out, _reference_attention(q, k, v, H), atol=1e-6, rtol=0)

    if not query_proj:
        assert not torch.allclose(out, _reference_attention(k, k, v, H), atol=1e-6, rtol=0), (
            "the query is the key, not the state"
        )
    if value == "tied":
        assert not torch.allclose(out, _reference_attention(q, k, q, H), atol=1e-6, rtol=0), (
            "the value is tied to the query, not to the key"
        )


def test_num_parameters_accounts_for_both_branches_of_the_mixture_readout():
    """``d_label == n_embd`` ties ``S`` to ``wte``; anything else pays for a second matrix.

    Only the untied branch was exercised before, because every rung in :data:`RUNGS` runs at
    ``d_label`` 8 against ``n_embd`` 16. The distinction is not cosmetic: at the shape used by
    ``experiments/exp6_switches/spec_switches.py`` (``n_embd`` 64, ``d_label`` 32) every
    mixture rung is untied and carries an extra ``V x d_label`` embedding matrix, so a row's
    budget can only be read literally if both branches are pinned.
    """
    V, C, P = BASE["vocab_size"], BASE["n_embd"], BASE["block_size"]
    tied = sw(readout="mixture", d_label=C)
    untied = sw(readout="mixture", d_label=C // 2)
    for m in (tied, untied):
        p = m.num_parameters()
        assert p["embedding"] + p["non_embedding"] == p["total"]
        assert p["total"] == sum(q.numel() for q in m.parameters())
        assert torch.isfinite(m.loss(sample()))
    assert tied.tied() and not untied.tied()
    # wte + wpe + the word unary b; the tied S is the same tensor as wte and is not recounted
    assert tied.num_parameters()["embedding"] == V * C + P * C + V
    # the same, plus the second word matrix S
    assert untied.num_parameters()["embedding"] == V * C + P * C + V * (C // 2) + V
