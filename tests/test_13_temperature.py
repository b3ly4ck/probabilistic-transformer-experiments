"""Check 13 - the attention temperature of the head update, ``PTConfig.temp_mode``.

Wu & Tu App. A.5 fixes the head temperature at ``lambda_H = 1/d``. The head logit is
``F_c(i, j) = <q_i, B^(c)_{j,.}>`` with ``q_i`` a *distribution* over the d labels, so
``||q_i||_2`` is not a constant of the architecture: it runs from ``1/sqrt(d)`` at the
uniform belief to ``1`` at a one-hot. ``1/d`` is therefore the temperature calibrated for
the uniform-belief limit and mis-calibrated by up to a factor ``d`` away from it - the loop
this project measured running away at ``d = 32`` (msg/unary 2.88 -> 20.57 within 500 steps,
REPORT_2026-08). ``temp_mode`` is the response, and this file is what says the response is
what it claims to be:

* ``"fixed"``   must reproduce the source formula *bitwise*. It is the mainline and every
  number already in the report was produced by it, so a refactor that moves it silently
  invalidates the run log. That is the first test below.
* ``"qnorm"``   divides by ``||q_i||_2 * sqrt(d)``, which is exactly 1 at the uniform belief.
  The two claims that make this the *right* modification rather than an arbitrary one are
  that it AGREES with Wu & Tu where Wu & Tu is calibrated (uniform query) and DIVERGES,
  in the direction of a flatter attention, as the query sharpens. Both are asserted.
* ``"qknorm"`` standardises the logit vector over the head domain and applies ``qk_gain``
  as the inverse temperature. Its defining property is invariance to a global rescaling of
  the score table, which is asserted directly by rescaling the table.

Everything here calls the real ``_arc_message`` / ``_slot_message`` with small real tensors.
Where two modes are compared, ONE model is built and ``cfg.temp_mode`` is mutated between
the two calls, so the parameters are identical by construction and the difference measured
is the temperature alone.

The NaN test is a regression test for a bug that was hit: slot 0's head domain is the single
position ROOT, so the variance of the logit vector over that domain is exactly zero, and
``sqrt(0)`` has an infinite derivative. See the ``eps``-inside-the-sqrt comment in
``_scaled_logits``.
"""

import math

import pytest
import torch

from conftest import toy_model

MODES = ["fixed", "qnorm", "qknorm"]


# --------------------------------------------------------------------------- helpers --


def head_logits(m, query, Bk):
    """Rebuild the unmasked head logit tensor ``F_c(i, j)``, ``(B, h, n, 1 + n)``.

    An independent re-derivation of what ``_arc_message`` assembles before the temperature
    is applied: column 0 is ROOT (``<q_i, r^(c)>``), column ``1 + j`` is the arc score of
    head position ``j`` in its clipped distance bucket. The near buckets are written with a
    plain Python loop over positions rather than the vectorised scatter of the model, so
    that a transcription error in the model's indexing cannot also be present here.
    """
    B, n, d = query.shape
    K = m.cfg.n_dist
    logit = torch.einsum("bia,bcja->bcij", query, Bk[-1]).clone()
    for k in range(K - 1):
        delta = k + 1
        for i in range(delta, n):
            logit[:, :, i, i - delta] = (
                query[:, i, :].unsqueeze(1) * Bk[k][:, :, i - delta, :]
            ).sum(-1)
    root = torch.einsum("bia,ca->bci", query, m.r_root)
    return torch.cat([root.unsqueeze(-1), logit], dim=-1)


def attn_entropy(alpha):
    """Entropy of every attention row. Masked entries are exactly 0 and contribute 0."""
    return -(alpha * alpha.clamp(min=1e-30).log()).sum(-1)


def uniform_query(m, B, n):
    """``q = 1/d`` everywhere - the belief Wu & Tu's ``lambda_H = 1/d`` is calibrated for.

    Hand-constructed on purpose. A fresh model's beliefs are ``softmax(S_w / lambda_Z)``,
    which is *near* uniform but not uniform, and the whole point of the qnorm claim is an
    exact agreement at exactly that point.
    """
    return torch.full((B, n, m.cfg.d), 1.0 / m.cfg.d, dtype=m.S.dtype)


def sharp_query(m, B, n):
    """A near one-hot belief, ``||q||_2 ~ 1``: the far end of the simplex from uniform.

    The label carried varies with position so that the resulting logit vector is not
    constant over the head domain - a constant one would make every temperature agree and
    the comparison vacuous.
    """
    d = m.cfg.d
    eps = 1e-4
    q = torch.full((B, n, d), eps / (d - 1), dtype=m.S.dtype)
    lab = (torch.arange(n) % d).view(1, n, 1).expand(B, n, 1)
    q.scatter_(-1, lab, 1.0 - eps)
    return q


# ------------------------------------------------------- "fixed" is the source formula --


@pytest.mark.parametrize("lambda_H", [1.0, None])  # None -> Wu & Tu's 1/d
def test_fixed_mode_reproduces_the_original_temperature(idx, lambda_H):
    """Regression guard: the refactor into ``_scaled_logits`` did not move the mainline.

    ``softmax(masked F / lambda_H)`` is computed here from the same query and the same
    contracted keys, with the mask written before the division exactly as the pre-refactor
    code did. Agreement to 1e-12 in float64 is bitwise up to the reduction order of the
    near-bucket assembly, which is the only part re-derived differently above.
    """
    m = toy_model(lambda_H=lambda_H, temp_mode="fixed")
    q = m.content_stream(idx)
    Bk = m.contract(q)
    _, alpha = m._arc_message(q, Bk)

    full = head_logits(m, q, Bk)
    allowed, _ = m._causal_masks(idx.shape[1], idx.device)
    want = torch.softmax(full.masked_fill(~allowed, float("-inf")) / m.cfg.lam_H, dim=-1)

    torch.testing.assert_close(alpha, want, atol=1e-12, rtol=0)


@pytest.mark.parametrize("mode", MODES)
def test_attention_is_a_distribution_on_the_head_domain(idx, mode):
    """Under every mode ``Q_c`` is a distribution over ``D_i``, and over nothing else."""
    m = toy_model(temp_mode=mode)
    q = m.content_stream(idx)
    _, alpha = m._arc_message(q, m.contract(q))
    allowed, _ = m._causal_masks(idx.shape[1], idx.device)

    torch.testing.assert_close(
        alpha.sum(-1), torch.ones_like(alpha.sum(-1)), atol=1e-12, rtol=0
    )
    assert (alpha >= 0).all()
    # nothing outside D_i carries mass: mask out the allowed entries, the rest must be 0
    assert float(alpha.masked_fill(allowed, 0.0).abs().max()) == 0.0


# --------------------------------------------------------------------- "qnorm" claims --


def test_qnorm_agrees_with_fixed_at_the_uniform_belief(idx):
    """``||q||_2 * sqrt(d) == 1`` at ``q = 1/d``, so the two temperatures coincide there.

    This is the claim that qnorm is a re-derivation of Wu & Tu's constant rather than a
    replacement of it: at initialisation, where the beliefs are near uniform, the model is
    the source's model.
    """
    m = toy_model()
    B, n = idx.shape
    q = uniform_query(m, B, n)
    Bk = m.contract(q)

    m.cfg.temp_mode = "fixed"
    G_fixed, a_fixed = m._arc_message(q, Bk)
    m.cfg.temp_mode = "qnorm"
    G_qnorm, a_qnorm = m._arc_message(q, Bk)

    torch.testing.assert_close(a_qnorm, a_fixed, atol=1e-6, rtol=0)
    torch.testing.assert_close(G_qnorm, G_fixed, atol=1e-6, rtol=0)


def test_qnorm_is_strictly_flatter_than_fixed_at_a_sharp_belief(idx):
    """Away from uniform, ``||q||_2 * sqrt(d) > 1``, so qnorm divides by more.

    One model, one query, one set of contracted keys; only ``cfg.temp_mode`` moves between
    the two calls. Dividing a logit vector by a larger number moves its softmax towards
    uniform, so the observable consequence is a strictly higher attention entropy - which
    is the mis-calibration this mode exists to remove.
    """
    m = toy_model()
    B, n = idx.shape
    q = sharp_query(m, B, n)
    Bk = m.contract(q)

    m.cfg.temp_mode = "fixed"
    _, a_fixed = m._arc_message(q, Bk)
    m.cfg.temp_mode = "qnorm"
    _, a_qnorm = m._arc_message(q, Bk)

    assert not torch.allclose(a_qnorm, a_fixed, atol=1e-6), (
        "qnorm and fixed agree at a near one-hot query: the query norm is not reaching "
        "the temperature"
    )
    e_fixed = float(attn_entropy(a_fixed).mean())
    e_qnorm = float(attn_entropy(a_qnorm).mean())
    assert e_qnorm > e_fixed, (
        f"qnorm did not flatten the attention: entropy {e_qnorm:.6f} vs fixed {e_fixed:.6f}"
    )
    # sanity on the construction itself: the query really is at the sharp end
    assert float((q.norm(dim=-1) * m.cfg.d**0.5).min()) > 1.9  # sqrt(d) = 2 at a one-hot


# -------------------------------------------------------------------- "qknorm" claims --


@pytest.mark.parametrize("rank", [None, 3])
def test_qknorm_is_invariant_to_a_global_rescaling_of_the_score_table(idx, rank):
    """The whole point of the mode: only the *pattern* of the arc scores sets attention.

    The score table is scaled by 10 with the query held fixed, so the head logit vector is
    scaled by exactly 10 and standardising it must give back the same distribution.

    Both carriers are scaled, ``T`` (or ``U``) *and* the root column ``r``. ``B^(c)_{ROOT,a}
    = r^(c)_a`` is column 0 of the same score table - the arc_regulariser docstring makes
    the same point when it explains why penalising ``T`` alone only moved the message onto
    ``r``. Scaling ``T`` alone would not be a global rescaling of the logit vector, it would
    change the *ratio* between ROOT and the positions, and qknorm neither claims nor should
    have invariance to that.
    """
    m = toy_model(rank=rank)
    q = m.content_stream(idx).detach()

    def both(Bk):
        m.cfg.temp_mode = "qknorm"
        a_qk = m._arc_message(q, Bk)[1]
        m.cfg.temp_mode = "fixed"
        a_fx = m._arc_message(q, Bk)[1]
        return a_qk, a_fx

    qk_before, fx_before = both(m.contract(q))
    with torch.no_grad():
        (m.T if m.T is not None else m.U).mul_(10.0)
        m.r_root.mul_(10.0)
    qk_after, fx_after = both(m.contract(q))

    # Measured drift 1.5e-7 in float64, not 1e-16: the invariance is exact in exact
    # arithmetic and the residual is the eps = 1e-8 added inside the square root, whose
    # relative weight against the variance changes by 100x when the table is scaled by 10.
    torch.testing.assert_close(qk_after, qk_before, atol=1e-4, rtol=0)
    moved = float((fx_after - fx_before).abs().max())
    assert moved > 1e-2, (
        f"the fixed temperature barely moved under a 10x score table ({moved:.2e}): the "
        "invariance above is then not evidence of anything"
    )


def test_qknorm_is_not_invariant_to_the_arc_scores_alone(idx):
    """The control for the test above, and the reason it scales ``r`` as well as ``T``.

    Scaling ``T`` without ``r`` changes the *ratio* between the ROOT column and the position
    columns of the head domain, which is a change of the score pattern and not of its scale.
    qknorm neither claims nor should have invariance to that: measured, the attention moves
    by 0.76 of probability mass. Asserting invariance under a ``T``-only rescaling would
    therefore be asserting something false, and this test pins the distinction.
    """
    m = toy_model(temp_mode="qknorm")
    q = m.content_stream(idx).detach()
    before = m._arc_message(q, m.contract(q))[1]
    with torch.no_grad():
        m.T.mul_(10.0)
    after = m._arc_message(q, m.contract(q))[1]
    assert float((after - before).abs().max()) > 1e-2


def test_qk_gain_is_the_inverse_temperature_of_qknorm(idx):
    """``qk_gain`` is a temperature, not a parameter - raising it sharpens the attention."""
    m = toy_model(temp_mode="qknorm", qk_gain=1.0)
    q = m.content_stream(idx).detach()
    Bk = m.contract(q)

    low = attn_entropy(m._arc_message(q, Bk)[1]).mean()
    m.cfg.qk_gain = 4.0
    high = attn_entropy(m._arc_message(q, Bk)[1]).mean()
    assert float(high) < float(low), f"qk_gain did not sharpen: {float(high)} vs {float(low)}"


# --------------------------------------------- the arithmetic itself, written out --


def closed_form_alpha(m, full, query, allowed):
    """Second, naive implementation of ``_scaled_logits`` followed by the softmax.

    The reduction over the head domain ``D_i`` is a plain Python loop over the allowed
    columns of one row, where the model builds a mask, zeroes outside it and reduces over
    the whole axis at once. Two different routes to the same number is what makes an
    agreement evidence; a re-derivation that re-used the masked reduction would only be
    checking that the model calls itself twice.

    This is here because everything above pins ``qnorm`` and ``qknorm`` only up to a
    *family* of functions. The invariance asserted for qknorm holds just as well for the
    cosine normalisation that ``PTConfig.temp_mode``'s comment describes ("divides by
    ||q_i||_2 ||B_j||_2"), which is a different function of the same inputs: measured on
    this fixture the two disagree by 0.34 of probability mass, and the only existing test
    that separates them does so incidentally. What is implemented - and what this asserts,
    to 1e-10 in float64 - is a standardisation of the logit vector over ``D_i``. If the
    cosine reading is the intended one, this test is where that has to be settled.
    """
    mode, d = m.cfg.temp_mode, m.cfg.d
    B, h, n, D = full.shape
    out = torch.full_like(full, float("-inf"))
    for b in range(B):
        for c in range(h):
            for i in range(n):
                cols = [j for j in range(D) if bool(allowed[i, j])]
                row = [float(full[b, c, i, j]) for j in cols]
                if mode == "fixed":
                    scaled = [x / m.cfg.lam_H for x in row]
                elif mode == "qnorm":
                    qn = float(query[b, i].norm()) * math.sqrt(d)
                    scaled = [x / (m.cfg.lam_H * qn) for x in row]
                else:  # "qknorm"
                    mu = sum(row) / len(row)
                    var = sum((x - mu) ** 2 for x in row) / len(row)
                    den = math.sqrt(var + 1e-8)  # eps INSIDE the root, as the model does
                    scaled = [m.cfg.qk_gain * (x - mu) / den for x in row]
                for j, x in zip(cols, scaled):
                    out[b, c, i, j] = x
    return torch.softmax(out, dim=-1)


@pytest.mark.parametrize("mode", MODES)
def test_each_mode_matches_its_closed_form(idx, mode):
    """Every mode's attention, against the formula spelled out one row at a time."""
    m = toy_model(temp_mode=mode)
    q = m.content_stream(idx).detach()
    Bk = m.contract(q)
    _, alpha = m._arc_message(q, Bk)
    allowed, _ = m._causal_masks(idx.shape[1], idx.device)

    want = closed_form_alpha(m, head_logits(m, q, Bk).detach(), q, allowed)
    torch.testing.assert_close(alpha, want, atol=1e-10, rtol=0)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("t", [0, 4])
def test_each_mode_matches_its_closed_form_on_the_single_slot_path(idx, mode, t):
    """``_slot_message``, the serial schedule's path, where ``allowed`` is ``None``.

    ``_scaled_logits`` has a second branch for the case where the head domain is already
    assembled and needs no mask, and it is the branch the serial schedule and the slot
    readouts take. Nothing else in this file asserts a *value* on it - only that a serial
    forward pass stays finite - so a temperature that was wrong only here would move every
    ``schedule="serial"`` cell of a sweep and no test would say so.

    ``t = 0`` is the zero-variance domain: one column, so centring gives exactly 0 and the
    epsilon inside the root gives ``0 / sqrt(eps) = 0``. That is asserted as a value here,
    which is stronger than the absence of NaN asserted further down.
    """
    m = toy_model(temp_mode=mode)
    qbar = m.content_stream(idx).detach()
    keys = [m.contract(qbar[:, j : j + 1]) for j in range(idx.shape[1])]
    B_full = m._slot_keys_from(keys, t, idx.shape[0])
    query = qbar[:, t]

    ctx, alpha = m._slot_message(query, B_full)
    assert alpha.shape == (idx.shape[0], m.cfg.h, 1 + t)

    logit = (query.unsqueeze(1).unsqueeze(1) * B_full).sum(-1)  # (B, h, 1 + t)
    scaled = torch.empty_like(logit)
    for b in range(logit.shape[0]):
        for c in range(logit.shape[1]):
            row = [float(x) for x in logit[b, c]]
            if mode == "fixed":
                new = [x / m.cfg.lam_H for x in row]
            elif mode == "qnorm":
                qn = float(query[b].norm()) * math.sqrt(m.cfg.d)
                new = [x / (m.cfg.lam_H * qn) for x in row]
            else:
                mu = sum(row) / len(row)
                var = sum((x - mu) ** 2 for x in row) / len(row)
                new = [m.cfg.qk_gain * (x - mu) / math.sqrt(var + 1e-8) for x in row]
            for j, x in enumerate(new):
                scaled[b, c, j] = x
    torch.testing.assert_close(alpha, torch.softmax(scaled, dim=-1), atol=1e-10, rtol=0)

    # the message back to Z is the same attention read against the same keys
    torch.testing.assert_close(
        ctx, (alpha.unsqueeze(-1) * B_full).sum(dim=(1, 2)), atol=1e-10, rtol=0
    )

    if t == 0:
        assert float(alpha.min()) == 1.0, "the single column of D_0 did not take all the mass"


# ------------------------------------------------------ full forward / backward passes --


def _assert_gradients_are_usable(m, require_nonzero=True, allow_missing=()):
    """Every learnable factor came back with a finite, non-NaN, non-trivial gradient.

    ``allow_missing`` names the factors that a particular conditioning legitimately
    disconnects from the loss; anything not named must be present and, unless
    ``require_nonzero`` is off, must actually be moving.
    """
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None and name in allow_missing:
            continue
        assert p.grad is not None, f"{name} received no gradient"
        assert not torch.isnan(p.grad).any(), f"{name} has NaN in its gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
        if require_nonzero:
            assert float(p.grad.abs().max()) > 0.0, f"{name} has an all-zero gradient"


@pytest.mark.parametrize("mode", MODES)
def test_forward_and_backward_are_finite_under_every_mode(idx, mode, readout, schedule):
    """A real forward pass, a real loss, a real backward: shape, finiteness, gradients.

    The batch starts at slot 0, whose head domain is the single position ROOT, so the
    zero-variance case is exercised on the way through under ``qknorm``.
    """
    m = toy_model(temp_mode=mode, readout=readout, schedule=schedule)
    logits = m(idx)
    assert logits.shape == idx.shape + (m.cfg.vocab_size,)
    assert torch.isfinite(logits).all()

    loss = m.loss(idx)
    assert torch.isfinite(loss)
    m.zero_grad(set_to_none=True)
    loss.backward()
    _assert_gradients_are_usable(m)


@pytest.mark.parametrize("mode", MODES)
def test_a_sequence_of_length_one_produces_no_nan(mode, readout):
    """Slot 0 alone: the head domain is ``{ROOT}``, so the logit variance is exactly 0.

    ``sqrt(0)`` has an infinite derivative and clamping the result after the root is not
    enough - autograd still evaluates ``0 * inf``. This produced NaN gradients in the first
    version of ``_scaled_logits``; the fix was to add the epsilon *inside* the square root,
    and this is the test that says so.

    The arc-score carriers ``T`` / ``U`` / ``V`` legitimately do not move at ``n = 1``: no
    arc reaches a head position, every column of ``D_0`` but ROOT is masked, and under the
    exact readout ``log mu_0(a) = r^(c)_a`` with the far and near terms constant ``-inf``,
    which disconnects the content stream from the loss entirely (measured: ``T.grad`` is
    ``None`` under ``readout="exact"`` and exactly ``0.0`` under ``"mfvi"``). The
    zero-variance *backward* path is therefore exercised separately, in the test below.
    """
    m = toy_model(temp_mode=mode, readout=readout)
    idx1 = torch.tensor([[2], [5]])

    logits = m(idx1)
    assert logits.shape == (2, 1, m.cfg.vocab_size)
    assert torch.isfinite(logits).all()

    loss = m.loss(idx1)
    assert torch.isfinite(loss)
    m.zero_grad(set_to_none=True)
    loss.backward()
    _assert_gradients_are_usable(
        m, require_nonzero=False, allow_missing=("T", "U", "V")
    )
    # the carriers that ROOT-only inference does reach must still be moving
    for name in ("S", "r_root", "b"):
        p = dict(m.named_parameters())[name]
        assert float(p.grad.abs().max()) > 0.0, f"{name} did not move at n = 1"


@pytest.mark.parametrize("mode", MODES)
def test_backward_through_a_zero_variance_head_domain_has_no_nan(mode):
    """The sqrt(0) regression proper: differentiate the content stream itself at ``n = 1``.

    The loss-level test above cannot reach this under the exact readout, because at
    ``n = 1`` the readout does not read the content stream at all. Here the content stream
    is the objective, contracted against a fixed random vector so the objective is not
    degenerate (the rows of ``q`` sum to 1, so ``q.sum()`` would have zero gradient by
    construction and would prove nothing).
    """
    m = toy_model(temp_mode=mode)
    idx1 = torch.tensor([[2], [5]])
    q = m.content_stream(idx1)
    assert torch.isfinite(q).all()

    torch.manual_seed(7)
    w = torch.randn_like(q)
    m.zero_grad(set_to_none=True)
    (q * w).sum().backward()

    for name, p in m.named_parameters():
        if p.grad is None:
            continue
        assert not torch.isnan(p.grad).any(), f"{name} has NaN after a zero-variance domain"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
    grads = dict(m.named_parameters())
    assert float(grads["r_root"].grad.abs().max()) > 0.0
    assert float(grads["S"].grad.abs().max()) > 0.0


# ------------------------------------------------------------------------- causality --


@pytest.mark.parametrize("mode", MODES)
def test_causality_holds_under_every_temperature_mode(idx, mode, readout):
    """Check 3, re-run per mode. A temperature that reads the query must not read ahead.

    ``qknorm`` is the one at risk: it reduces over the whole head-domain axis, so a mean or
    a variance taken before the causal mask would pull ``q_j`` for ``j > i`` into slot
    ``i``'s logits. Bitwise equality is the right assertion for the same reason as in
    ``test_03``: the arithmetic at slot ``s <= t`` is identical, not merely close.
    """
    m = toy_model(temp_mode=mode, readout=readout)
    n = idx.shape[1]
    base = m(idx)
    for t in range(n):
        alt = idx.clone()
        alt[:, t] = (idx[:, t] + 3) % m.cfg.vocab_size
        other = m(alt)
        assert torch.equal(base[:, : t + 1], other[:, : t + 1]), (
            f"temp_mode={mode}: changing token {t} moved the logits at some slot <= {t}"
        )
        if t + 1 < n:
            assert not torch.equal(base[:, t + 1 :], other[:, t + 1 :])


# -------------------------------------------------------------- the exact readout path --


@pytest.mark.parametrize("mode", MODES)
def test_exact_readout_works_under_every_temperature_mode(idx, mode, schedule):
    """The temperature sits in the content stream, upstream of the exact readout.

    ``readout="exact"`` never calls ``_scaled_logits`` itself - it is a log-sum-exp over the
    contracted keys - but the keys are contractions of ``q_bar``, which the temperature
    shapes. So the mode has to be exercised through this path too, and it has to leave a
    proper distribution behind.
    """
    m = toy_model(temp_mode=mode, readout="exact", schedule=schedule)
    logits = m(idx)
    assert logits.shape == idx.shape + (m.cfg.vocab_size,)
    assert torch.isfinite(logits).all()
    p = torch.softmax(logits, dim=-1)
    torch.testing.assert_close(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-12, rtol=0)
    assert torch.isfinite(m.next_token_logits(idx)).all()


def test_the_temperature_reaches_the_exact_readout_through_the_content_stream(idx):
    """Same seed, same parameters, three temperatures: the exact readout must differ.

    If it did not, the modes would be untested by every experiment run at the mainline
    readout. Slot 0 is the exception and is asserted as one: its ``log mu`` is ``r^(c)``
    alone, no attention enters it, so all three modes must agree there *bitwise*.
    """
    out = {mode: toy_model(temp_mode=mode, readout="exact")(idx) for mode in MODES}

    assert not torch.allclose(out["fixed"], out["qnorm"], atol=1e-8)
    assert not torch.allclose(out["fixed"], out["qknorm"], atol=1e-8)
    assert not torch.allclose(out["qnorm"], out["qknorm"], atol=1e-8)

    for mode in ("qnorm", "qknorm"):
        assert torch.equal(out["fixed"][:, 0], out[mode][:, 0]), (
            f"{mode} moved slot 0, whose head domain is ROOT alone and carries no attention"
        )
