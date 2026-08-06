"""Check 9 -- the exact readout agrees with brute-force enumeration.

The slot graph is a star centred at Z, hence a tree, so sum-product is exact and
the closed form of §17.2 must equal an explicit sum over the joint.  This is the
strongest correctness test in the project: it validates the factorisation
itself, not an approximation to it.  Under §23.3 the exact readout is the
mainline, so this tests the model rather than an oracle for it.
"""

import pytest
import torch

from src import exact, mfvi
from src.pt_decoder import causal_key_mask


GLOBAL = [pytest.param(None, id="no_global"), pytest.param(4, id="global")]


def _small(seed=0, h=2, K=4, d=3, V=5, m=None):
    g = torch.Generator()
    g.manual_seed(seed)
    Bkey = torch.randn(h, K, d, generator=g)
    S = torch.randn(V, d, generator=g)
    b = torch.randn(V, generator=g)
    Bg = torch.randn(m, d, generator=g) if m else None
    return Bkey, S, b, Bg


@pytest.mark.parametrize("m", GLOBAL)
def test_closed_form_equals_enumeration(m):
    """With a global head the enumeration gains an explicit loop over k, so
    agreement is evidence that the slot is still a tree once G_t is attached."""
    for seed in range(5):
        Bkey, S, b, Bg = _small(seed, m=m)
        closed = exact.exact_logits(exact.log_mu_slot(Bkey, Bg), S, b)
        brute = exact.brute_force_logits(Bkey, S, b, Bg)
        # both are unnormalised; compare the distributions they define
        assert torch.allclose(
            torch.softmax(closed, dim=-1), torch.softmax(brute, dim=-1), atol=1e-6
        ), f"seed {seed}"
        # and the log-scale offset must be constant, not drifting per word
        offset = closed - brute
        assert torch.allclose(offset, offset[0].expand_as(offset), atol=1e-5)


@pytest.mark.parametrize("m", GLOBAL)
def test_enumeration_is_independent_of_the_closed_form(m):
    """Guard: the oracle must disagree when the model is really perturbed.

    The perturbation has to break the *shape* of the score matrix, not shift it.
    Adding a constant to every arc score raises log mu by the same constant at
    every label, which cancels in the softmax -- the model is genuinely invariant
    to that, so it makes a worthless guard.  Perturbing a single label does not
    cancel.
    """
    Bkey, S, b, Bg = _small(m=m)
    brute = exact.brute_force_logits(Bkey, S, b, Bg)
    perturbed = Bkey.clone()
    perturbed[:, :, 0] += 0.5
    closed_wrong = exact.exact_logits(exact.log_mu_slot(perturbed, Bg), S, b)
    assert not torch.allclose(
        torch.softmax(closed_wrong, dim=-1), torch.softmax(brute, dim=-1), atol=1e-3
    )


def test_uniform_shift_of_arc_scores_is_a_no_op():
    """The invariance the guard above tripped over, asserted deliberately."""
    Bkey, S, b, _ = _small()
    base = torch.softmax(exact.exact_logits(exact.log_mu_slot(Bkey), S, b), dim=-1)
    shifted = torch.softmax(exact.exact_logits(exact.log_mu_slot(Bkey + 0.5), S, b), dim=-1)
    assert torch.allclose(base, shifted, atol=1e-6)


def test_prefix_scan_equals_per_slot_reduction(model, tokens):
    """The logcumsumexp scan must equal the naive per-slot LSE over D_t.

    The scan is the efficient implementation (§23.3, O(n d h), fully parallel);
    the per-slot reduction is the definition.  An off-by-one in the mask or in
    the ROOT offset shows up here and nowhere else.
    """
    qbar = model.content_stream(tokens)
    Bkey = mfvi.contract_prefix(qbar, model.T, model.r)
    Bg = model.B_global
    scanned = exact.log_mu_sequence(Bkey, Bg)

    n = tokens.shape[1]
    mask = causal_key_mask(n)
    for t in range(n):
        keys = Bkey[:, :, mask[t], :]  # exactly D_t = {ROOT} u {j < t}
        assert keys.shape[2] == t + 1
        naive = exact.log_mu_slot(keys, Bg)
        assert torch.allclose(scanned[:, t], naive, atol=1e-6), f"slot {t}"


def test_exact_readout_beats_the_rank_bound_that_binds_mfvi():
    """The mean-field readout is affine in Q_Z, so its logit matrix has rank <= d+1.

    The exact readout is a mixture of exponentials over labels and is not bound
    the same way.  Demonstrated numerically: build many contexts and compare the
    ranks of the two logit matrices.
    """
    g = torch.Generator()
    g.manual_seed(0)
    d, V, n_ctx = 3, 12, 40
    S = torch.randn(V, d, generator=g)
    b = torch.randn(V, generator=g)
    Q_Z = torch.softmax(torch.randn(n_ctx, d, generator=g) * 3, dim=-1)
    log_mu = torch.randn(n_ctx, d, generator=g) * 3

    from src.config import PTConfig

    cfg = PTConfig(vocab_size=V, d=d)
    mf = mfvi.mfvi_readout_logits(Q_Z, S, b, cfg)
    ex = exact.exact_logits(log_mu, S, b)

    assert torch.linalg.matrix_rank(mf, tol=1e-5).item() <= d + 1
    assert torch.linalg.matrix_rank(ex, tol=1e-5).item() > d + 1


@pytest.mark.parametrize("chunk", [0, 1, 3, 5, 1000])
def test_chunked_readout_is_exact(chunk):
    """Chunking bounds memory; it must not change a number."""
    Bkey, S, b, _ = _small(d=4, V=11)
    log_mu = exact.log_mu_slot(Bkey)
    reference = exact.exact_logits(log_mu, S, b)
    assert torch.allclose(exact.exact_logits(log_mu, S, b, chunk=chunk), reference, atol=1e-6)


def test_chunked_readout_gives_the_same_gradients():
    Bkey, S, b, _ = _small(d=4, V=11)
    log_mu = exact.log_mu_slot(Bkey)
    grads = []
    for chunk in (0, 3):
        S_ = S.clone().requires_grad_(True)
        exact.exact_logits(log_mu, S_, b, chunk=chunk).sum().backward()
        grads.append(S_.grad)
    assert torch.allclose(grads[0], grads[1], atol=1e-6)
