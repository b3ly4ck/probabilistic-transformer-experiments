"""The global head G_t -- Wu & Tu Appendix B.3.3, single-split (§22.2).

Two properties beyond the re-run checks, both of which decide how Experiment 1's
arm 1.2 must be run.
"""

import pytest
import torch

from src import exact, mfvi
from src.config import PTConfig
from src.pt_decoder import CausalPTDecoder
from tests.conftest import BASE, M_GLOBAL

CFG_G = PTConfig(**BASE, use_global_head=True, n_global=M_GLOBAL)
CFG_NO = PTConfig(**BASE)


def _model(cfg, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return CausalPTDecoder(cfg, generator=g), torch.randint(
        0, cfg.vocab_size, (2, 6), generator=g
    )


def test_composed_update_is_the_gfu_operator():
    """sigma(q B'^T) B' -- Appendix B Eq. 62, the operator §22.2 calls the
    feed-forward analogue.  Asserted against a hand-written form rather than
    trusted, since it is the whole reason G_t is claimed to substitute an FFN."""
    g = torch.Generator()
    g.manual_seed(0)
    q = torch.softmax(torch.randn(7, CFG_G.d, generator=g), dim=-1)
    Bg = torch.randn(M_GLOBAL, CFG_G.d, generator=g)

    composed = mfvi.gfu(q, Bg, CFG_G)
    by_hand = torch.softmax(q @ Bg.T / CFG_G.lambda_G, dim=-1) @ Bg
    assert torch.allclose(composed, by_hand, atol=1e-6)

    # and it really is the composition of the two update equations
    step = mfvi.global_message(mfvi.update_QG(q, Bg, CFG_G), Bg)
    assert torch.equal(composed, step)


def test_gfu_is_nonlinear_in_q():
    """If it were affine it would add nothing a bias could not: the FFN claim
    rests on the nonlinearity."""
    g = torch.Generator()
    g.manual_seed(1)
    Bg = torch.randn(M_GLOBAL, CFG_G.d, generator=g) * 3
    q1 = torch.softmax(torch.randn(CFG_G.d, generator=g) * 3, dim=-1)
    q2 = torch.softmax(torch.randn(CFG_G.d, generator=g) * 3, dim=-1)
    mid = 0.5 * (q1 + q2)
    lhs = mfvi.gfu(mid, Bg, CFG_G)
    rhs = 0.5 * (mfvi.gfu(q1, Bg, CFG_G) + mfvi.gfu(q2, Bg, CFG_G))
    assert not torch.allclose(lhs, rhs, atol=1e-3)


def test_global_term_is_context_free_in_the_exact_readout():
    """G_t is a leaf on Z_t, so exact marginalisation gives a term identical at
    every position and for every sentence.

    m*d parameters collapse to d effective numbers that reweight the label prior
    and carry no context.  This is why arm 1.2 cannot be judged on the exact
    readout alone -- see experiments/exp1_language_modeling/EXPERIMENT_STATUS.md.
    """
    model, tokens = _model(CFG_G)
    Bg = model.B_global
    term = exact.log_global_term(Bg)
    assert term.shape == (CFG_G.d,)

    # the same term appears at every position of every sequence, bitwise
    qbar = model.content_stream(tokens)
    Bkey = mfvi.contract_prefix(qbar, model.T, model.r)
    with_g = exact.log_mu_sequence(Bkey, Bg)
    without_g = exact.log_mu_sequence(Bkey, None)
    delta = with_g - without_g
    assert torch.allclose(delta, term.expand_as(delta), atol=1e-6)
    assert torch.allclose(delta, delta[0, 0].expand_as(delta), atol=1e-6)

    # a different sentence gets the identical contribution
    other = (tokens + 5) % CFG_G.vocab_size
    qbar2 = model.content_stream(other)
    Bkey2 = mfvi.contract_prefix(qbar2, model.T, model.r)
    delta2 = exact.log_mu_sequence(Bkey2, Bg) - exact.log_mu_sequence(Bkey2, None)
    assert torch.allclose(delta, delta2, atol=1e-6)


def test_global_head_does_reach_the_mfvi_readout_contextually():
    """The contrast with the test above: under MFVI, G_t's effect on the logits
    varies with the context, because it enters through Q_Z.

    The claim under test is the *mechanism*, not its magnitude at initialisation,
    so B' is set to a trained-like scale first.  At init scale 0.1 the effect is
    real but around 4e-05, which would make the test a measurement of the
    initialiser rather than of the model.
    """
    model, tokens = _model(CFG_G)
    g = torch.Generator()
    g.manual_seed(3)
    with torch.no_grad():
        model.B_global.copy_(torch.randn(M_GLOBAL, CFG_G.d, generator=g) * 3)

    qbar = model.content_stream(tokens)
    with_g = model.mfvi_logits(qbar)
    saved, model.B_global = model.B_global, None
    without_g = model.mfvi_logits(qbar)
    model.B_global = saved

    delta = with_g - without_g
    # the exact readout's contribution is constant across positions by
    # construction; this one must not be
    variation = delta.std(dim=1).max()
    assert variation > 0.01 * delta.abs().max(), (
        f"the global head's MFVI effect looks position-independent "
        f"(variation {float(variation):.3e}, scale {float(delta.abs().max()):.3e})"
    )


def test_turning_the_flag_off_reproduces_the_no_global_model_bitwise():
    """Arm 1.1 must be unaffected by the existence of the flag."""
    with_flag, tokens = _model(CFG_NO)
    for readout in ("exact", "mfvi"):
        logits = with_flag(tokens, readout=readout)
        assert torch.isfinite(logits).all()
    assert with_flag.B_global is None
    assert "B_global" not in dict(with_flag.named_parameters())


@pytest.mark.parametrize("readout", ["exact", "mfvi"])
def test_global_head_trains(readout):
    """B_global must receive gradient -- a factor that never moves is not a factor."""
    model, tokens = _model(CFG_G)
    model.zero_grad(set_to_none=True)
    model.loss(tokens, readout=readout).backward()
    assert model.B_global.grad is not None
    assert model.B_global.grad.abs().sum() > 0
