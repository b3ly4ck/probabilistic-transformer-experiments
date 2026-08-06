"""Check 6 -- the model can overfit a fixed batch.

If it plateaus, the model cannot represent its own training data and something
in the update equations is wrong.  Two properties of the construction shape what
"overfit" can mean here, and both are asserted rather than worked around:

  * ``D_0 = {ROOT}``.  The first position has no prefix, so its logits are the
    same for every sequence in the batch.  Two sequences with different first
    tokens therefore have an information-theoretic floor of log 2 at t = 0.
  * The exact readout needs a wider label space than the mean-field one at this
    scale.  ``log mu_t`` is pooled evidence with no query (§23.3 removes the
    query stream), so the context signal is d numbers accumulated by LSE.
"""

import math

import pytest
import torch

from src.config import PTConfig
from src.pt_decoder import CausalPTDecoder

V, N = 12, 6


def _train(readout, d, batch=1, steps=1200, lr=0.05, seed=0, use_global=False, lambda_G=5.0):
    cfg = PTConfig(
        vocab_size=V,
        d=d,
        n_channels=2,
        n_rounds=3,
        use_global_head=use_global,
        n_global=5 if use_global else 0,
        lambda_G=lambda_G,
    )
    g = torch.Generator()
    g.manual_seed(seed)
    model = CausalPTDecoder(cfg, generator=g)
    tokens = torch.randint(0, V, (batch, N), generator=g)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    first = float(model.loss(tokens, readout=readout).detach())
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = model.loss(tokens, readout=readout)
        loss.backward()
        opt.step()
    return first, float(loss.detach()), model, tokens


@pytest.mark.parametrize("readout,d", [("exact", 32), ("mfvi", 8)])
def test_overfits_a_single_sequence(readout, d):
    first, last, model, tokens = _train(readout, d)
    assert last < first
    assert last < 0.05, f"{readout} (d={d}) plateaued at {last}"
    assert torch.equal(model(tokens, readout=readout).argmax(dim=-1), tokens)


def test_overfits_with_the_global_head_under_mfvi():
    """Arm 1.2 memorises under the mean-field readout.

    Needs 2400 steps rather than 1200 -- the global head slows convergence -- and
    ``lambda_G = 5`` rather than the config default of 1, since at 1 the head
    collapses the label posterior entirely (asserted separately below).
    """
    first, last, model, tokens = _train("mfvi", d=8, use_global=True, steps=2400)
    assert last < first
    assert last < 0.05, f"mfvi with the global head plateaued at {last}"
    assert torch.equal(model(tokens, readout="mfvi").argmax(dim=-1), tokens)


def test_global_head_is_degenerate_under_the_exact_readout():
    """G_t cannot act as a feed-forward layer under exact inference.

    Not a bug and not a tuning failure: G_t is a leaf on Z_t, so exact inference
    integrates it out and what survives -- LSE_k B'[k,a] -- cannot depend on
    position.  What it *can* do is perturb the content stream, and it does: the
    same model that memorises a sequence to 0.002 without the head stops at 0.45
    with it, because the near-constant GFU term compresses the spread of qbar
    across positions that the LSE-pooling readout depends on.

    Measured to be out of reach of hyperparameters: lambda_G at 5, 20 and 100
    give a final loss identical to four decimals, with ||B'|| landing on 21.9 in
    all three, because Q_G converges near-uniform (max ~0.20 at m=5).

    Asserted here so the property is executable rather than a note.  Arm 1.2 of
    Experiment 1 runs under the MFVI readout for this reason.
    """
    _, without, _, _ = _train("exact", d=32, use_global=False)
    _, with_head, _, _ = _train("exact", d=32, use_global=True, steps=2400)

    assert without < 0.05, f"the baseline stopped memorising ({without})"
    assert with_head > 0.3, (
        f"the global head no longer degrades the exact readout ({with_head}); "
        "if this changed, arm 1.2's readout decision must be revisited"
    )


def test_position_zero_carries_an_irreducible_floor():
    """D_0 = {ROOT}: t=0 is predicted from no context at all, so two sequences
    with different first tokens cannot both be fitted.  Everything after t=0
    still must be."""
    _, _, model, tokens = _train("exact", d=32, batch=2)
    assert tokens[0, 0] != tokens[1, 0], "fixture no longer exercises the floor"
    with torch.no_grad():
        logits = model(tokens)
        per_token = torch.nn.functional.cross_entropy(
            logits.reshape(-1, V), tokens.reshape(-1), reduction="none"
        ).reshape(2, N)
    at_zero = float(per_token[:, 0].mean())
    assert abs(at_zero - math.log(2)) < 0.1, f"t=0 loss {at_zero}, expected ~log 2"
    assert float(per_token[:, 1:].mean()) < 0.3


def test_exact_readout_needs_more_label_width_than_mean_field():
    """An empirical property of the two readouts, asserted so it cannot silently
    change: at d=8 the exact readout fails to memorise a sequence the mean-field
    readout memorises.  Widening d to 32 fixes it.  Relevant to Experiment 3."""
    _, exact_narrow, _, _ = _train("exact", d=8)
    _, exact_wide, _, _ = _train("exact", d=32)
    _, mfvi_narrow, _, _ = _train("mfvi", d=8)

    assert exact_narrow > 0.3, f"exact at d=8 unexpectedly memorised ({exact_narrow})"
    assert exact_wide < 0.05
    assert mfvi_narrow < 0.05


def test_global_head_collapses_at_low_lambda_G():
    """At lambda_G = 1 the global head prevents memorisation entirely.

    Mechanism: the composed update is sigma(q B'^T) B'.  As B' grows the softmax
    saturates onto one global feature k*, and the message becomes a large
    constant vector B'[k*, :] added to every slot's label logits.  Q_Z is then
    pinned regardless of the word or the prefix, and the loss stops at what a
    context-free predictor achieves.  Measured: |B'| reaches ~10-20 within 300
    Adam steps and the loss plateaus near 0.48 where the same model without the
    global head reaches 0.005.

    lambda_G is not pinned by the source (see PTConfig.lambda_G), so this is a
    live open question for Experiment 1, recorded as an executable fact rather
    than a note.
    """
    _, collapsed, _, _ = _train("mfvi", d=8, use_global=True, lambda_G=1.0)
    _, healthy, _, _ = _train("mfvi", d=8, use_global=True, lambda_G=5.0)
    _, baseline, _, _ = _train("mfvi", d=8, use_global=False)

    assert collapsed > 0.3, f"the collapse at lambda_G=1 has changed ({collapsed})"
    assert healthy < 0.1
    assert baseline < 0.05
