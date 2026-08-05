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


def _train(readout, d, batch=1, steps=1200, lr=0.05, seed=0):
    cfg = PTConfig(vocab_size=V, d=d, n_channels=2, n_rounds=3)
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
