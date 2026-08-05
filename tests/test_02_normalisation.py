"""Check 2 -- posteriors normalise to 1 along their variable axis, every round."""

import torch

from src import mfvi
from src.pt_decoder import causal_key_mask
from tests.conftest import TOY


def test_qbar_normalised_every_round(model, tokens):
    for rounds in range(1, TOY.n_rounds + 1):
        qbar = model.content_stream(tokens, n_rounds=rounds)
        total = qbar.sum(dim=-1)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-6)


def test_head_posterior_normalised_and_masked(model, tokens):
    """Q_c sums to 1 over D_t, and puts exactly zero mass outside it."""
    qbar = model.content_stream(tokens)
    Bkey = mfvi.contract_prefix(qbar, model.T, model.r)
    n = tokens.shape[1]
    mask = causal_key_mask(n)
    scores = torch.einsum("bta,bcka->bctk", qbar, Bkey) / TOY.lambda_H
    Q_c = torch.softmax(scores.masked_fill(~mask, mfvi.NEG_INF), dim=-1)

    total = Q_c.sum(dim=-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)
    # every disallowed key carries exactly zero, not merely a small number
    assert (Q_c.masked_select(~mask.expand_as(Q_c)) == 0).all()
