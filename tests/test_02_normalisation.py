"""Check 2 -- posteriors normalise to 1 along their variable axis, every round."""

import torch

from src import mfvi
from src.pt_decoder import causal_key_mask


def test_qbar_normalised_every_round(model, tokens):
    for rounds in range(1, model.cfg.n_rounds + 1):
        qbar = model.content_stream(tokens, n_rounds=rounds)
        total = qbar.sum(dim=-1)
        assert torch.allclose(total, torch.ones_like(total), atol=1e-6)


def test_head_posterior_normalised_and_masked(model, tokens):
    """Q_c sums to 1 over D_t, and puts exactly zero mass outside it."""
    qbar = model.content_stream(tokens)
    Bkey = mfvi.contract_prefix(qbar, model.T, model.r)
    n = tokens.shape[1]
    mask = causal_key_mask(n)
    scores = torch.einsum("bta,bcka->bctk", qbar, Bkey) / model.cfg.lambda_H
    Q_c = torch.softmax(scores.masked_fill(~mask, mfvi.NEG_INF), dim=-1)

    total = Q_c.sum(dim=-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)
    # every disallowed key carries exactly zero, not merely a small number
    assert (Q_c.masked_select(~mask.expand_as(Q_c)) == 0).all()


def test_global_posterior_normalised(model, tokens):
    """Q_G sums to 1 over the global domain {1..m}."""
    if not model.cfg.use_global_head:
        return
    from src import mfvi

    qbar = model.content_stream(tokens)
    Q_G = mfvi.update_QG(qbar, model.B_global, model.cfg)
    assert Q_G.shape == (*qbar.shape[:-1], model.cfg.n_global)
    total = Q_G.sum(dim=-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)
