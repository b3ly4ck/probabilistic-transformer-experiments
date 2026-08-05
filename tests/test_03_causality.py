"""Check 3 -- changing token t leaves the logits at positions <= t bitwise unchanged.

Run on CPU with a fixed seed.  If the implementation is genuinely independent of
future tokens the arithmetic at earlier positions is identical, so bitwise
equality must hold; on GPU, non-deterministic reductions could perturb the last
bit on correct code.
"""

import torch

from tests.conftest import SEQ


def test_future_token_cannot_move_earlier_logits(model, tokens):
    for readout in ("exact", "mfvi"):
        base = model(tokens, readout=readout)
        for p in range(SEQ):
            perturbed = tokens.clone()
            perturbed[:, p] = (perturbed[:, p] + 7) % model.cfg.vocab_size
            other = model(perturbed, readout=readout)
            # position t is predicted from qbar_{<t}, so every t <= p is untouched
            assert torch.equal(base[:, : p + 1], other[:, : p + 1]), (
                f"{readout}: changing token {p} moved logits at positions <= {p}"
            )


def test_the_change_does_reach_later_positions(model, tokens):
    """Guards the test above: if nothing ever moved, it would pass vacuously."""
    perturbed = tokens.clone()
    perturbed[:, 0] = (perturbed[:, 0] + 7) % model.cfg.vocab_size
    base = model(tokens)
    other = model(perturbed)
    assert not torch.allclose(base[:, 1:], other[:, 1:])


def test_first_position_sees_only_root(model, tokens):
    """D_0 = {ROOT}: the logits at t=0 cannot depend on any token at all."""
    other_tokens = (tokens + 3) % model.cfg.vocab_size
    assert torch.equal(model(tokens)[:, 0], model(other_tokens)[:, 0])
