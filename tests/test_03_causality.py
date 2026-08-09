"""Check 3 — causality. The single most important test in the project.

Change the token at position t; the logits at every slot s <= t must be *bitwise*
unchanged. Slot s reads only the frozen prefix marginals q_bar_{<s}, which by the
induction of Part II §12.3 are functions of w_{1:s-1} only, so the arithmetic at those
slots is identical and bitwise equality is the right assertion — not allclose.

This runs on CPU with a fixed seed on purpose. On GPU, non-deterministic reduction
kernels can perturb the last bit of *correct* code, which costs a day chasing a bug
that does not exist.
"""

import torch

from conftest import toy_model


def _perturbed(idx, t, new):
    out = idx.clone()
    out[:, t] = new
    return out


def test_future_token_cannot_change_present_logits(idx, readout, schedule):
    m = toy_model(readout=readout, schedule=schedule)
    n = idx.shape[1]
    base = m(idx)

    for t in range(n):
        for new in range(m.cfg.vocab_size):
            alt = _perturbed(idx, t, new)
            if torch.equal(alt, idx):
                continue
            other = m(alt)
            # slot s reads w_{<s}; changing w_t leaves every s <= t untouched
            assert torch.equal(base[:, : t + 1], other[:, : t + 1]), (
                f"changing position {t} to {new} moved the logits at some slot <= {t}"
            )
            # and it must actually matter later, or the mask is over-zealous
            if t + 1 < n:
                assert not torch.equal(base[:, t + 1 :], other[:, t + 1 :])


def test_content_marginals_are_prefix_functions(idx, schedule):
    m = toy_model(schedule=schedule)
    n = idx.shape[1]
    base = m.content_stream(idx)
    for t in range(n):
        alt = _perturbed(idx, t, (int(idx[0, t]) + 3) % m.cfg.vocab_size)
        other = m.content_stream(alt)
        assert torch.equal(base[:, :t], other[:, :t])


def test_appending_a_token_does_not_move_earlier_logits(readout):
    """The generation loop of §18 Check 6 requires this: the prefix is never revised.

    Deliberately ``allclose`` and not ``equal``: the two calls contract tensors of
    different length, so the reduction order is not the model's to promise. Bitwise
    equality is asserted above, where the shapes are identical and it is meaningful.
    """
    m = toy_model(readout=readout)
    torch.manual_seed(4)
    long = torch.randint(0, m.cfg.vocab_size, (2, 7))
    for k in range(1, 8):
        assert torch.allclose(m(long[:, :k]), m(long)[:, :k], atol=1e-12, rtol=0)
