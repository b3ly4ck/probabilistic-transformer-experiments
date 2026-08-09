"""Check 6 — overfit a single batch.

Train on one sequence for a few hundred steps; the loss must fall to near zero. If it
plateaus, the model cannot represent its own training data and something in the update
equations is wrong.

Batch size is 1 on purpose. With two sequences in the batch, slot 0 of both is predicted
from D_0 = {ROOT} — an identical context — so two different first tokens would put an
irreducible floor on the loss and the test would be measuring the wrong thing.
"""

import torch

from conftest import toy_model


def _overfit(readout: str, steps: int = 400, lr: float = 0.05):
    m = toy_model(seed=3, dtype=torch.float32, readout=readout, d=16, h=2, n_iters=2, vocab_size=12)
    torch.manual_seed(7)
    idx = torch.randint(0, 12, (1, 8))
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    curve = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = m.loss(idx)
        loss.backward()
        opt.step()
        curve.append(float(loss))
    return m, idx, curve


def test_exact_readout_overfits_one_batch():
    m, idx, curve = _overfit("exact")
    assert curve[-1] < 0.05, f"loss plateaued at {curve[-1]:.4f}; curve {curve[::80]}"
    assert curve[-1] < curve[0]
    pred = m(idx).argmax(-1)
    assert torch.equal(pred, idx)


def test_overfit_is_not_a_single_lucky_seed():
    """A correction forced by the previous implementation's post-mortem.

    That implementation's check 6 passed on seed 0 and had a fit rate of 1/5 across seeds;
    the suite reported a healthy model because it happened to sit on the seed that works.
    A single-seed memorisation test cannot distinguish a model that fits from one that fits
    sometimes, so the check is over seeds and asserts a *rate*.

    The full sweep over the round count and the RPE table lives in
    ``experiments/exp0_decoder_validation/fit_rate.py``; this is its cheap guard.
    """
    finals = []
    for seed in range(4):
        m = toy_model(
            seed=seed, dtype=torch.float32, readout="exact",
            d=16, h=2, n_iters=2, vocab_size=12, gamma=3,
        )
        torch.manual_seed(100 + seed)
        idx = torch.randint(0, 12, (1, 8))
        opt = torch.optim.Adam(m.parameters(), lr=0.05)
        loss = torch.tensor(float("nan"))
        for _ in range(400):
            opt.zero_grad(set_to_none=True)
            loss = m.loss(idx)
            loss.backward()
            opt.step()
        finals.append(float(loss))

    n_fit = sum(1 for f in finals if f < 0.05)
    assert n_fit >= 3, f"memorisation fit rate {n_fit}/4 across seeds; losses {finals}"


def test_mfvi_readout_overfits_one_batch():
    m, idx, curve = _overfit("mfvi")
    assert curve[-1] < 0.05, f"loss plateaued at {curve[-1]:.4f}; curve {curve[::80]}"
    pred = m(idx).argmax(-1)
    assert torch.equal(pred, idx)


def test_loss_is_finite_and_decreasing_on_average():
    _, _, curve = _overfit("exact", steps=120)
    assert all(c == c for c in curve)  # no NaN
    assert sum(curve[-20:]) < sum(curve[:20])
