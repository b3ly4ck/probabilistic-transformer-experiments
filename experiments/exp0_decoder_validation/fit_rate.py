"""Multi-seed memorisation fit rate against the number of MFVI rounds and the RPE table.

Why this exists. The previous implementation of this project (commit `9c77f94`, removed in
`2e38ef9`) passed every one of checks 1-9 and still failed to learn on PTB: it converged to
val ppl 664 against a unigram baseline of 687, and could not fit its own training data. The
diagnosis recorded there found two things this repository must not inherit:

1. **The number of MFVI rounds was the driver.** Toy memorisation fit rate over five seeds
   fell monotonically: 8/20 at one round, 7/20 at two, 4/20 at three, **1/20 at four**.
2. **Check 6 passed on a single seed and was not representative** — its configuration had a
   fit rate of 1/5. A single-seed memorisation test cannot distinguish a model that fits
   from one that fits sometimes.

That implementation also had `T` of shape `(h, d, d)` — no distance dimension, hence **no
relative positional encoding at all**. Its attention `F_c(i,j)` depended on `j` only through
`q̄_j`, never through `i - j`, so the content stream was a bag of prefix labels and word order
was invisible to it. This repository implements the clipped RPE table of Wu & Tu Eqs. 9/10,
so `gamma` is swept here alongside the round count to see whether that is the difference.

Run:  python -m experiments.exp0_decoder_validation.fit_rate
"""

import time

import torch

from src import CausalPTDecoder, PTConfig

STEPS = 1200
LR = 0.05
FIT = 0.05
SEEDS = (0, 1, 2, 3, 4)


def fit_once(n_iters: int, gamma: int, seed: int, readout: str = "exact", d: int = 16) -> float:
    torch.manual_seed(seed)
    cfg = PTConfig(
        vocab_size=12,
        d=d,
        h=2,
        rank=None,
        gamma=gamma,
        n_iters=n_iters,
        readout=readout,
        init_std=0.5,
    )
    m = CausalPTDecoder(cfg)
    g = torch.Generator().manual_seed(1000 + seed)
    idx = torch.randint(0, 12, (1, 8), generator=g)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    loss = torch.tensor(float("nan"))
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = m.loss(idx)
        loss.backward()
        opt.step()
    return float(loss.detach())


def main() -> None:
    print(f"memorisation fit rate over {len(SEEDS)} seeds, {STEPS} Adam steps, lr {LR}")
    print(f"fit = final loss < {FIT}; d=16 h=2 V=12 n=8 batch=1, exact readout\n")
    for gamma in (0, 3):
        label = "no RPE (gamma=0)" if gamma == 0 else "RPE gamma=3"
        print(f"{label:>18} | " + " ".join(f"T={t:<10}" for t in (1, 2, 3, 4, 5)))
        row, losses = [], []
        for n_iters in (1, 2, 3, 4, 5):
            t0 = time.time()
            final = [fit_once(n_iters, gamma, s) for s in SEEDS]
            n_fit = sum(1 for f in final if f < FIT)
            row.append(f"{n_fit}/{len(SEEDS)} ({time.time() - t0:.0f}s)")
            losses.append(f"{min(final):.3f}/{sorted(final)[len(final) // 2]:.3f}")
        print(f"{'fit rate':>18} | " + " ".join(f"{r:<12}" for r in row))
        print(f"{'best/median loss':>18} | " + " ".join(f"{r:<12}" for r in losses))
        print()


if __name__ == "__main__":
    main()
