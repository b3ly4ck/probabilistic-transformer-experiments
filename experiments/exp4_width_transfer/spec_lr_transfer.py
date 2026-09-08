"""Experiment 4c --- does the optimal learning rate transfer across width?

This is the diagnostic that maximal update parametrisation is judged by
\\citep{yang2022tensor}, and that \\citet{kuang2026scaling} apply to the probabilistic
transformer *encoder*: plot validation loss against learning rate for several widths, and look
at whether the minima line up. Under a parametrisation that transfers, the curves for different
widths have their minimum at the same learning rate and one small-model search suffices for
every larger model. Under one that does not, the minimum walks, and every width needs its own
search --- which is the "seven hyperparameters that drift with parameter count" that Penghao
Kuang names as the practical obstacle.

The 2026-08 report has one slice of this and it is the reason the whole project has a working
region at all: on a synthetic chain, the causal PT reached the oracle at `d = 16, lr = 0.02`
and got nothing at `d = 256` at any rate tested, while a GPT control was insensitive to the
combination. That is a *joint* dependence in `(d, lr)`, i.e. exactly a failure of transfer,
observed but never mapped on real data.

The grid
--------
`d in {16, 32, 64, 128}` x `lr in {5e-3, 1e-2, 2e-2, 4e-2, 8e-2}` x two parametrisations:

* `fixed` --- Wu & Tu's `lambda_H = 1/d`, undamped;
* `qkn2`  --- the standardised head temperature at gain 2, undamped.

40 cells, seed 0, 6,000 steps, otherwise the 2026-08 working region (`h = 2`, `gamma = 3`,
`T = 3`, `tau = 2`, MFVI readout, frozen `b`, `rank = d`).

Undamped on purpose in both arms. `alpha_Z` is itself one of the drifting hyperparameters
(\\Cref{sec:exp-width}), so damping the grid would let a well-chosen `alpha_Z` hide a failure of
learning-rate transfer, and the question here is what the *temperature* alone does.

Prediction
----------
If the head temperature is the mechanism behind the width sensitivity, then under `fixed` the
argmin over `lr` should move to the left as `d` grows (a wider model needs a smaller rate to
keep the message from running away), and under `qkn2` it should not move. A flat argmin under
both would say the width sensitivity lives somewhere else entirely --- in the initialisation
scale, say, which \\Cref{sec:exp-init} varies --- and would be a clean negative result for the
temperature account.

One seed per cell, which is enough to see a minimum walk across a decade of learning rate but
not enough to call a 5\\% difference; the winning column is re-run with three seeds in
`spec_gain.py`, and any claim made from this grid is about the *location* of the minimum, not
about the value at it.
"""

from experiments.sweep import Cell

WIDTHS = [16, 32, 64, 128]
LRS = [5e-3, 1e-2, 2e-2, 4e-2, 8e-2]
ARMS = [("fixed", dict(temp_mode="fixed")),
        ("qkn2", dict(temp_mode="qknorm", qk_gain=2.0))]
STEPS = 6000


def build_cells(*args):
    cells = []
    for arm, kw in ARMS:
        for d in WIDTHS:
            for lr in LRS:
                cells.append(Cell(
                    name=f"LR_{arm}_d{d}_lr{lr:g}",
                    model="pt", d=d, rank=d, h=2, gamma=3, n_iters=3, tau=2,
                    readout="mfvi", freeze_b=True, alpha_Z=1.0,
                    lr=lr, steps=STEPS, eval_every=500, l2_arc=5e-4,
                    batch_size=16, block_size=64, seed=0,
                    tags={"grid": "lr_transfer", "arm": arm, "d": d, "lr": lr},
                    **kw))
    return cells
