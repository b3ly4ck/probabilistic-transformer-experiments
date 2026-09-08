"""Experiment 4b --- is the standardised-attention gain a constant, or does it drift with width?

Why this grid exists
--------------------
Grid B of `spec_width.py` showed that standardising the head logits over the head domain
(`temp_mode="qknorm"`) revives widths that are entirely dead under Wu & Tu's `lambda_H = 1/d`:
at `d = 32` and `alpha_Z = 1`, 700.7 (the unigram) becomes 293.6, and at `d = 96`, 698.1
becomes 234.4 --- the best number in the project, undamped and at 6,000 rather than 15,000
steps. But the *gain* that works appeared to move: `qk_gain = 2` won at `d = 32` and `d = 96`,
`4` at `d = 48`, `8` at `d = 16`, and the losing gains did not merely lose, they landed exactly
on the unigram. A hyperparameter that has to be re-searched per width is not a transfer rule,
so this has to be settled rather than reported from one seed per cell.

The prediction, stated before the runs
--------------------------------------
Under `qknorm` the logits over the head domain are standardised to zero mean and unit variance
and then multiplied by `qk_gain`, so the attention distribution is `softmax(g z)` with
`z` approximately standard normal over `|D_t|` candidates. The entropy of that softmax depends
on `g` and on `|D_t|` --- **and not on `d` at all**. Selectivity needs the gain to beat the
scale of the maximum of `|D_t|` standard normals, `sqrt(2 ln |D_t|)`, which for a block of 64
positions is about 2.9. So the theory says: the useful gain is a constant of order 2--4, set by
the context length, and the apparent width dependence in grid B is seed noise from a model the
reproduction check has independently shown to be chaotic when undamped.

That is a falsifiable prediction with a clean alternative. If the argmin over `g` is flat in
`d` across three seeds, `qknorm` is a genuine transfer rule and the seven-hyperparameter
problem \\citep{kuang2026scaling} loses one entry. If the argmin moves monotonically with `d`,
the rule is not a rule, the honest report is a width-dependent gain, and grid B's revival is
still a result but a weaker one.

Design
------
`d in {32, 64, 96, 128}` x `qk_gain in {1, 1.5, 2, 3, 4, 6}` x seeds `{0, 1, 2}`, undamped
(`alpha_Z = 1`) so the temperature is doing all the work, 6,000 steps, everything else at the
2026-08 working region (`h = 2`, `gamma = 3`, `T = 3`, `tau = 2`, MFVI readout, frozen `b`,
`lr = 0.02`, `rank = d`). 72 cells.

Three seeds per cell is the point of the grid, not a refinement of it: grid B's every cell was
a single seed, and a single seed cannot distinguish "this gain fails at this width" from "this
run fell into the bad basin". The deliverable is an argmin *with a spread*, per width.
"""

from experiments.sweep import Cell

WIDTHS = [32, 64, 96, 128]
GAINS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
SEEDS = [0, 1, 2]
STEPS = 6000


def build_cells(*args):
    cells = []
    for d in WIDTHS:
        for g in GAINS:
            for seed in SEEDS:
                cells.append(Cell(
                    name=f"G_d{d}_g{g:g}_s{seed}",
                    model="pt", d=d, rank=d, h=2, gamma=3, n_iters=3, tau=2,
                    readout="mfvi", freeze_b=True, alpha_Z=1.0,
                    temp_mode="qknorm", qk_gain=g,
                    lr=2e-2, steps=STEPS, eval_every=500, l2_arc=5e-4,
                    batch_size=16, block_size=64, seed=seed,
                    tags={"grid": "gain", "d": d, "qk_gain": g}))
    return cells
