"""Experiment 2, part C --- the causal PT ladder at low Kruskal rank.

Why this exists, and why it is a separate ladder
------------------------------------------------
`spec_pt.py` ran the width ladder at `rank = d`, inherited from Wu & Tu's Table 2 rule
`rank = min(64, d)`. Experiment 4d then measured what that costs and the answer inverts the
choice: at `d = 96`, `rank = 48` gives **242.6 ± 3.3** against `rank = 96`'s 275.6 ± 27.1 --
*better perplexity at half the arc parameters and a fifth of the seed variance* -- and
`rank = 12` gives 257.8 ± 4.5 at 18,624 non-embedding parameters, which is 8x smaller than
`rank = 96` and still ahead of the best `d = 32` configuration at a comparable budget
(282.2 ± 6.6 at 16,448).

Two things follow and both belong on the scaling curve rather than in a footnote. The rank is
not a capacity dimension the model is using --- the arc score reaches inference only contracted
against a belief, `B^(c)_{j,a} = E_{qbar_j}[T^(c)_{a,.}]`, so what is usable is its rank on the
affine hull of the reachable beliefs, and the measured belief sharpness says those are
concentrated. And a full-rank arc table is not merely wasteful but actively harmful at the top of
the width range, presumably because it gives the runaway loop more directions to run away in;
the variance figures say so more clearly than the means.

So the honest scaling curve for this model is width *and* rank, and the configuration the paper
should report is large `d` at low `r`, not the `r = d` line every earlier run used.

Grid: `d in {32, 48, 64, 96, 128, 192, 256}` x `r in {d/8, d/4}` (dropping `r < 8`), three
seeds, standardised head temperature at gain 1, **undamped**, 15,000 steps --- the budget the
baselines of `spec_baselines.py` use, so the two curves are directly comparable. Everything else
at the working region: `h = 2`, `gamma = 3`, `T = 3`, `tau = 2`, MFVI readout, frozen `b`,
`lr = 0.02`, blocks 16 x 64, validation checkpoint selection.

Falsification
-------------
If perplexity at `r = d/8` is materially worse than at `r = d/4` at every width, the rank is a
real capacity axis and the 4d result was a `d = 96` accident. If the whole family is flat in
`r`, then the arc factors are massively over-parameterised in the source configuration and the
paper should say so plainly, because it is the single largest parameter-efficiency finding
available here.
"""

from experiments.sweep import Cell

WIDTHS = [32, 48, 64, 96, 128, 192, 256]
FRACTIONS = [8, 4]
SEEDS = [0, 1, 2]
STEPS = 15000


def build_cells(*args):
    cells = []
    for d in WIDTHS:
        for f in FRACTIONS:
            r = d // f
            if r < 8:
                continue
            for seed in SEEDS:
                cells.append(Cell(
                    name=f"LR_d{d}_r{r}_s{seed}",
                    model="pt", d=d, rank=r, h=2, gamma=3, n_iters=3, tau=2,
                    readout="mfvi", freeze_b=True, alpha_Z=1.0,
                    temp_mode="qknorm", qk_gain=1.0,
                    lr=2e-2, steps=STEPS, eval_every=500, l2_arc=5e-4,
                    batch_size=16, block_size=64, seed=seed,
                    tags={"ladder": "lowrank", "d": d, "rank": r, "frac": f}))
    return cells
