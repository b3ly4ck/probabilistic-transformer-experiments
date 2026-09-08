"""Experiment 4d --- the Kruskal rank, which is PT's cheapest parameter axis.

The arc score of channel `c` is decomposed as `T^(c) = U^(c) V^(c)^T` with `U, V` of shape
`d x r` (Wu & Tu Eqs. 14/21), so the arc factors cost `2 K h d r` parameters against `K h d^2`
for a full table --- and every experiment in this project so far has run at `r = d`, which is
the *most expensive* setting of the decomposition and buys nothing a full `T` would not.
The 2026-08 configurations inherited `rank = min(64, d)` from Wu & Tu's Table 2 row, where it
saves parameters only while `2r < d`; at `d <= 64` that rule sets `r = d` and doubles the arc
cost relative to a full table.

That matters here for one reason: the scaling curve of \\Cref{sec:exp-scaling} plots perplexity
against parameter count, and a factor of two on the axis is a factor of two on the axis. If a
lower rank costs no perplexity, the causal PT's whole curve moves left by that factor, and the
matched-budget comparison changes accordingly. If it does cost perplexity, then the rank is a
real capacity dimension and the curve is where it is --- either way the axis has to be measured
rather than inherited.

Grid: `d in {32, 96}` x `r in {d/8, d/4, d/2, d}` (dropping `r < 4`) x 3 seeds, at the
standardised temperature and gain 1 undamped --- the configuration the width experiment
selected --- 6,000 steps, everything else at the working region.

Prediction: the arc score enters the model only through `B^(c)_{j,a} = E_{qbar_j}[T^(c)_{a,.}]`,
a contraction against a distribution, so what the model can use is the rank of `T` restricted to
the affine hull of the reachable beliefs. If the beliefs are far from spanning the simplex ---
and the measured belief sharpness says they are concentrated --- then `r` well below `d` should
cost little. A sharp drop at some `r` would locate the effective dimensionality of the label
dynamics, which is a quantity nothing else in this paper measures.
"""

from experiments.sweep import Cell

WIDTHS = [32, 96]
FRACTIONS = [8, 4, 2, 1]  # r = d // fraction
SEEDS = [0, 1, 2]
STEPS = 6000


def build_cells(*args):
    cells = []
    for d in WIDTHS:
        for f in FRACTIONS:
            r = d // f
            if r < 4:
                continue
            for seed in SEEDS:
                cells.append(Cell(
                    name=f"R_d{d}_r{r}_s{seed}",
                    model="pt", d=d, rank=r, h=2, gamma=3, n_iters=3, tau=2,
                    readout="mfvi", freeze_b=True, alpha_Z=1.0,
                    temp_mode="qknorm", qk_gain=1.0,
                    lr=2e-2, steps=STEPS, eval_every=500, l2_arc=5e-4,
                    batch_size=16, block_size=64, seed=seed,
                    tags={"grid": "rank", "d": d, "rank": r, "frac": f}))
    return cells
