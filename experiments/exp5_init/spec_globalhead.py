"""Experiment 5b --- the B.3.3 global head, measured where the measurement can be trusted.

Why the first pass is not enough
--------------------------------
`spec_init.py` grid A varied the global head's initialisation scale and its L2 treatment at one
seed per cell, at Wu & Tu's head temperature and undamped. That is the region the reproduction
check independently showed has a seed standard deviation of **130 perplexity**, and the grid
duly came back scattered: at `d = 32` the same head at `b_glob_init_std = 0.02` gives 337.7
without the L2 and 465.0 with it, at 0.1 it gives 587.2 and 675.7, and at 0.25 it gives 383.2
and 323.0 -- an ordering that reverses twice and cannot be read.

One thing in that grid is nonetheless unambiguous and worth carrying forward: at `d = 32` the
configuration **without** the head sits exactly on the unigram (700.7), while several cells
*with* it land between 323 and 380. The head is not merely a capacity addition there; it rescues
a configuration that otherwise collapses, presumably because a second message into `Z_t` that
does not pass through the arc factors breaks the runaway loop. That is a mechanism claim and it
deserves a measurement that can support it.

So this grid re-asks Prof. Tu's question in the configuration the width experiment selected ---
standardised head temperature at gain 1, undamped, which gives a seed spread of 1.9 at `d = 64`
against 130 in the old region --- and at three seeds.

    "B.3.3 global head: Uniform initialisation (of what? B?) does not sound like a good choice.
     How does it work with random initialisation?"  -- Prof. Kewei Tu, 2026-08-11

Two things are answered separately, because the 2026-08 report conflated them under the word
"uniform". **The distribution**: every factor here has always been Gaussian, and `init_dist`
now also offers a matched-variance uniform and a semi-orthogonal draw --- the last being the
*maximally asymmetric* initialisation at a fixed scale, which the row-collapse mechanism
predicts should survive a shared L2 that kills a Gaussian one. **The treatment**: the scale of
`B'` and whether it enters the same L2 term as the other factors, which is what the 2026-08
report actually varied.

Grid, all at `d in {32, 64}`, `h = 2`, standardised temperature gain 1, undamped, 15,000 steps,
three seeds:

* no head (the reference at each width);
* `m = 64` x `b_glob_init_std in {0.02, 0.1, 0.5}` x L2 on/off;
* at the winning treatment, `m in {16, 256}` --- the capacity axis, never run in 2026-08;
* at the winning treatment, `init_dist in {uniform, orthogonal}` --- the literal question, and
  the sharp prediction: an orthogonal `B'` should survive the shared L2.

Falsification
-------------
The row-collapse account says the head dies when its rows start close together relative to their
own length and the L2 keeps them there. It is wrong if the orthogonal draw dies exactly as the
small-scale Gaussian one does, or if the head's survival turns out not to depend on the
initialisation scale once the variance is small enough to see.
"""

from experiments.sweep import Cell

WIDTHS = [32, 64]
SEEDS = [0, 1, 2]
STEPS = 15000


def _base(**kw):
    d = dict(model="pt", h=2, gamma=3, n_iters=3, tau=2, readout="mfvi", freeze_b=True,
             alpha_Z=1.0, temp_mode="qknorm", qk_gain=1.0, lr=2e-2, steps=STEPS,
             eval_every=500, l2_arc=5e-4, batch_size=16, block_size=64)
    d.update(kw)
    return Cell(**d)


def build_cells(*args):
    cells = []
    for d in WIDTHS:
        for seed in SEEDS:
            cells.append(_base(name=f"H_d{d}_nohead_s{seed}", d=d, rank=d, n_global=0,
                               seed=seed, tags={"grid": "H", "d": d, "m": 0}))
            for std in (0.02, 0.1, 0.5):
                for reg in (True, False):
                    cells.append(_base(
                        name=f"H_d{d}_m64_bg{std:g}_{'L2' if reg else 'noL2'}_s{seed}",
                        d=d, rank=d, n_global=64, b_glob_init_std=std,
                        regularise_global_head=reg, seed=seed,
                        tags={"grid": "H", "d": d, "m": 64, "std": std, "reg": reg}))
    # capacity axis and the literal distribution question, at the width that works best
    for seed in SEEDS:
        for m in (16, 256):
            cells.append(_base(name=f"H_d64_m{m}_s{seed}", d=64, rank=64, n_global=m,
                               b_glob_init_std=0.1, regularise_global_head=False, seed=seed,
                               tags={"grid": "M", "d": 64, "m": m}))
        for dist in ("uniform", "orthogonal"):
            cells.append(_base(name=f"H_d64_{dist}_L2_s{seed}", d=64, rank=64, n_global=64,
                               b_glob_init_std=0.02, regularise_global_head=True,
                               init_dist=dist, seed=seed,
                               tags={"grid": "D", "d": 64, "init_dist": dist}))
    return cells
