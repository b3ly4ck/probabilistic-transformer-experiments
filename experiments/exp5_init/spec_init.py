"""Experiment 5 — initialisation, and the B.3.3 global head.

The questions, verbatim
-----------------------
Prof. Tu, 2026-08-11, on the global head:

    "B.3.3 global head: Uniform initialisation (of what? B?) does not sound like a good
     choice. How does it work with random initialisation?"

Penghao Kuang, 2026-08-14:

    "Regarding the hyperparameter initialization scheme, I noticed that you are using uniform
     distribution initialization. In fact, when attempting to derive quantitative laws, the
     initialization method is also a critical component. ... Therefore, parameter
     initialization is not only relevant to experimental outcomes but may also hold
     underlying theoretical significance for PT, given its rigorous mathematical foundations."

A terminological correction is owed first, and it is the reason this experiment exists in
this shape. **Every factor in this implementation has always been initialised from a Gaussian**
`N(0, init_std^2)`, `init_std = 0.02` (the nanoGPT convention; neither paper specifies one).
The word "uniform" in the 2026-08 report named the *treatment*, not the distribution: `B'` was
given the same scale and the same L2 coefficient as every other factor, and under that
treatment its rows collapse onto one row replicated `m` times (measured row spread
0.0197 -> 0.0037 at `d = 16` while within-row structure stayed at 0.90). The report should
have said "symmetric" or "undifferentiated". The literal question is nonetheless a good one
and is answered here as asked, by making the distribution a knob (`init_dist`).

The mechanism the grids test
----------------------------
The global head's message is `sum_k Q_g(k) B'_{k,.}` with `Q_g = softmax(q B'^T / lambda_G)`.
If the rows of `B'` are close together relative to their own length, the logits `q B'^T` are
nearly equal, `Q_g` is uniform, the message is the row mean, and then
`d(message)/d B'_{k,.} = 1/m` for **every** `k` — an exactly symmetric gradient that cannot
separate the rows. The only row-separating term comes through the softmax Jacobian and is of
the order of the row spread itself, and a shared L2 penalty shrinks that spread further. The
degeneracy is therefore self-locking from any near-symmetric start, and the two things that
set "near-symmetric" are the initialisation scale and the penalty. Grid A varies exactly
those two and nothing else.

Grids
-----
* **A -- the global head, scale x penalty.** `b_glob_init_std` x `regularise_global_head` at
  `d = 24` and `d = 32`, `m = 64`. This is the direct answer to Prof. Tu: the head is run
  across two decades of random-initialisation scale, with and without the L2, and the row
  spread and `H(Q_g)` are logged alongside perplexity so the verdict is mechanistic rather
  than a single number. The 2026-08 report has exactly two points of this grid (0.02 with
  L2, which died; 0.5 without, which revived) and calls the verdict "conditional at best".
* **B -- the number of features `m`.** At the best cell of A, `m` in {16, 64, 256}. The
  2026-08 report never ran `m = 256` because the conditional rule for running it was not
  met; if the head lives, its capacity axis is a real question.
* **C -- the distribution, literally.** `init_dist` in {normal, uniform, orthogonal} for the
  whole factor list, at two widths. Orthogonal is included because it is the natural
  *maximally asymmetric* initialisation: semi-orthogonal rows are as far from collapsed as a
  fixed scale allows, so if the symmetry argument above is right, orthogonal `B'` should
  survive the shared L2 that kills a Gaussian one.
* **D -- the overall scale.** `init_std` at `d = 32`. The 2026-08 sweep varied this at
  `d = 256, lr = 1e-3` -- outside the working region on two axes -- and found nothing; inside
  the region it has never been varied.
* **E -- a width rule for the arc scores.** `arc_init_std` following `0.02`, `0.02 sqrt(32/d)`
  and `0.02 (32/d)` across `d`. This is the initialisation half of the transfer question
  (exp4 is the temperature half): if the head message scales with `d` at fixed `init_std`,
  the scale that keeps it invariant is a rule, and a rule is what transfers.

Everything else is held at the 2026-08 working region: PTB, `h = 2`, `gamma = 3`, `T = 3`,
`tau = 2`, MFVI readout, frozen `b`, `lr = 0.02`, `rank = d`, 6,000 steps, validation
checkpoint selection. Seed 0 for the maps; the winning cells are replicated in `spec_pt`.
"""

from experiments.sweep import Cell

STEPS = 6000


def _base(**kw):
    defaults = dict(
        model="pt",
        h=2,
        gamma=3,
        n_iters=3,
        tau=2,
        readout="mfvi",
        freeze_b=True,
        lr=2e-2,
        steps=STEPS,
        eval_every=500,
        l2_arc=5e-4,
        batch_size=16,
        block_size=64,
        seed=0,
    )
    defaults.update(kw)
    return Cell(**defaults)


GLOB_STDS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]


def build_cells(*args):
    cells = []

    # --- A: the global head, initialisation scale x L2 -----------------------------
    for d in (24, 32):
        for std in GLOB_STDS:
            for reg in (True, False):
                cells.append(
                    _base(
                        name=f"A_d{d}_bg{std:g}_{'L2' if reg else 'noL2'}",
                        d=d,
                        rank=d,
                        n_global=64,
                        b_glob_init_std=std,
                        regularise_global_head=reg,
                        tags={"grid": "A", "d": d, "b_glob_init_std": std, "reg": reg},
                    )
                )
        # the reference without the head, at the identical budget
        cells.append(
            _base(name=f"A_d{d}_nohead", d=d, rank=d, n_global=0,
                  tags={"grid": "A", "d": d, "b_glob_init_std": None, "reg": None})
        )

    # --- B: how many global features ------------------------------------------------
    for m in (16, 64, 256):
        for std, reg in ((0.5, False), (0.02, True)):
            cells.append(
                _base(
                    name=f"B_m{m}_bg{std:g}_{'L2' if reg else 'noL2'}",
                    d=32,
                    rank=32,
                    n_global=m,
                    b_glob_init_std=std,
                    regularise_global_head=reg,
                    tags={"grid": "B", "m": m, "b_glob_init_std": std, "reg": reg},
                )
            )

    # --- C: the distribution, taken literally ---------------------------------------
    for dist in ("normal", "uniform", "orthogonal"):
        for d in (16, 32):
            cells.append(
                _base(name=f"C_{dist}_d{d}", d=d, rank=d, init_dist=dist,
                      tags={"grid": "C", "init_dist": dist, "d": d, "n_global": 0})
            )
        # and with the global head under the *shared* treatment that kills a Gaussian one
        cells.append(
            _base(name=f"C_{dist}_d32_head", d=32, rank=32, init_dist=dist, n_global=64,
                  b_glob_init_std=0.02, regularise_global_head=True,
                  tags={"grid": "C", "init_dist": dist, "d": 32, "n_global": 64})
        )

    # --- D: overall initialisation scale, inside the working region -----------------
    for std in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        cells.append(
            _base(name=f"D_init{std:g}", d=32, rank=32, init_std=std,
                  tags={"grid": "D", "init_std": std})
        )

    # --- E: a width rule for the arc-score scale ------------------------------------
    for d in (16, 32, 64, 128):
        for rule, std in (
            ("const", 0.02),
            ("sqrt", 0.02 * (32.0 / d) ** 0.5),
            ("linear", 0.02 * (32.0 / d)),
        ):
            cells.append(
                _base(
                    name=f"E_d{d}_{rule}",
                    d=d,
                    rank=d,
                    arc_init_std=std,
                    tags={"grid": "E", "d": d, "rule": rule, "arc_init_std": std},
                )
            )
    return cells
