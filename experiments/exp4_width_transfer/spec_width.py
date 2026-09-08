"""Experiment 4 — does the causal PT transfer across width, and what has to move with `d`?

The question, and why it is the one that unblocks everything else
----------------------------------------------------------------
The 2026-08 report ends with the matched-budget comparison **blocked**: undamped, the model
collapses above `d = 24` labels; damped at `alpha_Z = 0.25` it reaches `d = 32` and then goes
non-monotone (`d = 48` worse than both 32 and 64). Every scaling curve the reviewers asked
for runs through this. Penghao Kuang put the general form of it plainly:

    "Hyperparameter tuning for PT is indeed a cumbersome task. The original PT architecture
    involves seven hyperparameters that drift with changes in parameter count (whereas
    Transformers typically involve only one: the learning rate)."

So: which hyperparameters drift with `d`, and is the drift a law or a fit? A law is a
transfer rule and makes the scaling curve one sweep; a fit makes it a grid per point.

The mechanism under test
------------------------
Wu & Tu App. A.5 sets the head-update temperature to `lambda_H = 1/d`. The head logit is

    F_c(i, j) = <q_i, B^(c)_{j,.}>,      B^(c)_{j,a} = E_{q_bar_j}[T^(c)_{a,.}]

and **both** vectors in that inner product are expectations under label *distributions*. A
distribution on `d` labels has Euclidean norm between `1/sqrt(d)` (uniform) and `1` (one-hot).
At initialisation both beliefs are near uniform, `F ~ sigma_T / d`, and dividing by `1/d`
leaves an `O(sigma_T)` logit: the source's choice is exactly right *there*. After training
sharpens the beliefs, `F ~ sigma_T` and the same division leaves `O(d sigma_T)`. The
temperature is calibrated for a limit the model leaves in the first few hundred steps, and
the error is linear in `d`.

That predicts, in order:

1. The runaway `q_bar -> B -> G -> q_bar` loop should get worse with `d` — measured: it does
   (`d = 32` undamped, msg/unary 2.88 -> 20.57 within 500 steps).
2. `alpha_Z` should have to *fall* with `d`, because damping the label update is exactly what
   keeps `||q||_2` near the uniform value where the temperature is calibrated. Grid A maps
   `alpha*(d)` and tests this.
3. A temperature that tracks the belief norm should remove the `d`-dependence at its source.
   `temp_mode="qnorm"` divides by `||q_i||_2 sqrt(d)`, which is **1 at the uniform belief** —
   so it agrees with Wu & Tu at initialisation and differs only where the source's derivation
   does not reach. Grid B tests this. If it works, `alpha*(d)` should flatten to a constant
   in Grid C, and that constant is the transfer rule.

Grids
-----
* **A — `alpha*(d)`**, `temp_mode="fixed"`: `d x alpha_Z`. The map the 2026-08 report lists
  as open, and the direct answer to "does the optimal step size scale with width".
* **B — the temperature rule**, `alpha_Z = 1`: `d x temp_mode`. Undamped on purpose: the
  claim is that the right temperature removes the *need* for damping, and a grid run under
  damping could not show that.
* **C — interaction**: the two survivors of A and B crossed, at the widths that matter.
* **D — channels**: `h x temp_mode` at `d = 32`. The 2026-08 deconfound found channels
  degrade the model at every width, with the bound `|G_i(a)| <= h max|T|` as the mechanism.
  Sharper attention makes the message closer to a single arc-score row and therefore larger;
  if the temperature is what drives the sharpening, `h` should hurt less once it is fixed.

Held fixed across every cell: PTB, `lr = 0.02`, `h = 2` (except D), `gamma = 3`, `T = 3`,
`tau = 2`, MFVI readout, frozen `b` at the corpus log-unigram, `rank = d`, blocks 64 x 16,
6,000 steps with checkpoint selection on validation. `rank = d` keeps the non-embedding count
an exact `2 K h d^2 + h d`, so the width axis is a clean `d^2` law, and reproduces the
2026-08 record configuration at `d = 32` (16,448 non-embedding parameters).

6,000 steps rather than the record's 15,000 because this is a *comparative* map: the 2026-08
runs show the same configuration at 6,000 and 15,000 steps ranking the same way (473.28 vs
336.89 at `d = 16`) while costing 2.5x less. The winner is re-run at full budget in `spec_pt`.

Falsification
-------------
The mechanism is wrong if (a) `alpha*(d)` in Grid A is flat — the drift is then not width-
driven at all; or (b) `qnorm` in Grid B does not push the collapse point above `d = 32` while
`fixed` at the same `alpha_Z = 1` collapses at `d = 32` as previously measured. Either
outcome is reportable and both are recorded whichever way they fall.
"""

from experiments.sweep import Cell

WIDTHS = [16, 24, 32, 48, 64, 96, 128]
ALPHAS = [1.0, 0.5, 0.25, 0.125]
STEPS = 6000


def _base(**kw):
    """Everything the 2026-08 working region fixed, in one place."""
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


TEMPS = [
    ("fixed", dict(temp_mode="fixed")),
    ("qnorm", dict(temp_mode="qnorm")),
    ("qkn2", dict(temp_mode="qknorm", qk_gain=2.0)),
    ("qkn4", dict(temp_mode="qknorm", qk_gain=4.0)),
    ("qkn8", dict(temp_mode="qknorm", qk_gain=8.0)),
]


def build_cells(*args):
    cells = []

    # --- Grid A: alpha*(d) at the source temperature -------------------------------
    for d in WIDTHS:
        for a in ALPHAS:
            cells.append(
                _base(
                    name=f"A_d{d}_a{a:g}",
                    d=d,
                    rank=d,
                    alpha_Z=a,
                    tags={"grid": "A", "d": d, "alpha_Z": a, "temp": "fixed"},
                )
            )

    # --- Grid B: the temperature rule, undamped ------------------------------------
    for d in WIDTHS:
        for tname, tkw in TEMPS:
            if tname == "fixed":
                continue  # already in grid A at alpha_Z = 1
            cells.append(
                _base(
                    name=f"B_d{d}_{tname}",
                    d=d,
                    rank=d,
                    alpha_Z=1.0,
                    tags={"grid": "B", "d": d, "alpha_Z": 1.0, "temp": tname},
                    **tkw,
                )
            )

    # --- Grid C: does a corrected temperature remove the need for damping? ---------
    for d in [32, 64, 128]:
        for tname, tkw in TEMPS:
            if tname == "fixed":
                continue
            for a in [0.5, 0.25]:
                cells.append(
                    _base(
                        name=f"C_d{d}_{tname}_a{a:g}",
                        d=d,
                        rank=d,
                        alpha_Z=a,
                        tags={"grid": "C", "d": d, "alpha_Z": a, "temp": tname},
                        **tkw,
                    )
                )

    # --- Grid D: channels, at the source temperature and at the corrected one ------
    for h in [2, 4, 8]:
        for tname, tkw in [("fixed", dict(temp_mode="fixed")), ("qnorm", dict(temp_mode="qnorm")),
                           ("qkn4", dict(temp_mode="qknorm", qk_gain=4.0))]:
            cells.append(
                _base(
                    name=f"D_h{h}_{tname}",
                    d=32,
                    rank=32,
                    h=h,
                    alpha_Z=1.0,
                    tags={"grid": "D", "d": 32, "h": h, "temp": tname},
                    **tkw,
                )
            )
    return cells
