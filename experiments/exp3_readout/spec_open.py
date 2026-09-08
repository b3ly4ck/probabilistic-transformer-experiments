"""Open questions from the 2026-08 report, closed inside the working region.

The 2026-08 report's "Open" table lists nine items. Six of them are cheap, are prerequisites
for a preprint that claims to know what its own model does, and share one property: every
earlier attempt at them was run *outside* the working region -- at `d = 256`, `h = 8`,
`lr = 1e-3` -- which the region probe later showed is outside on all four axes at once. A
sweep of `tau` that finds nothing at `d = 256` has not measured `tau`.

Everything here is at the 2026-08 working region and changes exactly one thing per cell:
PTB, `d = 32`, `h = 2`, `rank = d`, `gamma = 3`, `T = 3`, `tau = 2`, MFVI readout, frozen `b`,
`lr = 0.02`, `alpha_Z = 0.25`, 15,000 steps, blocks 64 x 16, validation checkpoint selection.
`d = 16` is carried alongside where the 2026-08 record for it exists, because it is the one
width with an undamped reference number (315.46).

The items, and what each settles
--------------------------------
* **R -- replication.** The frozen-`b` gain (473.28 -> 430.04) and the damped record are
  single- and three-seed results respectively; the `d = 48` anomaly (the only damped run with
  `train > val`) is one seed. Three seeds each. A gain that does not survive three seeds is
  not a gain -- this session already withdrew one record (241.89) for exactly that reason.
* **S -- budget vs. schedule.** The report's largest unexplained jump, 473 -> 337, is
  confounded: the cosine schedule is defined over `max_steps`, so the 15,000-step run is not
  "the same run, longer". `min_lr_frac = 1.0` makes the rate constant after warmup and
  separates the two; run at both budgets it says how much of the gain was length.
* **T -- inference depth `T` and `tau`.** `T` is the content-stream iteration count (the
  depth of the equivalent weight-shared transformer) and `tau` the predictive rounds. Both
  were swept at `d = 256` and moved nothing. Inside the region they are the axis PT shares
  with a Looped Transformer, and Experiment 2's whole framing depends on knowing whether
  PT's depth buys anything.
* **G -- the RPE clip `gamma`.** Correction #4 of the report attributes the *predecessor*
  implementation's failure to having no relative positional encoding, on a toy fit-rate
  table. `gamma = 0` (distance-blind) against `gamma in {1, 3, 7}` on PTB inside the region
  turns that from an inherited explanation into a measurement.
* **X -- the exact readout, trained.** Experiment 3 proper. The only trained exact-readout
  number in the record is 1555.04 at `d = 256, lr = 1e-3`, i.e. from the dead region. Two
  things are needed and are different: training *with* the exact readout, and swapping the
  readout at evaluation on weights trained with the other one. The swap is now measured
  automatically for every PT cell in the sweep runner, so this grid supplies the missing
  half -- models actually trained under `readout="exact"`.
* **L -- the L2 coefficient on the arc scores.** Wu & Tu Table 2 gives 5e-4 for PTB masked
  LM. The report notes `train < val` for the first time at the damped record, which is the
  first time regularisation could matter at all; the coefficient has never been varied inside
  the region.
"""

from experiments.sweep import Cell

FULL = 15000


def _base(**kw):
    defaults = dict(
        model="pt",
        d=32,
        rank=32,
        h=2,
        gamma=3,
        n_iters=3,
        tau=2,
        readout="mfvi",
        freeze_b=True,
        alpha_Z=0.25,
        lr=2e-2,
        steps=FULL,
        eval_every=500,
        l2_arc=5e-4,
        batch_size=16,
        block_size=64,
        seed=0,
    )
    defaults.update(kw)
    return Cell(**defaults)


def build_cells(*args):
    cells = []

    # --- R: replication of every single-seed claim that the report leans on ---------
    for seed in (0, 1, 2):
        cells.append(_base(name=f"R_freezeb_on_s{seed}", d=16, rank=16, alpha_Z=1.0,
                           freeze_b=True, seed=seed, tags={"grid": "R", "what": "freeze_b_on"}))
        cells.append(_base(name=f"R_freezeb_off_s{seed}", d=16, rank=16, alpha_Z=1.0,
                           freeze_b=False, seed=seed, tags={"grid": "R", "what": "freeze_b_off"}))
        cells.append(_base(name=f"R_d48_s{seed}", d=48, rank=48, seed=seed,
                           tags={"grid": "R", "what": "d48_anomaly"}))
        cells.append(_base(name=f"R_record_s{seed}", seed=seed,
                           tags={"grid": "R", "what": "record_d32"}))

    # --- S: budget vs schedule -------------------------------------------------------
    for steps in (6000, 15000):
        for floor in (0.1, 1.0):
            cells.append(
                _base(name=f"S_steps{steps}_floor{floor:g}", steps=steps, min_lr_frac=floor,
                      tags={"grid": "S", "steps": steps, "min_lr_frac": floor})
            )

    # --- T: inference depth ----------------------------------------------------------
    for T in (1, 2, 3, 5, 8):
        cells.append(_base(name=f"T_iters{T}", n_iters=T, tags={"grid": "T", "n_iters": T}))
    for tau in (1, 2, 4):
        cells.append(_base(name=f"T_tau{tau}", tau=tau, tags={"grid": "T", "tau": tau}))

    # --- G: the relative positional encoding -----------------------------------------
    for g in (0, 1, 3, 7):
        cells.append(_base(name=f"G_gamma{g}", gamma=g, tags={"grid": "G", "gamma": g}))

    # --- X: the exact readout, trained -----------------------------------------------
    for d in (16, 32):
        for ro in ("exact", "mfvi"):
            cells.append(
                _base(name=f"X_d{d}_{ro}", d=d, rank=d, readout=ro,
                      alpha_Z=0.25 if d == 32 else 1.0,
                      tags={"grid": "X", "d": d, "readout": ro})
            )

    # --- L: the L2 coefficient on the arc scores -------------------------------------
    for l2 in (0.0, 5e-5, 5e-4, 5e-3, 5e-2):
        cells.append(_base(name=f"L_l2{l2:g}", l2_arc=l2, tags={"grid": "L", "l2_arc": l2}))

    return cells
