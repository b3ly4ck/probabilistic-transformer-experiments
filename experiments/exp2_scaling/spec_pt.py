"""Experiment 2, part B --- the causal PT scaling curve on PTB.

This is the half of the matched-budget comparison that the 2026-08 report could not run. It is
written after `exp4_width_transfer` because a PT ladder at one fixed `alpha_Z` measures the
mis-transfer of `alpha_Z`, not the architecture --- which is exactly what the 2026-08 ladder
did, and why it came out non-monotone (`d = 48` worse than both 32 and 64).

Three ladders, so the curve says which part of the result is the architecture and which is the
hyperparameter rule:

* **`fixed`** --- Wu & Tu's `lambda_H = 1/d` with `alpha_Z` held at the value tuned at
  `d = 32`. This is the 2026-08 protocol extended over width, and it is the control: whatever
  the other two ladders do, this one shows what happens if the constants are not transferred.
* **`alpha`** --- `lambda_H = 1/d` with `alpha_Z` set per width to the argmin of exp4's grid A,
  passed in as `--alpha-map`. A tuned-per-point curve: the strongest thing the source
  parameterisation can do, at the cost of a grid per point.
* **`rule`** --- the belief-norm temperature (`temp_mode` passed in), `alpha_Z` fixed at one
  value for every width. If this ladder tracks `alpha` without a per-width search, the rule
  transfers and the seven-hyperparameter problem Penghao Kuang describes has one fewer entry.

Held fixed: PTB, `h = 2`, `gamma = 3`, `T = 3`, `tau = 2`, MFVI readout, frozen `b`,
`lr = 0.02`, `rank = d`, `l2_arc = 5e-4`, blocks 16 x 64, **15,000 steps** (the budget the
baselines of `spec_baselines.py` use, so the two curves are comparable), validation checkpoint
selection, three seeds at every point.

Three seeds is not decoration here. The reproduction check found the undamped configuration to
be seed-sensitive enough that a single-seed curve could show a slope that is noise; every point
of every ladder therefore carries a spread, and the paper plots the mean with the spread drawn.

Usage::

    python -m experiments.sweep --spec experiments.exp2_scaling.spec_pt \\
        <temp_mode> <qk_gain> <alpha_fixed> <alpha_map>

with `alpha_map` a comma-separated `d:alpha` list, e.g. `16:0.25,32:0.25,64:0.125`.
"""

from experiments.sweep import Cell

WIDTHS = [16, 24, 32, 48, 64, 96, 128]
SEEDS = [0, 1, 2]
STEPS = 15000


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
    )
    defaults.update(kw)
    return Cell(**defaults)


def build_cells(temp_mode="qknorm", qk_gain="1.0", alpha_fixed="0.25", alpha_map="",
                alpha_rule="1.0"):
    qk_gain = float(qk_gain)
    alpha_fixed = float(alpha_fixed)
    # The rule ladder is undamped by default and that is the claim being tested: if the
    # temperature is what the damping was compensating for, the rule should not need it.
    alpha_rule = float(alpha_rule)
    amap = {}
    for part in alpha_map.split(","):
        if part.strip():
            d, a = part.split(":")
            amap[int(d)] = float(a)

    cells = []
    for d in WIDTHS:
        for seed in SEEDS:
            # control: the source parameterisation, constants not transferred
            cells.append(_base(
                name=f"PTfixed_d{d}_s{seed}", d=d, rank=d, alpha_Z=alpha_fixed,
                temp_mode="fixed", seed=seed,
                tags={"ladder": "fixed", "d": d}))
            # tuned per width from exp4 grid A
            if d in amap:
                cells.append(_base(
                    name=f"PTalpha_d{d}_s{seed}", d=d, rank=d, alpha_Z=amap[d],
                    temp_mode="fixed", seed=seed,
                    tags={"ladder": "alpha", "d": d, "alpha_Z": amap[d]}))
            # the transfer rule: one setting for every width
            cells.append(_base(
                name=f"PTrule_d{d}_s{seed}", d=d, rank=d, alpha_Z=alpha_rule,
                temp_mode=temp_mode, qk_gain=qk_gain, seed=seed,
                tags={"ladder": "rule", "d": d, "temp": temp_mode,
                      "alpha_Z": alpha_rule, "qk_gain": qk_gain}))
    return cells
