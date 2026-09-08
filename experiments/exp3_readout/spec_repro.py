"""The reproduction check of the 2026-08 record configuration, over seeds.

See `experiments/repro_check.slurm` for what this is separating. Part C: the same
configuration through the new sweep runner at three seeds, plus the damped configuration at
three seeds as a control -- if the undamped run is chaotic and the damped one is not, that is
the sharpest possible statement of what `alpha_Z` buys, and it is a claim about *variance*
that no single-seed table can make.
"""

from experiments.sweep import Cell


def build_cells(*args):
    cells = []
    for seed in (0, 1, 2):
        cells.append(Cell(name=f"repro_d16_a1_s{seed}", model="pt", d=16, rank=16, h=2,
                          gamma=3, n_iters=3, tau=2, readout="mfvi", freeze_b=True,
                          alpha_Z=1.0, lr=2e-2, steps=6000, eval_every=500, seed=seed,
                          tags={"grid": "repro", "alpha_Z": 1.0, "d": 16}))
        cells.append(Cell(name=f"repro_d16_a025_s{seed}", model="pt", d=16, rank=16, h=2,
                          gamma=3, n_iters=3, tau=2, readout="mfvi", freeze_b=True,
                          alpha_Z=0.25, lr=2e-2, steps=6000, eval_every=500, seed=seed,
                          tags={"grid": "repro", "alpha_Z": 0.25, "d": 16}))
    return cells
