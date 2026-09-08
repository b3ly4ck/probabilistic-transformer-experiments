"""One cell whose only purpose is to leave a checkpoint of the current best configuration.

The interpretability probes of `experiments/interpretability.py` read a trained model; the
sweep runner does not save one unless asked, and the only checkpoints on disk are from the
2026-08 configuration (`d = 32`, `alpha_Z = 0.25`, the source head temperature). Probing those
is legitimate but answers a question about the old configuration, so this trains the one the
width and rank experiments selected -- `d = 96`, `rank = 12`, standardised temperature at
gain 1, undamped -- and saves it.

It is a duplicate of a cell in `exp2_scaling/spec_pt_lowrank.py` except for `save_ckpt`, and
that is deliberate: adding `save_ckpt` to the ladder itself would change every cell's identity
and invalidate the resume of a grid already half finished.
"""

from experiments.sweep import Cell


def build_cells(*args):
    return [Cell(
        name="probe_d96_r12", model="pt", d=96, rank=12, h=2, gamma=3, n_iters=3, tau=2,
        readout="mfvi", freeze_b=True, alpha_Z=1.0, temp_mode="qknorm", qk_gain=1.0,
        lr=2e-2, steps=15000, eval_every=500, l2_arc=5e-4, batch_size=16, block_size=64,
        seed=0, save_ckpt=True, tags={"grid": "probe"})]
