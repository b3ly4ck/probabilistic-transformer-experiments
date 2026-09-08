"""Experiment 2, part A — the GPT and Looped Transformer scaling curves on PTB.

What this answers
-----------------
Prof. Tu, 2026-08-11: *"A fair comparison is needed for PT vs. looped transformer with the
same model size."* Penghao Kuang, 2026-08-14: *"plot a series of scaling curves showing model
performance as a function of parameter count ... this would provide a comprehensive
comparison between Causal PT and Looped Transformers."*

The 2026-08 report compared one PT point (16,448 non-embedding parameters) against one GPT
and one Looped point (1,237,440 and 309,600). That is not a comparison of architectures, it
is a comparison of budgets. A curve makes the question well posed: at each budget, which
architecture reaches the lower perplexity, and how do the curves' *slopes* differ.

This spec produces the two baseline curves. The PT curve is `spec_pt.py`, and it is written
after the width-transfer experiment (exp4) fixes how PT's hyperparameters move with `d` --
because a PT curve built at one fixed `alpha_Z` measures the mis-transfer of `alpha_Z`, not
the architecture.

Design
------
* **Width family**: `n_embd in {16, 24, 32, 48, 64, 96, 128, 160, 256}` at `n_layer = 4`.
  The bottom of the range is where the total parameter count meets PT's: at `n_embd = 32`
  the GPT holds ~371k parameters against PT's 346k at `d = 32`.
* **Depth family**: `n_layer in {1, 2, 8}` at `n_embd = 64`. Depth is the axis the Looped
  model shares with PT (its loop count is PT's `T`), so a curve over depth alone separates
  "more computation" from "more parameters" -- for the Looped model the depth axis adds
  *no* parameters at all, which is exactly PT's situation.
* **Learning rate**: every point runs at `lr in {1e-3, 3e-3}` and the better of the two is
  the point on the curve. The 2026-08 baselines used a single rate taken from Wu & Tu's
  Table 2, which is PT's tuned rate and not necessarily a transformer's; a curve whose
  points are individually under-tuned understates the baseline, and the reviewers' question
  is precisely about fairness.
* **Budget**: 15,000 steps for every model, batch 16, block 64 -- the budget the PT record
  used. Checkpoint selection is by validation inside the shared loop (`keep_best`), so a
  model that reaches its optimum early is not graded past it. PTB has 929,589 training
  tokens; 15,000 steps of 1,024 tokens is 16.5 epochs, and a 1.2M-parameter transformer is
  well past its validation optimum by then.

Falsification
-------------
If the Looped curve lies *below* the GPT curve at PT's budget, the "structure without weight
sharing" framing of Experiment 2 is wrong and weight sharing is a win, not a cost, at this
scale. If the two baseline curves are flat below ~100k parameters, then PT's budget region is
one where nothing works well and the comparison says little; the curve is what tells us.
"""

from experiments.sweep import Cell

WIDTHS = [16, 24, 32, 48, 64, 96, 128, 160, 256]
DEPTHS = [1, 2, 8]
LRS = [1e-3, 3e-3]
STEPS = 15000


def build_cells(*args):
    cells = []
    for model in ("gpt", "looped"):
        for e in WIDTHS:
            for lr in LRS:
                cells.append(
                    Cell(
                        name=f"{model}_e{e}_L4_lr{lr:g}",
                        model=model,
                        n_embd=e,
                        n_layer=4,
                        n_head=4,
                        lr=lr,
                        steps=STEPS,
                        eval_every=500,
                        l2_arc=0.0,
                        tags={"family": "width", "axis_value": e},
                    )
                )
        for L in DEPTHS:
            for lr in LRS:
                cells.append(
                    Cell(
                        name=f"{model}_e64_L{L}_lr{lr:g}",
                        model=model,
                        n_embd=64,
                        n_layer=L,
                        n_head=4,
                        lr=lr,
                        steps=STEPS,
                        eval_every=500,
                        l2_arc=0.0,
                        tags={"family": "depth", "axis_value": L},
                    )
                )
    return cells
