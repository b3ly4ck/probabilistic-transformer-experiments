"""Experiment 8 --- does the PTB rule transfer to WikiText-2, unchanged?

Why this experiment is small on purpose
---------------------------------------
Penghao Kuang, 2026-08-14:

    "our technical report on scaling PT addresses the challenge of avoiding repeated tuning
     across different parameter counts within the same dataset. However, if both the dataset
     and parameter count are changed simultaneously, our proposed method becomes ineffective.
     At the dataset level, we have not yet found a suitable workaround. Therefore, my
     suggestions are as follows: (1) To avoid repeated tuning, start by training on a single
     dataset rather than multiple different ones."

We follow that advice: every hyperparameter is tuned on PTB and **nothing is retuned here**.
That makes this a test of the rule rather than a second main experiment, and it is the only
form in which a second corpus is worth running at all --- a WikiText-2 number produced by a
fresh grid search would say nothing about transfer, which is the property under test.

The corpus changes two things at once and both are stated rather than controlled away: the
vocabulary goes from 10,000 to about 33,000 types, and the training set from 929,589 to about
2.1M tokens. The parameter *count* therefore moves even at fixed `d`, because the word--label
factor is `|V| x d`. That is unavoidable --- it is what changing corpus means for a model whose
parameters are almost all embedding --- and it is why the comparison here is against baselines
run on the same corpus through the same loop, never against PTB numbers.

What is being asked
-------------------
1. Does the configuration selected on PTB still learn on WikiText-2 --- i.e. land far below the
   corpus unigram --- without any retuning? A no here bounds the generality of everything in
   the paper and must be said plainly.
2. Does the ordering of the three ladders (constants not transferred / tuned per width / one
   transfer rule) survive the corpus change? The rule is only a rule if it does.
3. Is the PT-vs-baseline gap the same size as on PTB at a matched budget?

Design: two PT widths, both ladders, three seeds; transformer and looped baselines at three
widths bracketing the PT budget, at two learning rates each, best taken. 15,000 steps, blocks
16 x 64, validation checkpoint selection --- identical to the PTB protocol, which is the point.

Usage::

    python -m experiments.sweep --spec experiments.exp8_transfer.spec_wt2 \\
        <temp_mode> <qk_gain> <alpha_fixed>
"""

from experiments.sweep import Cell

SEEDS = [0, 1, 2]
STEPS = 15000
PT_WIDTHS = [32, 64]
BASE_WIDTHS = [32, 64, 128]
BASE_LRS = [1e-3, 3e-3]


def build_cells(temp_mode="qnorm", qk_gain="1.0", alpha_fixed="0.25"):
    qk_gain = float(qk_gain)
    alpha_fixed = float(alpha_fixed)
    cells = []

    for d in PT_WIDTHS:
        for seed in SEEDS:
            for ladder, tm in (("fixed", "fixed"), ("rule", temp_mode)):
                cells.append(Cell(
                    name=f"WT2_pt_{ladder}_d{d}_s{seed}",
                    model="pt", corpus="wikitext2",
                    d=d, rank=d, h=2, gamma=3, n_iters=3, tau=2, readout="mfvi",
                    freeze_b=True, alpha_Z=alpha_fixed, temp_mode=tm, qk_gain=qk_gain,
                    lr=2e-2, steps=STEPS, eval_every=500, l2_arc=5e-4,
                    batch_size=16, block_size=64, seed=seed,
                    tags={"grid": "WT2", "ladder": ladder, "d": d}))

    for model in ("gpt", "looped"):
        for e in BASE_WIDTHS:
            for lr in BASE_LRS:
                cells.append(Cell(
                    name=f"WT2_{model}_e{e}_lr{lr:g}",
                    model=model, corpus="wikitext2",
                    n_embd=e, n_layer=4, n_head=4,
                    lr=lr, steps=STEPS, eval_every=500, l2_arc=0.0,
                    batch_size=16, block_size=64, seed=0,
                    tags={"grid": "WT2", "ladder": model, "n_embd": e}))
    return cells
