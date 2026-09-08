"""Experiment 7 --- factored labels: does the product-of-mixtures readout buy anything?

The proposal, and why it has never been tested
----------------------------------------------
Part IV 22.2 of the technical note calls factored labels "the flagship option" for widening
the model without leaving the graph, and 26 lists them as rung 4 of the localisation ladder.
Neither the note nor Wu & Tu ever runs them. The construction replaces one label variable of
width `d` by `K` label variables of width `d/K` sharing the word variable, with each head
channel assigned a child component `k(c)` and a head component `k'(c)`. The slot stays a tree,
so the exact readout becomes a *product* of `K` mixtures instead of one mixture.

The note states its own falsifiable prediction, which is what makes this an experiment rather
than a sweep:

    "Stated honestly: under the mean-field readout, factoring buys nothing over flat width D
     (logits = b + sum_k S^(k) q^(k), affine dimension <= D); the gain exists only through the
     exact readout."

So the design is a factorial in (readout, K) and the prediction is an **interaction**: flat in
`K` under `mfvi`, improving in `K` under `exact`. A main effect of `K` under `mfvi` would
falsify the note's own analysis; no effect under `exact` would say the richer family is not
reachable by this optimiser at this scale, which is a different and equally reportable
conclusion.

What the parameter count actually does --- corrected
----------------------------------------------------
The note says the parameter count is "identical to a flat label of width D". That is true of
the **word--label factors only**: `|V| K (d/K) = |V| d`, so the embedding is untouched by `K`.
It is *not* true of the arc factors. Those are per-component `(d/K) x (d/K)` matrices, so with
`rank = d/K` the non-embedding count falls as `1/K^2`:

======  ===  =========  ==============
`d`     `K`  embedding  non-embedding
======  ===  =========  ==============
32      1    330,000    32,896
32      2    330,000     8,256
64      2    650,000    32,896
64      4    650,000     8,256
128     4    1,290,000  32,896
======  ===  =========  ==============

(`h = 4`, `gamma = 3`, `rank = d/K`; measured, not derived from memory.) Read down the table
and the useful fact appears: **at a fixed non-embedding budget, factoring buys total label
width.** `(d{=}64, K{=}2)` and `(d{=}32, K{=}1)` cost the same 32,896 arc parameters, but the
first has twice the label width and, under the exact readout, a product of two mixtures rather
than one. That, and not "same parameters, richer family", is the honest claim.

The grids therefore run along two directions, and both are needed:

* **Family P --- iso-parameter diagonals.** `(d, K)` pairs with identical non-embedding counts:
  `{(16,1), (32,2), (64,4)}` at 8,256 and `{(32,1), (64,2), (128,4)}` at 32,896. Along a
  diagonal the arc budget is constant and only the factorisation changes, so a difference is
  attributable. The embedding grows with `d` along the diagonal and that is reported, not
  hidden --- a wider label set costs `|V|` more parameters whatever `K` is.
* **Family W --- fixed total width.** `d = 32` and `d = 64` with `K in {1,2,4}`. This is the
  note's literal comparison, in which the factored model is strictly *cheaper*; if it matches
  the flat model here it wins on both axes, and if it loses, the loss is bounded by a known
  budget reduction.

Held fixed: PTB, `h = 4`, `gamma = 3`, `T = 3`, `tau = 2`, `alpha_Z = 0.25`, frozen `b`,
`lr = 0.02`, `rank = d/K`, blocks 16 x 64, 15,000 steps, validation checkpoint selection,
three seeds. `h = 4` rather than the usual 2 because the channel assignment is what carries
the factorisation: under the round-robin rule with `h = K^2` every (child, head) component
pair occurs exactly once, and at `h = 2` with `K = 4` most pairs never occur at all. Holding
`h` fixed across the `K` axis keeps the two from confounding; a `h = 2` bridge row is kept so
the grid connects to the configuration the rest of the paper uses.

The sweep runner additionally scores every cell under the *other* readout at no training cost,
so the evaluation-time swap arrives alongside the trained comparison and the two can be told
apart --- "the exact readout is a better family" and "training under the exact readout finds
better parameters" are different claims.
"""

from experiments.sweep import Cell

SEEDS = [0, 1, 2]
STEPS = 15000
READOUTS = ("mfvi", "exact")

# (d, K) pairs sharing a non-embedding budget; measured with h = 4, gamma = 3, rank = d/K.
DIAGONALS = {
    8256: [(16, 1), (32, 2), (64, 4)],
    32896: [(32, 1), (64, 2), (128, 4)],
}
FIXED_WIDTH = {32: [1, 2, 4], 64: [1, 2, 4]}


def _base(**kw):
    defaults = dict(
        model="pt",
        h=4,
        gamma=3,
        n_iters=3,
        tau=2,
        freeze_b=True,
        alpha_Z=0.25,
        lr=2e-2,
        steps=STEPS,
        eval_every=500,
        l2_arc=5e-4,
        batch_size=16,
        block_size=64,
    )
    defaults.update(kw)
    return Cell(**defaults)


def build_cells(*args):
    cells = []
    seen = set()

    def add(d, K, readout, seed, family, budget=None, h=4):
        key = (d, K, readout, seed, h)
        if key in seen:
            return
        seen.add(key)
        cells.append(_base(
            name=f"F{family}_d{d}_K{K}_{readout}_h{h}_s{seed}",
            d=d, rank=d // K, h=h, n_components=K, readout=readout, seed=seed,
            tags={"grid": family, "d": d, "K": K, "readout": readout, "h": h,
                  "budget": budget}))

    for budget, pairs in DIAGONALS.items():
        for d, K in pairs:
            for readout in READOUTS:
                for seed in SEEDS:
                    add(d, K, readout, seed, "P", budget)

    for d, Ks in FIXED_WIDTH.items():
        for K in Ks:
            for readout in READOUTS:
                for seed in SEEDS:
                    add(d, K, readout, seed, "W")

    # bridge row: h = 2, the configuration every other experiment in the paper uses
    for K in (1, 2):
        for readout in READOUTS:
            for seed in SEEDS:
                add(32, K, readout, seed, "B", h=2)
    return cells
