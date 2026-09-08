# Experiment 7 — factored labels: does the product-of-mixtures readout buy anything?

**Written on 2026-09-09, after the grid started.** Question, design and falsification criterion
are copied from the docstring of `spec_factored.py`, written before any cell ran.

## The question, and the note's own prediction

Part IV §22.2 calls factored labels "the flagship option" for widening the model without leaving
the graph, and neither the note nor Wu & Tu ever runs them. One label variable of width `d`
becomes `K` variables of width `d/K` sharing the word variable; the slot stays a tree, so the
exact readout becomes a *product* of `K` mixtures instead of one mixture.

The note states its own falsifiable prediction, which is what makes this an experiment:

> Stated honestly: under the mean-field readout, factoring buys nothing over flat width D
> (logits = b + Σ_k S^(k) q^(k), affine dimension ≤ D); the gain exists only through the exact
> readout.

So the design is a factorial in (readout, `K`) and **the prediction is an interaction**: flat in
`K` under `mfvi`, improving in `K` under `exact`.

## What would falsify it

A main effect of `K` under `mfvi` falsifies the note's own analysis. No effect under `exact` says
the richer family is not reachable by this optimiser at this scale — a different and equally
reportable conclusion, and one the readout result of Experiment 3 now makes likely in advance.

## The correction this experiment already produced

The note says the parameter count is "identical to a flat label of width D". True of the
word–label factors only. The arc factors are per-component `(d/K) × (d/K)`, so at `rank = d/K`
the non-embedding count falls as `1/K²`. The grid is therefore built around **iso-parameter
diagonals** — `(d,K)` pairs with identical non-embedding counts — rather than around the note's
comparison, and the honest claim is not "same parameters, richer family" but *at a fixed
non-embedding budget, factoring buys total label width*.

## What is held fixed

PTB, `h = 4`, `gamma = 3`, `T = 3`, `tau = 2`, `alpha_Z = 0.25`, frozen `b`, `lr = 0.02`,
`rank = d/K`, blocks 16 × 64, 15,000 steps, validation checkpoint selection, three seeds.
`h = 4` rather than the usual 2 because the channel assignment is what carries the
factorisation. The sweep runner scores every cell under the *other* readout at no training cost,
so "the exact readout is a better family" and "training under it finds better parameters" can be
told apart.

## Structural half, already confirmed

At `d = 8` the affine dimension of the log-conditionals is 8, 7 and 5 for `K = 1, 2, 4` under the
mean-field readout — never above `d`, whatever `K` is — and strictly above `d` under the exact
readout on the same contexts (`tests/test_18_factored.py`). `K = 1` reproduces the flat decoder
to 6.9e-18.

## Run log

| date | commit | job | shards | result |
|---|---|---|---|---|
| 2026-09-09 | e0052d3 | 999035, 999037, 999038 | 0-2 of 6 | **In progress**, 3/48 cells. Last grid in the queue. |
