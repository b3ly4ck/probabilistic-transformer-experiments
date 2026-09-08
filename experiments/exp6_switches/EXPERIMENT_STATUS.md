# Experiment 6 — the switch ladder: which difference from a transformer costs what?

## Question

Penghao Kuang, 2026-08-14, relaying Haoyi Wu's encoder-side experiment:

> "Haoyi Wu ... previously conducted an experiment summarizing ten key differences between PT
> and Transformer models, treating them as ten 'switches.' When all switches are turned on, the
> model behaves identically to a Transformer; when all are turned off, it behaves identically to
> the original PT. This framework was used to systematically analyze the distinctions between
> the two architectures."

`src/switches.py` is that ladder for the **decoder**. "PT is worse than a transformer" is not a
finding; which of the ten differences buys back how much of the gap is.

## What is being measured, and what is not

This is analysis code and it deliberately breaks the project's modelling discipline: rungs carry
`W_Q`, `W_V`, `W_O`, a feed-forward block and a free projection into label space — every one of
which names no factor, and the last of which is exactly the output mechanism the construction
rejects. You cannot price a constraint without building the model that violates it. **No rung is
a proposed model.**

## Fidelity of the two endpoints — stated, because it bounds what the ladder can conclude

* **All-transformer is exact.** The module names match `src/gpt.py` so a GPT state dict loads
  with `strict=True`, and `tests/test_15_switches.py` asserts the logits agree to 1e-6.
* **All-PT is an approximation**, with four named residues:
  1. the mixture readout keeps a free `n_embd x d_label` projection producing `mu`, where the
     causal PT reads `mu` straight off the label posterior. Removing it would make the readout
     switch conditional on the state switch and destroy the factorial design;
  2. the distance switch gives a transformer's analogue — a scalar bias per (head, bucket) —
     where PT has a full arc-score matrix per bucket, so content-times-distance is absent;
  3. PT sums `h` channel messages each `d` wide; a transformer concatenates `n_head` heads each
     `n_embd/n_head` wide, and turning concatenation into a sum would change the budget and
     confound the switch it isolates;
  4. `alpha_Z` and `lambda_H` have no switches, being already sweepable in `PTConfig`.

**The residue is measured rather than only stated.** The all-PT rung and the real causal PT
decoder run at the same width, the same corpus, the same loop and the same budget, so the
difference between them is the size of everything the ladder could not express. That number
belongs in the paper next to the ladder, and if it is large the ladder's attributions are
correspondingly weaker.

`n_embd = d_label = 64`. The word–label factor is the transformer's `wte` read in the other
direction, so tying — which `CLAUDE.md` constraint 2 calls forced, not chosen — is only
expressible when the widths agree; at `n_embd = 64, d_label = 32` the ladder silently ran its PT
anchor untied, which was caught by the verification pass on `src/switches.py` before any cell
ran. Matching the widths also makes the readout switch isolate the mixture-versus-linear family
difference with no width reduction confounded into it, since by Prop. 22.1 the two bottlenecks
are equal at matched width.

## Design

Three families at `n_embd = d_label = 64`, `n_layer = 4`, `n_head = 4`, PTB, 15,000 steps,
blocks 16 x 64, validation checkpoint selection, seed 0, **each rung at three learning rates
(1e-3, 3e-3, 1e-2) and scored at its best** — a ladder run at one rate measures the rate's
mismatch with the rung, and the PT end of this ladder wants a rate twenty times the transformer
end's.

* **S1** — the transformer with exactly one PT property. The marginal cost of each.
* **S2** — all-PT with exactly one transformer property restored. The marginal benefit of each.
  A property whose S1 cost is small while its S2 benefit is large only matters in combination,
  which is the interesting case and the one a single ladder hides.
* **L** — the cumulative ladder, flipping in a declared order that puts the two suspected
  carriers last: `weight_sharing, position, norm, attn_out_proj, attn_query_proj, attn_value,
  ffn, residual, state, readout`.

96 cells.

## Falsification

The paper attributes the gap to the simplex state and the rank-`d` mixture readout. That
predicts `state` and `readout` account for most of the cumulative drop and that every other
single switch is cheap. A drop spread evenly across the ten falsifies it, and the honest
conclusion is then that the construction costs a little everywhere. A different switch
dominating is a finding that points at a specific repair.

## Run log

| date | commit | job | shard | notes |
|---|---|---|---|---|
| 2026-09-08 | 2afa95d | queued | 0-5 of 6 | 96 cells |
