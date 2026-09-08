# Experiment 3 — the readout, and the 2026-08 report's open questions

## Two things live here

### 1. Exact vs. mean-field readout (Experiment 3 of the research plan)

Part III offers two ways to turn the slot posterior into a word distribution: the mean-field
readout of 17.1 and the exact sum-product readout of 17.2 (the slot graph is a tree, so
sum-product is exact). Part IV 23.3 walks the recommendation from the first to the second.
The only trained exact-readout number on record is **1555.04** at `d = 256, lr = 1e-3` — from
the dead region, before the working region was known, and therefore uninformative.

Two measurements are needed and they are different:

* **train with each readout** — `spec_open.py` grid X;
* **swap the readout at evaluation on the same trained weights** — now measured automatically
  for *every* PT cell in `experiments/sweep.py`, at the cost of one extra pass. It is only
  meaningful at `lambda_W = 1` (17.2 Check 5): any other value makes the swap measure a
  temperature rather than the price of the mean-field approximation. `n_global > 0` is
  excluded because the head's contribution to the exact readout is a measured constant (22.2).

### 2. The reproduction check (`spec_repro.py`, `experiments/repro_check.slurm`)

The first cell of the exp4 grid is bit-for-bit job 940848 of 2026-08-10 (`d = 16, h = 2,
lr = 0.02`, frozen `b`, `alpha_Z = 1`, 6,000 steps, seed 0), recorded at **val 430.04**.
Through the new runner it landed at **610.88** — first evaluation agreeing to 0.1
(711.61 vs 711.7), traces separating after it. Three candidate causes, separated on one GPU:

| | old `src` (cc28e96) | new `src` |
|---|---|---|
| old driver (`ptb_attack.py`) | A | B |
| new runner (`sweep.py`) | — | C, three seeds |

* A != B -> the refactor changed the mathematics.
* B != C -> the harness differs from the old driver.
* A == B == C, with a large spread across seeds -> **the undamped configuration is chaotic
  and 430.04 was one draw of it.**

The third is a live possibility rather than an excuse: the 2026-08 report documents this very
configuration reverting 516.1 -> 693.9 between two evaluations and then recovering, and names
that excursion as the evidence that the message/unary balance gates learning. A run passing
through such an excursion is not expected to reproduce across two GPUs with different
reduction orders. If it holds, **no single-seed number from the undamped configuration can
appear in a paper**, and the damped configuration's three-seed spread (2.32) becomes the
sharpest available statement of what `alpha_Z` buys.

`spec_repro.py` runs the undamped and the damped configuration at three seeds each, so the
variance comparison is the deliverable, not a by-product.

## The other open questions of the 2026-08 report (`spec_open.py`)

Every earlier attempt at these ran at `d = 256, h = 8, lr = 1e-3` — outside the working
region on all four axes at once, so a null result there measured nothing.

| grid | question | what it settles |
|---|---|---|
| R | replication of the frozen-`b` gain, the record, and the `d = 48` anomaly, 3 seeds each | a gain that does not survive three seeds is not a gain; this session already withdrew one record (241.89) for exactly that |
| S | budget vs. schedule (`min_lr_frac = 1.0` vs `0.1` at both budgets) | the report's largest unexplained jump, 473 -> 337, is confounded by the cosine schedule being defined over `max_steps` |
| T | `T in {1,2,3,5,8}` and `tau in {1,2,4}` inside the region | the axis PT shares with a Looped Transformer; Experiment 2's framing depends on whether PT's depth buys anything |
| G | `gamma in {0,1,3,7}` on PTB inside the region | correction #4 blames the predecessor's failure on missing RPE using a *toy* fit-rate table; this measures it on real data |
| X | exact vs MFVI readout, trained, at `d in {16, 32}` | Experiment 3 proper |
| L | `l2_arc in {0 .. 5e-2}` | never varied inside the region; only relevant since `train < val` first appeared at the damped record |

## Run log

| date | commit | job | notes |
|---|---|---|---|
| 2026-09-08 | 14419ae | 998640 | reproduction check A/B/C |
| 2026-09-08 | 14419ae | queued | `spec_open`, 37 cells, 5 shards |
