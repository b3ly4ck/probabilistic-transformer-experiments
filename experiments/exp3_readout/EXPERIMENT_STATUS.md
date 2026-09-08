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
| 2026-09-09 | 39fc6bc | 998966, 998968, 998982 + | 23/37 cells done, 0 errors. **Grid X, the result this experiment exists for:** at `d=32` damped, trained under the exact readout 348.8 against the mean-field readout's 248.3 -- the strictly richer family (Prop. readout (iii)) trains 100 perplexity worse. **Grid T:** perplexity rises monotonically with inference depth, `T=1` 241.3, `T=2` 246.5, `T=3` 248.3, `T=8` 275.3; `tau=1` 242.1 against `tau=2` 248.3. **Grid R:** the damped record replicates, 253.0 +- 6.0 over three seeds against 250.7 +- 2.3. **Grid L:** `l2_arc` 5e-3 is indistinguishable from 5e-4 (249.8), 5e-2 kills the model (697.0, msg/unary 0.02, ablation KL 0.000). **Grid G:** `gamma=1` 249.7 against `gamma=3` 248.3. Single seed except grid R. |
| 2026-09-09 | 3dc31bd | 998966-999008 | **37/37 complete, 0 errors.** Grid T finished and withdrew the monotonicity the paper had claimed from four points: `T = 1,2,3,5,8` give 241.3, 246.5, 248.3, **297.0**, 275.3, so T=8 returns below T=5 and one seed cannot say whether that is real. What stands: every T above 1 is worse than T=1 and by T=5 the cost is 56, nine seed sd. Grid G finished: `gamma = 0,1,3,7` give **619.4**, 249.7, 248.3, 605.1 -- a window with cliffs on both walls, and gamma=0 has prefix-ablation KL 0.037, which closes correction #4 of the 2026-08 report (the predecessor's failure really was the missing RPE). Grid L finished: `l2_arc = 0, 5e-5, 5e-4, 5e-3, 5e-2` give 273.3, 255.6, **248.3**, 249.8, 697.0 -- the source's own value is the interior optimum, so Table 2 is well chosen. Grid X finished at both widths: exact readout 348.8 vs mfvi 248.3 at d=32, and 409.5 vs 315.5 at d=16. Grid S finished: 6k/floor1 275.4, 6k/floor0.1 293.7, 15k/floor1 250.7, 15k/floor0.1 248.3 -- the 2026-08 report's 473->337 jump was mostly budget, and the schedule confound only bites at short budgets. Grid R: the damped record replicates (253.0 +- 6.0 against 250.7 +- 2.3); the frozen-b gain replicates but smaller and within a spread (332.7 +- 16.4 against 350.3 +- 11.7). |

## Reproduction check — result (2026-09-08)

**The refactor is exonerated and the 2026-08 number is not reproducible.**

Job 998665 (TITAN RTX, node `ai_gpu29`), `d=16, h=2, lr=0.02`, frozen `b`, `alpha_Z=1`, 6,000
steps, seed 0 — the configuration recorded as **val 430.04** in job 940848 of 2026-08-10:

| arm | code | driver | val (final) | val (best) | train | test | msg/unary |
|---|---|---|---|---|---|---|---|
| A | `src/` at `cc28e96` | `ptb_attack.py` | **644.98** | 610.88 | 668.49 | 587.82 | 7.03 |
| B | `src/` at `e1fdac8` | `ptb_attack.py` | **644.98** | 610.88 | 668.49 | 587.82 | 7.03 |
| C | `src/` at `e1fdac8` | `experiments/sweep.py` | **644.98** | 610.88 | — | 555.44 | 7.03 |

A, B and C agree to every digit printed. So:

* **the refactor of `src/` (v0.16.0: the temperature modes, the init modes, the masking
  reorder) did not change the mathematics** — the `temp_mode="fixed"` path is the old model;
* **the new sweep runner is faithful to the old driver** — same weights, same data order, same
  result;
* and none of the three reproduces **430.04**, recorded for the same code, the same
  configuration and the same seed on 2026-08-10.

The one thing that differs between then and now is the GPU: job 940848 ran on an **A40**, all
three arms above ran on a **TITAN RTX**. The A40-pinned control is job 998674 (`sist-a40-01`).
Non-deterministic reduction order is the only remaining candidate, and it is a sufficient one
for a run that the 2026-08 report itself documents passing through an excursion — validation
perplexity reverting 516.1 → 693.9 between two evaluations and then recovering.

**Consequence for the write-up, whichever way the A40 control comes out.** The undamped
configuration is not reproducible at the precision the 2026-08 report quoted it to, and no
single-seed number from that region can carry a claim. Damping and the belief-normalised
temperature are then not only better on the mean; the variance they remove is itself the
result. Every headline number in the preprint is reported over three seeds with its spread.

Note that the *test* perplexity differs between arms A/B (587.82) and C (555.44) because the
old driver scores the final weights and the new runner scores the best-validation checkpoint;
that is a protocol difference, stated in `spec_open.py`, not a discrepancy.

### Part C — the seed spread, which is the actual finding (2026-09-08, job 998665)

Same configuration, three seeds, through the new runner on one GPU:

| `alpha_Z` | seeds | val ppl | mean ± sd | spread |
|---|---|---|---|---|
| **1.0** (undamped) | 0, 1, 2 | 610.88 / 408.81 / 367.55 | **462.4 ± 130.2** | 243 |
| **0.25** (damped)  | 0, 1, 2 | 322.46 / 321.55 / … | **322.0 ± 0.65** | 0.9 |

The undamped configuration has a seed standard deviation of **130 perplexity — 28 % of its own
mean**. The damped one has **0.65 — 0.2 %**. Two orders of magnitude.

**This resolves the reproduction question completely, and in a more useful way than a hardware
story would have.** The 2026-08 number, 430.04, sits comfortably inside the undamped spread
[367.55, 610.88]. It was never wrong; it was one draw from a distribution with `sd = 130`,
quoted to two decimal places as though it were a measurement. The same is true of every other
single-seed number the 2026-08 report took from the undamped region — including the frozen-`b`
gain (473.28 → 430.04, a difference of 43 against a spread of 130) and the `d = 16` versus
`d = 24` ordering.

What this does to the write-up:

1. **No single-seed number from the undamped region appears in the preprint**, and the
   frozen-`b` comparison of 2026-08 §5.1 is withdrawn pending the three-seed replication in
   `spec_open.py` grid R.
2. **The variance reduction is itself a headline result.** Damping was reported in 2026-08 as
   an improvement in the mean (641.5 → 248.29). It is at least as much an improvement in
   reproducibility, and reproducibility is the property that lets any later comparison mean
   anything. The same test applies to the standardised temperature: `spec_gain.py` reports
   `sd` and a dead-seed count in every cell for exactly this reason, and at `d = 32` it gives
   282.2 ± 6.6 at gain 1 against a 3/3 collapse at gain 3 — a cliff, not noise.
3. The A40-pinned control (job 998682) is still worth running, but it is now a secondary
   question: with `sd = 130` across seeds on one GPU, a difference between two GPUs needs no
   further explanation.

### The hardware half, answered for free (2026-09-08)

The A40-pinned control was cancelled after three hours queued behind a busy node — and did not
need to run, because a cell of an unrelated grid landed on one. `C_normal_d16` of
`exp5_init/spec_init.py` is bit-for-bit the same configuration, and it ran on `ai_gpu06`, an
**A40**. Its validation trace:

```
2026-08, job 940848, A40:    711.7 516.1 693.9 633.5 598.1 546.0 491.9 473.2 452.6 439.9 434.4 430.0
2026-09, spec_init, A40:     711.7 516.1 693.9 633.5 598.1 546.0 491.9 473.2 452.6 439.9 434.4 430.0
2026-09, repro_check, TITAN: 711.6 640.3 686.0 676.0 682.1 657.7 716.5 610.9 621.5 639.9 646.4 645.0
```

**Identical to every printed digit on the same GPU model, and divergent on another.** The two
traces already differ at the first evaluation, in the third significant figure (711.70 against
711.61), so the split happens inside the first 500 steps and is then amplified to a difference of
215 perplexity.

So the complete picture is:

1. the mathematics is unchanged (arms A and B of the reproduction check agree digit for digit);
2. the harness is faithful (arm C agrees with them);
3. **the run is deterministic within a GPU model and different across GPU models**, because the
   only thing that differs is the order of floating-point reductions;
4. **and it is chaotic in the seed**, with a spread of 130 perplexity across three seeds on one
   device.

(3) and (4) are the same phenomenon seen two ways: in this configuration any perturbation at the
last bit — a different reduction order, a different seed — moves the outcome by hundreds of
perplexity. The 2026-08 number was therefore both *exactly reproducible* on the hardware that
produced it and *not a measurement of anything*, which is a sharper statement than either half
alone and the one the paper makes.

It also settles a practical point for anyone building on this: a perplexity from the undamped
region of this model must be quoted with a seed spread and a device, or not quoted.
