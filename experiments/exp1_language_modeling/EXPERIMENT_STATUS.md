# Experiment 1 — Causal PT vs. a GPT-style decoder on PTB

## Question

Does the causal PT decoder train on real data, and is its perplexity in a reasonable
corridor against a matched GPT baseline?

This is not a controlled comparison — everything differs at once. It measures the *total
cost of the construction*, not the effect of any single component. Experiment 2 (PT vs.
Looped) is what isolates structure. Without Experiment 1 there is no paper; with only it, a
reviewer asks "so what — another decoder that is slightly worse".

## Stage 0 — the pilot, and why it is a separate stage

**Before any comparison: does PT beat the unigram model?**

A previous implementation of this project (commit `9c77f94`, removed in `2e38ef9`) passed
all nine validation checks and reached **val ppl 664 against a unigram baseline of 687** —
a 3 % margin that looks like a number and is not learning. Its *training* perplexity sat at
611 after 88 epochs, i.e. it could not fit the corpus at all, and its samples were function
words in sequence with no content words. GPT on the identical pipeline reached train ppl
5.4 and val 131.

So stage 0 has one gate, and it is not "perplexity went down":

| | value |
|---|---|
| **Unigram baseline, val, `ignore_first=1`, block 64** | **688.82** |
| Unigram baseline, val, `ignore_first=0` | 687.45 |
| Gate to proceed | val ppl **well below** the baseline, and train ppl falling far below it |
| Falsification | val ppl plateaus anywhere near 690, or train ppl stops above ~400 |

The training perplexity is the sharper of the two. A model that cannot fit its own training
data is not being regularised, it is failing to represent the data, and a sweep over such a
model measures nothing.

### What the previous failure was diagnosed as, and what is different here

The diagnosis recorded at `9c77f94` was: training drives the MFVI inner loop into a
saturated fixed point where the head message dominates the word unary; attention entropy
fell to 0.353 nats of a possible 4.159, and `q̄`'s spread across positions collapsed to
0.0022, so the content stream stopped carrying position. Measured fit rate on the toy
memorisation task fell monotonically with the number of MFVI rounds — 8/20 at one round,
1/20 at four.

Two things differ in this implementation, and both are load-bearing for whether the failure
reproduces:

1. **That implementation had no relative positional encoding.** Its `T` was `(h, d, d)`,
   with no distance dimension, so `F_c(i,j)` depended on `j` only through `q̄_j` and never
   through `i − j`. The content stream was a bag of prefix labels and word order was
   invisible to it — which is close to a description of a unigram model with extra steps.
   This repository implements the clipped RPE table of Wu & Tu Eqs. 9/10 (`γ = 3`).
2. **The L2 penalty on `T` is now in the training loop** (`5e-4`, Wu & Tu §4.2 and Table 2).
   It is the only mechanism restraining `‖T‖`, and `‖T‖` is exactly what drives the message
   domination the diagnosis identified. The previous loop did not have it.

`experiments/exp0_decoder_validation/fit_rate.py` re-runs the multi-seed fit-rate table on
this implementation, with and without RPE, before any GPU time is spent.

### Diagnostics logged at every evaluation

The previous session found the collapse by instrumenting a checkpoint afterwards. Here the
same readings are taken every evaluation, so a collapse is visible while it happens:
`msg_over_unary`, `attn_entropy_frac`, `label_entropy`, `root_mass_over_uniform`,
`max_abs_T`, `qbar_std_over_positions`, `rho`.

## Setup, held fixed across every model

| Item | Setting |
|---|---|
| Data | PTB word level, Mikolov preprocessing, `<eos>` per line |
| Vocabulary | 10,000 types, built from train |
| Tokens | train 929,589 / valid 73,760 / test 82,430 |
| Batching | fixed-length blocks from the `<eos>`-joined stream, identical for all models |
| Context length | 64 |
| Scored tokens | `ignore_first=1` — see below |
| Optimiser | AdamW, `β = (0.9, 0.999)`, lr `1e-3`, weight decay `1.4e-6` (Table 2, PTB) |
| Schedule | linear warmup 100 steps, cosine decay to `0.1 ×` |
| Gradient clip | 1.0 |
| L2 on ternary scores | `5e-4` (Table 2, PTB) |
| Training loop | one implementation, `src/train.py`, shared |
| Metric | perplexity on the held-out split, deterministic non-overlapping blocks |

**`ignore_first=1` is not a detail.** PT predicts every `w_t` from `w_{<t}`, including `w_0`
from ROOT alone — `n` scored tokens per block. A GPT trained the usual way consumes
`w_0..w_{n-2}` and scores `w_1..w_{n-1}` — `n−1` tokens. Averaging both would compare
different token sets, and PT's extra slot is a first-word unigram prediction the baseline
never makes. Dropping it makes the sets identical.

## Arms

| Arm | Model | Purpose |
|---|---|---|
| 0 | PT, exact readout, no `G_t` | the pilot gate above |
| 1.1 | PT vs. GPT, no `G_t` | the headline comparison |
| 1.2 | PT vs. GPT, with `G_t` (`m` swept) | is the B.3 global head worth its parameters |
| — | PT, MFVI readout | Experiment 3's object; also 78× cheaper per step, so it is the sweep vehicle |

`G_t` is a **measured** variable, not an assumption. Wu & Tu propose the globals as an
in-graph substitute for the missing feed-forward structure and never test them — Appendix
B.3 is a derivation with no experiments. Baking them in would assume the answer.

**Known in advance about arm 1.2 under the exact readout:** `G_t` contributes a
position- and prefix-independent `d`-vector to `log μ_t`, so only `d` numbers of the `m × d`
matrix reach the readout directly (§22.2 states this; measured in exp0). Its
context-dependent work happens in the content stream. Arm 1.2 must therefore report the
MFVI arm too, or it measures a label prior.

## Parameter matching

Report embedding and non-embedding parameters **separately**, in every table. With tied
embeddings PT's budget sits almost entirely in `S` (`|V| × d`), while a GPT's sits in its
blocks; a single total hides that completely. The matching convention — total, non-embedding,
or both at fixed vocabulary — is to be stated explicitly, not left implied. Report wall-clock
and step time alongside, since PT shares parameters across iterations and equal parameters
does not mean equal compute.

## Run log

All runs: PTB, `|V| = 10,000`, block 64, batch 16, `ignore_first=1`, AdamW lr 1e-3, wd 1.4e-6,
warmup 100 + cosine, grad clip 1.0, `d=256 h=8 rank=64 γ=3`, seed 0, NVIDIA TITAN RTX,
torch 2.0.1+cu117. Reference: **unigram val ppl 688.82** on the identical token set.

| Date | Commit | Job | Config | Best val ppl | Test ppl | Wall-clock |
|---|---|---|---|---|---|---|
| 2026-08-09 | `334e09b` | 940422 | MFVI readout, `T=3`, `l2_arc=5e-4`, 6000 steps | **695.70** | 655.09 | 306 s |
| 2026-08-09 | `334e09b` | 940423 | **exact readout**, `T=3`, `l2_arc=5e-4`, 2000 steps | **1555.04** | 1508.89 | 793 s |
| 2026-08-09 | `334e09b` | 940436 | MFVI, `T=1`, `l2_arc=5e-4`, 6000 steps | **695.33** | 654.60 | 180 s |
| 2026-08-09 | `334e09b` | 940435 | MFVI, `T=3`, **`l2_arc=5.0`**, 6000 steps | **691.82** | 649.88 | 307 s |
| 2026-08-09 | `334e09b` | 940438 | MFVI, `T=3`, **`λ_Z=4`**, 6000 steps | **690.02** | 646.51 | 228 s |

**Gate: FAIL on every configuration.** Every MFVI run lands within 1 % of the unigram
baseline; the exact readout lands more than twice above it. Perplexity does fall — 5692 →
695 over 6000 steps for run 940422 — but it falls *to* the unigram model, which is not
learning. This reproduces the outcome of the implementation removed at `2e38ef9`
(val 664 against a baseline of 687), despite the RPE table that the fit-rate probe showed
was the cause of that implementation's toy-scale failure.

### Diagnostics at the end of each run

| Job | `msg_over_unary` | attn `H/H_max` | `label_entropy` (max 5.55) | `max abs T` | `ρ` | `q̄` std over positions |
|---|---|---|---|---|---|---|
| 940422 MFVI `T=3` | 39.3 | 0.880 | 0.17 | 4.22 | 1.4e6 | 0.0008 |
| 940423 exact `T=3` | 55.1 | 0.220 | 0.076 | 4.74 | 2.8e6 | 0.0009 |
| 940436 MFVI `T=1` | 15.4 | 0.855 | 5.51 | 5.67 | 7.0e5 | 0.0011 |
| 940435 MFVI `l2=5.0` | 18.6 | 0.016 | 1.15 | **0.46** | 1.5e4 | 0.0013 |
| 940438 MFVI `λ_Z=4` | 34.5 | 0.018 | 3.28 | 2.20 | 3.3e5 | 0.0009 |

Read these together, because they say something sharper than any one of them:

1. **`msg_over_unary` never falls below 15.** The head message outweighs the word's own
   unary by 15–55× in every configuration. `‖G‖ ≤ h · max|T|`, so at `max|T| = 0.46` the
   message can be at most 3.7 — and it is still 18.6× the unary. That puts `‖S_{w,·}‖`
   at roughly 0.2, i.e. **at or below its initialisation scale of `0.02·√256 = 0.32`.
   The word–label matrix `S` is not growing.**
2. **The three levers each moved the internals a great deal and the metric not at all.**
   `l2_arc=5.0` cut `max|T|` by 9× and `ρ` by 100×; `λ_Z=4` cut `max|T|` in half;
   `T=1` changed the label posterior from nearly one-hot (0.17 nats) to nearly uniform
   (5.51 nats). Perplexity moved from 695.70 to 691.82, 690.02 and 695.33 — under 1 %.
   **The failure is not a hyperparameter setting.**
3. **`q̄` is nearly identical at every position** in all five runs (std 0.0008–0.0013
   against a uniform value of `1/256 = 0.0039`). If `q̄_j` does not vary with `j`, then
   `B^(c)_{j,a}` does not either, and `log μ_t(a) = Σ_c LSE_{j∈D_t} B^(c)_{j,a}` differs
   between positions only by `log|D_t|` — a constant in `a`, which the readout's
   normalisation removes exactly. The logits then *cannot* depend on `t`, and the model is
   forced to the unigram distribution through `b_w`. That is the whole failure, and it is
   an identity, not a hypothesis.
4. **The collapse takes two different shapes with the same outcome.** At `T=3` the label
   posterior saturates to one label everywhere; at `T=1` it stays nearly uniform
   everywhere. Both give a constant `μ_t`. So "the inner loop saturates" is not a complete
   description — the invariant is that the *word* never gets into its own label posterior.

### Next, in order

1. **Drop the word unary** (`word_unary=False`, sanctioned by §16(c): "Set `b ≡ 0` to drop
   it"). `b_w` is a free per-word parameter that reproduces the unigram distribution
   exactly, so the fastest descent direction is to fit the marginal with `b` and leave the
   context path at its initialisation. Removing it forces every bit of probability mass
   through `S` and `μ_t`. This is one flag and it separates "the context path is dead" from
   "the context path is unused because something cheaper is available".
2. **Prefix-ablation on a trained checkpoint** — zero and shuffle the prefix, measure the KL
   on the output. The previous implementation measured KL 0.0115 for PT against 12.26 for
   GPT. This is the decisive read on whether context reaches the output at all, and it
   needs checkpoint saving, which the loop does not yet do.
3. **Why `S` does not grow.** Its gradient arrives through two roles; if it is small in
   both, the readout's `LSE_a(S_{w,a} + log μ_t(a))` is the place to look, since a peaked
   `μ_t` concentrates the gradient on a few labels.
4. Only then a GPT baseline on the identical pipeline. Comparing a model that has not
   learned against one that has measures nothing.
