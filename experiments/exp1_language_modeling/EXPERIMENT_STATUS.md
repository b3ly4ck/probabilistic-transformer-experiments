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

| Date | Commit | Config | Seed | Metric | Wall-clock |
|---|---|---|---|---|---|
