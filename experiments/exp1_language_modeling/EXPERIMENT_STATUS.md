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

### Step 1 (drop the word unary) and step 2 (prefix ablation) — 2026-08-09

| Job | Config | Best val ppl | Test ppl |
|---|---|---|---|
| 940445 | MFVI `T=3`, **with** `b` (control) | 695.73 | 655.10 |
| 940444 | MFVI `T=3`, **`word_unary=False`** | **985.02** | 945.01 |

The control reproduces run 940422 to two decimals (695.73 vs 695.70), so the pipeline is
deterministic. **Removing `b` makes the model worse, not better.** The context path does not
come alive when the cheap route to the marginal is removed; the model simply becomes worse at
the marginal. `b` is not the culprit.

(Correction to how this step was proposed: dropping `b` never made the unigram unreachable.
With `μ` constant, `logits(w) = LSE_a(S_{w,a} + log μ(a))` is still a function of `w` alone and
`S` has `d` free parameters per word. Removing `b` deletes the cheapest route, not the only one.)

**Prefix ablation on both checkpoints — the decisive measurement:**

| checkpoint | condition | KL | `max abs Δlogit` | argmax unchanged |
|---|---|---|---|---|
| with `b` | prefix shuffled | **0.0000** | **0.0000** | **1.000** |
| with `b` | prefix replaced by one repeated token | **0.0000** | **0.0000** | **1.000** |
| no `b` | prefix shuffled | **0.0000** | **0.0000** | **1.000** |
| no `b` | prefix replaced by one repeated token | **0.0000** | **0.0000** | **1.000** |

Not "a thousand times less than GPT" as in the previous implementation (KL 0.0115). **Exactly
zero.** The output does not depend on the prefix at all.

### Mechanism, traced on the trained checkpoint

An exact zero is a saturated softmax, not a weak signal, and the trace confirms it end to end:

```
content stream  q̄ : max prob 0.9922, entropy 0.0441 nats (of 5.55),
                    distinct argmax over the 64 positions = [1, 1, 2, 1]
scales           : ‖S_w‖ 1.78   ‖r‖ 4.05   max|T| 4.07   ‖b‖ 2.66
predictive Q_Z   : ‖s̄‖ 0.74   ‖G‖ 44.85
      round 1    : max prob 0.99967, entropy 0.0038 nats, 1 distinct argmax
      round 2    : max prob 0.99993, entropy 0.0009 nats, 1 distinct argmax
                   deviation of Q_Z across all 256 slots: 1.08e-3
```

The chain, each link measured:

1. `‖G‖ = 44.85` against `‖s̄‖ = 0.74` — the head message outweighs the word message **60×**.
   There is no normalisation between them, by construction.
2. `Q_Z = softmax((s̄ + G)/λ_Z)` with `λ_Z = 1` therefore saturates: entropy 0.0009 nats of a
   possible 5.55.
3. The winning label is **the same one at every position of every sequence**. In float32 the
   surviving deviation (1.08e-3) is below what moves the argmax and vanishes entirely in
   `Q_Z Sᵀ`, which is why the ablation KL is exactly 0.
4. The same collapse happens in the content stream, so `B^(c)_{j,·}` is the same for every
   `j`, `log μ_t` is context-free, and the model has nothing left but `b_w` — the unigram.

Once saturated the softmax gradient is ~0, so the state is self-locking, and it arrives early:
`msg_over_unary` was already 93 at step 500.

**Why the `l2_arc = 5.0` probe did not fix it, and what that reveals.** It crushed `max|T|` to
0.46 — but `arc_regulariser()` covers `T` only, and **not the root column `r`**. In that run
`root_mass_over_uniform` rose from 0.20 to **4.78**: with the arc scores penalised, the model
routed the message through the *unpenalised* root column instead. The message was conserved,
only its carrier changed. That is a defect in the regulariser, not a refutation of the
hypothesis.

**Relation to the source.** Wu & Tu set `λ_Z = 1` "for simplicity" and state plainly that a
fixed `λ_Z` cannot recover the message variance because it depends on sentence length
(App. A.5). Their variance argument also assumes near-uniform beliefs; under saturation it does
not apply. What was tolerable in a sentence-level MLM encoder is not tolerable in a causal
decoder over 64-token blocks.

### The `λ_Z` sweep falsified the saturation diagnosis — 2026-08-09

The prediction written above was: label entropy should recover *and* the ablation KL should
become non-zero; "if perplexity stays at 690 while the entropy recovers, saturation was not the
binding constraint and the diagnosis is wrong."

| Job | `λ_Z` | Best val ppl | Test ppl | `label_entropy` (max 5.55) | `q̄` variance over positions | ablation KL |
|---|---|---|---|---|---|---|
| 940445 | 1 | 695.73 | 655.10 | 0.044 | 3.0e-05 | **0.000e+00** |
| 940483 | 16 | 696.24 | 655.82 | **4.95** | 4.8e-07 | **0.000e+00** |
| 940484 | 32 | 697.76 | 657.63 | **5.26** | 1.9e-10 | **0.000e+00** |
| 940485 | 64 | 699.29 | 659.42 | **5.48** | 3.9e-11 | **0.000e+00** |

The entropy recovered completely — 0.0009 nats to 5.48 of a possible 5.55 — and perplexity did
not move, and the ablation KL stayed at exactly zero. **The diagnosis is wrong by its own
stated criterion.** Recorded rather than rewritten.

What actually happened is that the model traded one degenerate fixed point for another: at
`λ_Z = 1` the label posterior is the same *one-hot* at every position, at `λ_Z = 64` it is the
same *near-uniform* vector at every position, and the variance of `q̄` across positions got
**worse**, from 3.0e-05 to 3.9e-11. Saturation was a surface symptom. The invariant is that
`q̄` does not vary with position.

A second inference of the previous entry is also withdrawn: **`S` does grow.** `‖S_w‖` reaches
17.4–20.1 against an initialisation of 0.32. The earlier claim that it stays at initialisation
scale was read off the ratio `msg_over_unary` rather than measured, and was wrong. What
explodes instead is the **root column**: `‖r‖` reaches 48–58 while `max|T|` falls to 0.016–0.16,
i.e. the arc scores vanish and ROOT carries the message, by a factor of order 10³.

### The initialisation, not training, is where the context is lost

The control that should have been run first. Prefix ablation on **untrained** models:

| model | ablation KL | `max abs Δlogit` | argmax unchanged |
|---|---|---|---|
| untrained, `init_std = 0.02` (the default) | 9.5e-11 | 4.3e-09 | 1.000 |
| untrained, `init_std = 0.5` | **6.9e-02** | **2.405** | **0.375** |
| untrained, exact readout, `init_std = 0.02` | 9.3e-08 | 3.8e-06 | 1.000 |

**The forward path is not broken** — at `init_std = 0.5` an untrained model's prediction moves
substantially when the prefix is shuffled, and its argmax changes on 62 % of slots. The
mechanism works.

**At the default `init_std = 0.02` the model is already prefix-blind before a single gradient
step**, and no configuration tried has escaped that basin. This reframes every run above: it is
not a collapse *from* a working state, it is a failure to ever leave a context-free
initialisation, with the unigram solution available immediately through `b` and through `S`.

`init_std = 0.02` is the nanoGPT convention, is specified by neither paper, and was flagged as
an unjustified default of mine in the exp0 review of 2026-08-09, where the decision taken was
to keep it. That decision now has evidence against it.

### The initialisation hypothesis is also falsified, and the failure is localised

| Job | `init_std` | Best val ppl | Test ppl | `q̄` var over positions | ablation KL |
|---|---|---|---|---|---|
| 940445 | 0.02 | 695.73 | 655.10 | 3.0e-05 | 0.000e+00 |
| 940489 | 0.2 | 695.53 | 654.86 | 4.3e-05 | 0.000e+00 |
| 940490 | 0.5 | 702.57 | 660.80 | **7.3e-04** | **5.7e-08** |

Training from an initialisation that demonstrably uses context still ends at the unigram. So
the initialisation is not the binding constraint either — it only changes how much context
survives, not whether the model learns.

**Where the information actually dies.** Measured on the `init_std = 0.5` checkpoint, which is
the one with the most surviving signal:

```
logits std ACROSS SEQUENCES at a fixed slot :  8.9e-05
logits std ACROSS SLOTS within a sequence   :  1.5e-01      ← a factor of 1700
```

The prediction at slot `t` is **the same for every sequence**. The model has learned
`p(w | position in the block)` and nothing else — which, averaged over positions, is the
unigram. That is why nine configurations all land at 690–703 and why the ablation KL is zero:
there is no content to ablate.

It is not the RPE table collapsing — the distance buckets stay distinct (relative differences
1.40, 1.39, 1.41 between bucket 0 and buckets 1–3). Position is used; content is not.

And the content **is** present upstream. Tracing the content stream on the same checkpoint:

```
q^(0) = softmax(S_w)   std across sequences 1.9e-03   across slots 2.1e-03
q after iteration 1    std across sequences 2.3e-03   across slots 2.7e-03
q after iteration 2    std across sequences 3.8e-03   across slots 4.6e-03
q after iteration 3    std across sequences 2.1e-03   across slots 4.0e-03
```

`q̄` varies with the words about as much as it varies with position. **The word reaches `q̄`.
It does not reach the logits.** The loss of information is between the frozen prefix beliefs
and the readout, not in the content stream and not in the initialisation.

That is also a *learned* degeneracy rather than a structural impossibility: the same
architecture at `init_std = 0.5` **before training** has ablation KL 6.9e-02 with the argmax
changing on 62 % of slots. Training destroys a sensitivity that is there at the start.

### Every configuration tried, one table

Unigram baseline 688.82. All MFVI unless stated, `d=256 h=8 rank=64 γ=3`, 6000 steps.

| Variable | Value | Best val ppl |
|---|---|---|
| baseline | `T=3, λ_Z=1, l2=5e-4, init 0.02` | 695.70 / 695.73 |
| readout | exact (2000 steps) | 1555.04 |
| rounds | `T=1` | 695.33 |
| L2 on arc scores | `5.0` | 691.82 |
| `λ_Z` | 4 / 16 / 32 / 64 | 690.02 / 696.24 / 697.76 / 699.29 |
| word unary | `b` removed | 985.02 |
| `init_std` | 0.2 / 0.5 | 695.53 / 702.57 |

Nine runs, a 45× range in `λ_Z`, a 10⁴× range in the L2 coefficient, 25× in initialisation
scale, both readouts, with and without the word unary — and the metric never leaves the
interval [690, 703] except when it gets worse.

### Stage-by-stage across sequences — there is no single guilty stage

`where_content_dies.py`. The *content fraction* of a tensor is `std over sequences at a fixed
slot / overall std` — scale-free, so stages of very different magnitude compare directly. A
sharp drop marks where word identity is discarded. The untrained model at `init_std = 0.5` is
the control: its prefix ablation gives KL 6.9e-2 and its argmax changes on 62 % of slots, so
its readout demonstrably works.

| stage | untrained `init 0.5` | trained `init 0.5` | trained `λ_Z=16` | trained baseline |
|---|---|---|---|---|
| `q̄` (content stream) | 3.10e-01 | 3.61e-02 | 5.18e-02 | 9.10e-04 |
| `B` (contracted arcs) | 7.68e-01 | 2.41e-01 | — | 2.74e-03 |
| `α` round 1 | 4.77e-03 | 2.59e-01 | — | 1.41e-01 |
| `G` round 1 | 6.28e-01 | 4.75e-01 | **6.00e-05** | 3.08e-03 |
| `Q_Z` round 1 | **3.22e-01** | **5.87e-04** | 6.52e-05 | 7.15e-09 |
| `Q_Z` round 2 | 2.28e-01 | 1.64e-05 | — | 9.69e-10 |
| **logits** | **7.35e-01** | 9.33e-05 | 2.98e-09 | 1.33e-08 |

Read down the columns rather than across:

* **The control carries content the whole way** — 0.31 at `q̄`, 0.32 through the `Q_Z` update,
  0.73 at the logits. The architecture transmits word identity end to end. Nothing structural
  is blocking it.
* **Every trained model ends at 1e-5 to 1e-9 at the logits**, but they get there **at different
  stages**. In `init 0.5` the content survives the message (`G` = 0.475) and is annihilated by
  the `Q_Z` softmax — an 810× drop in one step. In `λ_Z = 16` the message itself is already
  content-free (`G` = 6.0e-05) while `q̄` still has 5.2e-02, so it dies in the attention. In the
  baseline `q̄` is already nearly content-free at 9.1e-04.
* So the hypothesis of a single guilty stage is **wrong**. Training reliably removes the
  content by whichever route is available, which points at the objective and the optimisation
  rather than at one line of the forward pass.

This also explains why the `λ_Z` sweep did nothing: raising `λ_Z` de-saturated the `Q_Z` softmax
(entropy 0.0009 → 5.48) and the content simply left one stage earlier instead.

### `τ` is not it either

`τ` had sat at 2 in all nine earlier runs. `init_std = 0.5`, everything else held:

| `τ` | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| best val ppl | 700.45 | 702.57 | 696.00 | 709.45 |

§17.1's concern — that at `τ = 1` the attention query is the fixed global probe `σ(s̄/λ_Z)` and
the readout sees the prefix only through the mask — is real but not binding here: making the
query context-dependent for eight rounds changes the metric by 1 %.

### Where this stands: twelve runs, one conclusion

| Variable | Range covered | Best val ppl range |
|---|---|---|
| `λ_Z` | 1 → 64 (45×) | 690.02 – 699.29 |
| `l2_arc` | 5e-4 → 5.0 (10⁴×) | 691.82 – 695.70 |
| `init_std` | 0.02 → 0.5 (25×) | 695.53 – 702.57 |
| `n_iters` (`T`) | 1, 3 | 695.33 – 695.70 |
| `τ` | 1, 2, 4, 8 | 696.00 – 709.45 |
| word unary `b` | on, off | 695.73 / 985.02 |
| readout | MFVI, exact | 695.70 / 1555.04 |

**Unigram baseline 688.82.** Nothing moves it. The interval is [690, 710] except where a
change makes it worse.

**What is established:**

1. The architecture transmits word identity end to end — the untrained control has content
   fraction 0.73 at the logits and prefix-ablation KL 6.9e-2.
2. The trained models do not — content fraction 1e-5 to 1e-9 at the logits, ablation KL ~0.
3. They lose it at *different stages* depending on the configuration, so it is not one broken
   operation.
4. The trained model has learned `p(w | position in block)`: logits vary 1700× more across
   slots than across sequences.
5. **The same model, the same loop and the same loss do learn a context-dependent task at small
   scale** — `tests/test_11::test_training_reduces_loss_and_perplexity_on_a_learnable_stream`
   trains this decoder on a deterministic stream where `w_{t+1} = f(w_t)` and drives validation
   perplexity below uniform, at `|V| = 11`, `d = 16`, block 16. So neither the training loop nor
   the model is incapable of using context in principle.

The failure is therefore **scale-dependent**: context is used at `|V| = 11, d = 16` and
abandoned at `|V| = 10⁴, d = 256`. That is the axis to probe next, and it is a much sharper
question than any remaining hyperparameter.

### The pipeline is exonerated — 2026-08-10, job 940799

Identical loop, identical blocks, identical `ignore_first=1` token set, identical unigram
reference of **688.82**. 6000 steps each.

| Model | Params (emb / non-emb) | Best val ppl | Test ppl | Gate |
|---|---|---|---|---|
| **GPT**, `n_embd=160, L=4, H=4` | 1,610,240 / 1,237,440 | **115.43** | 107.16 | **PASS** |
| **Looped**, same shape, one shared block ×4 | 1,610,240 / 309,600 | **126.46** | 117.93 | **PASS** |
| PT, MFVI, `d=256 h=8 rank=64 T=3` | 2,570,000 / 1,050,624 | 695.84 | 655.25 | FAIL |

PT has the **largest** non-embedding budget of the three and the largest total (3.62 M against
2.85 M and 1.92 M), so this is not a capacity deficit. Looped wins on 4× fewer non-embedding
parameters than PT.

The GPT reaches 115.43 — better than the 131 the previous implementation obtained on its own
pipeline. So the training loop, the PTB pipeline, the deterministic evaluation, the metric and
the slot alignment are all sound, and every PT diagnosis in this file was measuring PT rather
than a defect the models share.

**Looped is the sharper control.** It has PT's weight sharing — one block applied four times —
and none of PT's structure, and it costs only 9.6 % against the GPT. So PT's failure is not
weight sharing either. What is left is the construction itself: the label bottleneck and the
readout through it.

This is also the first real data point of Experiment 2, obtained for free: at a matched shape,
weight sharing costs about 10 % perplexity.

### Prefix ablation, all three models, identical code and blocks

| Model | shuffled KL | `max abs Δlogit` | argmax unchanged |
|---|---|---|---|
| GPT | 3.9461 | 14.04 | 0.117 |
| Looped | 3.6469 | 14.91 | 0.125 |
| **PT** | **0.0000** | **0.0000** | **1.000** |

Replacing the prefix with a single repeated token gives the same picture (GPT 3.77, Looped
3.63, PT 0.0000). The two baselines change their prediction on ~88 % of slots; PT never
changes it, on any slot, at float precision.

### The scale bisection: both axes excluded — 2026-08-10, job 940799

Order-1 Markov chain, 5 successors per state, 3000 steps, fraction of the unigram→oracle gap
closed (1.0 = oracle, 0.0 = unigram).

| cell | oracle | unigram | **GPT** | **PT** |
|---|---|---|---|---|
| `V=11, d=256` | 3.71 | 7.84 | **1.149** | **0.000** |
| `V=100, d=256` | 3.65 | 81.19 | **1.000** | **0.000** |
| `V=1000, d=256` | 3.61 | 795.42 | **1.000** | **0.000** |
| `V=10000, d=256` | 3.61 | 7945.95 | **1.000** | **0.018** |
| `V=1000, d=16` | 3.61 | 795.42 | **0.984** | **0.062** |
| `V=1000, d=64` | 3.61 | 795.42 | **1.000** | **0.044** |

**The scale hypothesis is falsified.** PT extracts essentially nothing at *any* vocabulary and
*any* width, including `|V| = 11` where the entire next-token distribution is determined by a
single preceding token drawn from eleven symbols. GPT reaches the oracle in every cell. (The
`1.149` at `V = 11` exceeds 1 because the oracle is an estimate — the visit-weighted mean
conditional entropy of the generating chain — not a bound.)

This is the fourth diagnosis in a row to be falsified by the run that tested it: the word
unary, label saturation, the initialisation, and now scale. Each narrowed the space.

**What it buys is a minimal reproducing case.** `|V| = 11`, one channel of context, 400 k
training tokens, 115 s per run. The failure no longer needs PTB, a 10⁴ vocabulary, or a GPU
hour to study — it can be examined tensor by tensor by hand, which is what §17's worked
example was designed for and what the next step should use.

### The working region — 2026-08-10, jobs 940809 and 940814

**PT does learn.** `lr_probe` found it first on the minimal Markov task: at `d = 16, lr = 0.02`
the decoder reached perplexity 3.809 against an oracle of 3.84 — the whole unigram→oracle gap
closed — with `msg_over_unary` at **2.48** instead of the 12–79 of every failing configuration.

`region_probe` then mapped the region. Fraction of the gap closed, one chain, seed 0:

| `d` \ `lr` | 0.005 | 0.01 | 0.02 | 0.04 |
|---|---|---|---|---|
| 8 | 0.672 | **0.919** | 0.878 | 0.620 |
| **16** | **0.936** | 0.254 | 0.604 | 0.000 |
| 32 | **0.852** | 0.767 | 0.029 | 0.006 |
| 64 | 0.544 | 0.012 | 0.196 | 0.000 |
| 128 | 0.011 | 0.000 | 0.001 | 0.000 |
| 256 | 0.000 | 0.000 | 0.000 | 0.000 |

The region is **small `d`, small `lr`**, and it degrades monotonically in `d`. Nothing works at
`d ≥ 128`. This inverts the usual reading: more label capacity makes the model *worse*, which
is consistent with the message bound `|G_i(a)| ≤ h·max(max|T|, max|r|)` swamping the word
unary as the model widens.

**Replication of the best cell** `d = 16, lr = 0.005`, three seeds — because the previous
implementation's post-mortem turned on a single-seed check that had a real fit rate of 1/5:

| seed | gap closed | `msg/unary` |
|---|---|---|
| 0 | 0.936 | 2.86 |
| 1 | 0.886 | 2.87 |
| 2 | 0.780 | 5.67 |

**2 of 3 seeds close more than 80 % of the gap.** Not a lucky cell, and not a reliable one
either — the seed that fell short is also the one whose `msg/unary` left the healthy band.

**Carried to PTB** (unigram 688.82, GPT reference 115.43, 6000 steps):

| `d` | `lr` | val ppl | test ppl | `msg/unary` | verdict |
|---|---|---|---|---|---|
| 16 | 0.005 | 688.53 | 640.89 | 10.25 | unigram |
| **16** | **0.02** | **473.28** | **433.06** | **3.41** | **beats unigram by 31 %** |
| 32 | 0.005 | 688.47 | 640.88 | 28.14 | unigram |
| 32 | 0.02 | 696.14 | 642.49 | 15.41 | unigram |
| 64 | 0.005 | 688.43 | 640.84 | 39.61 | unigram |
| 64 | 0.02 | 694.83 | 642.01 | 65.70 | unigram |

**Against the gate: FAIL.** The gate written in this file is val ppl well below the baseline,
operationalised as below half of it (344.4). 473.28 does not reach it. It is nonetheless the
first run in the entire project to move off the unigram at all, and by 31 %.

**`msg_over_unary` predicts the outcome without exception.** It is below 10 for exactly one of
the six PTB runs, and that is exactly the one that learns. The healthy band from the minimal
task — 2–3 — brackets the working PTB run at 3.41. This is now the quantity to watch, and the
one whose departure from band should be reported first when a run fails.

**A caveat on the region probe's absolute numbers.** Its chain was generated before the
`make_chain` fix of `v0.8.2` (`Dirichlet.sample` ignores its generator argument and drew from
the global RNG), so its oracle/unigram pair — 1.96 / 3.09 — differs from later probes. Only
the gap-closed fractions are comparable across probes; the raw perplexities are not.

### Width and channels, deconfounded — 2026-08-10, job 940827

`region_probe` swept `d` with `h = 8 if d >= 64 else 2`, so its collapse at `d ≥ 64` could
have been width, channels, or both. Full cross grid, `lr = 0.005` fixed at the best replicated
cell, one chain (oracle 3.577, unigram 7.105), seed 0, 3000 steps per cell.

| `d` | `h` | val ppl | train ppl | gap closed | `msg/unary` | `H(q)` | ablation KL |
|---|---|---|---|---|---|---|---|
| 16 | **2** | 5.675 | 7.006 | 0.405 | 8.56 | 0.122 | 6.93e-02 |
| 16 | 4 | 6.715 | 6.671 | 0.110 | 18.81 | 0.089 | 1.45e-01 |
| 16 | 8 | 6.502 | 6.462 | 0.171 | 15.52 | 0.118 | 1.59e-01 |
| 32 | **2** | 4.551 | 4.485 | **0.724** | **6.78** | 1.796 | 9.31e-01 |
| 32 | 4 | 5.003 | 4.921 | 0.596 | 8.21 | 0.950 | 6.80e-01 |
| 32 | 8 | 7.104 | 7.047 | 0.000 | 36.68 | 0.083 | 0.000 |
| 64 | **2** | 7.023 | 6.974 | 0.023 | 22.77 | 1.079 | 1.69e-04 |
| 64 | 4 | 7.103 | 7.044 | 0.001 | 43.48 | 0.105 | 0.000 |
| 64 | 8 | 7.106 | 7.051 | 0.000 | 47.51 | 0.047 | 0.000 |
| 128 | 2 | 7.106 | 7.050 | 0.000 | 12.21 | 0.156 | 0.000 |
| 128 | 4 | 7.106 | 7.051 | 0.000 | 28.39 | 0.118 | 0.000 |
| 128 | 8 | 7.106 | 7.051 | 0.000 | 25.78 | 0.036 | 0.000 |

**Both factors are real and they are not symmetric.**

* **Channels degrade monotonically at every width.** At `d = 32`: 0.724 → 0.596 → 0.000 for
  `h` = 2 → 4 → 8, with `msg/unary` rising 6.78 → 8.21 → 36.68. That is the linear-in-`h` term
  of `|G_i(a)| ≤ h · max(max|T|, max|r|)` doing exactly what it says.
* **Width degrades independently of channels.** At `h = 2` held fixed: 0.405 → **0.724** →
  0.023 → 0.000 for `d` = 16 → 32 → 64 → 128. The collapse between 32 and 64 happens with the
  channel count untouched.

So the earlier "monotone in `d`" reading was **partly confounded but not wrong**: the width
effect survives the deconfound. It is not monotone, though — the optimum is at `d = 32`, not at
the smallest width tried.

**What this licenses the write-up to say:** both the number of channels and the size of the
label set degrade the causal PT, and at `h = 2` the width collapse lies between `d = 32` and
`d = 64`. Neither may be attributed to the other.

Three qualifications, all measured:

1. **No cell reaches the healthy `msg/unary` band of 2–5** on this chain; the best is 6.78. The
   best gap here is 0.724 against 0.936 for the nominally same configuration on the region
   probe's chain — a different chain instance (the `v0.8.2` fix), so only fractions compare,
   but a spread that large means the result is sensitive to the task instance and that has to
   be carried into any claim.
2. **`train ppl ≈ val ppl` in every cell** (4.485 vs 4.551 in the best). This is underfitting,
   not overfitting — consistent with everything since the first pilot.
3. **The ablation KL tracks the gap closed monotonically** — 0.93 at gap 0.724, 0.68 at 0.596,
   1.7e-4 at 0.023, exactly 0 at 0.000. An independent confirmation that "gap closed" is
   measuring context use and not something else.

### Step 3a/3b — the ceiling of the construction on PTB, 2026-08-10, job 940844

`h = 2` (from the deconfound), `lr = 0.02`, warmup 100, 6000 steps, everything else as the
PTB control. Unigram 688.82, gate 344.41, GPT reference 115.43.

| `d` | val ppl | train ppl | test ppl | `msg/unary` | `H(q)` / max | ablation KL |
|---|---|---|---|---|---|---|
| **16** | **473.28** | 488.85 | **433.06** | **3.41** | 0.030 / 2.77 | **9.13e-01** |
| 24 | 695.38 | 736.83 | 642.06 | 21.49 | 0.026 / 3.18 | **0.000e+00** |

**The ceiling is `d = 16`.** The ladder stopped at `d = 24` on both criteria at once —
`msg/unary` outside the 2-5 band and the ablation KL at exactly zero. The number for the
write-up is **val 473.28 / test 433.06**, 31 % below the unigram baseline. Against the gate
(344.41) this is still a **FAIL**.

**The traces show a phase transition, and it is the strongest causal evidence collected.**

```
d=16  msg/unary : 13.4  11.6  21.9  11.7  |5.06|  3.38  3.17  2.89  3.36  3.37  3.34  3.41
d=16  val ppl   : 698.4 701.2 706.9 703.6 |539.0| 511.8 500.4 488.9 480.5 476.5 474.3 473.3
                                    step 2500 ^
d=24  msg/unary : 13.5  15.5  86.2  32.9  20.8  20.4  41.5  32.6  25.6  23.7  23.2  21.5
d=24  val ppl   : 698.4 701.1 706.9 704.7 702.7 701.9 700.7 699.5 695.8 695.6 695.4 695.5
```

The model sits at the unigram for 2000 steps; at the very evaluation where `msg/unary` first
enters the band (5.06 at step 2500) validation perplexity drops 703.6 -> 539.0, and falls
monotonically thereafter. At `d = 24` it never enters the band and never leaves the unigram.
This is a within-run coincidence of timing, not a correlation across end-points.

**Two corrections this run forces.**

1. **Label entropy is not a discriminator.** The working run has `H(q) = 0.030` against a
   maximum of 2.77 — about 1 % of it, i.e. a nearly one-hot label posterior — and the failing
   run has 0.026/3.18, essentially the same. The band "entropy at neither extreme", carried
   over from the minimal task, **does not hold on PTB even where the model learns**.
   `msg_over_unary` is the only quantity that separates the outcomes.
2. **`first_out_of_band` as implemented reports the first evaluation** (step 500 in both runs),
   because both start out of band. The meaningful statistic is the *entry* step, not the exit.
   Recorded rather than re-run; the traces in `attack.json` carry the information.

Steps 3c (freeze `b`) and 3d (decouple `lambda_H`) were **not run**: both are conditional on
3a failing to beat the unigram, and it did not.

### Steps 3c and the budget — the gate is passed, 2026-08-10, jobs 940848 / 940850

Both were run as single-variable changes against the 3a configuration (`d=16, h=2, lr=0.02`).

| run | steps | `b` | val ppl | train ppl | test ppl | `msg/unary` | ablation KL | gate 344.41 |
|---|---|---|---|---|---|---|---|---|
| 3a | 6000 | learned | 473.28 | 488.85 | 433.06 | 3.41 | 0.913 | FAIL |
| 3c | 6000 | **frozen at log-unigram** | 430.04 | 450.35 | 393.09 | 3.96 | 0.799 | FAIL |
| budget | **15000** | learned | **336.89** | 338.90 | **308.04** | 1.09 | **1.873** | **PASS** |

Reference points: unigram 688.82, Looped 126.46, GPT 115.43.

**3c works, and the mechanism is the one predicted.** Prepaying the unigram moves the phase
transition from step 2500 to step 1000:

```
3c  msg/unary : 1.91  4.88  8.64 11.70  8.79  6.07  4.76  4.46  4.15  4.03  4.02  3.96
3c  val ppl   : 711.7 516.1 693.9 633.5 598.1 546.0 491.9 473.2 452.6 439.9 434.4 430.0
                      ^ step 1000
3a  val ppl   : 698.4 701.2 706.9 703.6 539.0 511.8 500.4 488.9 480.5 476.5 474.3 473.3
                                        ^ step 2500
```

The excursion is visible and reversible: at steps 1500-2000 `msg/unary` rises to 8.64 and
11.70 and validation perplexity goes *back up* from 516.1 to 693.9; when it descends again the
perplexity falls monotonically. That is a two-sided within-run coincidence, not just an onset.

Freezing `b` also changes the internal state qualitatively: label entropy 0.870 of 2.77 against
0.030 in 3a. The free `b` was not merely spending steps, it was driving `q̄` into a degenerate
state.

**The budget was the larger effect.** 473.28 → 336.89, and the run ends genuinely flat
(339.19 → 338.84 → 336.89 → 337.21) where the 6000-step run was cut off mid-descent.
`qbar_std_over_positions` reaches 0.126 against 0.0008 in every failing run.

**Caveat on the budget comparison.** The cosine schedule is defined over `max_steps`, so a
15000-step run is not "the same run, longer" — at step 6000 its learning rate is still high
while the 6000-step run has already decayed to `0.1×`. Part of the gain belongs to the
schedule, and separating the two needs a fixed-lr run. Stated rather than absorbed.

**A defect in the ladder's stop rule, and the band framing behind it.** The 15000-step run was
stopped after `d = 16` with `STOP: msg/unary 1.09 outside (2.0, 5.0)` — it fired on the *lower*
edge while the model was at its healthiest result in the project. The band was written as an
interval from two points on the minimal task; only the upper edge is a failure condition. Fixed
to a ceiling of 5.0. The cost was that `d = 24` went untested in that job.

### The final ladder: both interventions together — 2026-08-10, job 940858

`h = 2`, `lr = 0.02`, `b` frozen at the corpus log-unigram, 15000 steps, `d` walked upward.

| `d` | val ppl (final) | train ppl | test ppl | `msg/unary` | `H(q)` / max | ablation KL |
|---|---|---|---|---|---|---|
| **16** | **315.5** | 314.09 | **285.33** | 0.98 | 0.444 / 2.77 | **2.182** |
| 24 | 321.5 | 325.84 | 294.89 | 2.26 | 0.959 / 3.18 | 2.003 |
| 32 | 641.5 | 664.95 | 582.92 | 5.33 | 0.131 / 3.47 | 0.281 |

**The ceiling is between `d = 24` and `d = 32`, not at `d = 16`.** The earlier "ceiling `d=16`"
was an artefact of the 6000-step budget with a learnable `b`: at that budget `d = 24` gave
695.38, pure unigram, `msg/unary` 21.49 and ablation KL exactly 0. With the unigram prepaid and
a real budget it reaches 321.5. The band *is* attainable under late dynamics, as predicted.

**The two interventions compose:** 473.28 (neither) → 430.04 (freeze only) → 336.89 (budget
only) → **315.46** (both). A 33 % improvement over 3a.

**`train ppl` falls below `val ppl` for the first time in the project** — 314.09 against 315.46
at `d = 16`. Every earlier run had train *above* val (488.85 against 473.28 in 3a), which is
the signature of underfitting. The model has finally reached the regime where it fits its own
training data at least as well as held-out data.

**A defect in this file's own reporting, found while reading the traces.** `best val ppl` was
taken as the minimum over the trace. That is right for a converging run and wrong for a
diverging one: at `d = 32` the minimum 523.08 is the evaluation at **step 500**, before the
message exploded, while the run actually ends at 641.5.

```
d=32  val : 523.1  769.1  731.5  733.8  ...  646.8  641.5
d=32  msg : 2.88  20.57  17.74  16.78  ...   5.95   5.33
```

The runner now reports the final value as `val_ppl` and keeps the minimum as `val_ppl_min`.
The table above uses finals. Earlier rows in this file quote minima and were converging runs,
so they are unaffected — but the `d = 32` row must be read as **641.5**, not 523.08.

### Experiment 1 — where it stands

| model | val ppl | test ppl | embedding / non-embedding |
|---|---|---|---|
| GPT | **115.43** | 107.16 | 1,610,240 / 1,237,440 |
| Looped | **126.46** | 117.93 | 1,610,240 / 309,600 |
| **Causal PT** (`d=16, h=2`, frozen `b`, 15k steps) | **315.46** | **285.33** | 160,000 / **4,128** |
| unigram | 688.82 | — | — |

The causal PT trains, clears the gate by a wide margin, and is **2.7× worse than the GPT
baseline**. The parameter columns are the uncomfortable part of the result: the working PT
configuration has **4,128** non-embedding parameters against the GPT's 1,237,440 — three
orders of magnitude fewer. The matched-budget comparison the research plan requires has *not*
been made, because PT cannot currently be given a comparable budget: it collapses above
`d = 24` and above `h = 2`.

That is the finding, and it is a finding about the construction rather than about tuning. The
label bottleneck of Part IV stops being a caveat and becomes a measured boundary between
`d = 24` and `d = 32`.

### Next, in order

Steps 1–4 above are done. Three successive diagnoses — the word unary, label saturation, the
initialisation — were each stated with a falsifiable prediction and each falsified by the run
that followed. What survives is a *localisation*, not a cause:

> The word reaches `q̄` and does not reach the logits. The information is lost between the
> frozen prefix beliefs and the readout, and the sensitivity that is lost was present at
> initialisation.

The remaining candidates, ordered by how directly they attack that localisation. **None of
them should be run as a sweep** — each needs a prediction attached first, because three
plausible diagnoses have already died.

Items 1 and 2 are done, above: no single guilty stage, and `τ` is not binding. What remains,
reordered by what the evidence now supports.

1. **Bisect the scale gap.** The decisive new fact is that this decoder *does* learn context at
   `|V| = 11, d = 16` and not at `|V| = 10⁴, d = 256`. Walk the two axes separately on a
   synthetic stream where the answer is known — vocabulary 11 → 100 → 1000 → 10⁴ at fixed `d`,
   then `d` at fixed vocabulary — and find where it stops. This turns a diffuse "it does not
   learn" into a located transition, and needs no new code beyond a synthetic corpus generator.
   It is also the cheapest of everything left.
2. **A GPT baseline on the identical pipeline.** No longer for comparison: to confirm the loop,
   the data and the metric are sound. If a GPT reaches ~130 on this pipeline as it did for the
   previous implementation, everything outside the PT forward pass is exonerated. If it does
   not, the bug is somewhere all models share and every PT diagnosis above is suspect.
3. **The exact readout is pure pooling and has no query at all** —
   `log μ_t(a) = Σ_c LSE_{j∈D_t} B^(c)_{j,a}` is a soft max over prefix positions per label.
   §23.3 flags exactly this: "the optimisation behaviour of LSE-pooling gradients at LM scale
   is untested". It scored 1555 here, more than twice the unigram. That is consistent with the
   pooling being the weak link, and it is a claim about the *construction*, not the code.

**What must not happen next:** another hyperparameter sweep. Nine runs spanning 45× in `λ_Z`,
10⁴× in the L2 coefficient and 25× in initialisation scale moved the metric by less than 2 %.
The answer is not in that space.
