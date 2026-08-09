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

1. **Instrument the readout across sequences, not across positions.** Every diagnostic so far
   measured variation over *positions*, which is exactly the axis the model still uses. The
   quantity that matters is `std across sequences at a fixed slot`, stage by stage:
   `q̄ → B → α → G → Q_Z → logits`. `q̄` has it (1.9e-3 to 3.8e-3) and the logits do not
   (8.9e-5). One of those five stages destroys it, and the measurement is cheap and local.
2. **Check the predictive attention's query.** §17.1 initialises `Q_Z^(0) = σ(s̄/λ_Z)`, a
   single global vector — identical at every slot *and every sequence*. At `τ = 1` the whole
   readout would then see the prefix only through the mask; `τ = 2` is supposed to fix that by
   making round 2's query context-dependent. Whether it does at this scale is measurable, and
   `τ` has never been varied in any run above — it sat at 2 throughout.
3. **The exact readout is pure pooling and has no query at all** —
   `log μ_t(a) = Σ_c LSE_{j∈D_t} B^(c)_{j,a}` is a soft max over prefix positions per label.
   §23.3 flags exactly this: "the optimisation behaviour of LSE-pooling gradients at LM scale
   is untested". It scored 1555 here, more than twice the unigram. That is consistent with the
   pooling being the weak link, and it is a claim about the *construction*, not the code.
4. **A GPT baseline on the identical pipeline** — but now for a different reason than
   comparison: it would confirm the loop, the data and the metric are sound, which currently
   rests only on the synthetic-stream test in `tests/test_11`. Cheap and worth having before
   any further PT diagnosis.

**What must not happen next:** another hyperparameter sweep. Nine runs spanning 45× in `λ_Z`,
10⁴× in the L2 coefficient and 25× in initialisation scale moved the metric by less than 2 %.
The answer is not in that space.
