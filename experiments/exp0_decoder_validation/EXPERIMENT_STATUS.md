# Experiment 0 — Implement and validate the causal PT decoder

Nothing else exists until this passes. This is the only code written from scratch.

## Question

Does the construction of `causalprobabilistictransformer_1.pdf` Parts II–IV, implemented
literally, produce a decoder that is (a) the model that was declared, and (b) causal?

This is not a comparison and produces no perplexity number. It is the gate that every
later experiment stands on.

## What is being validated

| | Object | Specification |
|---|---|---|
| Content stream | `q̄_t ≈ p(Z_t | w_{1:t})`, MFVI on the directed chain | Part II §12.2, updates as boxed there |
| Exact readout (mainline) | `p̂(W_t=w) ∝ e^{b_w} Σ_a e^{S_{w,a}} μ_t(a)`, `log μ_t(a) = Σ_c LSE_{j∈D_t} B^(c)_{j,a}` | §17.2, made mainline by §23.3 |
| Mean-field readout (ablation) | `Q_W ∝ exp((b + Q_Z S)/λ_W)` after τ rounds of (2)–(3) | §17.1 |
| Arc contraction | `B^(c)_{j,a} = Σ_b q̄_j(b) T^(c)_{a,b}`, `B^(c)_{ROOT,a} = r^(c)_a` | Part II §12.2, Wu & Tu App. B.3.1 |
| RPE | distance-sensitive `T[f(i−j)]^(c)`, causal half of the clipped table | Wu & Tu Eqs. 9/10 |
| Global head (optional) | single-split `G_t`, `φ_b(G_t=k, Z_t=a) = exp(B'_{k,a})` | Wu & Tu App. B.3.3, Part IV §22.2 |

The parameter list is exactly the factor list: `S`, `{T^(c)}` (or `U`, `V`), `r`, `b`,
and `B'` when the global head is on. `tests/test_05_tying.py` asserts behaviourally that
no second vocabulary-sized matrix exists.

## Scale

Deliberately tiny and CPU-only, per the research plan: `d = 4`, `h = 2`, vocabulary 7,
sequence length 6, batch 3, 2 MFVI iterations, `float64`. The overfit check uses
`d = 16`, vocabulary 12, length 8, batch 1, `float32`.

Batch size 1 in the overfit check is deliberate: slot 0 of every sequence is predicted
from `D_0 = {ROOT}`, an identical context, so two sequences with different first tokens
would put an irreducible floor on the loss.

## Validation checks and where they live

| # | Check | File | Status |
|---|---|---|---|
| 1 | Shapes of every intermediate | `tests/test_01_shapes.py` | pass |
| 2 | Posteriors normalised, and on the right support | `tests/test_02_normalisation.py` | pass |
| 3 | Causality, bitwise | `tests/test_03_causality.py` | pass |
| 4 | Prefix gradient reaches `j < t` and exactly zero for `j ≥ t` | `tests/test_04_prefix_gradient.py` | pass |
| 5 | Tying — one matrix, both roles | `tests/test_05_tying.py` | pass |
| 6 | Overfit a single batch | `tests/test_06_overfit.py` | pass |
| 7 | Worked example of the note, §5 | `tests/test_07_worked_example.py` | pass |
| 8 | Free energy non-increasing | `tests/test_08_free_energy.py` | pass |
| 9 | Exact readout vs. brute-force enumeration | `tests/test_09_exact_vs_brute.py` | pass |

Two checks are worth calling out because they are the ones that test the *equations*
rather than the plumbing:

* **9** enumerates `p(W, Z, H^(1..h))` over every assignment with plain Python loops and
  compares against the closed-form readout. Agreement to `1e-12` in `float64` validates
  the factorisation itself. It is run for the plain slot and with the global head folded
  in as an extra leaf, and separately the vectorised `logcumsumexp` scan is checked
  against slot-by-slot assembly for `γ ∈ {0, 1, 2, 4, 9}` — that is where an off-by-one
  in the RPE bucketing would hide.
* **8** additionally contains a mutation test: the same monotonicity check is run against
  a sign-flipped H-update and must *fail*. A check that cannot fail proves nothing, and
  checks 1–7 survive that mutation untouched.

## Exit criterion

Checks 1–9 pass and the loss on a single batch goes to ~0. Met — see the run log.

## What would falsify the implementation

* A change to token `t` moving the logits at any slot `≤ t` (check 3).
* The exact readout disagreeing with brute-force enumeration by more than float noise
  (check 9) — that would mean the readout implements a different model than the one
  declared, which is the failure the whole output mechanism was designed to avoid.
* Free energy rising along the inner loop (check 8) — the update rule would not be the
  gradient of the energy written down.
* A second vocabulary-sized parameter appearing (check 5) — untied embeddings are
  unrepresentable in this model class, so their appearance is a bug, not a variant.

## Decisions taken, and why

**Exact readout is the mainline.** §17.2 recommends MFVI; §23.3 of Part IV inverts that
recommendation explicitly ("Verdict, inverted as the question anticipated: exact readout
mainline; mean-field two-stream as the ablation"). The later section wins. Both are
implemented — the mean-field readout is Experiment 3's comparison object, and it is what
reproduces the worked example's printed numbers.

**Two schedules, both causal, both from §12.3.** `schedule="parallel"` is the layer-parallel
one — iteration `t` computes all `Q_i^(t)` from `{Q_j^(t-1)}_{j<i}` under one strict
lower-triangular mask, giving exactly the computation graph of a depth-`T`,
parameter-shared causal transformer. That is the training path. `schedule="serial"` is
left-to-right filtering with a per-slot inner loop; it is slower, it is the clean
filtering story, and it is what the note's §5 worked example uses, so check 7 needs it.
The two are different schedules of the same updates and do **not** agree numerically —
that is expected, not a bug.

**Gradients through the frozen prefix — a conflict with `CLAUDE.md`, resolved toward the
paper.** `CLAUDE.md` constraint 3 and the research plan's check 4 say that *neither
gradients nor messages* may flow backwards from step `t` into the prefix posteriors. The
specification says the opposite about gradients, in two places:

> Part II §12.3 Check 2 — "*frozen* means constant with respect to step-`i`'s inference
> problem, *not* detached in autodiff. Training gradients flow backward through
> `B^(c)_{j,·}` into `q̄_j` exactly as they flow through cached activations in a causal
> transformer. Forward causality is what defines a decoder; backward gradient flow is
> what trains it."

> Part III §18 Check 5 "Gradients" — the same, adding that this is how the tied matrix
> `S` is trained in both of its roles.

`CLAUDE.md` states that the paper wins when code and paper disagree, so the mainline is
`detach_prefix=False`. The other reading is available as `PTConfig.detach_prefix=True`
and is covered by a test, so switching is one line. **This needs a human decision** — if
the intended reading is the stop-gradient one, the paper has to be changed first.

What is *not* in dispute and is asserted in check 4: the loss at slot `t` reaches prefix
beliefs `q̄_j` only for `j < t`, and exactly zero gradient arrives at `j ≥ t`.

**`λ_H` defaults to `1/d`,** the Wu & Tu default from §2.3.3 with the variance argument
of Appendix A.5. `λ_Z = 1`, `λ_W = 1` (§18 Check 5 fixes it; §22.1 reopens it as a lever,
which is a later experiment's variable, not a default).

**The global head is a flag, defaulting off.** §22.2 argues it "should ship in the causal
model from day one". The research plan makes it a *measured* variable — arm 1.1 without,
arm 1.2 with — and baking it in would assume the answer. It is implemented and tested so
that both arms are one config field apart.

## What is deliberately not here

* Training loop, data pipeline, GPT and Looped baselines — Experiment 1's scope.
* Sampling / the generation loop of §18 Check 6. `next_token_logits` is step 2 of it;
  steps 3–5 (sample, run the observed step, append to the cache) are not written.
* The `r`-space fast path. Arc scores are materialised as `d × d` per channel and bucket,
  so attention logits cost `O(n² d)` rather than `O(n² r)`, and the per-bucket contracted
  scores cost `O(n_dist · B · h · n · d)` memory. Correct but not cheap; this is the first
  thing to optimise before Experiment 1, and it changes no mathematics.
* Fixed-lag smoothing (§25.3), factored labels (§22.2), continuous labels. All named in
  Part IV as extensions, none of them needed for the empirical section.

## Measured diagnostics (2026-08-09, review pass)

`src/diagnostics.py`, `python -m src.diagnostics`. All numbers below are measurements,
not estimates.

### Message scale — what replaces layer norm

There is no layer norm and no residual, because either would be a *map* and §22.2 names
that as a tripwire. The bound comes from the simplex instead, and it is tight:

```
|G_i(a)| = |Σ_c Σ_j Q_c(j) B^(c)_{j,a}|  ≤  h · max(max|T|, max|r|)
```

since `Q_c` is a distribution over `D_i` and `B^(c)_{j,·}` is itself an expectation of a
row of `T^(c)` under `q̄_j`. Asserted in `tests/test_10_diagnostics.py` at three parameter
scales. On the overfitted toy model `max|G| = 14.03` against a bound of `2 × 7.388 = 14.78`
— 95 % of it, so the bound is not slack. Activations cannot diverge at fixed parameters;
only the parameters can grow, and the original's control for that is an explicit L2
penalty on the ternary scores (Wu & Tu §4.2, Table 2: `5e-4` on PTB). **Not implemented
here** — there is no training loop yet, but it belongs in it.

| | untrained, `d=384 h=16 r=64 γ=3 T=5`, len 64 | overfitted toy, `d=16 h=2 V=12` |
|---|---|---|
| `‖G‖/‖S_w‖`, iter 1 → last | 0.342 → 0.348 | 1.839 → 1.928 |
| `max|G|` | 0.247 | 14.03 (bound 14.78) |
| attention `H/H_max` | 0.982 | 0.246 → 0.223 |
| `H(Q_Z)` | 5.950 nats (`log 384 = 5.951`) | 0.442 → 0.003 nats |
| `ρ` (Lemma 23.1) | 233.2 | 206.5 |

Reading: at initialisation the model is essentially uniform and context contributes about
a third of the label belief; after fitting, the message *dominates* the unary roughly 2:1
and both the attention and the label posterior are nearly hard.

### `λ_H = 1/d` does not sharpen the softmax at initialisation

Measured `H/H_max = 0.982` at `d = 384`, i.e. attention within 2 % of uniform. That is
exactly what the weight is for: Wu & Tu App. A.5 shows `F_ic(j)` has variance `σ²/d²`
under uniform beliefs, so dividing by `1/d` restores variance `σ²`. Multiplying the
logits by `d` is not a sharpening, it is a *de*-shrinking.

The sharpening that does occur is driven by `‖T‖`, not by `d`. Feeding progressively
peaked beliefs into the same untrained model: `H(Q_Z)` 5.95 → 4.10 nats moves `H/H_max`
only 0.982 → 0.932. After training, where `max|T|` reached 7.4, `H/H_max` fell to 0.22.
So the answer to "does `1/d` harden the choice" is **no at init, yes once the arc scores
grow** — which is one more reason the L2 penalty on `T` is not optional.

### The root column is initialised on a different scale — measured variable, not a defect

**Superseded framing.** An earlier pass of this file called this an "open defect". It is
an arithmetic fact about the initialisation, and it inflates `ρ`, but the two consequences
it was suspected of causing were both tested and found absent: it does not cause
multistability (see the `ρ` section), and it does not produce an attention sink. Measured
ROOT attention mass, last slot: 1.10× uniform at the source's configuration untrained, and
**0.0001 against a uniform 0.125** on the overfitted toy model — after training the model
*avoids* ROOT rather than sinking into it. Default left unchanged, per the review; the
mass is now logged every iteration so that if a sink does appear on real data, the knob and
the reason are both already in place.

`r^(c)` enters the attention in raw `d`-space, while arc scores arrive contracted,
`B^(c)_{j,a} = E_{q̄_j}[T^(c)_{a,·}]`, which shrinks them by ≈`1/√d` for a near-uniform
prefix belief and again by the Kruskal product. Drawing both from `N(0, init_std²)`:

* ROOT row norm `0.390` vs. prefix row norm `0.0032` — **121×**;
* `ρ = 233.2` including ROOT, `ρ = 0.976` excluding it. The entire violation of the
  contraction condition is the root column.

At initialisation this is nearly invisible in the attention itself (ROOT gets 0.0174 of
the mass against 0.0154 for uniform) because a near-uniform `Q_Z` averages `r`'s
zero-mean entries away. It bites once `Q_Z` sharpens, at which point `⟨Q_Z, r⟩ → r_a`
while `⟨Q_Z, B_j⟩ → B_{j,a}`, and the ratio of the two is the 121×. That is an
attention-sink prior nobody chose.

`PTConfig.root_init_std` now exists so the scale can be set independently; the default is
unchanged (`= init_std`), because changing it is a modelling decision, not a bug fix I
should make silently. `init_std = 0.02` itself is **not from either paper** — it is the
nanoGPT convention. Neither Wu & Tu nor the causal document specifies an initialisation.

### `ρ ≫ 1` — the bound is vacuous, and that is all it means

**What Lemma 23.1 actually is.** §23 is "Defect 2: the prior/posterior divergence under
MFVI". §23.1 opens: "Within slot `t`, the predictive and observed runs differ only in the
word message `m_W`: `s̄` against `S_{w_t,·}`. Both share the constants `B^(c)` and the
schedule." The lemma bounds the distance between **those two runs of the same inner loop**
— it is not a convergence statement about one run. With `δq_s` the difference between the
two runs' `Q_Z` after round `s`:

```
δα_s^(c) ≤ ‖B^(c)‖₂/(2λ_H) · δq_{s-1}
δq_s     ≤ ‖ΠΔm_W‖/(2λ_Z) + ρ · δq_{s-1},     ρ := Σ_c ‖B^(c)‖₂² / (4 λ_Z λ_H)
ρ < 1  ⟹  δq_∞ ≤ (1/(1-ρ)) · ‖ΠΔm_W‖/(2λ_Z),   ΠΔm_W = Π(S_{w_t,·} − E_{Q_W} S_{w,·})
```

The proof is two applications of the softmax Lipschitz constant `‖diag(p) − ppᵀ‖₂ ≤ ½`
(Popoviciu) plus the mean value theorem, then a geometric series. The forcing term
`ΠΔm_W` is the *embedding-space surprisal of the observed word*, so the bound says the
prediction-time and encoding-time attention patterns diverge little on predictable tokens
and more on surprising ones — the prior/posterior behaviour Bayes prescribes.

Setting `Δm_W = 0` leaves `δq_s ≤ ρ δq_{s-1}` for two runs that differ only in
initialisation, so **`ρ` is also the contraction factor of the slot map itself**: `ρ < 1`
makes it a contraction, hence a unique fixed point reached from any start. That is the
sense in which it is a stability criterion, and it is *sufficient, not necessary*.

**Superseded claim.** An earlier pass of this file reported `ρ = 233` and treated the root
column's share of it as potentially central. That reading is withdrawn — it was an
inference from a vacuous bound, not a measurement. The direct test now exists
(`fixed_point_multiplicity`): run the slot inner loop from 48 random initialisations of
`Q_Z` to convergence and count distinct fixed points.

| config | `ρ` raw | `ρ` centred | distinct fixed points | max separation |
|---|---|---|---|---|
| toy `d=4 h=2`, `init_std=0.1` | 0.053 | 0.020 | 1, 1, 1 | 0.000 |
| `d=24 h=4`, `init_std=0.3` | 54.9 | — | 1, 1, 1 | 0.000 |
| `d=24 h=4`, `init_std=0.4` | 102.7 | — | 1, 1, 1 | 0.000 |
| `d=24 h=4`, `init_std=0.45` | 135.5 | — | 1, **2**, 1 | 0.221 |
| `d=24 h=4`, `init_std=0.6` | 292.8 | — | 2, 3, **5** | 0.720 |
| `d=24 h=4`, `init_std=2.0` | 10411 | 9486 | 15, 13, 19 | 1.000 |
| **`d=384 h=16 r=64`, source row, untrained** | **233.2** | **227.8** | **1, 1** | **0.000** |
| **overfitted toy, `d=16 h=2`** | **206.5** | — | **1** | **0.000** |

Readings, in order of importance:

1. **`ρ = 233` at the source's configuration produces exactly one fixed point.** So does
   `ρ = 207` after training. The bound being violated by two orders of magnitude is not,
   here, a symptom of anything. `ρ < 1` is very conservative; the empirical onset of a
   second fixed point in this setup is around `ρ ≈ 130`.
2. **The root column does not cause multistability.** Shrinking `root_init_std` by 100×
   and 1000× at `init_std = 2.0` left `ρ` at ~9500 and the count at 15–22 fixed points.
   Once the arc scores are large enough to matter, `ρ` is governed by `‖T‖`, exactly as
   §23.1 says ("`ρ` should be read as governed by `‖T‖²/(λ_Z λ_H)`"). The root column
   dominates `ρ` only in the regime where everything else is tiny — which is the regime
   with one fixed point anyway.
3. **`TV = 1.000` between the two schedules at `init_std = 2.0`, reported in the schedule
   section above, is now explained**: at that scale the slot map has 15–22 fixed points,
   so the two schedules landing on disjoint label mass is multistability, observed
   directly rather than inferred.
4. **Under the exact readout, the lemma's own subject largely disappears.** §23.3: the
   exact readout "removes the query stream entirely — no predictive inner loop, no
   `Q_Z^pred` iterations", and "with the exact readout, Defect 2 survives only in the mild
   warm-start form of Section 23.2". Lemma 23.1 bounds a divergence between two MFVI runs;
   with the mainline readout there is only one MFVI run. `ρ` still matters for the content
   stream's own uniqueness, and for the MFVI arm of Experiment 3.

So: worth logging, cheap, and now falsifiable by one call. Not a blocker and not a result.

### The two schedules can disagree completely

Both are legal causal schedules of the same updates (§12.3). Total variation between the
`q̄` they produce, `d=32 h=4 r=8 γ=3 T=5`, length 24:

| parameter scale | `τ_obs` | TV mean | TV max | `max|Δlogit|` | NLL parallel / serial |
|---|---|---|---|---|---|
| `init_std=0.02` | 1 | 3.4e-6 | 2.6e-5 | 1.3e-9 | 5.2976 / 5.2976 |
| `init_std=0.5` | 1 | 0.111 | 0.262 | 3.2e-2 | 5.2994 / 5.2985 |
| `init_std=0.5` | 5 | 0.016 | 0.098 | 1.0e-2 | 5.2994 / 5.2994 |
| `init_std=2.0` | 1 | 0.682 | **1.000** | 39.8 | 6.9491 / 6.9300 |
| `init_std=2.0` | 5 | 0.741 | **1.000** | 39.3 | 6.9491 / 7.2644 |
| overfitted toy | 1–5 | <1e-4 | <1e-4 | <1e-4 | 0.0022 / 0.0022 |

`TV = 1.000` means the two schedules put their label mass on disjoint labels — the
multistability `ρ ≥ 1` predicts, observed. They agree in the two degenerate regimes (near
uniform, and saturated one-hot) and can disagree totally in between. **Mainline for
training is `parallel`**, because §12.3 says its computation graph is exactly that of a
depth-`T` parameter-shared causal transformer, which is what makes the Looped comparison
of Experiment 2 an honest control; `serial` is the reference implementation and is needed
by check 7. Re-measure this on real data in Experiment 1 — it is cheap and it decides
whether the choice of schedule is a free parameter or a confound.

### The global head degenerates under the exact readout — partly

`log μ_t(a)` gains `LSE_k B'_{k,a}`, a single `d`-vector, independent of position and of
the prefix (§22.2 states this: "μ_t gains the multiplicative term `Σ_m e^{B'_{m,a}}`").
It does not cancel — the readout takes a log-sum-exp over labels, not a linear
combination — but only `d` numbers of the `m × d` matrix reach the readout directly,
whatever `m` is. Verified: `m = 9`, `d = 4` toy, the direct contribution is constant to
`1e-12` across every position and sequence when both models are given the identical
contracted prefix.

"`G_t` is therefore a constant" would still be the wrong summary. The global head also
enters the *content stream*, where the update gains the GFU term `σ(q B'ᵀ) B'` that Wu &
Tu identify as the feed-forward analogue; there it is position-dependent and it reshapes
`q̄`, so end to end it does move `log μ_t` position by position. Both effects are asserted
separately in `tests/test_10_diagnostics.py`.

### Parameter accounting, PT against a GPT layer

Non-embedding parameters, `12 d²` per GPT layer (4`d²` attention + 8`d²` MLP):

| PT config | embedding | non-embedding | in GPT layers |
|---|---|---|---|
| `d=384 h=16 rank=64 γ=3`, `V=10k` | 3,850,000 | 3,151,872 | 1.78 |
| `d=384 h=16 rank=None γ=3`, `V=10k` | 3,850,000 | 9,443,328 | 5.34 |
| `d=256 h=8 rank=32 γ=3`, `V=10k` | 2,570,000 | 526,336 | 0.67 |

`rank=None` (full `T`, the current default) costs `K·h·d²` and is 3× the decomposed form.
The source uses the decomposition on every task it reports (Table 2: `UV`, `r = 64` on
PTB MLM), and §18 Check 4 needs `T^(c) = U^(c)V^(c)ᵀ` for the attention correspondence to
be literal. **The default should probably be `rank = 64`, not `None`** — full `T` is kept
because it is the definitional form and the one the worked example uses, but it is not
what the source ran. Left as a decision rather than changed unilaterally.

### Evaluation alignment against a GPT baseline — must be fixed before Experiment 1

Given a block `w_0..w_{n-1}`, this model scores `n` tokens: every `w_t` from `w_{<t}`,
including `w_0` from ROOT alone. A GPT trained the usual way consumes `w_0..w_{n-2}` and
scores `n-1` tokens, `w_1..w_{n-1}`. Averaging both would compare different token sets,
and PT's extra slot is a first-word unigram prediction the baseline never makes.

`loss(idx, ignore_first=1)` now exists for this. Experiment 1 must use it. Giving the
baseline a BOS token is the other fix and the worse one — §18 Check 5 points out that PT
has a proper first-word distribution precisely so that no BOS hack is needed.

### Damping and step size are not implemented, and the source did not use them

Wu & Tu Appendix B.1 (step size `α_Z, α_H`) and B.2 (damping `β_Z, β_H`) exist, but the
appendix opens by saying these are "variants that we find do not bring significant
improvement empirically". The asynchronous update (§2.3.2), which *is* used by default and
*is* implemented here, is the one that matters — it is why `Q_c` is updated from the
current `Q_Z` before `Q_Z` is recomputed. Damping is a cheap thing to add if the `ρ ≫ 1`
regime turns out to cause oscillation on real data; it is not needed to reproduce the
source.

### Not implemented from the source's PTB configuration

Dropout (Table 2 gives 0.15 for PTB MLM), the L2 penalty on `T` (5e-4), and weight decay
(1.4e-6). All three are training-loop concerns and the training loop does not exist yet.
Recorded here so they are not silently omitted when it is written.

## Run log

| Date | Commit | Config | Seed | Metric | Wall-clock |
|---|---|---|---|---|---|
| 2026-08-09 | (this commit) | toy `d=4 h=2 V=7 n=6 B=3 γ=2 T=2`, float64, CPU | 0/1 | checks 1–9: 54/54 pass | 34 s |
| 2026-08-09 | (this commit) | overfit `d=16 h=2 V=12 n=8 B=1 γ=2 T=2`, exact readout, Adam lr=0.05, 400 steps, float32, CPU | 3/7 | loss 2.4533 → 0.1229 (50) → 0.0178 (100) → 0.0060 (200) → 0.0022 (400); argmax == target at every slot | 3.4 s |
| 2026-08-09 | (this commit) | same, mean-field readout, τ=2 | 3/7 | loss 2.4826 → 0.1016 (50) → 0.0144 (100) → 0.0049 (200) → 0.0017 (400); argmax == target at every slot | 5.5 s |
| 2026-08-09 | (this commit) | worked example: `V=4 d=2 h=1 γ=0`, serial, τ_obs=1, τ=2, float64 | — | every printed quantity of `causal_pt_output_note.pdf` §5 reproduced to 3 d.p. | <1 s |

| 2026-08-09 | (review pass) | diagnostics at `d=384 h=16 rank=64 γ=3 T=5`, len 64, batch 2, untrained, float32, CPU | 0 | `‖G‖/‖S_w‖` 0.342→0.348; `max|G|` 0.247; attn `H/H_max` 0.982; `H(Q_Z)` 5.950; `ρ` 233.2 (0.976 without ROOT); ROOT/prefix row norm 121× | 4 min |
| 2026-08-09 | (review pass) | same diagnostics on the overfitted toy model | 3/7 | `‖G‖/‖S_w‖` 1.839→1.928; `max|G|` 14.03 vs bound 14.78; attn `H/H_max` 0.246→0.223; `H(Q_Z)` 0.442→0.003; `ρ` 206.5 | <1 min |
| 2026-08-09 | (review pass) | schedule divergence, `d=32 h=4 rank=8 γ=3 T=5`, len 24 | 0 | TV(parallel, serial): 3.4e-6 at `init_std=0.02`, 0.111 at 0.5, 0.682 at 2.0 with `TV_max = 1.000`; `≈0` on the overfitted model | <1 min |
| 2026-08-09 | (review pass) | checks 1–10 after adding the diagnostics | 0/1 | 64/64 pass | 41 s |
| 2026-08-09 | (lemma pass) | fixed-point multiplicity, 48 random starts, 500 rounds, float64 | 0/1 | `ρ=0.05` → 1 fp; `ρ=103` → 1 fp; `ρ=135` → 2 fp; `ρ=293` → up to 5; `ρ=10411` → 15–22, separation 1.000. **Source row `ρ=233` → 1 fp.** Overfitted toy `ρ=207` → 1 fp | 3 min |
| 2026-08-09 | (lemma pass) | same, with `root_init_std` 100× and 1000× smaller at `init_std=2.0` | 0 | `ρ` 10411 → 9481/9490, fixed points 15–22 unchanged, ROOT mass → 0.0000. The root column is not the cause | 1 min |
| 2026-08-09 | (lemma pass) | ROOT attention mass, last slot | 0 | source row untrained 0.0232 vs uniform 0.0208 (1.10×); overfitted toy 0.0001 vs uniform 0.125 (0.15×) | <1 min |
| 2026-08-09 | (lemma pass) | full suite after adopting Table 2 defaults and adding the lemma checks | 0/1 | 70/70 pass | 44 s |

Initial loss ≈ 2.48 is `log 12`, i.e. the uniform distribution — the model starts
uninformative, as it should with `b = 0` and small `S`.

## Decisions taken in review, 2026-08-09

All five were ruled on. Nothing below is still open.

1. **`detach_prefix` stays `False`.** `CLAUDE.md` constraint 3 was rewritten instead of the
   code: it now separates the forward claim (no *message* may flow backwards — binding)
   from the gradient claim (gradients do flow, per Part II §12.3 Check 2 and Part III §18
   Check 5 — the previous wording was wrong).
2. **`rank = 64`, and the whole of Table 2's PTB row with it.** The instruction was
   `rank = 64`, but the decomposition only saves parameters while `2·rank < d`, so
   `rank = 64` at the previous default `d = 64` would have cost *more* than a full `T`.
   The coherent version is the source's row: `d = 384, h = 16, rank = 64, γ = 3, T = 5`.
   `PTConfig` now rejects `rank > d` outright. This also settles the performance question:
   3.15 M non-embedding parameters instead of 9.44 M.
3. **`n_iters = 5`** — the source's PTB value, replacing an arbitrary 4.
4. **`init_std = 0.02` stays.** Not from either paper, but at that scale the two schedules
   agree to 3.4e-6 and there is no better-argued alternative.
5. **`root_init_std` default unchanged; ROOT attention mass is now logged** every content
   stream iteration (`root_mass`, `root_mass_over_uniform`) and available standalone via
   `diagnostics.root_attention_mass`. Rationale, from the review: it is a measured variable
   with the cure already written, so changing it blind would destroy the observation.

Still not implemented, and required before Experiment 1 (Wu & Tu Table 2, PTB MLM):
dropout 0.15, L2 penalty on `T` 5e-4, weight decay 1.4e-6. The L2 term is the one that
matters most — it is the only mechanism restraining `‖T‖`, and `‖T‖` is what drives both
the message scale and `ρ`. Also mandatory: `loss(idx, ignore_first=1)`, or PT and the GPT
baseline compute perplexity over different token sets.
