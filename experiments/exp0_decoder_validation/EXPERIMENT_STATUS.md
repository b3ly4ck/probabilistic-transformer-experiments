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

## Run log

| Date | Commit | Config | Seed | Metric | Wall-clock |
|---|---|---|---|---|---|
| 2026-08-09 | (this commit) | toy `d=4 h=2 V=7 n=6 B=3 γ=2 T=2`, float64, CPU | 0/1 | checks 1–9: 54/54 pass | 34 s |
| 2026-08-09 | (this commit) | overfit `d=16 h=2 V=12 n=8 B=1 γ=2 T=2`, exact readout, Adam lr=0.05, 400 steps, float32, CPU | 3/7 | loss 2.4533 → 0.1229 (50) → 0.0178 (100) → 0.0060 (200) → 0.0022 (400); argmax == target at every slot | 3.4 s |
| 2026-08-09 | (this commit) | same, mean-field readout, τ=2 | 3/7 | loss 2.4826 → 0.1016 (50) → 0.0144 (100) → 0.0049 (200) → 0.0017 (400); argmax == target at every slot | 5.5 s |
| 2026-08-09 | (this commit) | worked example: `V=4 d=2 h=1 γ=0`, serial, τ_obs=1, τ=2, float64 | — | every printed quantity of `causal_pt_output_note.pdf` §5 reproduced to 3 d.p. | <1 s |

Initial loss ≈ 2.48 is `log 12`, i.e. the uniform distribution — the model starts
uninformative, as it should with `b = 0` and small `S`.
