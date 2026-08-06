# Experiment 0 — Implement and validate the causal PT decoder

| | |
|---|---|
| **Status** | checks 1–9 pass, **but check 6 is seed-fragile and the model does not train at scale** — diagnosis below |
| **Priority** | mandatory — nothing else exists until this passes |
| **Last updated** | 2026-08-05 |
| **Plan reference** | [RESEARCH_PLAN.md](../../developer%20files/RESEARCH_PLAN.md) § Experiment 0 |

## Question

Does the causal PT decoder forward pass, as derived in Part III and
`causal_pt_output_note.pdf` §4, implement the equations it claims to implement?

This is a correctness experiment, not a learning experiment. No perplexity number produced
here means anything.

## Success criterion

Validation checks 1–9 pass, and the loss on a single fixed batch falls to ~0. Only then does
work on real data begin.

## Design

Deliberately tiny and CPU-runnable, so every intermediate tensor can be printed and read:

| Item | Setting | Rationale |
|---|---|---|
| `d` (label dim) | 8 for most checks; **32** for the exact readout in check 6 | see the capacity finding below |
| Vocabulary | 20 (12 in checks 6, 4 in check 7) | check 7 is fixed by the note |
| Sequence length | 6 | |
| Batch size | 2 (1 for the overfit checks) | |
| Rounds `T` / `τ` | 3 | ≥2, so the attention query is context-dependent |
| Channels `h` | 1 (2 in checks 6 and 8) | |
| Device | CPU | check 3 requires bitwise reproducibility |
| Seed | 0, fixed | |
| Environment | `./.venv`, Python 3.11.15, torch 2.13.0+cpu, pytest 9.1.1 | created with `conda create -p ./.venv python=3.11`, then `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

## Validation checks

Record the commit at which each check first passed, and never delete a row — a check that
passed and later broke is the most valuable line in this file.

| # | Check | Status | Commit | Date | Notes |
|---|---|---|---|---|---|
| 1 | Shapes of every intermediate tensor match Part III | ☑ | `ab7c082` | 2026-08-05 | also asserts the parameter set is exactly `{S, T, r, b}` and that the ROOT key row is `r` |
| 2 | All posteriors normalise to 1 along their variable axis | ☑ | `ab7c082` | 2026-08-05 | disallowed keys carry *exactly* zero, not merely small mass |
| 3 | Causality: changing token `t` leaves logits at `≤ t` bitwise unchanged (CPU, fixed seed) | ☑ | `ab7c082` | 2026-08-05 | `torch.equal`, not `allclose`. Guarded by a test that later positions *do* move |
| 4 | No anti-causal path: gradient of logits at `t` w.r.t. `q̄_j` is zero for `j ≥ t` | ☑ | `ab7c082` | 2026-08-05 | restated — see the open-questions note. Also asserts `S`, `T`, `r` **do** receive gradient |
| 5 | Tying: input and output word matrices are the *same tensor object* | ☑ | `ab7c082` | 2026-08-05 | mutating the single parameter moves both roles; no `(d, V)` parameter exists |
| 6 | Overfit a fixed batch | ☑ | `ab7c082` | 2026-08-05 | per sequence, not per batch — see the `D_0` floor below |
| 7 | Worked example reproduces `causal_pt_output_note.pdf` §5 numbers | ☑ | `ab7c082` | 2026-08-05 | **digit for digit**, including the intermediate attention logits |
| 8 | Mean-field free energy non-increasing across MFVI iterations | ☑ | `ab7c082` | 2026-08-05 | plus: the Z-update is the argmin, not merely a descent step |
| 9 | Exact sum-product readout agrees with brute-force enumeration | ☑ | `ab7c082` | 2026-08-05 | agreement to `7.5e-08`; also validates the `logcumsumexp` scan against the per-slot reduction |

Checks 1–7 catch shape, axis and masking errors. Checks 8–9 are what test the update
equations themselves — a model with a typo in (2)–(4) passes 1–7 and overfits a batch.

### Check 7 — reference numbers

From `causal_pt_output_note.pdf` §5: `V = {the, cat, sat, mat}`, `d = 2`, labels {N, V},
`h = 1`, `λ_Z = λ_H = 1`. Observed pass with `τ = 1`, predictive with `τ = 2`.
Target values to reproduce (slot 4, predictive, after readout):

| | the | cat | sat | mat |
|---|---|---|---|---|
| logit | 2.368 | 1.824 | −1.824 | 1.824 |
| `p̂(W₄ \| w<4)` | .460 | .267 | .007 | .267 |

Reproduced: ☑ — exactly, to all three printed decimals, with no adjustment to the target. The
intermediate quantities match too: `s̄ = (1.063, −1.063)`, `Q_Z^(0) = (.893, .107)`, attention
logits `(.160, .413, .178, 1.776)` then `(.074, .387, .128, 1.890)`, `Q_Z` after each round
`(.951, .049)` then `(.956, .044)`, and the observed-pass `q̄` and cached `B` for all three
slots. Run `experiments/exp0_decoder_validation/worked_example.py` to print the full trace.

## Run log

| Run | Date | Commit | Config | What was run | Outcome |
|---|---|---|---|---|---|
| 1 | 2026-08-05 | `ab7c082` | toy, CPU, seed 0 | full check suite, `pytest tests` | **30 passed in 75.6 s** |
| 2 | 2026-08-05 | `ab7c082` | §5 numbers | `worked_example.py` | reproduces the note exactly; exact-vs-brute agreement `7.45e-08` |
| 3 | 2026-08-05 | `ab7c082` | `V=12, n=6, h=2, T=3`, Adam lr .05 | overfit sweep over `d`, readout, batch | see results |

## Results

**Overfit sweep** (single sequence unless stated, 1200–1500 Adam steps, lr 0.05):

| Readout | `d` | Batch | Final loss | Token accuracy |
|---|---|---|---|---|
| exact | 8 | 1 | 0.561 | 0.67 |
| exact | 32 | 1 | **0.002** | 1.00 |
| exact | 32 | 2 | 0.210 | — (see the `D_0` floor) |
| MFVI | 8 | 1 | **0.003** | 1.00 |
| MFVI | 32 | 1 | 0.693 | — |

Per-position loss, exact readout at `d=32`, batch of 2 with differing first tokens:
`[0.705, 0.007, 0.019, 0.275, 0.245, 0.007]` — the first entry is `log 2 = 0.693` to within
noise, everything after it fits.

## Decisions and justifications

**`D_0 = {ROOT}` imposes an information floor at position 0.** The head domain at the first
slot contains only ROOT, so the logits there are identical for every sequence in the batch —
the model has no context to condition on and structurally cannot. Two sequences with different
first tokens therefore carry an irreducible `log 2` at `t = 0`. The plan's "loss must fall to
near zero" is met per sequence; the floor is asserted as a property rather than tuned away.

**The exact readout needed `d = 32` where the mean-field readout memorised at `d = 8`.**
Diagnosed rather than patched: it is a capacity effect, not an optimisation failure — raising
steps, learning rate, channels and rounds did not close the gap, widening `d` did, immediately.
The mechanism is structural: with the query stream removed (§23.3), `log μ_t(a) = Σ_c LSE_j
B^(c)[j,a]` is *pooled* evidence with no query, so the whole context signal is `d` numbers
accumulated monotonically along the prefix. Mean-field attention *selects* a head; LSE pooling
*averages* over them.

**Both readouts are kept in one model.** They share `S`, `T`, `r`, `b` and differ only after
`q̄`, so Experiment 3 compares readouts on identical parameters rather than two models.

**The content stream is layer-parallel; the worked example is sequential.** §18 Check 4 licenses
the layer-parallel schedule (all positions in one pass under one triangular mask), and that is
what `content_stream` implements. §5 of the note uses the strictly sequential per-slot schedule
— clamp, one round, freeze, cache — so check 7 drives the slot primitives directly. The two
schedules are not expected to produce identical `q̄`; the relaxation is the one §18 already
accepts for the content chain. **Not yet measured: how far apart they actually are.**

**Two checks were rewritten after failing for the right reason.** Check 9's guard perturbed
every arc score by a constant, which cancels in the softmax — the model is genuinely invariant
to that, and the invariance is now asserted deliberately instead. Check 6's original form
demanded zero loss on a batch whose first tokens differ, which the `D_0` floor forbids.

## Open questions

- ~~§17.2 recommends MFVI as mainline; §23.3 walks this back toward exact readout.~~
  **Resolved 2026-08-05.** §17.2 and §23.3 read. §23.3 states the verdict outright: *exact
  readout mainline; mean-field two-stream as the ablation.* The query stream disappears
  (§24.1), and the exact readout is a causal `logcumsumexp` prefix scan over
  `B^(c)_{j,a}` seeded with `r^(c)_a` — `O(ndh)` and fully parallel, so the layer-parallel
  schedule survives. MFVI is still implemented, as the ablation and as Experiment 3's
  comparison object.

- **Gradient flow through the content stream — corrected 2026-08-05.** An earlier design here
  proposed running the filtering pass under `torch.no_grad()`, on the reading that "the prefix
  is frozen" meant stop-gradient. **That reading is wrong.** §18 Check 4 describes the schedule
  as layer-parallel two-stream attention in the XLNet sense, with both streams under one
  triangular mask; §25.1 frames freezing as *dropped evidence*, an inference-level statement.
  "Frozen" means `q̄_j` is a **conditioning constant of the variational problem at later slots**
  — it is not re-optimised there — not that it is detached from autograd. Causality is enforced
  by the strict lower-triangular mask, exactly as in a transformer, and gradients flow through
  the content stream normally. Running it under `no_grad` would starve `S` and `T^(c)` of the
  gradient that trains the representation.

- **Should the Appendix B.3 global variables ship from day one?** §22.2 argues yes: adding a
  global-head variable `G_t` per slot is the graph-faithful answer to "PT lacks an FFN", and
  without it the experiment "confounds *causal construction* with the known encoder-side
  capacity gap". It costs `O(md)` per position and stays inside the graph — `μ_t` simply gains
  a multiplicative term. This is a scope decision for Experiments 1–2, not for the toy
  validation. Resolution: —

## Finding to raise with the supervisors

**On the §5 example, the exact readout does not reproduce the behaviour the example was chosen
to illustrate.** The note's reading of slot 4 is that context "puts .993 of the mass on the
noun-like cluster and kills the verb": MFVI gives `p(sat) = .007`. The exact readout, on the
same frozen prefix and the same parameters, gives `p = (.365, .215, .206, .215)` — `p(sat) =
.206`, essentially flat. The cause is visible in one line: `log μ_4 = (2.382, 2.338)`, nearly
equal across the two labels, because the LSE over `j` washes out the contrast that MFVI's `Q_c`
concentrates (weight `.604` on slot 3, whose `B_3 = (1.987, .006)` is a strong N signal).

This is not an implementation discrepancy — the exact readout agrees with brute-force
enumeration to `7.5e-08`, so it is the correct marginal of the declared model. It is the
§23.3 caveat "the single summary belief `Q_Z^pred` is lost; `μ_t` replaces it (finer, but
different)" showing up immediately, and it is consistent with the capacity finding above.

Scope of the evidence, stated honestly: one hand-built example with `d = 2`, one channel and a
three-token prefix, plus toy memorisation runs. It says nothing yet about LM scale. But it
bears directly on the §23.3 verdict that made the exact readout the mainline, and it is cheap
to raise now rather than after Experiment 1.

## Reproduce

```bash
./.venv/bin/python -m pytest tests
```

```bash
./.venv/bin/python experiments/exp0_decoder_validation/worked_example.py
```


---

# Diagnosis, 2026-08-06 — why context does not reach the output

Opened because PT reaches train ppl 611 on PTB against a unigram baseline of 687, at 88 epochs,
with the curve flat for the last half of training. GPT on the identical pipeline reaches train
ppl 5.4, so data, tokenisation and the training loop are not at fault. Per `RESEARCH_PLAN.md`
this is a return to validation, not a language-modelling result.

## Mechanism

**Training drives the MFVI inner loop into a saturated fixed point in which the head message
dominates the word unary, and every additional round compounds it.** Once there, attention is
near-deterministic, `q̄` is nearly the same at every position, and nothing downstream can
recover context that the content stream no longer carries.

It is a self-reinforcing loop: `T` grows → attention sharpens → the message becomes one
full-magnitude `B` row instead of an average over the prefix → the message grows → it swamps the
word unary in the `Z`-update → `q̄` loses its positional spread → attention has nothing left to
discriminate on and stays saturated.

## Evidence

### 1. Prefix ablation on trained checkpoints (`diagnose_context.py`)

Logits at the last position, prefix zeroed and prefix shuffled:

| Model | shuffled: max Δlogit | KL | argmax agrees |
|---|---:|---:|---:|
| PT MFVI (val ppl 664) | 1.85 | **0.0115** | 0.75 |
| GPT (val ppl 131) | 28.67 | **12.26** | 0.00 |
| PT untrained | ~0 | ~0 | 1.00 |

Context moves PT's output about **a thousand times less** than GPT's. Shuffling the prefix leaves
three quarters of PT's predictions unchanged.

### 2. Logit decomposition — the unary does *not* dominate the readout

`logit_w = b_w + Σ_a Q_Z(a) S[w,a]`, on trained weights:

| Term | std over vocabulary | range |
|---|---:|---:|
| `b_w` | 0.774 | 6.45 |
| `Σ_a Q_Z(a) S[w,a]` | 0.676 | 6.46 |

Ratio 1.14 — the two terms are comparable, so the obvious hypothesis is **wrong**: the readout is
not drowned by the word unary. The problem is one level up. The same context term varies across
*positions* by only 0.193, and `Q_Z` itself has a spread across positions of **0.0022**. The
context term is large but nearly constant — it is a second bias, not a signal.

### 3. Trained posteriors, and where the collapse happens

Content-stream trace, PTB validation batch, per MFVI round:

| Round | ‖word unary‖ | ‖message‖ | msg/word | `q̄` std over t | `q̄` entropy | `Q_c` entropy |
|---:|---:|---:|---:|---:|---:|---:|
| init | 16.10 | — | — | 0.00072 | 5.520 | — |
| 1 | 16.10 | 18.18 | 1.13 | 0.00153 | 4.501 | 1.869 |
| 2 | 16.10 | 28.23 | 1.75 | 0.00206 | 3.822 | 0.726 |
| 3 | 16.10 | 67.18 | **4.17** | 0.00310 | **2.031** | **0.353** |

Untrained, same shape: msg/word reaches only 0.62 and both entropies stay flat (5.54, ~2.9).
**Training creates the domination**; it is not there at initialisation.

`Q_c` entropy 0.353 nats against a maximum of 4.159 — attention is effectively hard. 39.9 % of
its mass sits on ROOT, which is a constant. This also corrects an earlier note in the report:
attention entropy was checked *at initialisation* (3.1 nats, not saturated) and a conclusion
drawn from it; after training it is 0.35.

`q̄` still identifies its own word — variation between word types 0.285 against 0.102 within a
word type — so the word is not erased. What is destroyed is the *positional* dynamic range.

### 4. `λ_H` is not the mechanism

`λ_H = 1/d` multiplies attention logits by `d`, so it was the leading suspect. Decoupled and
pinned to `1/8` across the toy memorisation task at `d = 8 … 256`, it changes nothing: the task
fails at `d ≥ 16` with both the coupled and the pinned value. Ruled out.

### 5. The number of MFVI rounds is the driver

Toy memorisation (check 6's task), fit rate over five seeds, 1200 Adam steps, fit = final loss
below 0.05:

| `d` \ rounds | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 8 | 2/5 | 3/5 | 1/5 | **0/5** |
| 16 | 3/5 | 3/5 | 2/5 | **0/5** |
| 64 | 1/5 | 1/5 | 1/5 | 1/5 |
| 256 | 2/5 | 0/5 | 0/5 | **0/5** |
| **total** | **8/20** | **7/20** | **4/20** | **1/20** |

Monotone in rounds; at four rounds nothing fits at any `d`. The `d` dependence is weak and noisy
by comparison — the apparent cliff between `d=8` and `d=16` in the first pass was seed noise.

**More steps do not help.** At `d=16`, 6000 steps give 1.1102 against 1200 steps' 1.1103 — a hard
plateau, not slow convergence. The same is true on PTB, where the curve is flat over the last
10 000 steps.

## Correction to the validation suite

**Check 6 passes on one seed and is not representative.** Its configuration — `d=8`,
`n_rounds=3`, seed 0 — has a fit rate of **1/5**. The suite reported a healthy model because it
happened to sit on the seed that works. Any future claim from check 6 must be over several
seeds; a single-seed memorisation test cannot distinguish a model that fits from one that fits
sometimes.

## What remains ambiguous, and the experiment that separates it

Two readings survive the evidence:

- **(a) The fixed point is degenerate.** Iterating the updates converges to a state where the
  message dominates by construction, so no parameter setting reachable by gradient descent
  avoids it.
- **(b) The fixed point is fine and the optimisation cannot reach it.** Saturated softmaxes kill
  the gradient early, and the model is trapped before it can arrange a useful message.

Both predict everything measured above. **The separating experiment:** take a seed that fits at
`n_rounds=1`, freeze those weights, and evaluate at 2, 3 and 4 rounds without retraining. If the
fit survives more rounds, the fixed point is sound and the failure is optimisation — (b). If
adding rounds destroys the fit at fixed weights, the iteration itself is degenerate — (a). Under
(b) the remedies are annealing `λ_H`, warm-starting, or damping the update (Appendix B.1/B.2 of
Wu & Tu give step size and damping); under (a) the update schedule has to change.

## Not done, deliberately

No sweep, no hyperparameter tuning to improve perplexity, no change to the parameterisation of
`T` — the capacity question (0.27M non-embedding for PT against 1.23M for GPT, and the
full-rank versus factored `T`) is real but distinct, and changing it mid-diagnosis would destroy
the comparison. Evidence above says context is *present but crushed*, not absent, which is the
reading that argues against reaching for capacity first.

## Reproduce

```bash
./.venv/bin/python experiments/exp0_decoder_validation/diagnose_context.py
```

```bash
./.venv/bin/python experiments/exp0_decoder_validation/diagnose_content_stream.py
```

```bash
./.venv/bin/python experiments/exp0_decoder_validation/bisect_d.py
```
