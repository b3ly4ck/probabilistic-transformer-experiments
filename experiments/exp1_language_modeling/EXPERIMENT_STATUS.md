# Experiment 1 — Language modelling, split into arms 1.1 and 1.2

| | |
|---|---|
| **Status** | flag implemented, validation passes in both arms. Arm 1.2 **runs under the MFVI readout**. Data and baseline in progress; no training runs yet. |
| **Priority** | mandatory — without it there is no paper |
| **Blocked by** | [Experiment 0](../exp0_decoder_validation/EXPERIMENT_STATUS.md) — **complete**, checks 1–9 pass at `ab7c082` |
| **Supersedes** | [`exp1_pt_vs_gpt/`](../exp1_pt_vs_gpt/EXPERIMENT_STATUS.md), kept for the record |
| **Last updated** | 2026-08-05 |
| **Plan reference** | [RESEARCH_PLAN.md](../../developer%20files/RESEARCH_PLAN.md) § Experiment 1 |

## Question

Two questions, one per arm, at otherwise identical settings.

| Arm | Model | Question |
|---|---|---|
| **1.1** | causal PT **without** `G_t` vs. GPT-style decoder | does the causal construction train, and what is the gap? |
| **1.2** | causal PT **with** `G_t` vs. GPT-style decoder | does the in-graph FFN analogue close any of that gap? |

The **1.1 vs 1.2 delta is itself the result** — an ablation of `G_t` at fixed everything else.

## Why the split exists

Wu & Tu report PT underperforming on large datasets and *suspect* the absence of a feed-forward
structure. They propose global variables (Appendix B.3) as an in-graph substitute — and never
test them. Appendix B.3 is a derivation with no experimental results; the main text mentions
them once, hedged ("may have similar functionality to the feed-forward structure").

So whether `G_t` closes the gap is open, not settled. Baking `G_t` into Experiment 1 would
assume the answer. This supersedes the line in `RESEARCH_PLAN.md` treating the B.3 globals as a
prerequisite: **they are a measured variable, not an assumption.**

Either outcome is publishable:

- `G_t` helps → the original authors' conjecture confirmed empirically for the first time.
- `G_t` does not help → the PT capacity gap is *not* explained by the missing FFN, contradicting
  the paper's stated suspicion. The more interesting result.

## Source check — done before implementation

The task specification asked to confirm whether §22.2's "single-split form" means B.3.1
(all-dep) or B.3.2 (dep-split). **It means neither.** The Wu & Tu appendix has a third
subsection the specification does not list:

> **B.3.3 Single-split.** "…similar to the *dep-split* model but only allows **one global head
> for each word**. Denote `G_i` as the global head variable for `i`-th word with a label set of
> size `m`. We define a binary potential for `Z_i` and `G_i`:
> `φ_b(G_i = k, Z_i = a) = exp(B_{k,a})` (46), where `B ∈ R^{m×d}` is a score matrix."

That is exactly what §22.2 names, and exactly what the specification describes — one `G_t` per
slot, one score matrix of shape `(m, d)`. The distinction that matters:

| | Variable | Score matrix | Head domain |
|---|---|---|---|
| B.3.1 all-dep | global features `F_j` are **observed**; head domain widens | `T''^(c) ∈ R^{d×d}`, ternary over `(Z_i, H_i, F_j)` | `{1..n+m}` |
| B.3.2 dep-split | `G^(c)_i` — one global head **per channel** | `B^(c) ∈ R^{m×d}`, one per channel | `{1..m}` |
| **B.3.3 single-split** | `G_i` — **one** global head per word, channel-independent | `B ∈ R^{m×d}`, **shared across channels** | `{1..m}` |

**Chosen: B.3.3 single-split.** It is what §22.2 names, it is the cheapest of the three, and its
parameter count `m·d` is independent of `h`.

§22.2 verbatim, for the record: *"Add to each slot's conditional a global-head variable `G_t`
(single-split form) with the binary factor `exp(B'_{m,Z_t})`. The slot is one more leaf off
`Z_t`; `μ_t` gains the multiplicative term `Σ_m e^{B'_{m,a}}`; causality holds (position-local);
the mean-field update gains exactly the GFU term `σ(qB'^⊤)B'` that the paper shows is the
feed-forward analogue; cost `O(md)` per position."*

The MFVI updates in the specification also check out against the source: they are B.3.2's
equations (40), (41) and (44) with the channel index dropped, which is what single-split means.

## Finding that changes the design — read before running arm 1.2

**Under the exact readout, `G_t` is context-free and cannot act as a feed-forward layer.**

`G_t` is a leaf attached only to `Z_t`, so `G_t ⊥ everything | Z_t`. Summing it out analytically
gives

```
log μ_t(a)  =  Σ_c LSE_{j ∈ D_t} B^(c)[j,a]   +   LSE_k B'[k,a]
                └── depends on the prefix ──┘       └── constant in t ──┘
```

The second term is **the same at every position and for every sentence**. So in the exact path,
`m·d` new parameters collapse to `d` effective numbers, and those `d` numbers reweight the label
prior without carrying any context. That is not an FFN analogue; it is a learned label bias.

Under the MFVI readout the same variable behaves completely differently: the composed update
`σ(Q_Z B'^⊤) B'` is nonlinear in `Q_Z`, and `Q_Z` is context-dependent — which is precisely the
GFU operator the paper compares to a transformer FFN in its Figure 9.

The asymmetry is not a bug in either path. It is what mean-field iteration buys: coupling that
exact marginalisation of a leaf removes. It matches the Experiment 0 finding that the exact
readout is nearly flat on the note's own §5 example — the same mechanism, seen twice.

**Consequence for this experiment.** §23.3 made the exact readout the mainline. Arm 1.2 run
*only* under the exact readout would test a `G_t` that structurally cannot do the job, and a
null result would say nothing about the paper's conjecture. Therefore **arm 1.2 runs both
readouts**, and the write-up states which one carries the claim:

**Decision (2026-08-05): arm 1.2 runs under the MFVI readout, not the exact one.** Arm 1.1
runs under both, so the 1.1 vs 1.2 delta is taken within the MFVI readout, never across
readouts.

**The degeneracy under the exact readout is a result, not a gap in coverage.** It is not a bug
to be fixed and not a hyperparameter to be tuned: `G_t` is a leaf on `Z_t`, so exact inference
integrates it out and what survives cannot depend on position. The measurements confirm the
argument rather than merely illustrating it — `λ_G` at 5, 20 and 100 give a final loss identical
to four decimals, which is what "no hyperparameter reaches this" looks like in numbers.

That belongs in the write-up as a statement about the construction: *the in-graph feed-forward
analogue proposed in Appendix B.3 exists only under mean-field inference; exact inference on the
same graph removes it.* Since §23.3 makes the exact readout the mainline, the two
recommendations are in tension, and this experiment is where that shows up.

This is not licence to start Experiment 3 — the comparison here is `G_t` on/off within one
readout, not exact vs. MFVI as a scientific object.

## Design

**One codebase, one flag.** `G_t` is a config flag, never a fork:

```python
use_global_head: bool = False
n_global: int = m          # only read when use_global_head
lambda_G: float = 1.0      # see open questions
```

Both arms execute the same code path in `src/pt_decoder.py`. No second model class, no second
training script, no duplicated update function. If the arms can drift independently the
comparison is void.

**What is added.** One parameter `B'` of shape `(m, d)` — a factor between `G_t` and `Z_t`,
trained by backprop like `S`, `T`, `r`, `b`. The per-slot energy gains exactly one term,
`+ B'[G_t, Z_t]`. Nothing else changes. The slot stays a star centred on `Z_t`, so it stays a
tree and exact inference remains available.

**MFVI path** — two equations change:

```
Q_G(k) <- softmax_k ( (1/λ_G) Σ_a Q_Z(a) B'[k,a] )
Q_Z(a) <- softmax_a ( (1/λ_Z) [ ...existing terms... + Σ_k Q_G(k) B'[k,a] ] )
```

**Exact path** — `log μ_t(a) += LSE_k B'[k,a]`, no `Q_G` and no iteration.

**Correctness anchor.** Composing the two MFVI updates must reproduce `σ(Q_Z B'^⊤) B'`, the GFU
operator of Appendix B Eq. 62. Assert this directly against a hand-written GFU.

### Baseline alignment — the two families must score the same token set

The PT decoder scores position `t` on predicting `w_t` from `D_t = {ROOT, 0, …, t−1}`. A standard
GPT scores position `t` on predicting `w_{t+1}` from `w_{≤t}`. Left alone, the two compute
perplexity over **different token sets**, and a results table comparing them is meaningless.

The baseline is therefore given a **learned BOS vector, prepended internally, with the last
position dropped**, so both expose `logits(tokens) -> (B, n, |V|)` where entry `t` predicts
`tokens[t]` from `tokens[:t]`. This is the direct analogue of PT's root key `r`: in both models
position 0 is predicted from a learned constant and no context at all. Asserted in
`tests/test_11_data_and_baseline.py` — causality under the shared convention, and that position 0
is invariant to the input.

**Embeddings are tied in the baseline.** In PT tying is forced by the construction (§16(b));
untying the baseline would hand it free parameters PT cannot have. Asserted by test.

**Forbidden, and asserted against in tests.** Expressivity is bought by adding variables and
factors, never maps: no MLP from `Q_Z` to logits, no context-dependent `S`, no nonlinearity on
`q̄` between steps outside the update equations, no per-iteration untied parameters. `B'` is
legitimate because it parameterises a factor between two variables.

## Validation to re-run before any training run

The energy changed, so the Experiment 0 checks that test the equations are not valid as they
stand. **Both flag states must pass the full suite; a check that passes in only one arm is a
bug.**

| Check | With `use_global_head=True` | Status |
|---|---|---|
| 1 shapes | parameter set becomes `{S, T, r, b, B_global}`, asserted exactly; `B_global` is `(m, d)` | ☑ |
| 2 normalisation | `Q_G` sums to 1 over `{1..m}` | ☑ |
| 3 causality | bitwise, CPU — `G_t` reads `q` at `t` only, never the prefix | ☑ |
| 4 no anti-causal path | unchanged in form, re-run | ☑ |
| 5 tying | still no `(d, |V|)` parameter anywhere | ☑ |
| 6 overfit | passes under MFVI (2400 steps, `λ_G = 5`); under the exact readout the degeneracy below is asserted instead | ☑ |
| 7 worked example | §5 has no `G_t`; unchanged with the flag off, bitwise | ☑ |
| 8 free energy | non-increasing; the `Q_G` update is an exact **argmin** under perturbation | ☑ |
| 9 exact vs. brute force | enumeration gains an explicit loop over `k`; agreement holds, so the slot is still a tree | ☑ |
| new | composed MFVI update equals `σ(qB'^⊤)B'`, and is bitwise the composition of the two updates | ☑ |
| new | `LSE_k B'[k,a]` identical at every position and every sentence | ☑ |
| new | the GFU term is genuinely nonlinear in `q` | ☑ |
| new | `B_global` receives gradient under both readouts | ☑ |
| new | flag off leaves no `B_global` parameter and reproduces arm 1.1 | ☑ |

Suite: 67 tests, all passing.

### Measured consequence of the degeneracy — recorded as a result

**With `G_t` attached, the exact readout cannot memorise a single sequence.** Measured 0.4536
against 0.0021 for the identical model without it. It is not a step budget (1200 and 2400 steps
give 0.4537 and 0.4535) and it is not `λ_G`: at 5, 20 and 100 the final loss is identical *to
four decimals* and `‖B'‖` lands on 21.9 in all three.

Diagnosis: `Q_G` converges near-uniform — `max Q_G ≈ 0.20` at `m = 5` — so `σ(qB'^⊤)B'`
degenerates to a near-constant vector added to every slot's label logits, which is exactly why
`λ_G` stops mattering. That constant compresses the spread of `q̄` across positions
(0.0535 → 0.046), and the exact readout depends on that spread because it pools by LSE with no
query. The mean-field readout, which has a query, survives it.

This is the *second* mechanism pushing the same way as the context-free finding above: under the
exact readout the global head contributes nothing that varies with position, and here it
actively costs. Asserted directly as a model property in
`tests/test_06_overfit.py::test_global_head_is_degenerate_under_the_exact_readout`, with the
measured gap, so the property cannot silently change.

**Not investigated further, by decision.** The mechanism is understood from the construction and
is not hyperparameter-reachable; digging into why `Q_G` converges near-uniform would not change
what arm 1.2 does.

### Second finding — `λ_G` at 1 collapses the model entirely

At the config default `λ_G = 1` the global head prevents memorisation under **both** readouts:
`‖B'‖` reaches 10–20 within 300 Adam steps and the loss stops near 0.48 where the same model
without the head reaches 0.005. Raising `λ_G` to 5 fixes the mean-field arm (needs 2400 steps
rather than 1200); it does not fix the exact arm, for the separate reason above.

The source does not pin `λ_G`. Wu & Tu's calibration for `λ_H = 1/d` is about the message being
a sum over `d` labels; the message to `G` has the same form, so the principled analogue is
**`1/d`, not `1/m`** — which would be sharper still, and worse at toy scale. At realistic `d`
the calibration may matter more and the collapse less. Every run must record the value used.

## Configuration

Held identical across every run. Any difference other than the one being tested invalidates the
comparison.

| Item | Setting |
|---|---|
| Dataset | **PTB** (Mikolov preprocessed, word level) — see decisions |
| Tokenizer / vocabulary | identical across all models |
| Context length | identical |
| Optimizer | Adam, identical hyperparameters and schedule |
| Training loop | one implementation shared by PT and both baselines, commit: |
| Seeds | multiple; report mean ± std |
| Metric | perplexity on the held-out split |
| `m` (`n_global`) | |
| `λ_G` | |
| Hardware | Slurm GPU partition, to be recorded per run |

**Budget matching.** `G_t` adds `m·d` parameters, so arm 1.2 may **not** reuse arm 1.1's GPT
baseline configuration. Either recompute the matched GPT baseline separately per arm, or hold
total parameters fixed across arms by adjusting `d`.

**Measured budgets on PTB (`|V| = 10,000`), computed 2026-08-05.** The table below is why the
convention has to be chosen deliberately rather than defaulted:

| Model | total | embedding | non-embedding |
|---|---:|---:|---:|
| PT `d=512`, `h=4` | 6.2M | 5.1M (83%) | 1.1M |
| PT `d=1024`, `h=4` | 14.4M | 10.2M (71%) | 4.2M |
| PT `d=2048`, `h=4` | 37.3M | 20.5M (55%) | 16.8M |
| GPT `d=256`, `L=6` | 7.3M | 2.6M (35%) | 4.8M |
| GPT `d=512`, `L=6` | 24.1M | 5.1M (21%) | 19.0M |
| GPT `d=768`, `L=6` | 50.3M | 7.7M (15%) | 42.6M |

Against `GPT d=512, L=6` (24.1M), matching on **total** puts PT at `d ≈ 1500`; matching on
**non-embedding** puts it at `d ≈ 2180`. Either way the label set runs to well over a thousand
values.

> **Convention chosen: neither — sweep `d` and report the curve.** The two conventions disagree
> (`d ≈ 1500` vs `d ≈ 2180`), and picking one is arbitrary in a way that invites *"did you tune
> this?"*. Instead `d` is swept over at least three values (256, 512, 1024) and the GPT baseline
> is marked on the curve under **both** conventions. Every table reports total **and**
> non-embedding parameters for every model.
>
> This is what §22.3 already asks for — *"report the trade curve, not a single point"* — and it
> answers the interpretability objection head on: rather than sitting silently at `d ≈ 2000`
> while claiming a small interpretable label set, the curve shows where that claim holds and
> where it breaks.
>
> **The 20–50M figure is provisional.** It came from Penghao before the `|V|·d` coupling was
> understood; the sweep shows what is actually reachable. It is not a constraint on this
> experiment.

**Why the sweep, concretely: the 20–50M budget forces `d` into the thousands, and §22.2 names
that as a cost.**
"Raise `d`… this is the default lever; its real price is the one Defect 1(c) names: a `d` in the
thousands erodes *small interpretable label set*." At a matched 24M on PTB the erosion is not
hypothetical — `d ≈ 1500–2200` is a hidden dimension, not a syntactic label set, and the
interpretability half of the paper's framing weakens accordingly. Options, none of them free:
run at a smaller matched budget where `d` stays in the hundreds; keep the budget and drop the
interpretability claim to a measured trade curve; or buy non-embedding parameters through `h`
rather than `d`. The sweep subsumes the first two. On the third, see the correction below.

### Correction — how `T` is parameterised, and what `h` actually buys

An earlier note here claimed raising `h` buys non-embedding parameters at `h·d²`. **That is true
of the current implementation and false of the parameterisation the paper prices.**

Verified in the code: `src/pt_decoder.py` has `self.T = nn.Parameter(randn(cfg.n_channels,
cfg.d, cfg.d))` — a full `d × d` matrix per channel, hence `h·d²`.

Verified against the source: §22.2 states *"decomposed `T` is **linear in `d`**"* and *"attention
flops are governed by `r·h`"*; §18 Check 4 writes the decomposition explicitly, `T^(c) =
U^(c)V^(c)⊤`, with the logits becoming `F_c(j) = (Q_Z U^(c))(q̄_j V^(c))⊤` — queries and keys of
rank `r`. Under that form the count is `2·h·d·r`, not `h·d²`.

The gap is not small. At `d = 2048`, `h = 4`: full-rank `T` is 16.8M parameters; factored at
`r = 64` it is 1.05M — a factor of 16. Every budget-matching number computed so far assumes the
full-rank form.

Consequences, not yet acted on:

- The full-rank form is a strict superset in expressivity, so it is not *wrong* — but it is not
  what the paper costs, and a reviewer comparing parameter counts against Wu & Tu will be
  reading the factored numbers.
- Factoring changes which lever is cheap. Under `2·h·d·r`, buying non-embedding capacity through
  `h` or `r` is far cheaper than through `d`, which weakens the argument that the budget forces
  `d` into the thousands.
- **Decision required before the full sweep:** implement `T = U V⊤` with a rank parameter, or
  keep full-rank and state the deviation in the write-up. The calibration run uses the current
  full-rank form; it does not depend on this.

Report **both** parameter count and wall-clock / FLOPs for every run — PT shares parameters
across iterations, so equal parameter count does not imply equal compute. Report the
embedding / non-embedding split separately: with tied embeddings PT's budget sits almost
entirely in `S`.

## Run log

Never delete a row. Failed and abandoned runs stay, with the reason.

| Run | Date | Commit | Arm | Readout | Model | `m` | Seed | Params (total / emb / non-emb) | Val PPL | Wall-clock | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 939123 | 2026-08-06 | `74d4f71` | — | — | all four | — | 0 | — | — | 25 min | **cancelled.** Single job for four models would have hit the 4 h limit on PT-exact alone, and the JSON is written only at the end, so it would have taken the GPT row down with it |
| 939148 | 2026-08-06 | `74d4f71` | 1.1 | — | GPT `d=160 L=4` | — | 0 | 2.85M / 1.62M / 1.23M | **130.63** | 37 s | 2000 steps, curve flat by 1500. test 118.56, peak 0.68 GiB |
| 939149 | 2026-08-06 | `74d4f71` | 1.1 | MFVI | PT `d=256 h=4` | — | 0 | 2.83M / 2.56M / 0.27M | 1450.59 | 43 s | 2000 steps, **undertrained** — still falling ~10 % per eval. Superseded by 939156 |
| 939150 | 2026-08-06 | `74d4f71` | 1.2 | MFVI | PT + `G_t` | 64 | 0 | 2.85M / 2.56M / 0.29M | 1442.70 | 48 s | as above. Superseded by 939157 |
| 939151 | 2026-08-06 | `74d4f71` | 1.1 | exact | PT `d=256 h=4` | — | 0 | — | — | 3 s | **failed** — bad GPU on the node, `nvidia-smi` errored and `set -e` killed the job. GPU check made non-fatal |
| 939155 | 2026-08-06 | `cdd75e2` | 1.1 | exact | PT `d=256 h=4` | — | 0 | 2.83M / 2.56M / 0.27M | 7004.89 | 549 s | only **300 steps** — 1.83 s/step, **78× slower than MFVI**. Peak 4.28 GiB vs 0.70. Nowhere near converged |
| 939158 | 2026-08-06 | `cdd75e2` | 1.1 | — | GPT `d=160 L=4` | — | 0 | 2.85M / 1.62M / 1.23M | — | 16 min, cancelled | **ran on CPU without saying so.** Landed on `ai_gpu32` after its GPU failed; `torch.cuda.is_available()` returned False and the runner fell through to CPU. Making the `nvidia-smi` check non-fatal (after 939151) removed the only thing that caught this. Runner now aborts unless `--allow-cpu` is passed |
| 939170 | 2026-08-06 | `9c7ca4e` | 1.1 | — | GPT `d=160 L=4` | — | 0 | 2.85M / 1.62M / 1.23M | pending | — | resubmission of 939158 with `--exclude=ai_gpu32` |
| 939156 | 2026-08-06 | `cdd75e2` | 1.1 | MFVI | PT `d=256 h=4` | — | 0 | 2.83M / 2.56M / 0.27M | **664.19** | 471 s | 20 000 steps, **converged** — flat from step 10 000. test 612.14 |
| 939157 | 2026-08-06 | `cdd75e2` | 1.2 | MFVI | PT + `G_t` | 64 | 0 | 2.85M / 2.56M / 0.29M | **678.40** | 483 s | 20 000 steps, converged. test 621.29. **Worse than without `G_t`** |

## Results

### Calibration, 2026-08-06 — PT converges to the unigram baseline

**PTB unigram baseline: val ppl 687.0.** Measured on the same split, for reference.

| Model | Val PPL | Test PPL | vs unigram |
|---|---:|---:|---|
| GPT `d=160 L=4`, 2 000 steps | 130.63 | 118.56 | **5.3× better** |
| PT MFVI, no `G_t`, 20 000 steps | 664.19 | 612.14 | 3 % better |
| PT MFVI, with `G_t`, 20 000 steps | 678.40 | 621.29 | 1 % better |
| PT exact, 300 steps | 7004.89 | 6976.89 | not converged |

**This is a return to Experiment 0, not a result to report.** The research plan says so
explicitly: a catastrophic gap is an implementation problem. PT converged — flat from step
10 000, three evaluations within 1 ppl of each other — to a number indistinguishable from
predicting word frequencies and ignoring context entirely. Generated samples agree: function
words only, no content words, no local structure.

The machinery is not broken in the ways already tested: checks 1–9 pass, the §5 worked example
reproduces digit for digit, exact and brute-force agree to 7.5e-08, and PT memorises a single
sequence at toy scale. Something between "correct on four words" and "learns nothing on PTB"
is unaccounted for. Candidates not yet separated: `λ_H = 1/d` at `d = 256` (checked at
initialisation — attention is *not* saturated, entropy 3.1 of 4.16 nats — but not checked after
training); `n_rounds = 3` too few; the learning rate, shared with GPT, being wrong for a model
whose parameters are all in one tied matrix; or the rank-`d` readout binding harder than
expected at `|V| = 10 000`.

**`G_t` makes it worse**, 678.40 against 664.19 — 14 ppl in the wrong direction, well outside
the 1-ppl spread of the converged plateau. Not yet meaningful, since the base model is not
learning; recorded so the sign is on file.

**Device audit.** Every run in the table above was checked against the `device` field its
runner logs. All six completed runs used CUDA on a TITAN RTX; only 939158, cancelled, fell back
to CPU. The timings and the 78× figure below stand.

### Measured cost of the exact readout

| Readout | ms/step | Peak memory | Slowdown |
|---|---:|---:|---|
| MFVI | 23.6 | 0.70 GiB | — |
| exact (chunked, 250) | 1832 | 4.28 GiB | **78×** |

§23.3 calls this "the same FLOPs with worse hardware constants". Measured, the constant is 78,
and it is memory traffic rather than arithmetic: the readout materialises `(B, n, |V|, d)` and
runs elementwise exponentials over it, where the matmul form materialises `(B, n, |V|)` and
uses tensor cores. A 20 000-step exact run is ~10 h against 8 min for MFVI. The fused
cross-entropy §23.3 asks for is not optional at sweep scale.

PT under MFVI costs 23.6 ms/step against GPT's 18.5 — only 1.3×, so the construction itself is
not the expensive part.

| Model | Arm | Readout | Params (total / emb / non-emb) | FLOPs | Val PPL (mean ± std) | Seeds |
|---|---|---|---|---|---|---|
| GPT | — | — | | | | |
| Causal PT, no `G_t` | 1.1 | exact | | | | |
| Causal PT, no `G_t` | 1.1 | MFVI | | | | |
| Causal PT, with `G_t` | 1.2 | exact | | | | |
| Causal PT, with `G_t` | 1.2 | MFVI | | | | |

**The `1.1` vs `1.2` delta**, per readout, with the seed noise it must clear:

| Readout | PPL(1.1) | PPL(1.2) | Δ | Seed std | Larger than noise? |
|---|---|---|---|---|---|
| MFVI (carries the claim) | | | | | |
| exact | | | | | |

## Success criterion

**Arm 1.1.** PT converges stably and lands within a modest gap of GPT. A small gap is a good
result given the rank-`d` bottleneck. A catastrophic gap is an implementation problem, not a
finding — that is a return to Experiment 0, not a result to report.

**Arm 1.2.** No "winning" outcome is required. What is required is that the delta is reported
against seed noise and interpreted honestly:

- Δ larger than seed noise and favourable → Wu & Tu's conjecture confirmed empirically.
- Δ within seed noise → the capacity gap is not explained by the missing feed-forward
  structure. Contradicts the paper's stated suspicion, and is the stronger result.

**Falsification.** If arm 1.2 shows no effect under the *MFVI* readout — where `G_t` genuinely
is the GFU operator — then the FFN-substitute claim of Appendix B.3 fails on this data, and the
write-up says so.

## Decisions and justifications

| Decision | Choice | Why |
|---|---|---|
| Which B.3 construction | **B.3.3 single-split** | what §22.2 names; one `G_t` per slot, `B' ∈ R^{m×d}` shared across channels; cost independent of `h` |
| `G_t` as flag, not fork | `use_global_head` | arms that can drift independently make the ablation worthless |
| Arm 1.2 runs both readouts | yes | under the exact readout `G_t` is context-free and cannot be an FFN analogue; the MFVI arm carries the claim |
| Deltas taken within a readout | yes | never across readouts — that would confound Experiment 3 into this one |

## Open questions

- **`λ_G` is not pinned by the source.** Wu & Tu's Eq. (44) writes `Q''_{ic}(k) ∝ exp(H_{i,k,c})`
  with no explicit temperature, and §22.2 does not name one either. `λ_Z = 1` and `λ_H = 1/d`
  are stated; the analogous scaling for a head over `m` values would be `1/m` by the same
  variance argument the paper uses for `λ_H`. Default to `1.0`, expose it as config, and record
  which was used. Resolution: —
- **Value of `m`.** Not specified anywhere. It trades against `d` under the fixed-budget
  convention. Resolution: —
- ~~Why does `Q_G` converge near-uniform?~~ **Closed 2026-08-05, not investigated.** The
  degeneracy under the exact readout follows from `G_t` being a leaf and is not reachable by
  hyperparameters; arm 1.2 moves to the MFVI readout instead. Whether `Q_G` stays near-uniform
  *under MFVI at corpus scale* is a different question and is answered by the arm itself, since
  a uniform `Q_G` there would show up as a null delta.
- **Dataset: PTB or WikiText-2.** Resolution: —
- **Budget-matching convention.** Resolution: —

## Reproduce

```bash
./.venv/bin/python -m pytest tests
```
