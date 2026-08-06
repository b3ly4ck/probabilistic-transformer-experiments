# Experiment 1 — Language modelling, split into arms 1.1 and 1.2

| | |
|---|---|
| **Status** | specified, not started |
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

| Arm | Readout | What it measures |
|---|---|---|
| 1.2-exact | exact | the label-prior effect of `G_t` — expected small, reported for completeness |
| 1.2-mfvi | MFVI | the GFU / FFN analogue — **this is the arm that answers the question** |

Arm 1.1 also runs both, so the 1.1 vs 1.2 delta is taken within a readout, never across.

This is not licence to start Experiment 3 — the comparison here is `G_t` on/off within a
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
| 1 shapes | parameter set becomes `{S, T, r, b, B_global}`; check 1 updated to expect exactly that | ☐ |
| 2 normalisation | `Q_G` sums to 1 over `{1..m}` | ☐ |
| 3 causality | `G_t` is position-local and must open no path to the future — bitwise, CPU | ☐ |
| 4 no anti-causal path | unchanged in form, re-run | ☐ |
| 5 tying | still **no** `(d, |V|)` parameter anywhere | ☐ |
| 6 overfit | re-run both flag states | ☐ |
| 7 worked example | §5 has no `G_t`; must be unchanged with the flag **off**, and is not a reference with it on | ☐ |
| 8 free energy | re-run — the `Q_G` update must be an exact **argmin**, not a descent step | ☐ |
| 9 exact vs. brute force | re-run with `G_t` included in the enumeration; agreement confirms the slot is still a tree | ☐ |
| new | composed MFVI update equals the GFU operator `σ(qB'^⊤)B'` | ☐ |
| new | with the flag on, `LSE_k B'[k,a]` is identical at every position — the context-free property above, asserted so it cannot be misread later | ☐ |

## Configuration

Held identical across every run. Any difference other than the one being tested invalidates the
comparison.

| Item | Setting |
|---|---|
| Dataset | PTB or WikiText-2 — **not** WikiText-103 |
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

> Convention chosen for this experiment: **—** (state it here before the first run; leaving it
> unstated is what a reviewer attacks first.)

Report **both** parameter count and wall-clock / FLOPs for every run — PT shares parameters
across iterations, so equal parameter count does not imply equal compute. Report the
embedding / non-embedding split separately: with tied embeddings PT's budget sits almost
entirely in `S`.

## Run log

Never delete a row. Failed and abandoned runs stay, with the reason.

| Run | Date | Commit | Arm | Readout | Model | `m` | Seed | Params (total / emb / non-emb) | Val PPL | Wall-clock | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |

## Results

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
- **Dataset: PTB or WikiText-2.** Resolution: —
- **Budget-matching convention.** Resolution: —

## Reproduce

```bash
./.venv/bin/python -m pytest tests
```
