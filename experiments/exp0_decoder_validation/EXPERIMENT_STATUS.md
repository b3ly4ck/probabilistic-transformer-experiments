# Experiment 0 — Implement and validate the causal PT decoder

| | |
|---|---|
| **Status** | not started |
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
| `d` (label dim) | | plan says 8–16 |
| Vocabulary | | ~20 tokens |
| Sequence length | | 4–8 |
| Batch size | | 1–4 |
| MFVI iterations `τ` | | ≥2, so the attention query is context-dependent |
| Channels `h` | | start at 1 |
| Device | CPU | check 3 requires bitwise reproducibility |
| Seed | | fixed |

## Validation checks

Record the commit at which each check first passed, and never delete a row — a check that
passed and later broke is the most valuable line in this file.

| # | Check | Status | Commit | Date | Notes |
|---|---|---|---|---|---|
| 1 | Shapes of every intermediate tensor match Part III | ☐ | | | |
| 2 | All posteriors normalise to 1 along their variable axis | ☐ | | | |
| 3 | Causality: changing token `t` leaves logits at `< t` bitwise unchanged (CPU, fixed seed) | ☐ | | | |
| 4 | Prefix frozen: prefix leaves attached with `requires_grad=True`, `.grad` is None/zero after backward from step `t` | ☐ | | | |
| 5 | Tying: input and output word matrices are the *same tensor object* | ☐ | | | |
| 6 | Overfit a single batch to ~0 loss | ☐ | | | |
| 7 | Worked example reproduces `causal_pt_output_note.pdf` §5 numbers | ☐ | | | |
| 8 | Mean-field free energy non-increasing across MFVI iterations | ☐ | | | |
| 9 | Exact sum-product readout agrees with brute-force enumeration | ☐ | | | |

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

Reproduced: ☐ — record any discrepancy and its cause below, do not silently adjust the target.

## Run log

| Run | Date | Commit | Config | What was run | Outcome |
|---|---|---|---|---|---|
| | | | | | |

## Results

*(filled as checks pass — free energy curves, overfit loss curve, printed tensors from the
worked example)*

## Decisions and justifications

*(every non-obvious implementation choice, with the reason and the section of the paper or the
note that forced it — especially anywhere the paper was ambiguous and a reading had to be
chosen)*

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

## Reproduce

```bash
# command that runs the validation suite
```
