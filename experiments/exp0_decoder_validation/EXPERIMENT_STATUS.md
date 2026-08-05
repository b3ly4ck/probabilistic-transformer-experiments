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

- §17.2 recommends MFVI as mainline; §23.3 in Part IV walks this back toward exact readout.
  Both must be read before the implementation is frozen. Resolution: —

## Reproduce

```bash
# command that runs the validation suite
```
