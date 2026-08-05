# Experiment 3 — Exact readout vs. MFVI readout

| | |
|---|---|
| **Status** | not started |
| **Priority** | variant (a) survives most cuts; variant (b) is the first thing to drop |
| **Blocked by** | [Experiment 1](../exp1_pt_vs_gpt/EXPERIMENT_STATUS.md) — (a) needs only a trained PT |
| **Last updated** | 2026-08-05 |
| **Plan reference** | [RESEARCH_PLAN.md](../../developer%20files/RESEARCH_PLAN.md) § Experiment 3 |

## Question

What does the mean-field approximation cost?

## Why this experiment is available at all

The per-slot graph is a star centred on `Z_t`, hence a tree, so sum-product gives a
closed-form exact readout — a mixture of softmaxes — as an alternative to iterating MFVI. A
standard transformer offers no exact/approximate choice; this one is created by the tree
structure. Prof. Tu independently raised the comparison as worth doing empirically.

The exact readout function itself is **not written here** — it already exists as the oracle
for check 9 of [Experiment 0](../exp0_decoder_validation/EXPERIMENT_STATUS.md). This
experiment reuses it as a scientific object.

## Two variants

| | Variant (a) — evaluation-time swap | Variant (b) — train with each |
|---|---|---|
| Procedure | train with MFVI readout, substitute exact readout at eval on the *same* parameters | train one model with MFVI readout, one with exact |
| Measures | approximation error of the readout alone | end-to-end effect, including how the approximation shapes learning |
| Cost | nearly free | a full training run |
| Order | do first | only if time allows |

Both variants must be stated explicitly in the write-up — they answer different questions and
conflating them is a real risk when reporting a single "exact vs. MFVI" number.

## Success criterion

**There is no winning outcome here — either result is informative.** A small gap says
mean-field is adequate and cheap. A large gap says the tree structure should be exploited, and
that is a finding in its own right. Report whichever occurs.

## Configuration

| Item | Setting |
|---|---|
| Trained checkpoint used for (a) | run id / commit: |
| MFVI iterations `τ` at eval | |
| Exact readout implementation | from Experiment 0, commit: |
| Dataset / split | same held-out set as Experiment 1 |
| Seeds | |

## Run log

| Run | Date | Commit | Variant | Readout | Seed | Val PPL | Wall-clock | Notes |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

## Results

| Variant | Readout | Val PPL (mean ± std) | Δ vs. MFVI | Cost per token |
|---|---|---|---|---|
| (a) eval swap | MFVI | | — | |
| (a) eval swap | exact | | | |
| (b) trained | MFVI | | — | |
| (b) trained | exact | | | |

## The rank-`d` bottleneck angle

The MFVI readout is rank-`d` limited: logits `b_w + Σ_a Q_Z(a) S_{w,a}` are affine in `Q_Z`,
so across contexts the logit matrix has rank at most `d+1` — a softmax bottleneck. A mixture
of softmaxes is **not** rank-limited in the same way. Whether that shows up empirically is
worth checking here, and it is the most interesting thing this experiment could find.

Observed: —

## Decisions and justifications

## Open questions

## Reproduce

```bash
# eval-time swap, then the trained variants
```
