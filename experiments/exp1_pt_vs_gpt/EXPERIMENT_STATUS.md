# Experiment 1 — Causal PT vs. GPT-style decoder

| | |
|---|---|
| **Status** | not started |
| **Priority** | mandatory — without it there is no paper |
| **Blocked by** | [Experiment 0](../exp0_decoder_validation/EXPERIMENT_STATUS.md) |
| **Last updated** | 2026-08-05 |
| **Plan reference** | [RESEARCH_PLAN.md](../../developer%20files/RESEARCH_PLAN.md) § Experiment 1 |

## Question

Does the causal PT train on real data, and does its perplexity land in a reasonable corridor
relative to a standard decoder?

This is **not** a controlled comparison — everything differs between the two models. It
measures the total cost of the construction, not the effect of any one component. The
controlled comparison is Experiment 2.

## Success criterion

PT converges stably and lands within a modest gap of GPT. A small gap is a good result given
the rank-`d` bottleneck. A catastrophic gap is an implementation problem, not a finding —
treat it as a return to Experiment 0, not as a result to report.

## Design

| | Causal PT | GPT baseline |
|---|---|---|
| Source | written from scratch | nanoGPT, off the shelf |
| Weight sharing | yes (across MFVI iterations) | no |
| Syntactic structure | yes | no |

## Configuration

Held identical across both models. Any difference other than the model itself invalidates
the comparison.

| Item | Setting |
|---|---|
| Dataset | PTB / WikiText-2 — **not** WikiText-103 |
| Tokenizer / vocabulary | |
| Context length | |
| Total parameters | target 20–50M, matched |
| Embedding parameters | *report separately* |
| Non-embedding parameters | *report separately* |
| Matching basis chosen | total / non-embedding / both — **state explicitly** |
| Optimizer | Adam, identical hyperparameters and schedule |
| Seeds | |
| Hardware | |
| Training loop | shared implementation, commit: |

**Why the parameter split is reported.** With tied embeddings, PT's budget sits almost
entirely in the `|V| × d` word–label matrix `S`; `T^(c)`, `b` and `r` are negligible. GPT
spends a large share in its blocks. A single total-parameter figure hides this, and it is the
first thing a reviewer will attack.

## Run log

Never delete a row. A run that failed or was abandoned is recorded with the reason.

| Run | Date | Commit | Model | Seed | Config | Val PPL | Wall-clock | Notes |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

## Results

| Model | Params (total / emb / non-emb) | Val PPL (mean ± std) | Seeds | FLOPs | Wall-clock |
|---|---|---|---|---|---|
| GPT | | | | | |
| Causal PT | | | | | |

*(training curves referenced here)*

## Decisions and justifications

*(dataset choice, tokenizer, how budgets were matched and why that basis, any hyperparameter
that was tuned and on what — state whether it was tuned per-model or shared)*

## Open questions

## Reproduce

```bash
# commands for both models
```
