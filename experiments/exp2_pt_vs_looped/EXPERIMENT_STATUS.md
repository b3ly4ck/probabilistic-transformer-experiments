# Experiment 2 — Causal PT vs. Looped Transformer ★ core result

| | |
|---|---|
| **Status** | not started |
| **Priority** | the scientific core — cutting it weakens the paper's central claim |
| **Blocked by** | [Experiment 1](../exp1_pt_vs_gpt/EXPERIMENT_STATUS.md) |
| **Last updated** | 2026-08-05 |
| **Plan reference** | [RESEARCH_PLAN.md](../../developer%20files/RESEARCH_PLAN.md) § Experiment 2 |

## Question

Does syntactic structure contribute **beyond weight sharing**?

PT differs from GPT in two ways at once — shared parameters across iterations, *and* a
syntactic factor graph. PT vs. GPT cannot separate them. The Looped Transformer has the
sharing without the structure, which makes the three-point comparison decisive:

| Model | Weight sharing | Syntactic structure |
|---|---|---|
| GPT | no | no |
| Looped Transformer | **yes** | no |
| Causal PT | **yes** | **yes** |

- Looped vs. GPT isolates weight sharing.
- **PT vs. Looped isolates structure** — this is the claim of the paper.

## Success criterion

PT is **no worse than Looped at matched budget**. Then the paper can state that syntactic
structure carries information not explained by weight sharing alone. Without this experiment,
a reviewer can attribute any gain to sharing and the central claim does not stand.

## Design

Looped baseline is nanoGPT with the stack of distinct blocks replaced by one shared block
applied `T` times:

```python
# standard: a stack of distinct blocks
for block in self.blocks:
    x = block(x)

# looped: one shared block, applied T times
for _ in range(self.T):
    x = self.block(x)
```

**Control on depth.** `T` in the Looped baseline matches the number of MFVI iterations `τ` in
PT, so effective depth is comparable alongside parameter count. Report a **small sweep over
`T`**, not a single value — one lucky `T` is exactly the objection this experiment exists to
prevent.

## Configuration

Inherits the shared setup from Experiment 1 unchanged — same data, tokenizer, context length,
optimizer, schedule, seeds and training loop. Only the model differs.

| Item | Setting |
|---|---|
| Shared setup inherited from | Experiment 1, commit: |
| PT MFVI iterations `τ` | |
| Looped `T` sweep | |
| Total parameters | matched |
| Embedding / non-embedding split | *report separately* |
| Seeds per configuration | |

## Run log

| Run | Date | Commit | Model | `T` / `τ` | Seed | Val PPL | Wall-clock | Notes |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

## Results

| Model | `T` / `τ` | Params (total / emb / non-emb) | Val PPL (mean ± std) | Seeds |
|---|---|---|---|---|
| GPT | — | | | |
| Looped | | | | |
| Causal PT | | | | |

**Derived comparisons** (fill once the table above is complete):

- Looped − GPT = effect of weight sharing:
- PT − Looped = **effect of structure**:

## Decisions and justifications

*(choice of `T` range, whether Looped got its own hyperparameter tuning, how depth was matched
and what was traded away to match it)*

## Open questions

## Reproduce

```bash
# commands across the T sweep
```
