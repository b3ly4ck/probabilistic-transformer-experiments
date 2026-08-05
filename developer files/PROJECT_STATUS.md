# Project Status

Technical state of the project, maintained so work can be resumed without re-deriving history
from `git log`. Per-experiment records live in `experiments/<exp>/EXPERIMENT_STATUS.md`; this
file covers what spans experiments.

**Overall state as of 2026-08-05: documentation and structure complete, no code written.**
The theoretical construction is finished and approved by Prof. Tu. Experiment 0 is the next
step and nothing downstream of it can start.

## Recent changes

| Date | Version | Change |
|---|---|---|
| 2026-08-05 | 0.2.2 | §17.1 / §23.3 read: exact readout becomes the mainline, the query stream disappears, the no_grad filtering design is retracted. |
| 2026-08-05 | 0.2.1 | `PROJECT_STATUS.md` created. Runtime environment surveyed — see below. |
| 2026-08-05 | 0.2.0 | Experiment station: four folders under `experiments/`, each with `EXPERIMENT_STATUS.md`, plus `experiments/README.md` with the logging rules. |
| 2026-08-05 | 0.1.3 | Experiment 0 validation gates strengthened: check 4 rewritten (the old form was vacuous), check 3 pinned to CPU, checks 8–9 added, parameter split required in reporting. |
| 2026-08-05 | 0.1.2 | `RESEARCH_PLAN.md` moved into `developer files/`. |
| 2026-08-05 | 0.1.1 | Documentation and reference papers consolidated into `developer files/`. |
| 2026-08-05 | 0.1.0 | `CLAUDE.md` rewritten for this project (the committed one belonged to `NeuroLady_Final`). |
| 2026-08-05 | — | Repository created; reference PDFs, `PROJECT.md` and `RESEARCH_PLAN.md` added by the author. |

## Repository structure

```
.
├── CLAUDE.md                 # working rules — must stay at root, auto-loaded
├── README.md
├── developer files/          # all documentation and reference papers
│   ├── PROJECT.md            # research context, framing, success criteria
│   ├── RESEARCH_PLAN.md      # the executable plan: Exp 0-3, gates, cut order
│   ├── PROJECT_STATUS.md     # this file
│   ├── VERSION
│   └── *.pdf                 # four reference papers
└── experiments/
    ├── README.md             # index + logging rules
    ├── exp0_decoder_validation/
    ├── exp1_pt_vs_gpt/
    ├── exp2_pt_vs_looped/
    └── exp3_exact_vs_mfvi/
```

`src/`, `tests/`, `notebooks/` and `data/` are declared in the layout but do not exist yet.

## Code

None written. The first code is the causal PT decoder forward pass for Experiment 0 — the
only component written from scratch in the whole project. nanoGPT is used off the shelf for the
GPT baseline, and the Looped baseline is nanoGPT with one shared block applied `T` times.

## Runtime environment

Surveyed 2026-08-05 on the login node. **Nothing is installed yet** — this is the first
blocker for Experiment 0.

| Item | State |
|---|---|
| System Python | 3.7.1 at `/usr/local/bin/python3` — no `torch`, no `numpy`, no `pytest` |
| Anaconda | `/public/software/anaconda3/bin/python`, Python 3.8.8 |
| Conda user dir | `~/.conda` exists (package cache from earlier projects) |
| PyPI | reachable (HTTP 200) — packages can be installed directly |
| System git | 1.8.3.1; **2.37.0 available** via the module system |
| Module system | modulefiles at `/public/software/modules/modulefiles/` (cuda, gcc, anaconda3, git, cmake, …). `module` is a shell function from the login profile and is **not available in non-interactive shells** — `MODULEPATH` is unset there |
| Scheduler | Slurm. GPU partitions include `ShangHAI` (A40), `hexm_l40` (L40), `critical` |
| Disk | `/public` — 994G free |

**Consequence for Experiment 0:** it is specified as CPU-runnable at toy scale, so it needs
only a Python environment with PyTorch (CPU build) and pytest — no GPU allocation and no Slurm
job. GPU and Slurm become relevant at Experiment 1.

**Environment decision — not yet made.** Conda env vs. venv, Python version, torch build
(CPU-only for Exp 0 vs. CUDA for Exp 1), and whether the env lives inside the repo (must be
gitignored — `.venv/` already is) or in `~/.conda/envs`. Record the choice here once made,
with the exact creation command, so it is reproducible.

## Cross-experiment decisions

| Decision | Status | Notes |
|---|---|---|
| Which readout is mainline | **decided 2026-08-05: exact** | §23.3 states it outright — exact readout mainline, mean-field two-stream as the ablation. The query stream disappears entirely (§24.1); the readout is a causal `logcumsumexp` scan, `O(ndh)`, fully parallel |
| Exact readout written in Exp 0 | decided | It is now the mainline *and* the oracle for check 9; Exp 3 reuses it rather than implementing anything new |
| Gradient through the content stream | **decided 2026-08-05: flows normally** | "Frozen prefix" is variational (`q̄_j` is not re-optimised at later slots), not stop-gradient. Causality comes from the triangular mask, as in a transformer. Running the filtering pass under `no_grad` would starve `S` and `T^(c)` |
| Appendix B.3 globals (`G_t`) from day one | **open** | §22.2: the graph-faithful answer to "PT lacks an FFN". Without it, Experiments 1–2 confound the causal construction with the known encoder-side capacity gap. `O(md)` per position, stays inside the graph |
| Dataset | open | PTB or WikiText-2. Not WikiText-103 — that is the regime where the original PT is reported to fail |
| Parameter matching basis | open | total / non-embedding / both at fixed vocabulary. Must be stated explicitly; with tied embeddings PT's budget is almost entirely the `\|V\| × d` matrix `S` |
| Shared training loop | decided | One implementation across PT, GPT and Looped. A loop change that helps one model and not the others voids the comparison |

## Known risks

- **Cost profile, not parallelism.** An earlier reading of this section claimed the model
  filters sequentially over positions like an RNN. That is **wrong**: §18 Check 4 states that
  the single-pass layer-parallel schedule survives, since query slot `t` and content position
  `t` both need only content `j < t`, so all `2n` inference problems per layer run in one pass
  under one triangular mask. Positions are parallel; iterations are sequential, exactly as
  layers are in a transformer. What remains is a constant-factor cost question, and §24.2 puts
  PT at roughly 0.4–0.6× a transformer layer (no FFN), with the paper's measured ~3× wall-clock
  slowdown attributed to RPE table gathers — an axis orthogonal to stream count.
- **Rank-`d` bottleneck.** The MFVI readout is affine in `Q_Z`, so the logit matrix has rank at
  most `d+1`. This is acknowledged and expected to cost some perplexity; it is also what makes
  the mixture-of-softmaxes exact readout interesting in Experiment 3.
- **Internship timeline.** Ends late August 2026. Experiment 2 is the scientific core and must
  survive any schedule compression; cut from the bottom of the plan's priority list.
