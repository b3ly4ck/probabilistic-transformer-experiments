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
| MFVI as mainline readout | **open** | §17.2 recommends MFVI; §23.3 in Part IV walks this back toward exact readout. Both must be read before the implementation is frozen |
| Exact readout written in Exp 0 | decided | Serves as the oracle for check 9; Exp 3 reuses it rather than implementing anything new |
| Dataset | open | PTB or WikiText-2. Not WikiText-103 — that is the regime where the original PT is reported to fail |
| Parameter matching basis | open | total / non-embedding / both at fixed vocabulary. Must be stated explicitly; with tied embeddings PT's budget is almost entirely the `\|V\| × d` matrix `S` |
| Shared training loop | decided | One implementation across PT, GPT and Looped. A loop change that helps one model and not the others voids the comparison |

## Known risks

- **Sequential filtering.** Each slot's frozen marginal `q̄_t` depends on `q̄_{<t}`, so the
  observed passes are sequential in `t` — unlike a transformer, which parallelises over
  positions in training. Mitigation exists (the prefix is frozen, so the predictive passes for
  all `t` are independent given the cache and can run in parallel), but the cost profile has
  not been measured. Relevant from Experiment 1 onward.
- **Rank-`d` bottleneck.** The MFVI readout is affine in `Q_Z`, so the logit matrix has rank at
  most `d+1`. This is acknowledged and expected to cost some perplexity; it is also what makes
  the mixture-of-softmaxes exact readout interesting in Experiment 3.
- **Internship timeline.** Ends late August 2026. Experiment 2 is the scientific core and must
  survive any schedule compression; cut from the bottom of the plan's priority list.
