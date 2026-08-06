# Project Status

Technical state of the project, maintained so work can be resumed without re-deriving history
from `git log`. Per-experiment records live in `experiments/<exp>/EXPERIMENT_STATUS.md`; this
file covers what spans experiments.

**Overall state as of 2026-08-05: Experiment 0 complete — checks 1–9 pass.** The causal PT
decoder forward pass exists, reproduces the note's §5 worked example digit for digit, and its
exact readout agrees with brute-force enumeration to 7.5e-08. Experiment 1 is unblocked, with
two things settled since: the B.3 globals are now a measured variable (arms 1.1/1.2 of
Experiment 1), and the flat-`log μ` finding below has a second instance — see the `G_t` entry
in the decisions table.

## Recent changes

| Date | Version | Change |
|---|---|---|
| 2026-08-06 | 0.8.0 | Calibration on PTB. PT converges to the unigram baseline and cannot fit its training data; GPT overfits. Session report in `REPORT_2026-08-06.md`. |
| 2026-08-05 | 0.4.0 | Experiment 1 split into arms 1.1 (no `G_t`) and 1.2 (with `G_t`); `exp1_language_modeling/` specified, `exp1_pt_vs_gpt/` superseded. Source check found B.3.3 single-split and the context-free degeneracy of `G_t` under the exact readout. |
| 2026-08-05 | 0.3.1 | Experiment 0 results recorded: overfit sweep, the `D_0` floor, and the finding that the exact readout is nearly flat on the note's §5 example. |
| 2026-08-05 | 0.3.0 | Causal PT decoder implemented: content stream, exact and mean-field readouts, nine checks. All pass. |
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
├── src/                      # model code
├── tests/                    # nine checks, one file each
└── experiments/
    ├── README.md             # index + logging rules
    ├── exp0_decoder_validation/   # complete, checks 1-9 pass
    ├── exp1_language_modeling/    # arms 1.1 and 1.2, specified
    ├── exp1_pt_vs_gpt/            # superseded by the above, kept
    ├── exp2_pt_vs_looped/
    └── exp3_exact_vs_mfvi/
```

`notebooks/` and `data/` are declared in the layout but do not exist yet.

## Code

| Module | Contents |
|---|---|
| `src/config.py` | `PTConfig` — dimensions and the MFVI temperatures (`λ_Z = 1`, `λ_H = 1/d` by default, per the Notation section) |
| `src/mfvi.py` | slot-level updates (2)–(4) as pure functions over tensors, the prefix contraction, the per-slot energy `E_t` and the free energy `F = E − Σ λ_x H(Q_x)` |
| `src/exact.py` | the §17.2 tree readout — `log μ` by `logcumsumexp` prefix scan, `O(ndh)` and parallel — and a deliberately naive brute-force enumerator used as its oracle |
| `src/pt_decoder.py` | `CausalPTDecoder`: content stream over all positions under one causal mask, both readouts, NLL loss |
| `tests/` | nine checks, one file per check, 30 tests, ~76 s on CPU |

The parameter list is exactly the factor list — `S`, `T`, `r`, `b` — and a test asserts it, so a
parameter that names no factor cannot be added silently. `S` is one `nn.Parameter` used in both
directions, so tying is structural rather than asserted.

nanoGPT is still used off the shelf for the GPT baseline; the Looped baseline is nanoGPT with
one shared block applied `T` times. Neither is written yet.

## Runtime environment

Surveyed 2026-08-05 on the login node. The environment was empty at survey time; what was
installed since is recorded below the table.

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

**Environment — decided 2026-08-05.** Conda environment inside the repo at `./.venv`
(gitignored), Python 3.11.15, torch 2.13.0+cpu, pytest 9.1.1, numpy 2.4.6:

```
/public/software/anaconda3/bin/conda create -y -p ./.venv python=3.11
./.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./.venv/bin/pip install pytest pyyaml numpy
```

CPU-only is deliberate: Experiment 0 is CPU-runnable by design and check 3 needs bitwise
reproducibility, which a GPU does not guarantee. A CUDA build is installed separately for
Experiment 1, after checking the driver version on the GPU partitions.

## Cross-experiment decisions

| Decision | Status | Notes |
|---|---|---|
| Which readout is mainline | **decided 2026-08-05: exact** | §23.3 states it outright — exact readout mainline, mean-field two-stream as the ablation. The query stream disappears entirely (§24.1); the readout is a causal `logcumsumexp` scan, `O(ndh)`, fully parallel |
| Exact readout written in Exp 0 | decided | It is now the mainline *and* the oracle for check 9; Exp 3 reuses it rather than implementing anything new |
| Gradient through the content stream | **decided 2026-08-05: flows normally** | "Frozen prefix" is variational (`q̄_j` is not re-optimised at later slots), not stop-gradient. Causality comes from the triangular mask, as in a transformer. Running the filtering pass under `no_grad` would starve `S` and `T^(c)` |
| Appendix B.3 globals (`G_t`) from day one | **resolved 2026-08-05: measured, not assumed** | Experiment 1 splits into arm 1.1 (without) and arm 1.2 (with); the delta is the result. §22.2 calls `G_t` the graph-faithful answer to "PT lacks an FFN", but Wu & Tu never test it — B.3 is a derivation with no experiments. Construction is **B.3.3 single-split**, `B' ∈ R^{m×d}`, cost `O(md)` per position |
| `G_t` under the exact readout | **decided 2026-08-05: run both readouts in arm 1.2** | `G_t` is a leaf on `Z_t`, so exact marginalisation gives `log μ_t(a) += LSE_k B'[k,a]` — identical at every position, hence context-free. Only the MFVI path yields the GFU operator `σ(qB'^⊤)B'` that is the FFN analogue, so that path carries the claim |
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
- **The exact readout is nearly flat on the note's own §5 example.** MFVI gives `p(sat) = .007`
  ("context kills the verb", the behaviour the example illustrates); the exact readout on the
  same prefix and parameters gives `.206`, because `log μ_4 = (2.382, 2.338)` is almost equal
  across labels — the LSE over prefix positions averages where MFVI's `Q_c` selects. Not an
  implementation discrepancy: it matches brute-force enumeration to `7.5e-08`. It is §23.3's own
  caveat ("`μ_t` replaces `Q_Z^pred` — finer, but different") appearing at once, and it bears on
  the verdict that made the exact readout the mainline. Evidence is one toy example plus toy
  memorisation runs; raise with Penghao and Prof. Tu before Experiment 1.
- **Rank-`d` bottleneck.** The MFVI readout is affine in `Q_Z`, so the logit matrix has rank at
  most `d+1`. This is acknowledged and expected to cost some perplexity; it is also what makes
  the mixture-of-softmaxes exact readout interesting in Experiment 3.
- **Internship timeline.** Ends late August 2026. Experiment 2 is the scientific core and must
  survive any schedule compression; cut from the bottom of the plan's priority list.
