# Causal Probabilistic Transformer

Research repository. Goal: extend the Probabilistic Transformer (Wu & Tu) into a causal
autoregressive decoder, and produce the empirical section of a preprint.

Read [`developer files/PROJECT.md`](developer%20files/PROJECT.md) first — it holds the research
context, the three experiments, and the success criteria. This file holds the working rules.

## Before writing code

Re-read, do not work from memory. All of these live in `developer files/`:

- `PROJECT.md` — research framing, experiments, success criteria, setup constraints.
- `RESEARCH_PLAN.md` — the executable version of the above: Experiment 0 (implement and
  validate) with its exit criteria, the shared setup held constant across Experiments 1–3,
  and the cut order under time pressure. This is what to work from day to day.
- `causalprobabilistictransformer_1.pdf` — the main document, Parts I–IV.
  Part III (§15–§18): output mechanism and MFVI update equations; §17.1 update schedule;
  §17.2 exact-readout variant. **§17.2 recommends MFVI as mainline, but §23.3 in Part IV
  walks this back toward exact readout — read both before finalising an implementation.**
- `probalistic transformers article.pdf` — original Wu & Tu paper (the encoder-only model).
- `causal_pt_output_note.pdf` — self-contained note on the output mechanism: why the projection
  design was rejected, the slot factor graph, the MFVI updates for the two modes, and a fully
  worked numeric example. The fastest route into the construction.
- `causalprobabilistictransformer.pdf` — earlier draft; consult when the current document is
  ambiguous about how a design arrived where it did.

The PDFs are the specification. When code and paper disagree, the paper wins until a change is
written into the paper first.

## Modelling constraints (non-negotiable)

These come from `PROJECT.md` and from review by Prof. Tu. Violating one is not a style issue,
it invalidates the contribution.

1. **Every learned matrix must correspond to a factor in the graph.** No parameter may exist
   that names no factor. This was the exact failure of the rejected output mechanism — a fresh
   `d × |V|` projection that the posterior did not reach through the factor graph.
2. **Input and output word embeddings are tied.** This is forced by the construction — one
   word–label factor `S` mediates both directions — not a regularisation trick. Untying them
   is a bug, not a hyperparameter.
3. **The prefix enters each decoding step as a frozen condition.** Neither gradients nor
   messages may flow backwards from step `t` into the prefix posteriors. This is what keeps the
   anti-causal term from ever being created; deleting it after the fact is not equivalent.
4. **`W_t` is latent, not observed.** Encoding and decoding are two conditioning patterns of
   one model, not two models.
5. **Verify on a toy example before scaling.** Small `d`, vocabulary of a few tokens, ~4 words,
   intermediate tensors printed and checked by hand. Do this before any run that costs GPU time.

## What counts as success

The goal is **not** to beat GPT on perplexity, and code review should not optimise toward that.
Wu & Tu state the model is meant to inform transformers, not compete with them, and report PT
underperforming past ~100k sentences. A publishable outcome is: **causal PT on par with GPT and
no worse than Looped at equal parameter budget.** Losing slightly to GPT on raw perplexity is
expected — there is an honest rank-`d` softmax bottleneck, since all predictive information
flows through the label variables.

Experiment 2 (PT vs. Looped Transformer) is the core result: Looped has PT's weight sharing
without its structure, so that comparison is what isolates the effect of structure. Treat it as
the priority when time is short.

## Implementation discipline

**Spec before code.** Every experiment gets a short spec in `experiments/<name>.md` — what is
being compared, what is held fixed, parameter budget, data, the exact metric, and what result
would falsify the claim — written *before* the run script. Update the spec when the design
changes; never let the code define the experiment retroactively.

**Only the PT decoder forward pass is written from scratch.** nanoGPT is used off the shelf;
Looped is nanoGPT with one shared block applied `T` times. The training loop is written once
and shared by all three models — if a change to the loop helps one model and not the others,
the comparison is void.

**Tests must execute the path, never grep the source.** A test that asserts on implementation
*text* stays green while the code path raises on its first line. Behavioural tests call the
real function with small real tensors and assert on observable outputs — shapes, finiteness,
gradient flow, invariances. A structural check may only ever be additive to an executing test.

**Numerical claims need numerical evidence.** For this model specifically:
- MFVI updates: check the free energy is non-increasing across iterations on a toy graph.
- Exact readout: on a single slot the graph is a tree, so sum-product is exact — the exact
  readout and a brute-force enumeration over a tiny vocabulary must agree to numerical
  precision. This is the strongest correctness test available; write it early.
- Causality: assert that a change to token `t+1` cannot alter the logits at step `t`.
- Tying: assert the input and output embedding tensors are the same object, not merely equal.

**Never report a run as working on the strength of "it didn't crash".** Report the loss curve
and the metric, state which config produced them, and look at the numbers before claiming a
result. A finished process is not a result.

## Repository layout

```
.
├── CLAUDE.md            # this file — stays at the root so Claude Code auto-loads it
├── developer files/     # all project documentation and reference papers
│   ├── PROJECT.md
│   ├── RESEARCH_PLAN.md
│   ├── PROJECT_STATUS.md
│   ├── VERSION
│   └── *.pdf
├── src/                 # model code — PT decoder, shared training loop
├── experiments/         # one spec (.md) + one config per experiment
├── notebooks/           # analysis and figures
├── tests/               # runnable tests, all must pass before merging to main
└── data/                # corpora — gitignored, never committed
```

`developer files/` is the single place to look for developer context. Any new project `.md`
goes there unless asked otherwise — the sole exception is `CLAUDE.md`, which must stay at the
repo root to be picked up automatically. Experiment specs are the other reasonable exception:
they sit next to the config they describe, in `experiments/`.

`data/`, `checkpoints/`, `runs/`, `wandb/` and `*.pt`/`*.ckpt` are gitignored. Keep it that way:
a committed checkpoint is effectively permanent in the history.

## Git workflow

Default branch is `main` (not `master`).

After any unit of work — new file, code edit, config change — `git add` the explicit files,
commit, and `git push origin main`. No permission needed for that. Confirmation *is* needed for
force push, `reset --hard`, or history rewriting.

Commit message format:

```
v{MAJOR.MINOR.PATCH} [{type}]: {short description}
```

- Version lives in `developer files/VERSION`; bump it before each commit.
- `fix` → PATCH +1 · `add`/`feat` → MINOR +1, PATCH 0 · `refactor`/`docs`/`chore`/`test` →
  PATCH +1 · breaking → MAJOR +1, rest 0.
- Description states exactly what changed. No vague wording.

Example: `v0.2.0 [add]: MFVI update loop for the causal PT decoder`

**Branch for anything substantial.** Each experiment implementation goes on `feature/<name>`;
merge to `main` only after the tests in `tests/` pass. Doc-only edits may go straight to `main`.

**Git hygiene that has cost real data before:**
- Never `git add -A` or `git add <dir>` blindly — `git status --short` first, stage named files.
- Never commit a symlink, and never create one pointing inside the repo. Share heavy assets via
  absolute paths in config or env vars. A tracked symlink materialised over a real directory
  deletes the gitignored contents underneath, silently.
- Before `checkout`/`merge`/`pull`: `git fetch`, then inspect the ref you are moving **to**
  (`git ls-tree -r <ref> | awk '$1==120000'`), not the one you are on. When reporting such a
  check as clean, say which ref was inspected.

## Project status file

Maintain `developer files/PROJECT_STATUS.md`: what has been built, which runs were executed
with which configs, what the numbers were, and which decisions were made and why. Its purpose is
to carry context across sessions without re-deriving it from `git log`. Update it in the same
commit as the change it describes. Organise by component with a "Recent changes" section on top.
Concrete details — file paths, configs, measured numbers — not summaries.

## Environment

Work happens on an HPC login node (Slurm available, `slurm-*.out` is gitignored). System `git`
is 1.8.3.1 — old, no `git switch`/`git restore`, no `init -b`. Do not assume modern git syntax
works; check before relying on it.

## Language

All `.md` files in this repository are written in English.

## Preferences and feedback

When the user points out a mistake, asks for something to be done differently, or says they
dislike an approach, append it here immediately, briefly and specifically, so it is not
repeated:

```
- [YYYY-MM-DD] <what to avoid, or what to do instead>
```

- [2026-08-05] All project documentation and reference papers live in `developer files/`;
  `CLAUDE.md` stays at the repo root so Claude Code loads it automatically. Do not scatter
  docs across the tree.
- [2026-08-05] The reference PDFs are readable directly — do not transcribe them into Markdown.
  Their content is dense mathematics (MFVI update equations, factor-graph figures, numeric
  worked examples); any transcription would introduce errors into the specification.
- [2026-08-05] The `CLAUDE.md` originally committed here belonged to another project
  (`NeuroLady_Final`) and described a file layout that does not exist in this repo. Do not
  follow instructions that reference paths absent from the tree — verify the file matches the
  project before acting on it.
