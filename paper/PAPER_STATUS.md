# Preprint status

Deliverable: a preprint of the causal Probabilistic Transformer, to be extended into a
conference submission. The theoretical construction is complete and approved by Prof. Tu; the
empirical section waits on the experiments running on `feature/causal-pt-decoder`.

Work on the paper happens on branch `article`. The experiments happen on their own branch, in a
separate worktree, so the two do not disturb each other:

```
/public/home/belyack/work/pt          feature/causal-pt-decoder   (experiments)
/public/home/belyack/work/pt-article  article                     (this paper)
```

## Recent changes

- **2026-08-09** — `paper/` created: LaTeX skeleton, notation macros, section outlines,
  bibliography stub, Makefile. No prose drafted yet; every section is an outline with the
  argument fixed and `\TODO` / `\FROMPDF` markers where the reference PDFs must be consulted.

## Build

**There is no TeX distribution on the cluster.** Checked 2026-08-09 on the login node: no
`pdflatex`, `xelatex`, `latexmk`, `bibtex` or `tectonic` in `PATH`, no `tex`/`latex` module in
the cluster module system, nothing under `/public/software`. `make` in `paper/` fails with an
explanatory message rather than a confusing LaTeX error.

Consequence: **the source in `paper/` has never been compiled.** It is written to build against
a stock TeX Live, but that is an expectation, not a verified fact. The first compilation will
almost certainly surface errors, and that is not evidence of anything being wrong with the
content.

Three ways forward, in order of how much they cost:

1. **Overleaf.** Upload `paper/`, compile there. Zero setup; means the source of truth lives in
   git and Overleaf is only a renderer, which needs discipline to avoid divergence.
2. **Compile locally.** Any laptop with TeX Live; `make` in `paper/` works as written.
3. **Install TeX Live into `$HOME`.** Outbound network works from the login node
   (`mirror.ctan.org` and `arxiv.org` both reachable through the configured proxy) and there is
   ~979 GB free on `/public`. `scheme-basic` plus the packages in `preamble.tex` is on the order
   of a few hundred MB. This is the only option that lets the paper be built in the same place
   the numbers are produced.

Not yet decided — see the open questions below.

## Structure

| File | Contents | State |
|---|---|---|
| `main.tex` | document skeleton, `\input` order | done |
| `preamble.tex` | packages, theorem environments, `\TODO`/`\NOTE`/`\FROMPDF` | done |
| `notation.tex` | notation macros; **kept in sync with `src/config.py`** | done |
| `sections/00_abstract.tex` | claim fixed, numbers pending | draft |
| `sections/01_introduction.tex` | paragraph plan + contributions | outline |
| `sections/02_background.tex` | Wu & Tu model | outline |
| `sections/03_impossibility.tex` | the trilemma and its proof | outline |
| `sections/04_construction.tex` | chain of conditional CRFs, frozen prefix | outline |
| `sections/05_output.tex` | unclamping `W_t`; tying and bias as consequences | outline |
| `sections/06_inference.tex` | MFVI updates, exact readout, cost | outline |
| `sections/07_experiments.tex` | shared setup + Experiments 0–3 | outline, most complete |
| `sections/08_related.tex` | four groups of related work | outline |
| `sections/09_conclusion.tex` | limitations, conclusion | outline |
| `sections/A1_derivations.tex` | appendix: derivations, worked example, proof | outline |
| `refs.bib` | bibliography — **every entry unverified** | stub |

## Rules specific to the paper

- **Notation follows the reference PDFs, not transformer convention.** `d` is the size of a
  label set, not a hidden dimension; `h` counts channels. `notation.tex` and `src/config.py`
  use the same names deliberately — change them together or not at all.
- **No number appears in the paper without a run-log row.** Every figure in
  `sections/07_experiments.tex` must be traceable to a row in the corresponding
  `experiments/*/EXPERIMENT_STATUS.md`: date, commit, config, seed, metric, wall-clock.
- **`\FROMPDF{...}` marks a claim taken from a reference PDF that has not been re-read while
  writing.** The PDFs are the specification; a claim reconstructed from memory is exactly the
  kind of error that survives review by everyone who already knows the material. Clear these
  markers by opening the document, not by deciding the claim looks right.
- **The exact readout is the mainline; MFVI is the ablation.** §17.2 says the opposite but is
  superseded by §23.3 of its own document. Do not present them as two equal options.
- **Write the conclusion to the measured result.** A negative Experiment 2 is still a paper.

## Open questions

1. **Author list and order** — placeholder in `main.tex`, to be confirmed with Prof. Tu before
   any public posting.
2. **Venue and style file** — currently the stock `article` class, deliberately venue-neutral.
   An ACL or NeurIPS style file can be dropped in without touching the sections.
3. **Build route** — Overleaf, local, or a `$HOME` TeX Live install (see above).
4. **`developer files/VERSION` is forked.** `main` carries `1.0.0`; the working tree of
   `feature/causal-pt-decoder` sets it to `0.1.0`, apparently restarting the numbering after the
   `v1.0.0 [breaking]` commit that cleared the implementation. Both lines will conflict at merge.
   Decide which is the line of record before then.
