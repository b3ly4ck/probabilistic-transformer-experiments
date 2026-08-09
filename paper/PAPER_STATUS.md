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

## Format decision — 2026-08-09

**The paper follows Wu & Tu, "Probabilistic Transformer: A Probabilistic Dependency Model for
Contextual Word Representation" (Findings of ACL 2023), in style, section skeleton and length.**
That paper is in `developer files/probalistic transformers article.pdf`; it is the direct parent
of this work and Prof. Tu is its author. Where a structural question comes up, the answer is
"what did that paper do".

Its actual shape, measured from the PDF:

- **22 pages total: 8 pages of main body**, references starting on p. 9, appendices A–F.
- **7 numbered sections** plus an unnumbered `Limitations` after the Conclusion.
- **3 figures, 1 table.** All experiments — five tasks, six datasets — are in a *single* results
  table with two model columns and mean ± std over 5 random runs.
- Section 3, the conceptual comparison against transformers, is the **largest section** (~2.4 p),
  as long as the experiments. Section 4 (Experiments) is ~1.6 p, of which Results is ~0.5 p.
- Related Work is one third of a column. Conclusion is one paragraph.
- A **Discussion** section separate from the Conclusion, holding the concessions: that the goal
  is not to compete with transformers, and that the model degrades beyond ~100k sentences.
- **No theorem environments anywhere.** The paper is entirely constructive prose.

Three consequences we adopted:

1. **One results table for the whole paper.** Experiments 1 and 2 of `RESEARCH_PLAN.md` are one
   table with three model columns (Transformer / Looped / Causal PT), not two subsections.
   Experiment 3 and the global-variable arm are paragraphs under Results.
2. **A `Comparison with Causal Transformers` section (§3) that the first outline did not have.**
   This was the largest structural gap. It is also where the Looped baseline gets motivated as a
   consequence of the model comparison rather than asserted as an experimental choice.
3. **The trilemma is stated, not proved, in the main text.** One theorem environment and one
   proposition in the whole paper; the proof goes to Appendix C. A paper in this format with six
   numbered theorems does not look like the paper it is modelled on.

## Section map and page budget

Budget is the 8-page main body. Percentages follow the parent paper's allocation.

| § | Section | Budget | Parent paper's counterpart |
|---|---|---|---|
| — | Abstract | ~150 words | same length, ends on framing not on a number |
| 1 | Introduction | 1.2 p | §1, same six-paragraph rhythm |
| 2 | Causal Probabilistic Transformers | 2.3 p | §2 (2.1 recap · 2.2 obstruction · 2.3 chain · 2.4 output · 2.5 inference · 2.6 variants) |
| 3 | Comparison with Causal Transformers | 2.4 p | §3, their largest section |
| 4 | Experiments | 1.6 p | §4 (Tasks and Datasets · Settings · Results) |
| 5 | Related Work | 0.3 p | §5, one paragraph |
| 6 | Discussion | 0.6 p | §6, where the concessions live |
| 7 | Conclusion | 0.2 p | §7, one paragraph |
| — | Limitations | free | unnumbered, after Conclusion, outside the page limit |
| A–F | Derivations · Variants · Proof · Datasets · Validation · Worked example | — | their A, B, —, D, —, F |

Figures, mirroring their three:

1. **The factor graph, two panels** — encoder with the word variable shaded, decoder with it
   unshaded and the prefix boxed as a frozen condition. Same factors, same edges, different
   shading. This one figure carries the "one model, two conditioning patterns" claim. Their
   Figure 1 appears at exactly this point.
2. **Three computation graphs side by side** — causal transformer, Looped, causal PT. Their
   Figure 3 does the same for transformer / pre-LN / PT.
3. **Free energy against iteration** — the validation figure. No counterpart in their paper.

## Recent changes

- **2026-08-09** — restructured to the Wu & Tu skeleton: ACL style file, 7 sections plus
  Limitations, appendices A–F, one results table, new §3. The nine sections of the first
  scaffold were folded into §2 and §3; `02_background`, `03_impossibility`, `04_construction`,
  `05_output`, `06_inference`, `07_experiments`, `08_related`, `09_conclusion` and
  `A1_derivations` are gone. Still no prose.
- **2026-08-09** — `paper/` created: LaTeX skeleton, notation macros, section outlines,
  bibliography stub, Makefile.

## Build

TeX Live (`scheme-small`) is installed under `$HOME/texlive` — the cluster has no system TeX and
no `tex` module, so this private install is the only one. `paper/Makefile` puts it on `PATH`
itself, so `make` works from any shell.

```bash
cd /public/home/belyack/work/pt-article/paper && make
```

`make pages` reports the page count, which is worth checking often: in a two-column format an
overrun is discovered far too late otherwise.

The ACL style files (`acl.sty`, `acl_natbib.bst`) are committed, from
`github.com/acl-org/acl-style-files`. `main.tex` uses the `preprint` option — non-anonymous with
page numbers; switch to `review` for anonymous submission and `final` for camera-ready.

**Nothing here has been compiled yet.** The install was still running when the restructuring was
committed. The first build will surface errors, and that is not evidence of anything wrong with
the content.

## Rules specific to the paper

- **Notation follows the reference PDFs, not transformer convention.** `d` is the size of a
  label set, not a hidden dimension; `h` counts channels. `notation.tex` and `src/config.py` use
  the same names deliberately — change them together or not at all. Where a quantity is the same
  one as in the parent paper, reuse its symbol verbatim; deviating without reason costs the
  reader the comparison.
- **No number appears in the paper without a run-log row.** Every entry of the results table must
  be traceable to a row in the corresponding `experiments/*/EXPERIMENT_STATUS.md`: date, commit,
  config, seed, metric, wall-clock.
- **`\FROMPDF{...}` marks a claim taken from a reference PDF that has not been re-read while
  writing.** Clear these by opening the document, not by deciding the claim looks right.
- **The exact readout is the mainline; MFVI is the ablation.** §17.2 says the opposite but is
  superseded by §23.3 of its own document.
- **Write the Discussion and Conclusion to the measured result.** A negative Experiment 2 is
  still a paper.

## Open questions

1. **Author list, order and affiliation block** — placeholder in `main.tex`. The parent paper
   uses a single shared-institution block; confirm with Prof. Tu before any public posting.
2. **Target venue.** The format assumes the *ACL family (8 pages, mandatory Limitations). The
   parent paper is Findings of ACL 2023, so this is the natural target, but it has not been
   confirmed.
3. **Corpus.** PTB, WikiText-2, or both. The parent paper used PTB and BLLIP for its MLM task —
   BLLIP is worth considering for continuity, though it is not in `RESEARCH_PLAN.md`.
4. **Every entry in `refs.bib` is unverified** and marked as such. `wu2023probabilistic` must be
   copied from the ACL Anthology BibTeX export before anything is posted.
5. **`developer files/VERSION` is forked.** `main` carries `1.0.0` and this branch continues from
   it; the working tree of `feature/causal-pt-decoder` sets it to `0.1.0`, apparently restarting
   the numbering after the `v1.0.0 [breaking]` commit. Both lines will conflict at merge.
