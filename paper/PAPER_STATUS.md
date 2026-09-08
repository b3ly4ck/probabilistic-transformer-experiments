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

**First build: 2026-08-09, clean.** No LaTeX errors, no undefined references, no undefined
citations, no overfull or underfull boxes. Output is 8 pages including references and
appendices. One cosmetic warning remains — `name{Hfootnote.1} has been referenced but does not
exist` — from the `\thanks` corresponding-author footnote interacting with hyperref in the ACL
template; it does not affect the output.

That 8 pages says nothing about the final length: every section is an outline, and the draft
markers render as visible red text that will disappear. Re-check with `make pages` once prose
exists — the budget is 8 pages of *main body*, before the references.

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

## Verification of the appendices — 2026-09-08

Appendices A, B and D were written from the technical note and then independently checked by a
second pass that re-read the source sections and rebuilt the document. That pass found and
removed two invented claims in Appendix A about what the main text says (a parameter-matching
rule and a wall-clock report that Section 4.2 does not contain), which is exactly the failure
mode an appendix written alongside an unfinished results section invites.

**Appendix C was checked by hand instead**, its automated pass having died mid-run. Verified
against Part I of the technical note, section by section:

| appendix content | source | verdict |
|---|---|---|
| word-locality (A3) and the word-nonlocal witness energy | §1, Decision 1 | matches |
| C1 vs C2, and that the theorem is false under C1 | §1, Decision 2 | matches |
| the sharpness construction: Hermite $h$ with $h(\Delta(v))=c_v$, $h'(\Delta(v))=0$; the $(1,-1)$ tangent argument; $\neg$C2 off the critical set | Prop. 8.1 and Rem. 8.2(i) | matches, including the "parameter conspiracy, not an architecture" reading |
| setup, (A1)--(A4), interiority, the $xy(x^2-y^2)/(x^2+y^2)$ counterexample | §2 | matches |
| gauge reduction | Lem. 3.1 | matches, both directions |
| no one-way coupling, all three equivalences | Lem. 3.3 | matches, both proof directions and the Schwarz step |
| confinement, and the head-variable remark | Lem. 4.1, Rem. 4.2 | matches, including "necessity only, not transmission" |
| C2 $\iff$ no cross-position edge | Prop. 5.1 | matches, including the "crux" sentence |
| the theorem and C2 $\Rightarrow$ C1 | Thm. 5.2, Cor. 5.3 | matches |
| $\Pi T^{(c)} \Pi = 0$, non-degenerate $T$ $\Rightarrow$ no energy | Cor. 7.1 | matches |
| the three witnesses | §6 | matches |

No claim in Appendix C is unsourced. The one liberty taken is presentational: the note's numbered
lemmas become run-in paragraphs with two numbered environments, because this paper's format
allows very few.

## All five grids complete — 2026-09-09, 05:00

globalhead 54/54 · exp3 open questions 37/37 · switch ladder 160/160 · WikiText-2 24/24 ·
factored labels 48/48. **Zero errors in any of them**, and the paper has **zero TODO markers**:
`\draftfalse` builds clean, verified.

The finding to lead with is the one the technical note **registered before any of it ran**. The
exact readout's penalty against the mean-field one falls monotonically with the number of label
components — $+52.2, +30.0, -3.5$ at 8,256 arc parameters and $+21.7, -6.2$ at 32,896 — and at
fixed total width the two readouts separate outright: cutting the arc budget sixteenfold, the
exact readout improves the whole way down (313.0 → 260.6 → 253.0) while the mean-field one is
best in the middle. That reframes Experiment 3's result that the exact readout trains 100
perplexity worse: it was not punished for being richer, it had nothing extra to be rich about at
`K=1`, because the product of one mixture is a mixture.

**What a reader should take from the session as a whole.** The construction is a package — nine
of ten transformer properties cannot be restored inside it without making it worse. Its width
sensitivity has a mechanism, the repair transfers to a second corpus untouched, and a
semi-orthogonal draw removes almost all of the seed variance. Its slope is shallower than both
baselines, four fifths of that attributable to weight sharing. And two of its own design
choices — the exact readout and multiple inference iterations — cost at `K=1`, which is where
everyone has run it.

## Grids finished, 2026-09-09 (early morning)

Four of the five grids completed and two produced findings that changed the paper's argument
rather than filling a gap in it.

5. **The switch ladder's S2 column: the construction is a package.** Restoring *any* single
   transformer property inside the all-PT rung makes it worse -- the mildest, weight sharing,
   costs 2.1; layer normalisation, the feed-forward block, an unconstrained state and the
   residual stream cost 205 to 390, the last landing on the unigram. The properties that are
   free to *adopt* (S1: three at or below zero) are not free to *remove*. Pricing each
   difference twice is what made this visible; a ladder pricing each once would have concluded
   there was no structure there. S1's optima are all interior; six of S2's were at the top of
   the swept grid, so a fifth learning rate (1e-1) was appended and is running.
6. **The factored-label interaction, which the technical note registered in advance.** Along an
   iso-parameter diagonal the exact readout's penalty against the mean-field one falls
   monotonically: +52.2 at K=1, +29.9 at K=2, -3.7 at K=4. This is the one result in the paper
   that cuts against its own thesis that the limit is optimisation rather than expressiveness,
   and it suggests the readout was punished not for being richer but for having nothing extra
   to be rich about at K=1. Reported as a trend, not a crossover: the last point is two seeds.
7. **WikiText-2: both halves of the width finding transfer** with nothing retuned, on three
   times the vocabulary. The gap to the baselines transfers too and is wider.

**Two operational bugs found and fixed, both of which had been quietly misleading.** The
Makefile did not depend on `figures/`, so regenerating a table did not rebuild the PDF and
several overfull-box measurements were stale readings of an unchanged file. And
`submit_queue.sh` removed a submitted line with `grep -v ... && mv`, which fails on the *last*
line in the queue because grep exits 1 when it filters everything out -- so the queue resubmitted
its final entry once a minute. Caught by the shard lock refusing every duplicate.

## Results added overnight, 2026-09-09

Four findings the paper did not have when the verification pass above was written. All are in
Section 4 with their seed counts; the first two are the most consequential.

1. **Training under the exact readout is worse, by a hundred perplexity.** $348.8$ against the
   mean-field readout's $248.3$ at `d=32` damped, same loop and budget. Proposition 2(iii) says
   the exact readout's image strictly contains the other's at zero parameter cost, so this is a
   statement about what the optimiser reaches, not about what the family contains. The paper's
   own mainline readout is the one that trains worse.
2. **Inference depth is monotonically harmful.** `T=1` 241.3, `T=2` 246.5, `T=3` 248.3, `T=8`
   275.3, with `tau` behaving the same way. At this scale the model is best as a one-iteration
   model, so the property it shares with looped transformers is present in the construction and
   not exercised by the data. Single seed; the `T=1`–`T=3` step is one seed sd and is not read
   alone, the span to `T=8` is five.
3. **The initialisation draw matters, through variance.** Semi-orthogonal $219.3 \pm 0.6$
   against Gaussian $225.8 \pm 10.2$, with matched-variance uniform collapsing on one seed in
   three. The registered prediction (orthogonal survives an $L_2$ that kills a small Gaussian
   draw) is *not* what this shows and the paper says so.
4. **The global head buys stability, not capacity.** Nothing at `d=32`; at `d=64` it changes
   which parameterisations are stable without making the model stable. `H(Q_g)/max` between
   0.93 and 1.000 in every cell of both grids — a variable whose posterior is uniform is
   sending a constant.

Two knobs with no gentle regime were also measured (`l2_arc` inert at 5e-3 and fatal at 5e-2;
the position clip inert at 1 and 3 and costing 357 at 7), and the Discussion now carries an
inventory of which of the model's seven hyperparameters matter.

**Process changes made alongside.** `\draftfalse` no longer hides outstanding items, it refuses
to build while any exist. Both grid tables share one rule for refusing to average a seed group
that is not one distribution. The title page states that the author list is provisional. Three
passages that cited listed authors' private review comments by name were rephrased.

## Number-by-number verification — 2026-09-08 (evening)

Every numeral in the paper was checked against the run JSONs it claims to come from, rather
than against the draft that introduced it. Seven were wrong. They are listed because the
pattern matters more than the individual fixes: **every one was a number that was correct when
typed and went stale as a grid filled in**, and prose does not announce when that happens.

| where | claimed | actual | fixed in |
|---|---|---|---|
| §4 scaling | PT "scales like a looped transformer" | shallower than looped in every shared window | v1.13.0 |
| §4 scaling | decomposition across three different windows | one three-way window, 13k–148k | v1.13.2 |
| §4 width | "argmin of α_Z falls from 0.25 at d=16" | 0.125 at d=16, and at five of seven widths | v1.13.3 |
| §4 rank | "a fifth of the seed variance" | an eighth of the spread (3.3 against 27.1) | v1.14.2 |
| §4 rank | "seed spreads are 1.5 to 5.6" | 1.5 to 10.4 | v1.14.2 |
| §4 rank | "the width ceiling disappears" | it moves; d=256, r=64 has a collapsed seed | v1.14.2 |
| §4 readout | seed mean against one seed's swapped value | both sides seed means, over 11 configs | v1.14.1 |
| App. F | prior–posterior r = 0.16 for the selected model | 0.47; 0.16 is the *earlier* model | v1.14.3 |

**The structural fix, so this class of error cannot recur.** `experiments/make_report.py` now
emits `paper/figures/numbers.tex`, a set of `\newcommand` macros for every headline number the
prose quotes, computed from the same rows that build the tables beside them. The prose carries
macros, not digits. Adding a number to the prose means adding it to `emit_numbers` first.

**What verified clean, checked and not merely assumed:**

* Appendix E — re-ran `experiments/validation_report.py`: exact readout against brute-force
  enumeration 6.66e-16, free energy 11 updates and 0 increases with total decrease 5.681e-3,
  the sign-flipped mutation 11 increases with the largest 6.5e-2, causality bitwise equal in
  all six readout × temperature combinations, `S` one object, factored K=1 against the flat
  decoder 6.94e-18, belief sums 1.2e-7. Every number reproduces.
* Appendix D — recomputed from the corpora: all eleven PTB figures and all eleven WikiText-2
  figures reproduce exactly, including the scored-token counts and both unigram conventions.
* Appendix F attachment probe — root mass, mean distance, distance-1 mass, mass beyond 3, and
  the uniform reference all reproduce from the probe JSONs for both models.
* §4 baselines — 129.2 / 124.7 / 125.3, train 68.5 → 10.2, and the looped crossover
  (164.7 at 12,768 against 195.2 at 13,152; 135.7 at 198,528 against 129.2 at 200,064).
* §4 control ladder — 253.0 ± 6.0, 287.4 ± 52.5, 470.0 ± 160.7, and 250.7 ± 2.3 traced to its
  row in `experiments/exp1_language_modeling/EXPERIMENT_STATUS.md`.

## Length — 2026-09-08

Main body currently runs to **page 16**; references start on 17; total 38 with appendices.
Word counts from `make words`, 2026-09-09: abstract 220, intro 836, §2 2403, §3 1569, **§4
5815**, related 540, discussion 1165, conclusion 196, limitations 696 --- 13,440 in the main
body. §4 has roughly doubled since this section was written, by additions each of which the
paper needed: which readout the results use, the readout rung of the ladder, the ladder figure,
the reproducibility figure, the initialisation grid that cannot be read, and the pairwise
exponent apparatus. None of them is padding and none should be cut before the four steps below
are taken in order. The
format budget adopted in the 2026-08-09 decision is 8 pages of main body, matching Wu & Tu.

The overrun is not padding. Their Section 4 reports one table over five tasks; ours reports a
scaling comparison over three architectures and nine widths, a width-transfer mechanism with
four grids behind it, a ten-rung ablation ladder, an initialisation study, a readout comparison
and a corpus transfer. Sections 2 and 3 are close to their budget (2431 and 1448 words against
~1950 and ~2040); Section 4 is at 2847 against ~1360.

**Plan, in the order it should be executed, once the results are final.**

1. Move the per-grid detail of Section 4 to appendices, leaving in the main text one paragraph
   per finding with its headline number and pointing at the appendix table. The grids that
   should go: the gain sweep table, the learning-rate transfer table, the rank table, and the
   per-switch S1/S2 tables (keeping the cumulative ladder figure).
2. Cut Section 2.1 (Background) hard --- it currently restates more of the encoder model than
   Sections 2.2 and 2.4 actually use.
3. Fold the two mechanism figures into one two-panel figure.
4. Only then trim prose.

Do **not** cut the reproducibility paragraph, the statement that the working repair leaves the
graph, or the width-ceiling limitation. Those are the three places where the paper says
something against its own interest, and they are what make the rest credible.

For an arXiv preprint the present length is acceptable; the cut is for a conference version, and
should be made against a specific venue's limit rather than in the abstract.

### Appendix D verified independently, 2026-09-08

Every derived count in `D_datasets.tex` was recomputed from the loaders rather than read back
from the appendix: vocabulary and split sizes for both corpora, evaluation blocks
(1{,}152 / 1{,}287 for PTB validation and test; 3{,}400 / 3{,}837 for WikiText-2), scored tokens
(72{,}576 / 81{,}081 and 214{,}200 / 241{,}731), and both unigram baselines with and without slot
0 (688.82 / 687.45 and 964.82 / 965.37). All match.
