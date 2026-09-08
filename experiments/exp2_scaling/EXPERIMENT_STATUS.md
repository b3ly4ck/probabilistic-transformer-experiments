# Experiment 2 — scaling curves: causal PT vs. GPT vs. Looped Transformer at matched budget

## Question

**At each parameter budget, which architecture reaches the lower validation perplexity on
PTB, and how do the curves' slopes differ?**

This replaces the single-point comparison of the 2026-08 report, which put one PT
(16,448 non-embedding parameters) against one GPT (1,237,440) and one Looped Transformer
(309,600) and reported the perplexity gap as if it were an architectural result. It is not:
it is a budget difference. Both reviewers said so independently.

> Prof. Kewei Tu, 2026-08-11: "A fair comparison is needed for PT vs. looped transformer with
> the same model size."

> Penghao Kuang, 2026-08-14: "you might consider plotting a series of scaling curves showing
> model performance as a function of parameter count. This would be similar to Figures 2(a)
> and (b) in [arXiv:2604.25409]. Such an analysis would provide a comprehensive comparison
> between Causal PT and Looped Transformers."

Kuang also gave a fallback for the case where a full PT curve is not affordable: "tune and
train a single PT model, then train one Transformer with slightly fewer parameters and
another with slightly more parameters than the PT model. On a log-log scaling plot, represent
the PT as a single point and the Transformer results as a line segment." The baseline grid
here is dense enough to contain that bracket as a special case, so the fallback is available
whatever happens to the PT curve.

## Two parameter axes, both reported

PT's parameters are overwhelmingly the tied word-label factor `S` (`V x d`): at `d = 32`,
330,000 embedding against 16,448 non-embedding. A transformer at `n_embd = 32` has 322,048
embedding against ~49,000 non-embedding. Reporting only the non-embedding count flatters PT
by a factor of 75; reporting only the total hides that PT's *computational* parameters are
tiny. Both curves are drawn, and the paper shows both.

## Held fixed across every cell

| Item | Setting |
|---|---|
| Data | PTB word level, Mikolov preprocessing, `<eos>` per line, vocab 10,000 |
| Batching | 16 x 64 blocks from the `<eos>`-joined stream, identical for all models |
| Scored tokens | `ignore_first = 1` — identical token set for PT and the baselines |
| Budget | 15,000 steps (15.4M tokens, 16.5 epochs) for every model |
| Checkpoint | selected on validation inside the shared loop (`TrainConfig.keep_best`) |
| Optimiser | AdamW, betas (0.9, 0.999), wd 1.4e-6, clip 1.0, warmup 100, cosine to 0.1x |
| Dropout | 0.0 for every model (PT has no specified dropout placement) |
| Metric | validation perplexity at the selected checkpoint; test measured on the same one |

## Grids

* **Part A, `spec_baselines.py`** — GPT and Looped over `n_embd in {16..256}` at `n_layer=4`,
  and over `n_layer in {1,2,8}` at `n_embd=64`, each at `lr in {1e-3, 3e-3}` with the better
  of the two taken as the curve point. 48 cells.
* **Part B, `spec_pt.py`** — the PT curve, written *after* exp4 fixes how PT's constants move
  with `d`. A PT ladder run at one fixed `alpha_Z` measures the mis-transfer of `alpha_Z`
  rather than the architecture; the 2026-08 ladder (`d = 48` worse than both 32 and 64) is
  what that looks like.

## Falsification

* If the Looped curve lies *below* the GPT curve at PT's budget, then weight sharing is a win
  at this scale, and Experiment 2's framing ("Looped has PT's weight sharing without its
  structure, so the comparison isolates structure") needs restating in the other direction.
* If both baseline curves are flat below ~100k parameters, PT's budget region is one where
  nothing works well and the comparison says little about architecture.
* If PT's curve is *parallel* to the baselines but offset, the cost of the construction is a
  constant factor and can be quoted as one. If it is *steeper*, PT scales worse and the
  honest conclusion is that the construction does not carry to larger budgets.

## Run log

| date | commit | job | shard | cells | notes |
|---|---|---|---|---|---|
| 2026-09-08 | 9a9527a | 998612-998615 | 0-3 of 4 | 48 | baselines, first submission |

## Baselines — complete (2026-09-08, jobs 998612–998615, 48/48 cells)

15,000 steps, blocks 16 x 64, validation checkpoint selection, best of `lr ∈ {1e-3, 3e-3}`.

| `d_model` | GPT val | GPT test | GPT non-emb | Looped val | Looped test | Looped non-emb |
|---|---|---|---|---|---|---|
| 16  | 195.23 | 181.04 | 13,152    | 214.20 | 196.46 | 3,312   |
| 24  | 159.41 | 147.18 | 28,944    | 183.09 | 168.44 | 7,272   |
| 32  | 144.79 | 133.85 | 50,880    | 164.75 | 150.07 | 12,768  |
| 48  | 133.89 | 122.28 | 113,184   | 149.98 | 137.17 | 28,368  |
| 64  | 129.25 | 118.20 | 200,064   | 141.78 | 129.67 | 50,112  |
| 96  | 128.76 | 118.72 | 447,552   | 136.41 | 125.91 | 112,032 |
| 128 | 126.45 | 118.09 | 793,344   | 135.67 | 124.43 | 198,528 |
| 160 | 124.69 | 116.59 | 1,237,440 | 133.77 | 125.34 | 309,600 |
| 256 | 125.27 | 116.43 | 3,159,552 | 128.41 | 120.66 | 790,272 |

Depth family at `d_model = 64`: GPT `L=1/2/4/8` → 141.45 / 135.12 / 129.25 / 124.46.

**Two features of the baselines reshape the whole comparison, and both were pre-registered as
falsification conditions in this file.**

1. **The transformer's own curve is flat above ~2×10⁵ non-embedding parameters** (129.2 at
   `d_model=64` against 124.7 at 160 and 125.3 at 256), while its training perplexity falls
   from 68.5 to 10.2 over the same range. On 929,589 training tokens the baseline stops being
   parameter-limited and becomes data-limited. A comparison made only at the top of that range
   compares two models in a regime where one has stopped improving — which is what the 2026-08
   report did when it put PT's 16,448 non-embedding parameters against GPT's 1,237,440.
2. **At matched non-embedding budget the Looped transformer beats the standard one below about
   5×10⁴ parameters** (164.75 at 12,768 against 195.23 at 13,152; 149.98 at 28,368 against
   159.41 at 28,944) and loses above it (135.67 at 198,528 against 129.25 at 200,064). At
   matched *total* parameters the standard transformer wins everywhere. Both statements are
   true and they are about different axes, which is why both axes are plotted.

Consequence for the framing of Experiment 2: at this scale weight sharing is not a handicap
that the causal PT pays in exchange for structure — it is a saving. That makes the
PT-versus-Looped comparison the *sharper* test rather than the more forgiving one, and it means
"causal PT loses to GPT" cannot be attributed to sharing.

## Run log

| date | commit | job | shard | cells | notes |
|---|---|---|---|---|---|
| 2026-09-08 | 9a9527a | 998612-998615 | 0-3 of 4 | 48/48 | baselines, complete, no failures |
| 2026-09-08 | 2afa95d | queued | 0-5 of 6 | 42 | PT ladder: control at the 2026-08 constants, and the transfer rule at gain 1 undamped |

## The PT ladders (2026-09-08)

### Part B — the width ladder at `rank = d`, 42 cells, three seeds each

| `d` | control (source constants) | rule (standardised temp., undamped) |
|---|---|---|
| 16  | 282.3 ± 0.6 | 281.2 ± 8.5 |
| 24  | 263.5 ± 7.3 | 262.5 ± 13.6 |
| 32  | **253.0 ± 6.0** | 246.7 ± 2.7 |
| 48  | 287.4 ± 52.5 | **229.8 ± 8.5** |
| 64  | 470.0 ± 160.7 | **219.8 ± 4.2** |
| 96  | 424.0 ± 205.5 (1 collapsed) | 372.4 ± 277.0 (1 collapsed) |
| 128 | 515.3 ± 159.0 (1 collapsed) | 508.9 ± 186.9 (1 collapsed) |

The control reproduces the 2026-08 record at `d = 32` (253.0 ± 6.0 against 250.7 ± 2.3), which
is the check that the two sessions are measuring the same thing. Above that width it does not
gently worsen — **it becomes unpredictable**, and the standard deviations say so more clearly
than the means. The rule extends the usable width to `d = 64` and improves the best from 253.0
to 219.8, then hits a second ceiling at `d ≥ 96`.

### Part C — the same ladder at low Kruskal rank, and the ceiling goes away

| `d` | `r` | non-emb | val |
|---|---|---|---|
| 32  | 8  | 4,160   | 255.7 |
| 64  | 8  | 8,320   | 230.5 |
| 96  | 12 | 18,624  | 217.6 |
| 128 | 16 | 33,024  | **212.8** |
| 192 | 24 | 74,112  | **208.0** |
| 256 | 32 | 131,584 | 208.8 |

**Two things at once.** The curve is uniformly *below* the `rank = d` ladder — 217.6 at 18,624
parameters against 372 ± 277 for `rank = 96` at 147,648 — and the width ceiling that stopped
both earlier ladders at `d = 96` is gone: `d = 192` trains stably and only at `d = 256` does the
curve turn over, on capacity rather than on collapse.

Read together with exp4d, the reading is that a full-rank arc table gives the runaway loop more
directions to run away in. Nothing in the source suggests `rank = d`; it is what Wu & Tu's
Table 2 rule `min(64, d)` produces at every width this project has used, and it was carried
along unexamined for the whole of 2026-08.

### Matched budget, which is what the reviewers asked for

Baselines interpolated in log-log space to the causal PT's own parameter count:

| causal PT | non-emb | val | transformer | ratio | looped | ratio |
|---|---|---|---|---|---|---|
| `d=96, r=12`  | 18,624  | 217.6 | 178.5 | **1.22** | 157.6 | 1.38 |
| `d=128, r=16` | 33,024  | 212.8 | 155.9 | 1.37 | 147.7 | 1.44 |
| `d=192, r=24` | 74,112  | 208.0 | 139.6 | 1.49 | 139.1 | 1.50 |
| `d=256, r=32` | 131,584 | 208.8 | 132.6 | **1.57** | 136.2 | 1.53 |

The gap is smallest where the models are smallest and widens with budget. That is what the
shallower exponent means in practice, and it is the honest form of the comparison: the 2026-08
report's single point gave 2.17 by comparing a 16k-parameter PT against a 1.2M-parameter GPT.

**Superseded 2026-09-08 (evening). The table above is seed 0 only; the row is kept because it is
what was read at the time.** Every causal PT figure in the paper is now a three-seed mean, which
moves the ratios: `d=96, r=12` is 223.1 ± 5.6 not 217.6, so its ratio against the transformer is
1.25 and not 1.22, and `d=192, r=48` at 201.9 ± 4.9 is now the best point rather than
`d=256, r=32`. Reading a seed minimum as the result is the specific error the reproduction check
of this session was run to prevent, and it had reached this table before the check was applied
to it. The live version is generated by `experiments/make_report.py` into
`paper/figures/table_matched.tex`; nothing here should be quoted in preference to it.

**Also superseded: "the curve turns over on capacity rather than on collapse"** (line 155). With
three seeds, `d=256, r=64` has one seed on the unigram, so the ladder's top is a collapse after
all — the ceiling moved with the rank rather than lifting. Recorded rather than edited, because
a check that passed and later broke is worth more in this file than a tidy one.
