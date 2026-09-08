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
