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
