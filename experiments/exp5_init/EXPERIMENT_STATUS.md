# Experiment 5 — initialisation, and the fate of the B.3.3 global head

## Questions, verbatim from the reviewers

> **Prof. Kewei Tu, 2026-08-11:** "B.3.3 global head: Uniform initialisation (of what? B?)
> does not sound like a good choice. How does it work with random initialisation?"

> **Penghao Kuang, 2026-08-14:** "Regarding the hyperparameter initialization scheme, I
> noticed that you are using uniform distribution initialization. In fact, when attempting to
> derive quantitative laws, the initialization method is also a critical component. ...
> Therefore, parameter initialization is not only relevant to experimental outcomes but may
> also hold underlying theoretical significance for PT, given its rigorous mathematical
> foundations."

## A correction that is owed first

**Every factor in this implementation has always been initialised from a Gaussian**
`N(0, init_std^2)` with `init_std = 0.02` (the nanoGPT convention; neither source paper
specifies one). The word "uniform" in the 2026-08 report named the *treatment* — `B'` given
the same scale and the same L2 coefficient as every other factor — not the distribution. It
should have said "symmetric" or "undifferentiated". The report is wrong in its wording, the
runs are what they are, and the correction goes in the corrections log.

The literal question is a good one anyway and is now answerable: `PTConfig.init_dist` takes
`normal`, `uniform` (matched variance, `U(-sqrt(3) s, sqrt(3) s)`) and `orthogonal`.

## The mechanism the grids test

`Q_g = softmax(q B'^T / lambda_G)` and the message back is `sum_k Q_g(k) B'_{k,.}`. If the
rows of `B'` are close together relative to their own length, the logits are nearly equal,
`Q_g` is uniform, the message is the row mean, and `d(message)/d B'_{k,.} = 1/m` for **every**
`k` — an exactly symmetric gradient that cannot separate the rows. The only row-separating
term comes through the softmax Jacobian and is of the order of the row spread itself; a
shared L2 shrinks that spread further. The degeneracy is self-locking from any near-symmetric
start. The two things that set "near-symmetric" are the initialisation scale and the penalty,
and Grid A varies exactly those.

Measured at `d = 16` in 2026-08: within-row std 0.904, across-row spread fallen 0.0197 ->
0.0037, logit range over 64 features 0.05, `H(Q_g)` uniform to four decimals.

## Grids

* **A** — `b_glob_init_std in {0.02 .. 1.0}` x `regularise_global_head in {on, off}` at
  `d in {24, 32}`, `m = 64`, plus the no-head reference at each width. 26 cells. This is the
  direct answer to Prof. Tu: two decades of random-initialisation scale, with and without the
  L2, with the row spread and `H(Q_g)` logged so the verdict is mechanistic.
* **B** — `m in {16, 64, 256}` at the two treatments. `m = 256` was never run in 2026-08.
* **C** — `init_dist in {normal, uniform, orthogonal}`. Orthogonal is the *maximally
  asymmetric* initialisation at a fixed scale, so if the symmetry argument is right, an
  orthogonal `B'` should survive the shared L2 that kills a Gaussian one. That is a sharp,
  falsifiable prediction of the mechanism, not a sweep.
* **D** — `init_std` at `d = 32`. Varied in 2026-08 only at `d = 256, lr = 1e-3`, i.e. two
  axes outside the working region.
* **E** — `arc_init_std` following `const`, `sqrt(32/d)` and `32/d` across `d`: the
  initialisation half of the transfer question (exp4 is the temperature half).

## Falsification

The symmetry account is wrong if an orthogonal `B'` under the shared L2 dies exactly as the
Gaussian one does (Grid C), or if the head's survival in Grid A is independent of the
initialisation scale.

## Run log

| date | commit | job | shard | notes |
|---|---|---|---|---|
| 2026-09-08 | 14419ae | queued | 0-3 of 4 | 59 cells, queued behind exp4 |

## First pass — and why it had to be re-run (2026-09-08, `spec_init.py`, 59 cells)

Grid A varied the global head's initialisation scale and L2 treatment at **one seed per cell, at
the source temperature, undamped** — the region the reproduction check independently showed has
a seed standard deviation of **130 perplexity**. The grid duly came back unreadable. At `d = 32`
the same head at `b_glob_init_std = 0.02` gives 337.7 without the L2 and 465.0 with it; at 0.1,
587.2 and 675.7; at 0.25, 383.2 and 323.0. The ordering reverses twice. No conclusion about the
initialisation scale can be drawn from it, and none is drawn.

**Two things in it are nonetheless unambiguous and are carried forward.**

1. **The head rescues a collapsing configuration.** At `d = 32` the reference *without* the head
   sits exactly on the unigram (700.7), while several cells *with* it land between 323 and 380.
   A second message into `Z_t` that does not pass through the arc factors evidently breaks the
   runaway loop. That is a mechanism claim; `spec_globalhead.py` measures it at three seeds in
   the stable configuration.
2. **`H(Q_g)` is 1.000 in almost every cell.** The head posterior is uniform, so its message is
   the row mean of `B'` — a constant vector added to the label logits, i.e. a **learned label
   prior**, not a feed-forward-like operator. The head can help perplexity while doing nothing
   the B.3.3 proposal was for. Perplexity alone cannot see this distinction, which is why
   `qg_entropy_frac` is reported beside it in every table.

## An unplanned but decisive by-product

`C_normal_d16` of grid C is bit-for-bit the configuration of job 940848 of 2026-08-10, and it
landed on `ai_gpu06`, an **A40** — the same GPU model as the original. It reproduced that run's
entire validation trace to every printed digit, ending at **430.04**. The same configuration on a
TITAN RTX ends at 645.0. See `experiments/exp3_readout/EXPERIMENT_STATUS.md`; this is the cell
that answered the hardware question and let the A40-pinned control be cancelled.

## Grid E — the initialisation half of the width rule

`arc_init_std` following `const`, `sqrt(32/d)` and `32/d` across `d`, one seed, same
high-variance region: the completed cells scatter (363.2 / 673.7 / 606.6) and support no rule.
The width mechanism this project found lives in the head *temperature*, not in the arc-score
initialisation scale, and exp4b/4c are where that is measured. Grid E is reported as
inconclusive rather than quietly dropped.

## Run log

| date | commit | job | shard | notes |
|---|---|---|---|---|
| 2026-09-08 | 14419ae | 998660, 998830-998833 | 0-3 of 4 | 59 cells; single seed, source temperature, undamped — superseded for the global-head question by `spec_globalhead.py` |
| 2026-09-08 | cc79156 | queued | 0-3 of 4 | `spec_globalhead.py`, 54 cells, three seeds, standardised temperature |
