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
