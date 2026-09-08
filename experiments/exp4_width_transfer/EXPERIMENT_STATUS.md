# Experiment 4 — width transfer: what has to move with `d`, and why

## Question

**Which of the causal PT's constants have to be retuned when the label-set size `d` changes,
and is the required change a law or a fit?**

A law is a transfer rule and turns the scaling curve of Experiment 2 into one sweep. A fit
turns it into a grid per point, which is what Penghao Kuang warned about:

> "Hyperparameter tuning for PT is indeed a cumbersome task. The original PT architecture
> involves seven hyperparameters that drift with changes in parameter count (whereas
> Transformers typically involve only one: the learning rate)."

The 2026-08 report leaves this open in two entries at once: `alpha*(d)` ("does the optimal
step size scale with width, and where is the ceiling under damping?") and the matched-budget
comparison, which is blocked on it.

## The mechanism under test, stated before the runs

Wu & Tu App. A.5 fixes the head-update temperature at `lambda_H = 1/d`. The head logit is
`F_c(i,j) = <q_i, B^(c)_{j,.}>`, and **both** vectors are expectations under label
*distributions*. A distribution on `d` labels has Euclidean norm in `[1/sqrt(d), 1]`. At
initialisation both beliefs are near uniform, so `F ~ sigma_T / d` and dividing by `1/d`
leaves an `O(sigma_T)` logit: the source's choice is exactly right there. Once training
sharpens the beliefs, `F ~ sigma_T` and the same division leaves `O(d sigma_T)`. **The
temperature is calibrated for a limit the model leaves within a few hundred steps, and the
error is linear in `d`.**

Three consequences, tested in order:

1. the runaway `q_bar -> B -> G -> q_bar` loop should worsen with `d` (already measured:
   `d = 32` undamped goes msg/unary 2.88 -> 20.57 within 500 steps);
2. `alpha*(d)` should *fall* with `d`, because damping the label update is what keeps
   `||q||_2` near the uniform value the temperature is calibrated for — **Grid A**;
3. a temperature that tracks the belief norm should remove the `d`-dependence at source, and
   then `alpha*(d)` should flatten to a constant — **Grids B and C**.

`temp_mode="qnorm"` divides the logit by `||q_i||_2 sqrt(d)`, which is exactly 1 at the
uniform belief: it **agrees with Wu & Tu at initialisation** and differs only where the
source's derivation does not reach. `temp_mode="qknorm"` standardises the logits over the
head domain and is the aggressive version.

## Held fixed

PTB, `h = 2` (except Grid D), `gamma = 3`, `T = 3`, `tau = 2`, MFVI readout, frozen `b` at
the corpus log-unigram, `lr = 0.02`, `rank = d`, `l2_arc = 5e-4`, blocks 16 x 64, 6,000
steps, seed 0, validation checkpoint selection. `rank = d` makes the non-embedding count
exactly `2 K h d^2 + h d`, so the width axis is a clean `d^2` law and `d = 32` reproduces the
2026-08 record configuration at 16,448 non-embedding parameters.

6,000 rather than 15,000 steps because this is a comparative map; the 2026-08 runs rank the
same way at both budgets (`d = 16`: 473.28 at 6,000, 336.89 at 15,000). Winners are re-run at
full budget.

## Falsification

The mechanism is wrong if **either**

* `alpha*(d)` in Grid A is flat — the drift is then not width-driven at all; **or**
* `qnorm` in Grid B does not raise the collapse width above `d = 32` while `fixed` at the
  same `alpha_Z = 1` collapses there, as previously measured.

Both outcomes are recorded whichever way they fall.

## Run log

| date | commit | job | shard | notes |
|---|---|---|---|---|
| 2026-09-08 | 14419ae | 998622-998625 | 0-3 of 6 | first submission, 89 cells |
| 2026-09-08 | 14419ae | 998625 | 3 of 6 | **DIED** — CUDA "device busy or unavailable"; three jobs landed on `ai_gpu28` and this one could not allocate. Wrote 15 error rows in 90 s. Fixed in the runner (abort the shard on a CUDA error instead of continuing; failed rows no longer count as done on resume) and in the batch script (allocate and matmul before training). Resubmitted. |
