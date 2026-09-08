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

## First reading (2026-09-08, grids A–D partial, seed 0, 6,000 steps)

**Grid A — `alpha*(d)` falls with width, as predicted.** Argmin over the damping grid:
`d=16 → 0.25` (322.5), `d=24 → 0.125` (281.2), `d=48 → 0.25` (295.2), `d=64 → 0.125` (375.5).
Undamped, `d ∈ {32, 96, 128}` sit on the unigram (700.7 / 698.1 / 701.4) with the collapse
signature the 2026-08 report described: the message runs away and then the label beliefs go
flat (`d=32, α=1`: msg/unary 1.72 → 16.73 → 1.06, belief sharpness 1.09 → 5.57 → 1.34, ablation
KL exactly 0).

**Grid B — the standardised head temperature revives widths that are dead at `lambda_H = 1/d`,
with no damping at all.** At `alpha_Z = 1`:

| `d` | `fixed` (Wu & Tu) | best standardised | gain |
|---|---|---|---|
| 32 | 700.73 (unigram) | **293.59** | 2 |
| 48 | 553.31 | **312.19** | 4 |
| 96 | 698.14 (unigram) | **234.35** | 2 |

234.35 at `d = 96` is the best number in the project — undamped, at 6,000 rather than 15,000
steps, against the 2026-08 record of 250.69 ± 2.32. Its trace falls monotonically from the
first evaluation (453 → 234) with belief sharpness rising to 3.65 and msg/unary near 1.

**But the gain does not obviously transfer.** The winning gain was 2 at `d=32` and `d=96`, 4 at
`d=48`, 8 at `d=16`, and the losing gains landed exactly on the unigram with a *different*
failure signature from the undamped collapse: the message *decays* (7.88 → 0.46 at `d=96,
gain=4`) and the beliefs stay uniform (sharpness 1.04), so the model never leaves the
initialisation rather than blowing up and falling back. One seed per cell cannot separate that
from the chaos the reproduction check found, so `spec_gain.py` re-runs `d × gain` at three
seeds, with the prediction registered first: under standardisation the attention entropy
depends on the gain and the context length and **not on `d`**, so the argmin should be flat.

`spec_lr_transfer.py` adds the muP-style diagnostic — validation loss against learning rate at
four widths, under both temperatures — because "the optimal learning rate stops moving with
width" is the operational form of the transfer claim.

## Run log

| date | commit | job | shard | notes |
|---|---|---|---|---|
| 2026-09-08 | 14419ae | 998622-998625 | 0-3 of 6 | first submission, 89 cells |
| 2026-09-08 | 14419ae | 998625 | 3 of 6 | **DIED** — CUDA "device busy or unavailable"; three jobs on `ai_gpu28`. 15 error rows in 90 s. Fixed in the runner (abort on CUDA error; failed rows no longer count as done) and the batch script (allocate and matmul before training). |
| 2026-09-08 | 2c58a7a | 998656, 998664, 998671 | 4, 3 of 6 | died in 2 s on the new allocation check — the fix working as intended. Re-queued automatically by `experiments/requeue.py`. |
| 2026-09-08 | 2c58a7a | 998658, 998662 | 5 of 6 | **submitted twice by hand**; two jobs interleaved writes to one JSON. 998662 cancelled; the runner now takes a per-shard lock naming the slurm job that holds it. |
| 2026-09-08 | e1fdac8 | 998675-998677 | 0-2 of 4 | `spec_gain`, 72 cells, three seeds |
