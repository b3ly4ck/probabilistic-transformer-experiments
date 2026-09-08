# The evaluation-time readout swap — a negative result

Every causal PT cell in this project scores its own trained weights under the *other* readout at
the cost of one extra pass (`experiments/sweep.py`; only at `lambda_W = 1`, and with the global
head excluded, since its contribution to the exact readout is a measured constant).

The result is unambiguous and it holds at every width and rank measured:

| `d` | `r` | trained (MFVI) | swapped to exact | factor |
|---|---|---|---|---|
| 32  | 8  | 255.7 | 4361.4 | 17.1 |
| 48  | 12 | 238.8 | 4834.1 | 20.2 |
| 64  | 8  | 232.0 | 4598.2 | 19.8 |
| 64  | 16 | 223.4 | 5434.2 | 24.3 |
| 96  | 12 | 223.1 | 1492.5 | 6.7 |
| 128 | 16 | 214.6 | 1671.9 | 7.8 |
| 192 | 24 | 206.9 | 1859.3 | 9.0 |
| 256 | 32 | 206.1 | 2293.1 | 11.1 |

**The two readouts are not interchangeable at evaluation.**

## What it is not

A first hypothesis — that the exact readout's log-sum-exp over a growing head domain concentrates
and washes out the label signal — was tested and **rejected**. Measuring the spread of the
label-dependent term across labels as the head domain grows, at initialisation:

| `d` | `h` | t=1 | t=4 | t=16 | t=63 |
|---|---|---|---|---|---|
| 32 | 2 | exact 0.078 / MFVI 0.083 | 0.030 / 0.032 | 0.022 / 0.022 | 0.022 / 0.022 |
| 96 | 2 | exact 0.075 / MFVI 0.073 | 0.031 / 0.030 | 0.012 / 0.012 | 0.013 / 0.013 |

Both readouts concentrate identically, so the decay is a property of the contracted arc scores
and not of the log-sum-exp pooling. The hypothesis is recorded because it was the obvious one
and it is wrong.

## What it is

On a trained model (`d=96, r=12`), the exact readout's logits are **not** degenerate. They
discriminate *less* between words at a fixed context (sd 1.63 against the mean-field readout's
2.64) and *more* between contexts for a fixed word (2.73 against 2.42). Parameters trained so
that one readout is accurate do not make the other accurate.

That is unsurprising once stated, but it is not what a reader would assume from a construction
that presents the two as alternative inferences in one model — and Part IV §23.3 recommends the
exact readout as the mainline partly on the grounds that it is a drop-in. It is not a drop-in.

## Consequence

The cheap half of Experiment 3 does not substitute for the expensive half. The readouts must be
compared by **training** under each, which is `spec_open.py` grid X.
