# Research Plan — Causal Probabilistic Transformer

Scope: empirical section of the preprint. The theoretical construction is complete and
approved; what follows establishes that it trains and that its structure earns its place.

**Central question of the paper:** does syntactic structure pay for itself in a causal decoder?

Not "does PT beat GPT". Losing slightly to GPT on raw perplexity is expected and honest —
there is a rank-`d` bottleneck, since all predictive information flows through the label
variables. The claim is about *structure*, not about *winning*.

---

## Experiment 0 — Implement and validate the causal PT decoder

**Nothing else exists until this passes.** This is the only code written from scratch.

### Goal

A working causal PT decoder forward pass, validated on a toy scale, with MFVI readout.

### Scale

Deliberately tiny, CPU-runnable: `d` small (e.g. 8–16), vocabulary ~20 tokens, sequence
length ~4–8, batch size 1–4, 2–3 MFVI iterations. The point is *correctness*, not learning.

### Validation checks (all must pass before scaling)

1. **Shapes.** Every intermediate tensor (`Q_W`, `Q_Z`, `Q_c`, contracted prefix scores)
   has the dimensionality stated in Part III. Print after each of the six update steps.

2. **Normalisation.** All posteriors sum to 1 along their variable axis, at every iteration.
   A drift here means a misplaced `softmax` axis.

3. **Causality.** Change the token at position `t`; logits at all positions `< t` must be
   **bitwise unchanged**. This is the single most important test — it is exactly what the
   model claims and exactly what is easy to break with an off-by-one in the mask.

4. **Prefix is frozen.** No gradient flows from step `t` backwards into prefix posteriors.
   Check by inspecting `.grad` on detached prefix quantities.

5. **Tying.** The word embedding matrix used on input is *the same tensor object* as the one
   used in the output factor. Not a copy — the same parameter.

6. **Overfit a single batch.** Train on one batch for a few hundred steps; loss must fall to
   near zero. If it plateaus, the model cannot represent its own training data and something
   in the update equations is wrong.

7. **Worked example.** A 4-word sequence with hand-set numbers, all intermediate tensors
   printed. This doubles as the numerical example referenced in correspondence with the
   supervisors, and is what to show if the derivation is probed.

### Exit criterion

Checks 1–7 pass. Loss on a single batch goes to ~0. Only then move to real data.

### Note on readout

MFVI readout is the mainline implementation here. Exact readout is Experiment 3 and is a
single additional function over the same trained machinery — not a second model.
(§17.2 recommends MFVI as mainline; §23.3 in Part IV walks this back toward exact readout.
Both sections must be read before the implementation is frozen.)

---

## Shared experimental setup (Experiments 1–3)

Everything below is held constant across all models. Any difference other than the one
being tested invalidates the comparison.

| Item | Setting |
|---|---|
| Data | PTB or WikiText-2 (small corpus) |
| Parameter budget | 20–50M, matched across models |
| Tokenizer / vocabulary | identical |
| Context length | identical |
| Training loop | one implementation, shared |
| Optimizer | Adam, identical hyperparameters and schedule |
| Random seeds | multiple runs; report mean ± std |
| Metric | perplexity on held-out set |

**On the dataset choice.** WikiText-103 is deliberately excluded. The original PT paper
reports that the model significantly underperforms transformers on datasets beyond ~100k
sentences, suspected to be the absence of a feed-forward structure. Running at that scale
would test a known failure mode rather than the causal extension.

**On matching budgets.** Report both parameter count *and* wall-clock / FLOPs. PT shares
parameters across iterations, so equal parameter count does not mean equal compute; being
explicit about both pre-empts the obvious reviewer objection.

---

## Experiment 1 — Causal PT vs. GPT-style decoder

**Question:** does the causal PT train, and is its perplexity in a reasonable corridor?

**Models:** causal PT decoder; standard GPT-style decoder (nanoGPT), matched budget.

**What differs:** everything — this is not a controlled comparison. It measures the *total
cost of the construction*, not the effect of any single component.

**Success criterion:** PT converges stably and lands within a modest gap of GPT. A small
gap is a good result given the bottleneck; a catastrophic gap indicates an implementation
problem, not a scientific finding.

**Why it is necessary but not sufficient:** without it there is no paper. With only it, a
reviewer asks "so what — another decoder that is slightly worse". That objection is what
Experiment 2 answers.

**Cost:** baseline is off the shelf; the expense is GPU time, not engineering.

---

## Experiment 2 — Causal PT vs. Looped Transformer  ★ core result

**Question:** does syntactic structure contribute *beyond* weight sharing?

### Why this comparison exists

PT differs from GPT in **two** ways simultaneously:

1. it applies one shared parameter set repeatedly across iterations (weight sharing);
2. it has a syntactic factor graph with label variables (structure).

Comparing PT to GPT cannot separate these. If PT does well, the cause is ambiguous.

A **Looped Transformer** is a single transformer block applied `T` times with shared
weights — property (1) without property (2). This gives a three-point comparison:

| Model | Weight sharing | Syntactic structure |
|---|---|---|
| GPT | no | no |
| Looped Transformer | **yes** | no |
| Causal PT | **yes** | **yes** |

- **Looped vs. GPT** isolates the effect of weight sharing.
- **PT vs. Looped** isolates the effect of structure. This is the claim of the paper.

### Controls

Number of loop iterations `T` in the Looped baseline should match the number of MFVI
iterations in PT, so that the two have comparable effective depth as well as comparable
parameter counts. Report results across a small sweep of `T` rather than a single value —
otherwise the comparison is vulnerable to a lucky choice.

**Success criterion:** PT is no worse than Looped at matched budget. Then the paper can
state that syntactic structure carries information not explained by weight sharing alone.

**Implementation cost:** low. Take nanoGPT, replace the list of distinct blocks with one
block applied in a loop:

```python
# standard: a stack of distinct blocks
for block in self.blocks:
    x = block(x)

# looped: one shared block, applied T times
for _ in range(self.T):
    x = self.block(x)
```

---

## Experiment 3 — Exact readout vs. MFVI readout

**Question:** what does the mean-field approximation cost?

### Why this is available at all

The per-slot graph is a **star** centred on the label variable `Z_t`, hence a tree.
Sum-product / variable elimination therefore yields a closed-form exact readout — a
mixture-of-softmaxes — as an alternative to iterating MFVI to convergence.

A standard transformer offers no such choice. This experiment is specific to this model and
demonstrates understanding of the construction rather than benchmark-running. Prof. Tu
independently raised exact vs. mean-field comparison as worth doing empirically.

### Two variants, to be stated explicitly in the write-up

- **(a) Evaluation-time swap.** Train with MFVI readout; at evaluation, substitute the exact
  readout on the same trained parameters. Isolates the approximation error of the readout.
- **(b) Train with each.** Train one model with MFVI readout and one with exact readout.
  Measures the end-to-end effect, including how the approximation shapes learning.

(a) is nearly free and should be done first. (b) costs a full training run.

**Success criterion:** there is no "winning" outcome here — either result is informative.
A small gap says mean-field is adequate and cheap. A large gap says the tree structure
should be exploited and is a finding in its own right.

**Note:** the mixture-of-softmaxes readout is also relevant to the rank-`d` bottleneck
discussed in Part IV, since a mixture of softmaxes is not rank-limited in the same way a
single softmax is. Whether this shows up empirically is worth checking.

---

## Priority and cut order

Under time pressure, cut from the bottom:

1. **Experiment 0** — mandatory. Nothing exists without it.
2. **Experiment 1** — mandatory. Establishes the model trains on real data.
3. **Experiment 2** — the scientific core. Cutting it weakens the paper's central claim.
4. **Experiment 3** — variant (a) is cheap and should survive most cuts; variant (b) is the
   first thing to drop.

A minimum viable result is: *the causal PT decoder trains, and its perplexity is in a
reasonable corridor relative to a matched GPT baseline.* Everything beyond that strengthens
the paper; nothing beyond that is a substitute for it.

## Reporting

For each experiment: model, parameter count, compute, dataset, mean ± std over seeds,
training curves. Negative and neutral results are reported as found — the framing of the
paper does not require PT to win, and overselling a marginal gap is the fastest way to lose
a reviewer.
