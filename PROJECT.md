# Causal Probabilistic Transformer

## Context

Research internship at ShanghaiTech University, NLP group of **Prof. Kewei Tu**.
Author: Viktor Beliakov (first-year undergraduate, MIPT).
Senior collaborator: **Penghao Kuang** (PhD student) — provides experimental guidance.

Timeline: internship ends late August 2026. Remote collaboration afterwards is intended
if sufficient progress is demonstrated.

## Research topic

Extend the **Probabilistic Transformer (PT)** (Wu & Tu) into a **causal autoregressive decoder**.

The original PT is a CRF-based *encoder*: word representations and syntactic dependency
structure are modelled jointly as a factor graph, and inference is performed by
**Mean Field Variational Inference (MFVI)**. Running MFVI to convergence produces a
computation graph strikingly similar to a transformer encoder — attention weights are
*derived* from the factor graph rather than learned as free parameters.

The original PT is masked-LM only. **No causal / autoregressive version exists.**
That gap is the contribution of this project.

## What is already done (theory — complete)

1. **Impossibility result.** Causality, a single global energy function, and contextuality
   cannot coexist. Resolution: a directed chain of *conditional* CRFs, where the prefix
   enters each step as a frozen condition, so the anti-causal term is never created
   (rather than being deleted after the fact).

2. **Graph-faithful output mechanism.** An earlier design projected the label posterior
   through a fresh `d × |V|` matrix. This was rejected by Prof. Tu: such a matrix
   corresponds to no factor in the graph, so the output does not traverse the factor graph.
   The fix: promote the word variable `W_t` from *observed* to *latent*. The existing
   word–label factor `S` was always a binary factor with the word clamped; making `W_t`
   latent simply **unclamps** it. No new factor is introduced.

   Consequences that follow automatically (not design choices):
   - input/output embedding tying is *forced* — one factor mediates both directions;
   - the LM-head bias equals the word unary term;
   - encoding mode and decoding mode are two conditioning patterns of **one** model.

3. **Exact inference is available.** The per-slot graph is a **star** centred on the label
   variable `Z_t`, hence a tree. Sum-product / variable elimination gives a closed-form
   readout (a mixture-of-softmaxes) in addition to the MFVI readout. Prof. Tu independently
   raised comparing exact vs. mean-field inference empirically.

**Status: the theoretical construction has been reviewed and approved by Prof. Tu.**

## What remains: experiments

The theory is written. What is missing is empirical evidence that the construction
actually trains and produces sensible language modelling numbers.

### Framing

The goal is **not** to beat GPT on perplexity. The original PT paper states explicitly that
its aim is not to compete with transformers but to inform and extend them; it also reports
that PT underperforms on large datasets (>100k sentences), likely due to the absence of a
feed-forward structure. This project inherits that framing.

The scientific question is: **does syntactic structure pay for itself in a causal decoder?**

### Three experiments

| # | Comparison | Question answered |
|---|-----------|-------------------|
| 1 | Causal PT vs. GPT-style decoder | Does the causal PT train at all, and is perplexity in a reasonable corridor? |
| 2 | Causal PT vs. **Looped Transformer** | Does syntactic structure contribute *beyond* weight sharing? |
| 3 | **Exact readout** vs. **MFVI readout** | What is the cost of the mean-field approximation? |

**Experiment 2 is the core result.** PT differs from GPT in two ways at once: it shares
parameters across iterations, *and* it has a syntactic factor graph. A Looped Transformer
(a single transformer block applied `T` times with shared weights) has the weight sharing
but **not** the structure. Therefore:

- Looped vs. GPT isolates the effect of weight sharing;
- **PT vs. Looped isolates the effect of structure**, which is the claim of the paper.

Without Experiment 2, a reviewer can object that any gain is merely a weight-sharing effect.

**Experiment 3 is unique to this model.** A standard transformer has no exact/approximate
choice; the tree structure of the PT slot graph creates one.

### Success criteria

A publishable outcome is: **causal PT is on par with GPT and no worse than Looped at equal
parameter budget** — i.e. structure carries information without costing quality.
Losing slightly to GPT on raw perplexity is acceptable and expected; there is an honest
rank-`d` softmax bottleneck (all predictive information flows through the label variables).

### Experimental setup

- **Scale:** 20–50M parameters (specified by Penghao Kuang).
- **Data:** a *small* corpus — PTB or WikiText-2. **Not WikiText-103**: 103M tokens is
  precisely the regime where the original paper reports PT failing.
- **Baselines:** nanoGPT off the shelf; Looped = nanoGPT with a single shared block looped
  `T` times (a few lines of change).
- **Only the PT decoder forward pass is written from scratch.** The training loop is written
  once and shared across all three models.
- **Readout:** MFVI is the mainline implementation; exact readout is added afterwards as
  Experiment 3 (one additional function, not a rewrite).

## Deliverable

A **preprint** describing the causal Probabilistic Transformer, to be extended into a full
conference submission. The experiments above supply the empirical section.

## Reference documents

- `causalprobabilistictransformer_1.pdf` — main document, Parts I–IV.
  Part III (§15–§18) contains the output mechanism and the MFVI update equations.
  §17.1 gives the update schedule; §17.2 the exact-readout variant.
  **Note:** §17.2 recommends MFVI as mainline, but §23.3 in Part IV walks this back
  toward exact readout — both must be read before finalising the implementation.
- `probalistic_transformers_article.pdf` — original Wu & Tu Probabilistic Transformer paper.

## Constraints for implementation

- Do not introduce parameters that correspond to no factor in the graph. Every learned
  matrix must be a factor. This was the exact failure of the rejected output mechanism.
- Input and output word embeddings **must** be tied — this is forced by the model, not a
  regularisation trick.
- The prefix enters each decoding step as a *frozen condition*; gradients and messages must
  not flow backwards from step `t` into the prefix posteriors.
- Verify correctness on a toy example (small `d`, small vocabulary, ~4 words) with printed
  intermediate tensors before scaling up.
