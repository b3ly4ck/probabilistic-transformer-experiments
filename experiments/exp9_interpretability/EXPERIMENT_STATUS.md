# Experiment 9 — what the latent variables learn

## Question

This model class is proposed because its latent variables mean something: labels are syntactic
categories, head posteriors are dependency attachments. A paper reporting only perplexity offers
no evidence for the one property that distinguishes it from a transformer. Wu & Tu spend their
Appendix F on exactly this and report honestly that the induced structures are only partially
consistent with intuition.

Three probes, all read off a trained checkpoint with no further training
(`python -m experiments.interpretability <ckpt>`):

1. what each label claims, by the contrastive score `S[w,a] - max_{a'≠a} S[w,a']`;
2. where the head posterior attaches — ROOT mass, mean distance, and the whole profile against
   what uniform attention over the same prefixes would give;
3. whether the prior-to-posterior divergence at a slot grows with the surprisal of the word
   observed there, which is the qualitative content of Part IV's contraction bound.

## The result, and the reason it needed running twice

| | 2026-08 record (`d=32`, source temp., damped) | selected (`d=96, r=12`, standardised, undamped) |
|---|---|---|
| ROOT mass | 0.486 | 0.277 |
| mean attachment distance | **1.05** | **11.06** |
| mass at distance 1 | 0.976 | 0.327 |
| mass beyond distance 3 | 0.001 | 0.497 |
| prior/posterior correlation `r` | 0.162 | **0.471** |
| mean KL, predictable → surprising quartile | 1.98 → 2.58 | 4.07 → 6.56 |

Uniform attention over the same prefixes has mean distance **17.01**.

**The first probe was run on the only trained model that existed when it was written, and it
gave the wrong answer about the model class.** On that checkpoint the head posterior has
collapsed onto the previous word — a bigram attachment with a large sink — and that went into
the paper's Discussion, Limitations, introduction and conclusion as a negative result. On the
configuration the width experiment actually selects, the same probe on the same corpus finds a
mean attachment distance of 11.06 against uniform's 17.01, half the mass beyond distance three,
and the contraction bound's predicted correlation nearly three times stronger.

Both are now reported side by side, because the difference is the finding: **the same probe
gives opposite answers about whether this model class learns dependency structure, and what
separates them is the head temperature and the arc-score rank.** An interpretability result from
a probabilistic transformer means little without the parameterisation that produced it.

**The labels run the other way.** At `d = 32` a majority of the 32 labels are readable —
bare infinitives (*reconsider, condemn, lend, hire, notify*), third-person singular present
(*prefers, manages, chooses, earns, requires*), prepositions, past participles, person names,
units of measure, abstract nouns. At `d = 96`, which gives the better perplexity, adverbs
(*hastily, mildly, extremely, widely*), gerunds (*completing, entering, joining, contemplating*),
units (*bottle, par, pound, capita, bushel*) and time nouns (*century, decade, phase, 1970s*)
survive, but most labels resist reading: 96 labels over a 10,000-word vocabulary fragments the
categories. Capacity and interpretability trade off along the width axis and no width was found
where both are best.

## Falsification

The interpretive claim would be dead if neither checkpoint's labels were readable, or if the
head posterior were uniform over the prefix at every parameterisation. Neither holds. It would
be *unconditional* if both checkpoints agreed; they do not, so the claim is conditional and is
stated that way.

## Run log

| date | commit | job | notes |
|---|---|---|---|
| 2026-09-08 | 0396521 | — (CPU) | probes on `checkpoints/RECseed1_d32_pt.pt`, the 2026-08 record |
| 2026-09-08 | 5497747 | 998922 | `spec_probe_ckpt`: trains `d=96, r=12` at the selected configuration and saves a checkpoint (val 217.56, test 197.35, 617 s) |
| 2026-09-08 | — | — (CPU) | probes on that checkpoint; the attachment result reverses |
