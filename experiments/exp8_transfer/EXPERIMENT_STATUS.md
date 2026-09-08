# Experiment 8 — does the PTB rule transfer to WikiText-2, unchanged?

**This file was written on 2026-09-09, after the grid had started.** The question, the design and
the falsification criterion below are copied from the docstring of `spec_wt2.py`, which was
written before any cell ran; nothing here is reconstructed from the results. The lateness of the
file is the deviation from the rule, not the lateness of the design.

## The question

Every hyperparameter is selected on PTB and **nothing is retuned here**. That makes this a test
of the rule rather than a second main experiment: a WikiText-2 number produced by a fresh grid
search would say nothing about transfer, which is the property under test.

1. Does the configuration selected on PTB still learn on WikiText-2 — land far below the corpus
   unigram — without retuning? A no bounds the generality of everything else in the paper.
2. Does the ordering of the ladders (constants carried across widths / one transfer rule)
   survive the corpus change? The rule is only a rule if it does.
3. Is the PT-versus-baseline gap the same size as on PTB at a matched budget?

## What is held fixed

Identical to the PTB protocol, which is the point: 15,000 steps, blocks 16 × 64, validation
checkpoint selection, the same shared training loop. Two PT widths (32, 64), both ladders, three
seeds; transformer baselines at three widths bracketing the PT budget, two learning rates each.

## What changes, and is not controlled away

Vocabulary 10,000 → 33,278 types and training set 929,589 → 2,088,628 tokens. The parameter count
therefore moves even at fixed `d`, because the word–label factor is `|V| × d`. Unavoidable — it is
what changing corpus means for a model whose parameters are almost all embedding — and it is why
every comparison here is against baselines run on *this* corpus through the same loop, never
against PTB numbers. Unigram 964.8, add-one bigram 692.5.

## What would falsify the claim

If the corrected arm were not better than the source-temperature arm here, or if it did not
improve with width here, the width mechanism of Experiment 4 would be a Penn Treebank artefact.

## Run log

| date | commit | job | shards | result |
|---|---|---|---|---|
| 2026-09-09 | 04ffc24 | 999015-999034 | 0-3 of 4 | **24/24 complete, 0 errors.** PT standardised 308.1 +- 2.6 at d=32 and 279.1 +- 9.0 at d=64; PT source temperature 343.1 +- 24.7 and 334.1 +- 27.6. **All three questions answered yes, yes, and no.** (1) The PTB configuration still learns here without retuning, far below the 964.8 unigram. (2) The ordering of the two ladders survives the corpus change, and so does the width behaviour -- the corrected arm is the only one that improves when widened. (3) The gap is *not* the same size: it is wider. Baselines on this corpus -- gpt 179.7 / 150.9 / 148.5 at d_model 32/64/128 and looped 203.0 / 167.9 / 157.2 -- put both below the PT at a smaller non-embedding budget. The looped-versus-standard crossover of exp2 reappears: looped worse at matched d_model, better at matched non-embedding budget. |
