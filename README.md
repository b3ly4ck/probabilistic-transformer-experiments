# Causal Probabilistic Transformer

Extending the Probabilistic Transformer (Wu & Tu) into a causal autoregressive decoder.

The original PT is a CRF-based encoder in which word representations and syntactic dependency
structure are modelled jointly as a factor graph, with inference by mean field variational
inference. It is masked-LM only. This project builds the causal version that does not yet
exist, and provides the empirical evidence that it trains.

See [developer files/PROJECT.md](developer%20files/PROJECT.md) for the research context and
success criteria, and [developer files/RESEARCH_PLAN.md](developer%20files/RESEARCH_PLAN.md)
for the experiment plan and its validation gates. See [CLAUDE.md](CLAUDE.md) for the working
rules and modelling constraints.

## Layout

```
.
├── developer files/     # documentation and reference papers
├── src/                 # model code — PT decoder, shared training loop
├── experiments/         # one spec (.md) + one config per experiment
├── notebooks/           # analysis and figures
├── tests/               # runnable tests
└── data/                # corpora — not versioned
```

## State

The causal PT decoder forward pass is implemented and validated (Experiment 0): checks 1–9 of
the research plan pass, and the worked example of `causal_pt_output_note.pdf` §5 is reproduced
number for number. See
[experiments/exp0_decoder_validation/EXPERIMENT_STATUS.md](experiments/exp0_decoder_validation/EXPERIMENT_STATUS.md)
for the validation record and
[developer files/PROJECT_STATUS.md](developer%20files/PROJECT_STATUS.md) for what exists.

The training loop, the data pipeline and the GPT / Looped baselines are not written yet.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch pytest
```

## Tests

```bash
python -m pytest
```

## The worked example

Reproduces `causal_pt_output_note.pdf` §5 with every intermediate tensor printed.

```bash
python -m experiments.exp0_decoder_validation.worked_example
```
