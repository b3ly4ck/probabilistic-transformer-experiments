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

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python -m src.train --config experiments/baseline.yaml
```
