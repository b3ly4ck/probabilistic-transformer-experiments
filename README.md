# Causal Probabilistic Transformer

Extending the Probabilistic Transformer (Wu & Tu) into a causal autoregressive decoder.

The original PT is a CRF-based encoder in which word representations and syntactic dependency
structure are modelled jointly as a factor graph, with inference by mean field variational
inference. It is masked-LM only. This project builds the causal version that does not yet
exist, and provides the empirical evidence that it trains.

See [PROJECT.md](PROJECT.md) for the research context, the three planned experiments, and the
success criteria. See [CLAUDE.md](CLAUDE.md) for the working rules and modelling constraints.

## Layout

```
.
├── src/            # model code — PT decoder, shared training loop
├── experiments/    # one spec (.md) + one config per experiment
├── notebooks/      # analysis and figures
├── papers/         # reference PDFs
├── tests/          # runnable tests
└── data/           # corpora — not versioned
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
