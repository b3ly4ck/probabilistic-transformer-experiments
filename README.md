# Causal Probabilistic Transformer

Extending the Probabilistic Transformer of Wu & Tu ([Findings of ACL
2023](https://aclanthology.org/2023.findings-acl.482/)) into a causal autoregressive decoder,
and producing the empirical section of a preprint.

The original PT is a CRF-based *encoder* in which word representations and syntactic dependency
structure are modelled jointly as a factor graph, with inference by mean-field variational
inference — its inference procedure turns out to have the computation graph of a transformer
encoder. It is masked-LM only. This project builds the causal version, which does not otherwise
exist, and measures what it costs.

Read [`developer files/PROJECT.md`](developer%20files/PROJECT.md) for the research context,
[`developer files/PROJECT_STATUS.md`](developer%20files/PROJECT_STATUS.md) for what has been
built and measured, and [`CLAUDE.md`](CLAUDE.md) for the working rules and the non-negotiable
modelling constraints.

## Layout

```
.
├── developer files/     # documentation and the reference PDFs
├── src/                 # model code
│   ├── pt_decoder.py    #   CausalPTDecoder — the construction, exact and mean-field readouts
│   ├── factored.py      #   FactoredPTDecoder — K label components sharing one word variable
│   ├── switches.py      #   the ten-switch ladder between a transformer and the PT decoder
│   ├── gpt.py           #   nanoGPT off the shelf, plus the Looped variant
│   ├── train.py         #   the training loop, written once and shared by every model
│   └── analysis.py      #   sweep JSONs -> tables and figures
├── experiments/         # one folder per experiment, each with EXPERIMENT_STATUS.md
│   ├── sweep.py         #   the grid runner every experiment goes through
│   ├── make_report.py   #   regenerates every figure and table of the preprint
│   └── exp*/            #   specs, run JSONs, and the run logs
├── tests/               # 12 behavioural test files
└── data/                # corpora — gitignored
```

## What is established

* **The construction.** Causality, a single global energy and contextuality are mutually
  incompatible, so the model is a directed chain of conditional CRFs in which the prefix enters
  each step as a frozen condition. Unclamping the word variable gives a language-model head with
  no parameter outside the factor graph; tied embeddings, the output bias and a
  mixture-of-softmaxes readout follow as consequences. The exact readout agrees with brute-force
  enumeration of the slot joint to `6.7e-16`.
* **A width mechanism.** Wu & Tu's head temperature `lambda_H = 1/d` is exactly right at the
  uniform belief and mis-calibrated by up to a factor `d` away from it — where training goes
  within a few hundred steps. Correcting it revives label-set sizes that otherwise sit exactly
  on the unigram baseline, roughly doubles the usable width, and improves the model's own
  scaling exponent from `-0.061` to `-0.082`. Per parameter it still gains less than either
  baseline (`-0.175` for a causal transformer, `-0.107` for a Looped one, on the range where all
  three are measured).
* **The Kruskal rank inverts the received configuration.** Every earlier run used `rank = d`.
  At `d = 96`, `rank = 12` reaches 217.6 validation perplexity with 18,624 non-embedding
  parameters, against `rank = 96`'s 275.6 ± 27.1 with 147,648.
* **A reproducibility result.** The configuration this project inherited has a seed standard
  deviation of **130 perplexity**, and is deterministic within a GPU model while differing by
  215 perplexity across models. A number from that region is not a measurement unless it comes
  with a seed spread and a device.

## Reproducing

GPU jobs go through one runner. A grid is a list of cells in a spec module:

```bash
sbatch --exclude=ai_gpu32,ai_gpu33 experiments/sweep.slurm experiments.exp2_scaling.spec_baselines --shard 0/4
```

Rows are written per cell, so a preempted job loses nothing and a re-run resumes.
`experiments/requeue.py` re-queues shards that died; `experiments/watchdog.sh` runs it on a
timer. Then:

```bash
python -m experiments.make_report      # every figure and table, from the committed JSONs
python -m experiments.validation_report # the checks behind the paper's validation appendix
```

## Tests

```bash
python -m pytest
```

Behavioural only: every test calls the real function with small real tensors and asserts on
observable outputs. Two of them test the derivation rather than the code — the free energy is
non-increasing along the inner loop (and *increases* under a deliberately sign-flipped update,
so the check is shown to have teeth), and the exact readout agrees with explicit enumeration.

## The worked example

Reproduces `causal_pt_output_note.pdf` §5 with every intermediate tensor printed.

```bash
python -m experiments.exp0_decoder_validation.worked_example
```

## The preprint

Branch `article`, in a separate worktree, under `paper/`. Build with `make` (TeX Live is a
private install under `$HOME/texlive`; the Makefile puts it on `PATH` itself).
