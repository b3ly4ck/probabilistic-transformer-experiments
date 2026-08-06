# Experiments

One folder per experiment. Each holds an `EXPERIMENT_STATUS.md` — the single record of what was
run, with what configuration, what came out, and why each decision was made — plus the configs
and run scripts for that experiment.

| Experiment | Question | Status |
|---|---|---|
| [exp0_decoder_validation](exp0_decoder_validation/EXPERIMENT_STATUS.md) | Does the causal PT decoder implement the equations it claims to? | **checks 1-9 pass** |
| [exp1_language_modeling](exp1_language_modeling/EXPERIMENT_STATUS.md) | Does it train on real data, and does the in-graph FFN analogue `G_t` close the gap? | specified |
| [exp1_pt_vs_gpt](exp1_pt_vs_gpt/EXPERIMENT_STATUS.md) | *superseded by the above, kept for the record* | never run |
| [exp2_pt_vs_looped](exp2_pt_vs_looped/EXPERIMENT_STATUS.md) ★ | Does structure contribute beyond weight sharing? | not started |
| [exp3_exact_vs_mfvi](exp3_exact_vs_mfvi/EXPERIMENT_STATUS.md) | What does the mean-field approximation cost? | not started |

Order is also the cut order under time pressure: cut from the bottom. Experiment 2 is the
scientific core — it is what separates structure from weight sharing, and without it a reviewer
can attribute any gain to sharing alone.

## Rules for the status files

- **Every run gets a row in the run log, in the same commit as the run.** A result that exists
  only in a terminal scrollback does not exist.
- **Every row records the commit it was produced at.** A number without a commit cannot be
  reproduced and cannot be defended.
- **Never delete a row.** Failed, abandoned and superseded runs stay, with the reason. A check
  that passed and later broke is the most valuable line in the file.
- **Negative and neutral results are recorded as found.** The paper's framing does not require
  PT to win; overselling a marginal gap is the fastest way to lose a reviewer.
- **Write the justification, not just the number.** Six months later the number is worthless
  without the reason the configuration was chosen.
- **Do not report a run as working because the process exited cleanly.** Look at the curve and
  the metric first.

Cross-experiment state — what exists in the codebase, which components are shared, decisions
that span experiments — belongs in
[`developer files/PROJECT_STATUS.md`](../developer%20files/PROJECT_STATUS.md), not here.
