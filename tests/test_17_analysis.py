"""Behavioural tests for `src.analysis`, the sweep-JSON to table-and-figure layer.

Every row used here is synthesised in the test itself, in the schema
`experiments/sweep.py:run_cell` writes. Nothing reads a real sweep output: those files are
produced by jobs that are still running, so a test that depended on one would be green or
red according to the state of the queue rather than the state of the code. The synthetic
rows are small enough that every expected number below is hand-checkable, which is the
point -- the assertions are on values worked out by hand, not on whatever the
implementation happened to return the first time it ran.

The plotting tests assert that a real PNG lands on disk with a non-trivial size. That is a
weak assertion about the *content* of a figure and a strong one about the code path: every
one of these functions builds axes, formats ticks, computes a colour per cell and writes a
file, and a broken key, an empty series or a masked-array mistake raises somewhere in
there. `matplotlib.use("Agg")` happens inside `src.analysis` at import, which is also
what these tests exercise -- there is no display on the cluster.
"""

import json
import math

import pytest

from src.analysis import (
    _pretty_number,
    aggregate_seeds,
    best_by,
    default_failed,
    fit_power_law,
    get_path,
    load_rows,
    markdown_table,
    plot_heatmap,
    plot_scaling,
    plot_traces,
    row_steps,
    scaling_frontier,
    to_table,
)

UNIGRAM = 688.818  # PTB, as `run_cell` records it


def make_row(
    name="cell",
    model="pt",
    d=32,
    alpha_Z=1.0,
    temp_mode="fixed",
    lr=2e-2,
    seed=0,
    val_ppl=300.0,
    non_embedding=16448,
    n_evals=4,
    msg=(1.2, 1.4, 1.6, 1.8),
    eval_every=500,
):
    """One sweep row in the schema of `experiments/sweep.py:run_cell`.

    Only the fields the analysis layer reads are filled in with meaningful values; the rest
    are present so that a dotted path into them resolves the way it does on a real file.
    """
    val_trace = [round(val_ppl * (1.0 + 0.35 * (n_evals - 1 - k)), 2) for k in range(n_evals)]
    return {
        "cell": {
            "name": name,
            "model": model,
            "d": d,
            "h": 2,
            "alpha_Z": alpha_Z,
            "temp_mode": temp_mode,
            "lr": lr,
            "seed": seed,
            "eval_every": eval_every,
            "steps": eval_every * n_evals,
            "tags": {"grid": "A", "d": d},
        },
        "unigram": UNIGRAM,
        "val_ppl": val_ppl,
        "val_ppl_final": val_trace[-1],
        "val_ppl_min": min(val_trace),
        "best_step": eval_every * n_evals,
        "test_ppl": val_ppl * 0.95,
        "train_ppl": val_ppl * 0.9,
        "val_loss": math.log(val_ppl),
        "params": {
            "embedding": 320000,
            "non_embedding": non_embedding,
            "total": 320000 + non_embedding,
        },
        "ablation_kl": 0.5,
        "ablation_argmax_same": 0.4,
        "seconds": 240.0,
        "steps_logged": [eval_every * (k + 1) for k in range(n_evals)],
        "val_trace": val_trace,
        "train_trace": [round(v * 0.9, 2) for v in val_trace],
        "diag_final": {"msg_over_unary": msg[-1], "attn_entropy_frac": 0.7},
        "diag_trace": [
            {"msg_over_unary": msg[k], "attn_entropy_frac": 0.7} for k in range(n_evals)
        ],
        "host": "test",
        "device": "cpu",
    }


def write_shard(path, rows, meta=None):
    """Write rows the way `run_sweep` does: one JSON object with `meta` and `rows`."""
    path.write_text(
        json.dumps({"meta": meta or {"commit": "deadbee", "host": "test"}, "rows": rows},
                   indent=1)
    )
    return path


# ------------------------------------------------------------------ nested access --


def test_get_path_reaches_nested_keys_and_defaults_on_missing():
    row = make_row(name="A_d32_a1")
    assert get_path(row, "cell.d") == 32
    assert get_path(row, "params.non_embedding") == 16448
    assert get_path(row, "diag_final.msg_over_unary") == pytest.approx(1.8)
    assert get_path(row, "cell.tags.grid") == "A"
    assert get_path(row, "val_trace.0") == row["val_trace"][0]
    # A GPT row has an empty diag_final; asking for a diagnostic must return the default,
    # not raise, or a mixed-model table cannot be built at all.
    assert get_path(row, "diag_final.rho") is None
    assert get_path(row, "nope.deeper", default=-1) == -1


# ------------------------------------------------------------------------ loading --


def test_load_rows_concatenates_shards_and_reports_dropped_errors(tmp_path):
    shard0 = [make_row(name="c0", seed=0, val_ppl=310.0),
              make_row(name="c1", seed=1, val_ppl=290.0)]
    shard1 = [make_row(name="c2", seed=2, val_ppl=270.0),
              {"cell": {"name": "c3_dead", "d": 128}, "error": "RuntimeError: out of memory"}]
    write_shard(tmp_path / "spec_width.shard0of3.json", shard0)
    write_shard(tmp_path / "spec_width.shard1of3.json", shard1)
    # shard2of3 is deliberately absent: a slurm array element that has not started yet.

    logged = []
    rows = load_rows(str(tmp_path / "spec_width.shard*of3.json"), log=logged.append)

    names = [get_path(r, "cell.name") for r in rows]
    assert names == ["c0", "c1", "c2"], "shards must concatenate in sorted file order"
    assert all("error" not in r for r in rows)

    assert rows.n_dropped == 1
    assert len(rows.dropped_errors) == 1
    assert rows.dropped_errors[0]["error"].startswith("RuntimeError")
    assert get_path(rows.dropped_errors[0], "cell.name") == "c3_dead"
    assert any("dropped 1" in line for line in logged), logged
    assert "c3_dead" in rows.error_report()

    assert len(rows.files) == 2
    assert rows.meta[str(tmp_path / "spec_width.shard0of3.json")]["commit"] == "deadbee"


def test_error_report_shows_one_line_per_dead_cell(tmp_path):
    """A CUDA failure carries four lines of boilerplate; fifteen of them drown the log.

    Observed on the live exp4 width sweep: 15 of 22 rows carried
    "RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable" followed
    by three lines about CUDA_LAUNCH_BLOCKING and TORCH_USE_CUDA_DSA.
    """
    verbose = (
        "RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable\n"
        "CUDA kernel errors might be asynchronously reported at some other API call.\n"
        "For debugging consider passing CUDA_LAUNCH_BLOCKING=1.\n"
        "Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n"
    )
    write_shard(
        tmp_path / "s.shard0of1.json",
        [{"cell": {"name": "A_d16_a0.125"}, "error": verbose},
         {"cell": {"name": "A_d32_a0.5"}, "error": verbose}],
    )
    rows = load_rows(tmp_path / "s.shard0of1.json", log=None)
    assert list(rows) == [] and rows.n_dropped == 2

    report = rows.error_report()
    assert len(report.splitlines()) == 2
    assert "A_d16_a0.125" in report and "busy or unavailable" in report
    assert "CUDA_LAUNCH_BLOCKING" not in report
    # the untruncated text is still available on the dropped row itself
    assert "CUDA_LAUNCH_BLOCKING" in rows.dropped_errors[0]["error"]
    assert "CUDA_LAUNCH_BLOCKING" in rows.error_report(full=True)


def test_load_rows_keep_errors_returns_them_in_the_list(tmp_path):
    write_shard(
        tmp_path / "s.shard0of1.json",
        [make_row(name="ok"), {"cell": {"name": "bad"}, "error": "ValueError: x"}],
    )
    rows = load_rows(tmp_path / "s.shard0of1.json", keep_errors=True, log=None)
    assert len(rows) == 2
    assert rows.n_dropped == 1  # still counted, even though it was kept


def test_load_rows_skips_a_half_written_shard_with_a_warning(tmp_path):
    """`run_sweep` rewrites the whole JSON after every cell and the write is not atomic."""
    write_shard(tmp_path / "s.shard0of2.json", [make_row(name="ok", val_ppl=250.0)])
    good = (tmp_path / "s.shard0of2.json").read_text()
    (tmp_path / "s.shard1of2.json").write_text(good[: len(good) // 2])  # truncated mid-write

    logged = []
    rows = load_rows(str(tmp_path / "s.shard*of2.json"), log=logged.append)

    assert [get_path(r, "cell.name") for r in rows] == ["ok"]
    assert len(rows.unreadable) == 1
    assert rows.unreadable[0][0].endswith("s.shard1of2.json")
    assert any("unreadable" in line for line in logged), logged


def test_load_rows_finds_shards_from_the_unsharded_path(tmp_path):
    """`--out spec.json --shard i/n` writes `spec.shardiofn.json`; asking for `spec.json`
    must find them, because that is the name a caller has in hand."""
    write_shard(tmp_path / "spec.shard0of2.json", [make_row(name="a")])
    write_shard(tmp_path / "spec.shard1of2.json", [make_row(name="b")])
    rows = load_rows(tmp_path / "spec.json", log=None)
    assert sorted(get_path(r, "cell.name") for r in rows) == ["a", "b"]


def test_load_rows_on_nothing_returns_empty_and_says_so(tmp_path):
    logged = []
    rows = load_rows(str(tmp_path / "absent*.json"), log=logged.append)
    assert list(rows) == []
    assert logged and "no files matched" in logged[0]


# ------------------------------------------------------------------------- tables --


def test_to_table_projects_dotted_and_derived_columns():
    rows = [make_row(name="A", d=32, val_ppl=300.0),
            make_row(name="B", d=64, val_ppl=250.0, non_embedding=65792)]
    header, body = to_table(
        rows,
        [
            ("run", "cell.name"),
            "cell.d",
            ("non-emb", "params.non_embedding"),
            ("msg/unary", "diag_final.msg_over_unary"),
            ("gap closed", lambda r: (UNIGRAM - r["val_ppl"]) / UNIGRAM),
            ("missing", "diag_final.rho"),
        ],
    )
    assert header == ["run", "cell.d", "non-emb", "msg/unary", "gap closed", "missing"]
    assert body[0][:3] == ["A", 32, 16448]
    assert body[1][2] == 65792
    assert body[0][4] == pytest.approx((UNIGRAM - 300.0) / UNIGRAM)
    assert body[0][5] is None


def test_markdown_table_renders_sorted_rows_with_a_separator():
    rows = [make_row(name="A", val_ppl=300.0), make_row(name="B", val_ppl=250.0)]
    text = markdown_table(
        rows, [("run", "cell.name"), ("val ppl", "val_ppl"), ("missing", "diag_final.rho")],
        sort_by="val ppl",
    )
    lines = text.splitlines()
    assert lines[0].startswith("| run")
    assert set(lines[1]) <= set("|-: ")  # the alignment row
    assert lines[2].split("|")[1].strip() == "B"  # 250 sorts before 300
    assert "250.000" in lines[2]
    assert lines[2].split("|")[3].strip() == "--"  # missing value placeholder
    assert len(lines) == 4


def test_markdown_table_sorts_a_text_column_alphabetically():
    """A table that says it is sorted must be sorted, whatever the column holds.

    Sorting used to coerce every key to float, so a text column (run name, model family,
    temp_mode) produced NaN for every row and the table came out in file order with no
    warning -- verified against the live exp2 rows, where `sort_by="run"` left the 19 GPT
    baselines in the order the shards happened to be written.
    """
    rows = [make_row(name="C_gamma"), make_row(name="A_alpha"), make_row(name="B_beta")]
    columns = [("run", "cell.name"), ("val ppl", "val_ppl")]

    names = [line.split("|")[1].strip() for line in
             markdown_table(rows, columns, sort_by="run").splitlines()[2:]]
    assert names == ["A_alpha", "B_beta", "C_gamma"]

    names = [line.split("|")[1].strip() for line in
             markdown_table(rows, columns, sort_by="run", reverse=True).splitlines()[2:]]
    assert names == ["C_gamma", "B_beta", "A_alpha"]


def test_markdown_table_puts_a_missing_sort_key_last_in_both_directions():
    rows = [make_row(name="has", val_ppl=300.0), make_row(name="none", val_ppl=250.0)]
    rows[1]["val_ppl"] = None
    for reverse in (False, True):
        text = markdown_table(rows, [("run", "cell.name"), ("val ppl", "val_ppl")],
                              sort_by="val ppl", reverse=reverse)
        last = text.splitlines()[-1].split("|")[1].strip()
        assert last == "none", "a row with no metric must never outrank one that has it"


# ---------------------------------------------------------------- grouping / bests --


def test_best_by_picks_the_minimum_per_key_combination():
    """The scaling specs run every point at two learning rates and take the better one."""
    rows = [
        make_row(name="gpt_e32_lr1e-3", model="gpt", lr=1e-3, val_ppl=260.0, non_embedding=13152),
        make_row(name="gpt_e32_lr3e-3", model="gpt", lr=3e-3, val_ppl=240.0, non_embedding=13152),
        make_row(name="gpt_e64_lr1e-3", model="gpt", lr=1e-3, val_ppl=210.0, non_embedding=52608),
        make_row(name="gpt_e64_lr3e-3", model="gpt", lr=3e-3, val_ppl=230.0, non_embedding=52608),
    ]
    best = best_by(rows, ["cell.model", "params.non_embedding"], metric="val_ppl")
    assert [get_path(r, "cell.name") for r in best] == ["gpt_e32_lr3e-3", "gpt_e64_lr1e-3"]
    assert [r["val_ppl"] for r in best] == [240.0, 210.0]


def test_best_by_ignores_non_finite_metrics_and_can_maximise():
    rows = [
        make_row(name="nan", val_ppl=float("nan")),
        make_row(name="good", val_ppl=280.0),
        make_row(name="better", val_ppl=250.0),
    ]
    assert [get_path(r, "cell.name") for r in best_by(rows, ["cell.model"])] == ["better"]
    worst = best_by(rows, ["cell.model"], lower_is_better=False)
    assert [get_path(r, "cell.name") for r in worst] == ["good"]


def test_aggregate_seeds_mean_and_sample_std_hand_checked():
    """values 100, 104, 108 -> mean 104, sample variance (16+0+16)/2 = 16, std 4."""
    rows = [
        make_row(name="rec", seed=s, val_ppl=v)
        for s, v in [(0, 100.0), (1, 104.0), (2, 108.0)]
    ]
    (group,) = aggregate_seeds(rows, ["cell.name"], metric="val_ppl")
    assert group.key == ("rec",)
    assert group.n == 3
    assert group.mean == pytest.approx(104.0)
    assert group.std == pytest.approx(4.0)  # ddof=1; the population std would be 3.265986
    assert group.std != pytest.approx(math.sqrt(32.0 / 3.0))
    assert group.seeds == [0, 1, 2]
    assert group.values == [100.0, 104.0, 108.0]
    assert group.format(".2f") == "104.00 +- 4.00 (n=3)"


def test_aggregate_seeds_single_seed_reports_zero_std_and_n_one():
    (group,) = aggregate_seeds([make_row(name="solo", val_ppl=241.89)], ["cell.name"])
    assert group.n == 1
    assert group.std == 0.0
    assert group.mean == pytest.approx(241.89)


def test_aggregate_seeds_separates_key_combinations():
    rows = [make_row(name="a", d=32, seed=0, val_ppl=250.0),
            make_row(name="a", d=32, seed=1, val_ppl=254.0),
            make_row(name="b", d=64, seed=0, val_ppl=300.0)]
    groups = aggregate_seeds(rows, ["cell.d"], metric="val_ppl")
    assert [g.key for g in groups] == [(32,), (64,)]
    assert [g.n for g in groups] == [2, 1]
    assert groups[0].mean == pytest.approx(252.0)
    assert groups[0].std == pytest.approx(math.sqrt(8.0))  # (4+4)/1


# ---------------------------------------------------------------- scaling analysis --


def test_scaling_frontier_drops_a_dominated_point():
    """The 100k point is beaten by the 50k point, so it is not on any scaling curve."""
    rows = [
        make_row(name="p10k", non_embedding=10_000, val_ppl=400.0),
        make_row(name="p50k", non_embedding=50_000, val_ppl=300.0),
        make_row(name="p100k_dominated", non_embedding=100_000, val_ppl=320.0),
        make_row(name="p200k", non_embedding=200_000, val_ppl=250.0),
    ]
    points = scaling_frontier(rows, x="params.non_embedding", y="val_ppl")
    assert [p.x for p in points] == [10_000, 50_000, 200_000]
    assert [p.y for p in points] == [400.0, 300.0, 250.0]
    assert get_path(points[-1].row, "cell.name") == "p200k"
    assert all(points[i].x < points[i + 1].x for i in range(len(points) - 1))
    assert all(points[i].y > points[i + 1].y for i in range(len(points) - 1))


def test_scaling_frontier_keeps_the_better_of_two_points_at_the_same_x():
    rows = [make_row(name="lo", non_embedding=1000, val_ppl=500.0),
            make_row(name="hi", non_embedding=1000, val_ppl=450.0)]
    points = scaling_frontier(rows)
    assert len(points) == 1
    assert points[0].y == 450.0
    assert get_path(points[0].row, "cell.name") == "hi"


def test_fit_power_law_recovers_a_known_exponent():
    xs = [1e3, 1e4, 1e5, 1e6]
    ys = [3.0 * x ** (-0.5) for x in xs]
    law = fit_power_law(xs, ys)
    assert law is not None
    assert law.exponent == pytest.approx(-0.5, abs=1e-6)
    assert law.intercept == pytest.approx(math.log10(3.0), abs=1e-6)
    assert law.r2 == pytest.approx(1.0, abs=1e-6)
    assert law.predict(1e4) == pytest.approx(3.0 * 1e4 ** (-0.5), rel=1e-9)
    # The legend qualifies the exponent as an own-range fit: on this project's data the same
    # curve gives a different exponent on a different window, so a bare "slope" in a legend
    # invites exactly the comparison the paper refuses to make.
    assert "-0.500" in law.label()
    assert "own range" in law.label()


def test_fit_power_law_needs_three_distinct_points():
    assert fit_power_law([1.0, 10.0], [1.0, 0.5]) is None
    assert fit_power_law([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None  # no spread in x
    # Non-positive and non-finite values are dropped before the log, leaving too few.
    assert fit_power_law([1.0, 10.0, -5.0, float("nan")], [1.0, 0.5, 2.0, 1.0]) is None


def test_fit_power_law_r2_falls_below_one_on_noisy_data():
    xs = [1e3, 1e4, 1e5, 1e6]
    ys = [3.0 * x ** (-0.5) for x in xs]
    ys[2] *= 2.0
    law = fit_power_law(xs, ys)
    assert law is not None
    assert 0.0 < law.r2 < 1.0


# ------------------------------------------------------------------- misc helpers --


def test_row_steps_prefers_the_logged_steps_and_falls_back_to_eval_every():
    row = make_row(n_evals=3, eval_every=500)
    assert row_steps(row) == [500, 1000, 1500]
    row.pop("steps_logged")
    assert row_steps(row) == [500, 1000, 1500]
    row["cell"].pop("eval_every")
    assert row_steps(row) == [1, 2, 3]


def test_pretty_number_formats_counts_and_perplexities():
    """The log-axis tick formatter: parameter counts abbreviated, perplexities plain.

    Matplotlib's default labels a sub-decade log axis with `2 x 10^4`, `3 x 10^4`, ... and
    those collide on the exp2 baseline range (13k-450k non-embedding parameters), which is
    what this formatter replaces.
    """
    assert _pretty_number(10_000) == "10k"
    assert _pretty_number(200_000) == "200k"
    assert _pretty_number(1_500_000) == "1.5M"
    assert _pretty_number(300) == "300"
    assert _pretty_number(1.5) == "1.5"
    assert _pretty_number(0.0) == ""


def test_default_failed_flags_a_cell_sitting_on_the_unigram():
    assert default_failed(make_row(val_ppl=687.0)) is True
    assert default_failed(make_row(val_ppl=250.0)) is False
    row = make_row(val_ppl=250.0)
    row.pop("unigram")
    assert default_failed(row) is False  # cannot be judged without a baseline


# ------------------------------------------------------------------------ figures --


def assert_png(path, minimum=2000):
    """A real PNG of non-trivial size landed on disk."""
    assert path.exists(), "{} was not written".format(path)
    head = path.read_bytes()[:8]
    assert head == b"\x89PNG\r\n\x1a\n", "not a PNG: {!r}".format(head)
    assert path.stat().st_size > minimum, "suspiciously small PNG: {} bytes".format(
        path.stat().st_size
    )


def test_plot_scaling_writes_a_png_with_two_fitted_series(tmp_path):
    gpt = [make_row(name="gpt{}".format(n), model="gpt", non_embedding=n,
                    val_ppl=3000.0 * n ** -0.3)
           for n in (10_000, 30_000, 100_000, 300_000)]
    looped = [make_row(name="lp{}".format(n), model="looped", non_embedding=n,
                       val_ppl=4000.0 * n ** -0.3)
              for n in (10_000, 30_000, 100_000, 300_000)]
    out = plot_scaling(
        {"GPT": gpt, "Looped": looped, "PT (hand)": [(16_448, 250.69), (65_792, 300.0)]},
        tmp_path / "fig_scaling.png",
        title="PTB scaling",
        hline=(UNIGRAM, "unigram baseline"),
    )
    assert_png(out)


def test_plot_scaling_survives_an_empty_series(tmp_path):
    out = plot_scaling(
        [("empty", []), ("one", [make_row(non_embedding=1000, val_ppl=400.0)])],
        tmp_path / "fig_empty.png",
    )
    assert_png(out)


def test_plot_heatmap_writes_a_png_with_missing_and_failed_cells(tmp_path):
    rows = []
    for d in (16, 32, 64):
        for a in (1.0, 0.5, 0.25):
            if d == 64 and a == 0.25:
                continue  # a cell that has not run yet -> drawn as n/a, not as a bad result
            collapsed = (d == 64 and a == 1.0)
            rows.append(
                make_row(name="A_d{}_a{}".format(d, a), d=d, alpha_Z=a,
                         val_ppl=700.0 if collapsed else 400.0 - d)
            )
    out = plot_heatmap(
        rows, "cell.d", "cell.alpha_Z", "val_ppl", tmp_path / "fig_heatmap.png",
        x_label="label count d", y_label="alpha_Z (label step size)",
        v_label="validation perplexity", title="grid A",
    )
    assert_png(out)


def colour_pixels(path, hex_colour, tol=0.02):
    """How many pixels of the rendered figure carry (approximately) this colour."""
    import matplotlib.image as mpimg
    import numpy as np

    rgb = np.array([int(hex_colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])
    image = mpimg.imread(str(path))[:, :, :3]
    return int(np.sum(np.all(np.abs(image - rgb) < tol, axis=-1)))


def test_each_series_reaches_the_canvas_in_its_own_colour(tmp_path):
    """The series are actually drawn, and drawn in different colours.

    Checking that a PNG landed on disk is not enough: an empty 6.4x4.4 figure saved at
    dpi 200 is 19 kB of valid PNG, so a plotting function that drew nothing at all would
    pass a file-exists-and-is-big-enough assertion. Counting the ink of each series colour
    separates "drew two series" from "drew an empty pair of axes" (0 pixels of either
    colour), and pins the greyscale rule from the module docstring: two series must not
    share one colour.
    """
    from src.analysis import _COLOURS

    scaling = plot_scaling(
        {"GPT": [make_row(name="g{}".format(n), non_embedding=n, val_ppl=3000.0 * n ** -0.3)
                 for n in (10_000, 30_000, 100_000, 300_000)],
         "Looped": [make_row(name="l{}".format(n), non_embedding=n, val_ppl=4000.0 * n ** -0.3)
                    for n in (10_000, 30_000, 100_000, 300_000)]},
        tmp_path / "two_series.png",
    )
    assert colour_pixels(scaling, _COLOURS[0]) > 500
    assert colour_pixels(scaling, _COLOURS[1]) > 500

    traces = plot_traces(
        [make_row(name="d16", d=16, val_ppl=315.46, msg=(0.4, 0.9, 2.1, 2.8)),
         make_row(name="d32", d=32, val_ppl=690.0, msg=(2.9, 20.6, 18.4, 17.9))],
        tmp_path / "two_traces.png", label="cell.name",
    )
    assert colour_pixels(traces, _COLOURS[0]) > 500
    assert colour_pixels(traces, _COLOURS[1]) > 500


def test_plot_heatmap_paints_a_missing_cell_flat_grey(tmp_path):
    """A combination with no row must render as flat grey, not as a colour off the scale.

    Asserted on the rendered pixels rather than on the existence of the file, because the
    whole content of the choice is what the reader sees: a cell that has not run yet drawn
    in a colormap colour reads as a result, and an unfinished shard then looks like a bad
    configuration. The grey is 0.85 -> (217, 217, 217), a value the viridis_r ramp never
    takes, so counting pixels within 0.01 of it separates the two cases cleanly.

    This also pins the portability fix: the grey used to come from a copied colormap
    (`Colormap.with_extremes` / `Colormap.copy`, both matplotlib >= 3.4) and the fallback
    raised `AttributeError` on the matplotlib 3.3.4 of the cluster python 3.8.8, so the
    figure was never written there at all. Painting a patch keeps one code path for both.
    """
    import matplotlib.image as mpimg
    import numpy as np

    full = [make_row(name="d{}_a{}".format(d, a), d=d, alpha_Z=a, val_ppl=400.0 - d)
            for d in (16, 32) for a in (1.0, 0.5)]
    holed = [r for r in full
             if not (get_path(r, "cell.d") == 32 and get_path(r, "cell.alpha_Z") == 0.5)]

    def grey_pixels(path):
        image = mpimg.imread(str(path))[:, :, :3]
        return int(np.sum(np.all(np.abs(image - 0.85) < 0.01, axis=-1)))

    dense = plot_heatmap(full, "cell.d", "cell.alpha_Z", "val_ppl", tmp_path / "full.png")
    sparse = plot_heatmap(holed, "cell.d", "cell.alpha_Z", "val_ppl", tmp_path / "holed.png")
    assert_png(dense)
    assert_png(sparse)
    assert grey_pixels(dense) < 500, "no cell is missing, so nothing should be grey"
    assert grey_pixels(sparse) > 5000, "the missing cell was not painted grey"


def test_plot_heatmap_handles_a_categorical_axis(tmp_path):
    rows = [
        make_row(name="B_d{}_{}".format(d, t), d=d, temp_mode=t, val_ppl=400.0 - d + i * 5)
        for d in (16, 32, 64)
        for i, t in enumerate(("fixed", "qnorm", "qknorm"))
    ]
    out = plot_heatmap(
        rows, "cell.d", "cell.temp_mode", "val_ppl", tmp_path / "fig_temp.png", agg="mean"
    )
    assert_png(out)


def test_plot_heatmap_raises_on_an_axis_that_is_not_in_the_rows():
    with pytest.raises(ValueError):
        plot_heatmap([make_row()], "cell.nope", "cell.d", "val_ppl", "/dev/null")


def test_plot_traces_writes_a_png_with_a_diagnostic_second_axis(tmp_path):
    """The 2026-08 phase-transition figure: val perplexity and msg/unary on shared steps."""
    learned = make_row(name="d16", d=16, val_ppl=315.46, n_evals=4,
                       msg=(0.4, 0.9, 2.1, 2.8))
    collapsed = make_row(name="d32", d=32, val_ppl=690.0, n_evals=4,
                         msg=(2.88, 20.57, 18.4, 17.9))
    out = plot_traces(
        [learned, collapsed], tmp_path / "fig_traces.png",
        diag="msg_over_unary", label="cell.name", diag_band=(None, 5.0),
        title="within-run transition",
    )
    assert_png(out)


def test_plot_traces_without_a_diagnostic_still_draws(tmp_path):
    """GPT and Looped rows have an empty diag_trace; the figure must still be drawable."""
    row = make_row(name="gpt", model="gpt", n_evals=5, msg=(0,) * 5)
    row["diag_trace"] = [{} for _ in row["val_trace"]]
    row["diag_final"] = {}
    out = plot_traces([row], tmp_path / "fig_plain.png", diag="msg_over_unary", logy=False)
    assert_png(out)

    # ... and it must come out as the *same figure* as one that never asked for a
    # diagnostic. A twin axis created for a diagnostic that turns out not to exist leaves
    # an unlabelled 0.0-1.0 scale down the right-hand side, which is the normal case for
    # GPT and Looped rows (the shared loop fills `diag_trace` only for models exposing
    # `content_stream`) and reads as a real quantity. Byte equality of the two Agg renders
    # is the observable form of "the empty second axis is not on the figure".
    plain = plot_traces([row], tmp_path / "fig_no_axis.png", diag=None, logy=False)
    assert out.read_bytes() == plain.read_bytes()


def test_end_to_end_shards_to_markdown_and_figures(tmp_path):
    """The path an actual report build takes: load shards, select, tabulate, plot."""
    rows_a = [make_row(name="A_d{}_a1".format(d), d=d, non_embedding=16 * d * d,
                       val_ppl=2000.0 * d ** -0.4, seed=0) for d in (16, 32, 64)]
    rows_b = [make_row(name="A_d{}_a025".format(d), d=d, alpha_Z=0.25,
                       non_embedding=16 * d * d, val_ppl=1800.0 * d ** -0.4, seed=1)
              for d in (16, 32, 64)]
    write_shard(tmp_path / "spec.shard0of2.json", rows_a)
    write_shard(tmp_path / "spec.shard1of2.json",
                rows_b + [{"cell": {"name": "dead"}, "error": "RuntimeError: nan loss"}])

    loaded = load_rows(tmp_path / "spec.json", log=None)
    assert len(loaded) == 6 and loaded.n_dropped == 1

    best = best_by(loaded, ["cell.d"], metric="val_ppl")
    assert [get_path(r, "cell.name") for r in best] == [
        "A_d16_a025", "A_d32_a025", "A_d64_a025"
    ]

    text = markdown_table(
        best,
        [("d", "cell.d"), ("alpha_Z", "cell.alpha_Z"), ("non-emb", "params.non_embedding"),
         ("val ppl", "val_ppl"), ("msg/unary", "diag_final.msg_over_unary")],
        sort_by="d",
    )
    assert text.count("\n") == 4  # header, separator, three rows

    frontier = scaling_frontier(best)
    law = fit_power_law([p.x for p in frontier], [p.y for p in frontier])
    assert law is not None
    # non_embedding = 16 d^2 and val = 1800 d^-0.4, so val ~ params^-0.2 exactly.
    assert law.exponent == pytest.approx(-0.2, abs=1e-6)

    assert_png(plot_scaling({"PT": list(loaded)}, tmp_path / "e2e_scaling.png"))
    assert_png(plot_heatmap(loaded, "cell.d", "cell.alpha_Z", "val_ppl",
                            tmp_path / "e2e_heatmap.png"))
    assert_png(plot_traces(best, tmp_path / "e2e_traces.png"))
