"""From sweep result JSONs to the tables and figures of the preprint.

Why this file exists
--------------------
`experiments/sweep.py` is deliberately dumb about presentation: it appends one fully
specified row per cell and writes `{"meta": {...}, "rows": [...]}` after every cell, so a
preempted allocation still leaves a usable result. Everything that turns those rows into a
number a reviewer can read -- a table, a scaling curve, a heatmap -- lives here, once,
instead of being re-derived in a notebook per experiment. The 2026-08 session produced its
six figures from five ad-hoc scripts, and the cost was visible: two of them computed "gap
closed" against different unigram baselines, and correction #11 of `REPORT_2026-08.md`
("best-val = min over the trace" hid a divergent run) was a presentation bug, not a
training bug. Presentation code is part of the experiment and gets the same treatment.

Three rules are baked in rather than left to the caller.

* **A failed cell is never silently invisible.** `run_sweep` catches an exception per cell
  and writes `{"cell": ..., "error": "..."}` so that a dead cell does not kill the grid.
  Dropping those rows quietly is exactly how a sweep starts looking better than it was, so
  `load_rows` reports what it dropped, both as a returned attribute and as a log line.
* **A sharded grid is one result.** `--shard i/n` splits a spec over a slurm array and
  writes `spec_width.shard0of6.json ... spec_width.shard5of6.json`. The loader globs and
  concatenates, and tolerates a shard that has not been written yet or is being written
  right now -- `out_path.write_text` is not atomic, so a read that lands mid-write sees
  truncated JSON. That is a warning, not a crash: partial results are the normal state of
  a running sweep.
* **Every figure must survive being printed in greyscale.** Series differ in marker and
  linestyle as well as colour, heatmap cells carry their value as text, and figures are
  saved at dpi >= 150 with `bbox_inches="tight"`.

Row schema consumed here (see `experiments/sweep.py:run_cell`)::

    cell            dict of the flat Cell dataclass, incl. nested `tags`
    unigram         float, the unigram-baseline perplexity of the corpus
    val_ppl         float, the *selected* checkpoint (best-validation when keep_best)
    val_ppl_final   float, end of schedule
    val_ppl_min     float, min over the trace
    best_step       int or None
    test_ppl        float, measured on the same checkpoint as val_ppl
    train_ppl       float
    val_loss        float
    params          {"embedding": int, "non_embedding": int, "total": int}
    ablation_kl     float
    seconds         float
    steps_logged    list[int] (the evaluation steps)
    val_trace       list[float], one entry per evaluation
    train_trace     list[float]
    diag_final      dict, the last diagnostics reading
    diag_trace      list[dict], one per evaluation
    swap_val_ppl    float, present on PT rows only: the same trained weights scored under
                    the other readout (Part III 17.1 vs 17.2), so `("mfvi", "swap_val_ppl")`
                    and `("exact", "val_ppl")` are directly comparable columns
    error           present only on a cell that raised

Nested fields are reached with dotted paths -- `cell.d`, `params.non_embedding`,
`diag_final.msg_over_unary`, `cell.tags.grid` -- everywhere a column or key is named.

Matplotlib is put into the Agg backend at import time, before `pyplot` is imported,
because this module runs on a cluster with no display; importing pyplot first would bind
an interactive backend and fail (or, worse, hang) inside a batch job.
"""

from __future__ import annotations

import glob as _glob
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import; see the module docstring

import matplotlib.pyplot as plt  # noqa: E402  (deliberately after matplotlib.use)
import numpy as np  # noqa: E402


__all__ = [
    "get_path",
    "LoadedRows",
    "load_rows",
    "Table",
    "to_table",
    "markdown_table",
    "best_by",
    "SeedGroup",
    "aggregate_seeds",
    "FrontierPoint",
    "scaling_frontier",
    "default_failed",
    "PowerLaw",
    "fit_power_law",
    "row_steps",
    "plot_scaling",
    "plot_heatmap",
    "plot_traces",
]


# ------------------------------------------------------------------ nested access --

_MISSING = object()


def get_path(obj: Any, path: str, default: Any = None) -> Any:
    """Read a dotted path out of nested dicts and lists.

    ``get_path(row, "params.non_embedding")`` and ``get_path(row, "cell.tags.grid")`` are
    the two shapes that actually occur in the sweep schema; ``get_path(row, "val_trace.0")``
    also works, because an integer-looking segment indexes a sequence.

    A missing key returns ``default`` rather than raising. That is the right behaviour for
    this schema and not laziness: `diag_final` is empty for every GPT and Looped row (the
    shared loop only computes diagnostics for models that expose `content_stream`), so a
    table that mixes model families must be able to ask for a diagnostic and get a blank.
    """
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            cur = cur.get(part, _MISSING)
            if cur is _MISSING:
                return default
        elif isinstance(cur, (list, tuple)):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return default
        else:
            return default
    return cur


def _as_float(value: Any) -> float:
    """Coerce a cell value to float, mapping anything unusable to NaN.

    Rows written by a crashed-then-resumed sweep, or by an older schema, can carry ``None``
    or a string where a number is expected. Propagating NaN keeps such a row visible in a
    table while keeping it out of a minimum or a mean, which is what `best_by` and
    `aggregate_seeds` want.
    """
    if isinstance(value, bool) or value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite(value: Any) -> bool:
    x = _as_float(value)
    return not (math.isnan(x) or math.isinf(x))


# ------------------------------------------------------------------------ loading --


class LoadedRows(List[Dict[str, Any]]):
    """The rows of a sweep, plus an account of everything the load discarded.

    It is a plain ``list`` of row dicts, so every other function here takes it directly.
    The extra attributes exist so that the drop count is *returned* and not only logged --
    a caller building a table can assert on `len(rows.dropped_errors)` and a test can check
    it without capturing stdout.

    Attributes
    ----------
    files : list of str
        Files that were read successfully, in load order.
    dropped_errors : list of dict
        Rows carrying an ``error`` key, dropped from the result. Kept whole so the reason
        and the offending cell can be printed or tabulated.
    unreadable : list of (str, str)
        ``(path, reason)`` for each file that could not be parsed -- a shard that is being
        written right now, or a truncated file left by a killed job.
    meta : dict
        ``meta`` blocks of the loaded files keyed by path (commit, host, torch, slurm job).
        The commit is what makes a number reproducible, so it travels with the rows.
    """

    def __init__(self, rows: Iterable[Dict[str, Any]] = ()) -> None:
        super().__init__(rows)
        self.files: List[str] = []
        self.dropped_errors: List[Dict[str, Any]] = []
        self.unreadable: List[Tuple[str, str]] = []
        self.meta: Dict[str, Dict[str, Any]] = {}

    @property
    def n_dropped(self) -> int:
        """How many cells were dropped for having raised during the sweep."""
        return len(self.dropped_errors)

    def error_report(self, full: bool = False) -> str:
        """One line per dropped cell: its name and the exception it died with.

        Only the first line of the exception text is shown unless ``full`` is set. A CUDA
        failure carries four lines of boilerplate about `CUDA_LAUNCH_BLOCKING` and
        `TORCH_USE_CUDA_DSA` after the actual message, and fifteen dead cells then produce
        sixty lines of log in which the one thing that matters -- which cells died and of
        what -- is no longer readable. The untruncated text stays in ``dropped_errors``.
        """
        lines = []
        for row in self.dropped_errors:
            message = str(row.get("error", ""))
            if not full:
                message = message.splitlines()[0] if message.splitlines() else message
            lines.append(
                "  dropped {}: {}".format(get_path(row, "cell.name", "<unnamed>"), message)
            )
        return "\n".join(lines)


def load_rows(
    *paths_or_globs: Union[str, os.PathLike],
    keep_errors: bool = False,
    log: Optional[Callable[[str], None]] = print,
) -> LoadedRows:
    """Load and concatenate sweep result files, dropping (and reporting) failed cells.

    Each argument is a path or a glob; a directory is expanded to ``<dir>/*.json``, and a
    path with no wildcard that does not exist is *also* tried as ``<stem>.shard*of*.json``
    so that ``load_rows("experiments/exp4_width_transfer/spec_width.json")`` finds the six
    shards a slurm array actually produced. Duplicate paths are read once, and the files
    are sorted so that concatenation order does not depend on the filesystem.

    A file that cannot be parsed is recorded in ``unreadable`` and skipped with a warning.
    This is expected, not exceptional: `run_sweep` rewrites the whole JSON after every
    cell with a non-atomic `write_text`, so reading a live sweep will occasionally catch a
    half-written file, and a slurm array that is still queued has shards that do not exist.
    Crashing there would make the analysis unusable exactly while the sweep is running.

    Rows carrying an ``error`` key are dropped unless ``keep_errors`` is set, and both the
    count and the rows themselves come back on the result (see :class:`LoadedRows`). A
    silently dropped failed cell is how a sweep starts looking better than it was: the
    cells that die are usually the aggressive corner of the grid, so discarding them
    without saying so biases every summary computed afterwards.

    Parameters
    ----------
    keep_errors : bool
        Keep the error rows in the returned list as well as in ``dropped_errors``.
    log : callable or None
        Where the warnings go. ``None`` silences them; the counts are still returned.
    """
    paths: List[str] = []
    for spec in paths_or_globs:
        spec = str(spec)
        if os.path.isdir(spec):
            matches = sorted(_glob.glob(os.path.join(spec, "*.json")))
        else:
            matches = sorted(_glob.glob(spec))
            if not matches and not any(ch in spec for ch in "*?["):
                # A plain, non-existent path: try the shard naming `run_sweep` uses.
                stem = spec[:-5] if spec.endswith(".json") else spec
                matches = sorted(_glob.glob(stem + ".shard*of*.json"))
        for m in matches:
            if m not in paths:
                paths.append(m)

    out = LoadedRows()
    if not paths and log is not None:
        log("load_rows: no files matched {}".format(list(paths_or_globs)))

    for path in paths:
        try:
            with open(path, "r") as fh:
                payload = json.load(fh)
        except (OSError, ValueError) as exc:
            out.unreadable.append((path, "{}: {}".format(type(exc).__name__, exc)))
            if log is not None:
                log("load_rows: skipping unreadable {} ({})".format(path, exc))
            continue
        if isinstance(payload, list):  # tolerate a bare list of rows
            rows, meta = payload, {}
        elif isinstance(payload, Mapping):
            rows, meta = payload.get("rows", []), dict(payload.get("meta", {}))
        else:
            out.unreadable.append((path, "unexpected top-level {}".format(type(payload))))
            if log is not None:
                log("load_rows: skipping {} (unexpected top-level object)".format(path))
            continue
        if not isinstance(rows, list):
            out.unreadable.append((path, "'rows' is not a list"))
            if log is not None:
                log("load_rows: skipping {} ('rows' is not a list)".format(path))
            continue

        out.files.append(path)
        out.meta[path] = meta
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            row = dict(row)
            row.setdefault("_source", path)
            if "error" in row:
                out.dropped_errors.append(row)
                if keep_errors:
                    out.append(row)
            else:
                out.append(row)

    if log is not None:
        if out.dropped_errors:
            log(
                "load_rows: dropped {} of {} rows that carried an error{}".format(
                    out.n_dropped,
                    len(out) + (0 if keep_errors else out.n_dropped),
                    ":" if out.dropped_errors else "",
                )
            )
            log(out.error_report())
        if out.unreadable:
            log(
                "load_rows: {} file(s) unreadable (still being written?)".format(
                    len(out.unreadable)
                )
            )
    return out


# ------------------------------------------------------------------------- tables --

ColumnSpec = Union[str, Tuple[str, Union[str, Callable[[Dict[str, Any]], Any]]]]


class Table(NamedTuple):
    """A rendered table: a header and a list of rows, both plain Python."""

    header: List[str]
    rows: List[List[Any]]


def _column_pair(col: ColumnSpec) -> Tuple[str, Callable[[Dict[str, Any]], Any]]:
    """Normalise a column spec to ``(label, getter)``.

    A bare string is both label and dotted path -- ``"val_ppl"``, ``"cell.d"``. A pair
    gives an explicit label and either a dotted path or a callable, which is how a derived
    column such as "gap closed against the unigram" enters a table without a new field
    having to exist in the row.
    """
    if isinstance(col, str):
        return col, (lambda row, p=col: get_path(row, p))
    label, accessor = col
    if callable(accessor):
        return label, accessor
    return label, (lambda row, p=accessor: get_path(row, p))


def to_table(rows: Sequence[Dict[str, Any]], columns: Sequence[ColumnSpec]) -> Table:
    """Project rows onto columns, returning a header and a plain list of lists.

    Values are returned unformatted -- ints stay ints, a missing field is ``None`` -- so
    that the same table can be rendered as Markdown here, written as CSV, or asserted on
    in a test without a formatting round trip in the way.

    ``columns`` accepts dotted paths (``"cell.d"``, ``"params.non_embedding"``,
    ``"diag_final.msg_over_unary"``), ``(label, path)`` pairs, and ``(label, callable)``
    pairs for derived quantities.
    """
    pairs = [_column_pair(c) for c in columns]
    header = [label for label, _ in pairs]
    body = [[getter(row) for _, getter in pairs] for row in rows]
    return Table(header, body)


def _format_cell(value: Any, floatfmt: str, missing: str) -> str:
    if value is None:
        return missing
    if isinstance(value, float):
        if math.isnan(value):
            return missing
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return format(value, floatfmt)
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def markdown_table(
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[ColumnSpec],
    floatfmt: str = ".3f",
    missing: str = "--",
    sort_by: Optional[Union[str, int]] = None,
    reverse: bool = False,
) -> str:
    """Render rows as a GitHub-flavoured Markdown table, ready to paste into a report.

    Numeric columns are right-aligned via the separator row, which is the only alignment
    control Markdown has. ``sort_by`` names either a column label, a dotted path, or a
    column index; rows whose sort key is missing or NaN go last, whichever direction is
    asked for, so a broken cell never displaces a good one at the top of a table. A numeric
    column sorts numerically and a text column (a run name, a model family, a temperature
    mode) alphabetically; when a column mixes the two, numbers come first, the same order
    :func:`best_by` uses for its key tuples.

    The floats in the sweep rows are already rounded at write time (`run_cell` rounds
    perplexities to three decimals), so ``floatfmt=".3f"`` re-prints them rather than
    inventing precision.
    """
    table = to_table(rows, columns)
    body = list(table.rows)

    if sort_by is not None:
        if isinstance(sort_by, int):
            index = sort_by
        elif sort_by in table.header:
            index = table.header.index(sort_by)
        else:
            index = None
        if index is None:
            keys = [get_path(r, str(sort_by)) for r in rows]
        else:
            keys = [r[index] for r in body]

        def sort_key(pair):
            value = pair[0]
            # Missing is None or a NaN float only. Everything else is ordered by
            # `_sortable`, which puts numbers before strings and compares strings as
            # strings: sorting a table by its run-name column used to coerce every key to
            # NaN and therefore return the rows in file order, silently -- a table that
            # says it is sorted and is not is exactly the class of presentation bug this
            # module exists to remove.
            missing = value is None or (isinstance(value, float) and math.isnan(value))
            # `reverse` flips the comparison, so the "bad last" flag is flipped with it.
            return (missing != reverse, (2, 0.0, "") if missing else _sortable(value))

        body = [row for _, row in sorted(zip(keys, body), key=sort_key, reverse=reverse)]

    text = [[_format_cell(v, floatfmt, missing) for v in row] for row in body]
    # Minimum width 3, because the alignment row of a right-aligned column is "-" * (w-1)
    # plus ":" and a one-character header such as "L" would otherwise emit a bare ":",
    # which GitHub does not accept as a separator and silently renders as a text row.
    widths = [max(3, len(h)) for h in table.header]
    for row in text:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    numeric = []
    for i in range(len(table.header)):
        column = [row[i] for row in table.rows]
        seen = [v for v in column if v is not None]
        numeric.append(
            bool(seen)
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in seen)
        )

    def line(cells: Sequence[str]) -> str:
        padded = []
        for i, cell in enumerate(cells):
            padded.append(cell.rjust(widths[i]) if numeric[i] else cell.ljust(widths[i]))
        return "| " + " | ".join(padded) + " |"

    sep = ["-" * (widths[i] - 1) + ":" if numeric[i] else "-" * widths[i]
           for i in range(len(table.header))]
    out = [line(table.header), "| " + " | ".join(sep) + " |"]
    out.extend(line(row) for row in text)
    return "\n".join(out)


# ---------------------------------------------------------------- grouping / bests --


def _key_tuple(row: Dict[str, Any], key_fields: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(get_path(row, f) for f in key_fields)


def _sortable(value: Any) -> Tuple[int, float, str]:
    """Order mixed key values deterministically: numbers first, then strings, then None."""
    if value is None:
        return (2, 0.0, "")
    if isinstance(value, bool):
        return (1, 0.0, str(value))
    if isinstance(value, (int, float)):
        return (0, float(value), "")
    return (1, 0.0, str(value))


def best_by(
    rows: Sequence[Dict[str, Any]],
    key_fields: Sequence[str],
    metric: str = "val_ppl",
    lower_is_better: bool = True,
) -> List[Dict[str, Any]]:
    """One row per distinct combination of ``key_fields``: the best on ``metric``.

    This is how a scaling curve takes the better of the two learning rates at each point.
    `experiments/exp2_scaling/spec_baselines.py` runs every architecture point at
    ``lr in {1e-3, 3e-3}`` and says so explicitly: "the better of the two is the point on
    the curve ... a curve whose points are individually under-tuned understates the
    baseline, and the reviewers' question is precisely about fairness". Selecting here,
    rather than in each plotting call, keeps that rule in one place.

    Rows whose metric is missing, NaN or infinite are ignored when choosing a winner; a
    group in which *every* row is like that yields no output row at all, which is louder
    than emitting a group with a blank metric.

    The result is sorted by the key tuple (numbers before strings) so that a table built
    from it is stable across runs and across shard arrival order.
    """
    groups: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()
    sign = 1.0 if lower_is_better else -1.0
    for row in rows:
        value = _as_float(get_path(row, metric))
        if math.isnan(value) or math.isinf(value):
            continue
        key = _key_tuple(row, key_fields)
        current = groups.get(key)
        if current is None or sign * value < sign * _as_float(get_path(current, metric)):
            groups[key] = row
    ordered = sorted(groups.items(), key=lambda kv: tuple(_sortable(v) for v in kv[0]))
    return [row for _, row in ordered]


class SeedGroup(NamedTuple):
    """A multi-seed cell reduced to the ``X +- Y (n=k)`` a table reports."""

    key: Tuple[Any, ...]
    mean: float
    std: float
    n: int
    seeds: List[Any]
    values: List[float]
    rows: List[Dict[str, Any]]

    def format(self, fmt: str = ".2f") -> str:
        """``"250.69 +- 2.32 (n=3)"`` -- the form used throughout REPORT_2026-08.md."""
        return "{} +- {} (n={})".format(format(self.mean, fmt), format(self.std, fmt), self.n)


def aggregate_seeds(
    rows: Sequence[Dict[str, Any]],
    key_fields: Sequence[str],
    metric: str = "val_ppl",
    seed_field: str = "cell.seed",
) -> List[SeedGroup]:
    """Group rows by ``key_fields`` and reduce the metric to mean, std, n and the seeds.

    The standard deviation is the **sample** standard deviation (``ddof=1``, i.e. dividing
    by ``n - 1``) when ``n > 1``, and exactly ``0.0`` when ``n == 1``. Sample rather than
    population because the seeds are a sample of the run-to-run distribution and the number
    is used as an error bar on the mean, not as a description of the three runs actually
    performed; ``0.0`` for a single seed because there is no estimate at all there, and
    reporting NaN would poison every downstream comparison while reporting a population
    std of 0 would claim a precision that does not exist. A single-seed cell must be read
    as ``n=1``, which is why ``n`` travels with the number everywhere.

    The 2026-08 record was ``250.69 +- 2.32`` over three seeds and the sequel replication
    withdrew a 241.89 single-seed record precisely because it was single-seed; the ``n``
    in the output is doing real work.

    The seed list comes back so that a missing or duplicated seed is visible: three rows
    all at ``seed=0`` are not three seeds, and only the list shows that.
    """
    groups: "OrderedDict[Tuple[Any, ...], List[Dict[str, Any]]]" = OrderedDict()
    for row in rows:
        value = _as_float(get_path(row, metric))
        if math.isnan(value) or math.isinf(value):
            continue
        groups.setdefault(_key_tuple(row, key_fields), []).append(row)

    out: List[SeedGroup] = []
    for key, group in sorted(groups.items(), key=lambda kv: tuple(_sortable(v) for v in kv[0])):
        values = [_as_float(get_path(r, metric)) for r in group]
        seeds = [get_path(r, seed_field) for r in group]
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in values) / (n - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        out.append(SeedGroup(key, mean, std, n, seeds, values, list(group)))
    return out


# ---------------------------------------------------------------- scaling analysis --


class FrontierPoint(NamedTuple):
    """One point of a Pareto lower envelope, unpackable as ``(x, y, row)``."""

    x: float
    y: float
    row: Dict[str, Any]


def scaling_frontier(
    rows: Sequence[Dict[str, Any]],
    x: str = "params.non_embedding",
    y: str = "val_ppl",
) -> List[FrontierPoint]:
    """The lower envelope of ``y`` against ``x``: best result achieved at or below each x.

    A scaling plot should draw the frontier, not the cloud. Every point in the cloud that
    is beaten by a *smaller* model is not on any architecture's scaling curve -- it is a
    configuration that wasted parameters -- and joining cloud points with a line produces
    a curve that goes up and down for reasons that have nothing to do with scaling. The
    2026-08 width ladder was non-monotone (``d = 48`` worse than both 32 and 64) exactly
    because of mis-transferred hyperparameters, and a frontier states that honestly: the
    dominated point is simply absent, rather than drawn as a bump in a "scaling law".

    Ties at the same ``x`` keep the better ``y``. Non-positive or non-finite ``x`` or ``y``
    are dropped, since the frontier feeds a log-log fit.

    Returns points sorted by increasing ``x``, each with the row it came from so that a
    caller can annotate a point with its configuration.
    """
    points: List[Tuple[float, float, Dict[str, Any]]] = []
    for row in rows:
        xv, yv = _as_float(get_path(row, x)), _as_float(get_path(row, y))
        if not (_finite(xv) and _finite(yv)) or xv <= 0 or yv <= 0:
            continue
        points.append((xv, yv, row))
    points.sort(key=lambda p: (p[0], p[1]))

    frontier: List[FrontierPoint] = []
    best = float("inf")
    for xv, yv, row in points:
        if yv < best:
            # Strictly better than everything at or below this x: a real frontier point.
            if frontier and frontier[-1].x == xv:
                frontier[-1] = FrontierPoint(xv, yv, row)
            else:
                frontier.append(FrontierPoint(xv, yv, row))
            best = yv
    return frontier


class PowerLaw(NamedTuple):
    """``log10(y) = exponent * log10(x) + intercept``, with the log-space ``r2``."""

    exponent: float
    intercept: float
    r2: float

    def predict(self, x: float) -> float:
        """The fitted ``y`` at ``x`` -- ``10**(intercept) * x**exponent``."""
        return 10.0 ** (self.exponent * math.log10(x) + self.intercept)

    def label(self) -> str:
        """``"own range: -0.43"`` -- what goes in a legend entry.

        Short on purpose: with R2 included the entries reach the left edge of the axes and
        overlap the baseline annotation. The goodness of fit belongs in the appendix table,
        which has room for it beside the point counts.

        The qualifier is not decoration. Each curve here is fitted over the parameter range it
        was measured on, and those ranges differ by more than an order of magnitude, so these
        slopes are *not* comparable with each other -- reading them as a comparison is the
        specific error the scaling section exists to correct. A legend is read before the text
        beside it, so the legend has to carry the caveat itself.
        """
        return "own range: {:.3f}".format(self.exponent)


def fit_power_law(xs: Sequence[float], ys: Sequence[float]) -> Optional[PowerLaw]:
    """Least squares on ``log10(x)`` against ``log10(y)``; ``None`` if underdetermined.

    This is what makes a scaling plot quantitative rather than decorative: the reviewers'
    question is not only "which curve is lower" but "how do the curves' *slopes* differ"
    (`experiments/exp2_scaling/spec_baselines.py`), and the slope is the exponent of
    ``y = 10**intercept * x**exponent``. A line drawn through three points with no
    reported exponent and no ``r2`` cannot answer it.

    The fit is the closed-form ordinary least squares on the log-transformed pairs, which
    for exactly power-law data recovers the exponent to floating-point precision -- the
    property `tests/test_17_analysis.py` asserts to 1e-6.

    Returns ``None`` when fewer than three distinct ``(x, y)`` points survive, or when all
    surviving points share one ``x`` (the slope is then undefined). Three is the smallest
    number at which ``r2`` says anything at all: two points always fit a line exactly, so
    an ``r2`` of 1.0 from two points is not evidence of a power law.

    ``r2`` is computed in log space, on the quantity actually fitted. When the ``y`` values
    are identical the total sum of squares is zero and there is no variance to explain;
    the result is then ``1.0`` for an exact fit (a flat line through flat data) and ``0.0``
    otherwise, which cannot happen for a least-squares line through constant data.
    """
    pairs = []
    for xv, yv in zip(xs, ys):
        xf, yf = _as_float(xv), _as_float(yv)
        if not (_finite(xf) and _finite(yf)) or xf <= 0 or yf <= 0:
            continue
        pairs.append((math.log10(xf), math.log10(yf)))
    distinct = set(pairs)
    if len(distinct) < 3:
        return None
    if len({p[0] for p in pairs}) < 2:
        return None

    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    if sxx == 0.0:
        return None
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((p[1] - (slope * p[0] + intercept)) ** 2 for p in pairs)
    ss_tot = sum((p[1] - my) ** 2 for p in pairs)
    if ss_tot == 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    return PowerLaw(slope, intercept, r2)


# ------------------------------------------------------------------------ plotting --

# Marker and linestyle cycles come before colour on purpose. Half of the readers of a
# preprint print it, and a figure whose series differ only in hue is unreadable in
# greyscale; varying three channels at once means the series stay separable in colour,
# in greyscale, and for a red-green colour-blind reader.
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
_LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1))]
_COLOURS = [
    "#1b3a5c", "#c1440e", "#2b7a3d", "#7d3c98", "#b58900",
    "#0e7c7b", "#8b1a3a", "#4a4a4a",
]


def _pretty_number(value: float, _pos: Any = None) -> str:
    """Tick label for a count or a perplexity: ``13k``, ``200k``, ``1.5``, ``300``.

    A parameter count reads better abbreviated than as ``2 x 10^5``, and a perplexity in
    the 100-700 band reads better as a plain integer than as ``6 x 10^2``.
    """
    if value <= 0:
        return ""
    for divisor, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if value >= divisor:
            scaled = value / divisor
            body = ("{:.0f}" if abs(scaled - round(scaled)) < 0.05 else "{:.1f}").format(scaled)
            return body + suffix
    if abs(value - round(value)) < 0.05:
        return "{:.0f}".format(value)
    return "{:g}".format(value)


def _log_ticks(axis, values: Sequence[float]) -> None:
    """Label a log axis with a readable number of plain ticks.

    Matplotlib's default on a log axis labels the decades and then, when fewer than two
    decades are shown, adds minor labels such as ``2 x 10^4`` and ``3 x 10^4`` that collide
    with each other on a narrow figure -- observed directly on the exp2 baseline curve,
    where the x range 13k-450k produced overlapping labels. The fix is to choose the
    subdivisions from the span actually plotted and to label majors only.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    finite = [v for v in values if _finite(v) and v > 0]
    if not finite:
        return
    span = math.log10(max(finite) / min(finite)) if min(finite) > 0 else 0.0
    if span >= 2.0:
        subs = (1.0,)
    elif span >= 1.0:
        subs = (1.0, 2.0, 5.0)
    else:
        subs = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)
    axis.set_major_locator(LogLocator(base=10.0, subs=subs, numticks=12))
    axis.set_major_formatter(FuncFormatter(_pretty_number))
    axis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=64))
    axis.set_minor_formatter(NullFormatter())


def _style(index: int) -> Dict[str, Any]:
    return {
        "marker": _MARKERS[index % len(_MARKERS)],
        "linestyle": _LINESTYLES[index % len(_LINESTYLES)],
        "color": _COLOURS[index % len(_COLOURS)],
    }


def _save(fig, out_path: Union[str, os.PathLike], dpi: int) -> Path:
    """Write a figure and close it.

    dpi is floored at 150 and `bbox_inches="tight"` is always applied: a figure that is
    legible on screen at 100 dpi is a blurred mess in a two-column PDF, and the default
    bounding box clips rotated tick labels and the outer axis label of a twin axis.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=max(150, int(dpi)), bbox_inches="tight")
    plt.close(fig)
    return path


def _curve_points(
    data: Any, x: str, y: str, frontier: bool
) -> Tuple[List[float], List[float]]:
    """Turn one entry of ``curves`` into ``(xs, ys)``.

    Accepts a list of sweep rows (reduced to its frontier unless ``frontier=False``) or a
    ready sequence of ``(x, y)`` pairs, so a caller can draw a curve that was computed by
    hand -- a theoretical ``2 K h d^2 + h d`` parameter count, say -- next to a measured one.
    """
    seq = list(data)
    if seq and isinstance(seq[0], Mapping):
        if frontier:
            pts = scaling_frontier(seq, x=x, y=y)
            return [p.x for p in pts], [p.y for p in pts]
        pairs = sorted(
            (_as_float(get_path(r, x)), _as_float(get_path(r, y))) for r in seq
        )
        pairs = [(a, b) for a, b in pairs if _finite(a) and _finite(b) and a > 0 and b > 0]
        return [a for a, _ in pairs], [b for _, b in pairs]
    pairs = sorted((_as_float(a), _as_float(b)) for a, b in seq)
    return [a for a, _ in pairs], [b for _, b in pairs]


def plot_scaling(
    curves: Union[Mapping[str, Any], Sequence[Tuple[str, Any]]],
    out_path: Union[str, os.PathLike],
    x: str = "params.non_embedding",
    y: str = "val_ppl",
    x_label: str = "non-embedding parameters (count)",
    y_label: str = "validation perplexity (per token, lower is better)",
    title: Optional[str] = None,
    frontier: bool = True,
    fit: bool = True,
    hline: Optional[Tuple[float, str]] = None,
    figsize: Tuple[float, float] = (6.4, 4.4),
    dpi: int = 200,
) -> Path:
    """Draw several labelled series on log-log axes, with the fitted exponent in the legend.

    ``curves`` maps a label to either a list of sweep rows or a sequence of ``(x, y)``
    pairs; passing a sequence of ``(label, data)`` pairs instead fixes the drawing and
    legend order, which matters when the reader is meant to compare two architectures in a
    particular direction. Rows are reduced to their Pareto lower envelope by default (see
    :func:`scaling_frontier`), so the drawn line is a scaling curve rather than a tour of
    the grid.

    Log-log because that is the axis pair on which a power law is a straight line, and the
    question being asked of these curves -- "how do the slopes differ" -- is a question
    about that line. Each legend entry carries the fitted exponent and ``r2`` from
    :func:`fit_power_law` (omitted for a series with too few distinct points to fit), so
    the quantitative claim is on the figure and not only in the caption.

    ``hline`` draws a horizontal reference with a label, which is where the unigram
    baseline goes: on PTB it is 688.8 and it is the line that separates "learned something"
    from "learned the unigram" -- the distinction the whole 2026-08 report turns on.
    """
    items = list(curves.items()) if isinstance(curves, Mapping) else list(curves)
    fig, ax = plt.subplots(figsize=figsize)
    all_x: List[float] = []
    all_y: List[float] = []

    for i, (label, data) in enumerate(items):
        xs, ys = _curve_points(data, x, y, frontier)
        if not xs:
            continue
        all_x.extend(xs)
        all_y.extend(ys)
        style = _style(i)
        text = label
        if fit:
            law = fit_power_law(xs, ys)
            if law is not None:
                text = "{} ({})".format(label, law.label())
        ax.plot(
            xs, ys, label=text, markersize=6, linewidth=1.6,
            markerfacecolor="none", markeredgewidth=1.4, **style
        )

    if hline is not None:
        value, hlabel = hline
        ax.axhline(value, color="0.35", linestyle=(0, (1, 2)), linewidth=1.2)
        # Left-aligned and *below* the line. Above it the label is overlapped by the legend,
        # whose entries grew long enough to reach the left edge; below it there is empty space
        # between the reference and the topmost curve in every plot that uses this.
        ax.annotate(
            hlabel, xy=(0.01, value), xycoords=("axes fraction", "data"),
            xytext=(0, -3), textcoords="offset points",
            ha="left", va="top", fontsize=8, color="0.25",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    _log_ticks(ax.xaxis, all_x)
    _log_ticks(ax.yaxis, all_y + ([hline[0]] if hline is not None else []))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
    if ax.get_legend_handles_labels()[0]:
        # Framed and placed away from the reference line: with five series the legend is large
        # enough that an unframed one sitting on top of the curves is unreadable.
        ax.legend(fontsize=8, frameon=True, framealpha=0.92, edgecolor="0.8",
                  loc="upper right", borderpad=0.5)
    return _save(fig, out_path, dpi)


def _axis_values(rows: Sequence[Dict[str, Any]], key: str) -> List[Any]:
    """Distinct values of an axis key: numeric ones sorted, categorical in first-seen order.

    Sorting ``temp_mode`` alphabetically would put ``qkn2`` before ``qnorm`` and separate
    the two ``qknorm`` gains for no reason; the spec lists them in a meaningful order
    (`fixed, qnorm, qkn2, qkn4, qkn8`) and first-appearance order preserves it. Numeric
    axes such as ``d`` and ``alpha_Z`` are sorted, because their order is their meaning.
    """
    seen: List[Any] = []
    for row in rows:
        v = get_path(row, key)
        if v is not None and v not in seen:
            seen.append(v)
    if seen and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in seen):
        return sorted(seen)
    return seen


def default_failed(row: Dict[str, Any], fraction: float = 0.98) -> bool:
    """Did this cell fail to learn -- i.e. land on the unigram baseline?

    The criterion is the one the 2026-08 report used to separate its runs: a model whose
    validation perplexity is within ``fraction`` of the corpus unigram perplexity has not
    learned anything a bigram could not, whatever its loss curve looks like. The unigram
    number travels in the row (`run_cell` writes ``unigram``), so the comparison is against
    *that cell's own* corpus baseline and not against a constant copied between scripts --
    which is how two figures of the 2026-08 session ended up with different baselines.

    A row without a ``unigram`` field cannot be judged and is reported as not failed.
    """
    ppl = _as_float(get_path(row, "val_ppl"))
    base = _as_float(get_path(row, "unigram"))
    if math.isnan(ppl) or math.isnan(base) or base <= 0:
        return False
    return ppl >= fraction * base


def plot_heatmap(
    rows: Sequence[Dict[str, Any]],
    xkey: str,
    ykey: str,
    vkey: str,
    out_path: Union[str, os.PathLike],
    agg: str = "min",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    v_label: Optional[str] = None,
    title: Optional[str] = None,
    fmt: str = ".0f",
    cmap: str = "viridis_r",
    failed: Optional[Callable[[Dict[str, Any]], bool]] = default_failed,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 200,
) -> Path:
    """A ``xkey x ykey`` grid of ``vkey``, annotated per cell and marking failed cells.

    Built for the two grids of the width-transfer experiment -- ``d x alpha_Z`` (grid A,
    "does the optimal step size scale with width") and ``d x temp_mode`` (grid B, "does a
    belief-norm temperature remove the need for damping") -- and for any other two-factor
    grid with the same shape.

    Every cell carries its value as text, which is what makes the figure work in
    greyscale: the colour is a fast index, the number is the datum. Two states are marked
    rather than left to be inferred from a colour near the top of the scale:

    * **failed to learn** -- hatched, with the value suffixed ``*``. A cell at the unigram
      baseline and a cell slightly above it are the same colour but not the same result,
      and the whole 2026-08 diagnosis rests on that distinction.
    * **missing** -- a combination with no row, drawn in flat grey with ``n/a``. A shard
      that has not finished must not look like a bad result.

    ``agg`` (``min``/``max``/``mean``) decides what to do when several rows fall in one
    cell -- typically seeds, where ``min`` reports the best and ``mean`` the expected one.
    ``min`` is the default to match :func:`best_by`, and a mean over an unknown number of
    seeds would be the more misleading default.
    """
    xs = _axis_values(rows, xkey)
    ys = _axis_values(rows, ykey)
    if not xs or not ys:
        raise ValueError(
            "plot_heatmap: no values for {!r} / {!r} in {} rows".format(xkey, ykey, len(rows))
        )

    buckets: Dict[Tuple[Any, Any], List[Tuple[float, Dict[str, Any]]]] = {}
    for row in rows:
        xv, yv = get_path(row, xkey), get_path(row, ykey)
        value = _as_float(get_path(row, vkey))
        if xv is None or yv is None or math.isnan(value):
            continue
        buckets.setdefault((xv, yv), []).append((value, row))

    grid = np.full((len(ys), len(xs)), np.nan)
    fail_grid = np.zeros((len(ys), len(xs)), dtype=bool)
    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            entries = buckets.get((xv, yv))
            if not entries:
                continue
            values = [v for v, _ in entries]
            if agg == "min":
                pick = min(range(len(values)), key=lambda k: values[k])
                grid[j, i] = values[pick]
            elif agg == "max":
                pick = max(range(len(values)), key=lambda k: values[k])
                grid[j, i] = values[pick]
            elif agg == "mean":
                pick = 0
                grid[j, i] = sum(values) / len(values)
            else:
                raise ValueError("plot_heatmap: unknown agg {!r}".format(agg))
            if failed is not None:
                # A cell is marked failed if *every* run in it failed; one surviving run in
                # a seed group means the configuration can learn, and hatching it would
                # overstate the result.
                fail_grid[j, i] = all(failed(r) for _, r in entries)

    if figsize is None:
        figsize = (max(4.0, 1.05 * len(xs) + 2.2), max(3.0, 0.75 * len(ys) + 1.8))
    fig, ax = plt.subplots(figsize=figsize)

    masked = np.ma.masked_invalid(grid)
    # Missing cells get flat grey, never a colour taken from the value scale -- but the grey
    # is painted as an explicit patch in the annotation loop below rather than as the
    # colormap's "bad" colour. Routing it through the colormap needs a *copy* of the
    # colormap (mutating the registered instance recolours every later figure in the
    # process), and both spellings of that copy, `Colormap.with_extremes` and
    # `Colormap.copy`, arrived in matplotlib 3.4. The cluster interpreter
    # (/public/software/anaconda3, python 3.8.8) carries matplotlib 3.3.4, where
    # `plt.get_cmap("viridis_r").copy()` raises `AttributeError` and no figure is written
    # at all. A rectangle over the masked cell needs no colormap surgery, renders the same
    # on both versions, and -- the reason it is worth the four lines -- means the login-node
    # tests exercise exactly the code path the cluster runs, with no version branch between
    # them.
    image = ax.imshow(masked, cmap=cmap, aspect="auto", origin="lower")
    bar = fig.colorbar(image, ax=ax)
    bar.set_label(v_label or vkey)

    finite = masked.compressed()
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    span = (hi - lo) or 1.0
    for j in range(len(ys)):
        for i in range(len(xs)):
            value = grid[j, i]
            if np.isnan(value):
                # zorder 1 puts the patch above the image (zorder 0) and below the text
                # (zorder 3), so the cell reads as flat grey whatever the colormap does
                # with a masked entry.
                ax.add_patch(
                    plt.Rectangle(
                        (i - 0.5, j - 0.5), 1, 1, facecolor="0.85", edgecolor="none",
                        zorder=1,
                    )
                )
                ax.text(i, j, "n/a", ha="center", va="center", fontsize=8, color="0.35")
                continue
            # Dark text on the light half of the scale and vice versa, so the annotation
            # is readable at both ends and after a greyscale conversion.
            shade = (value - lo) / span
            rgba = plt.get_cmap(cmap)(shade)
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text = format(value, fmt) + ("*" if fail_grid[j, i] else "")
            ax.text(
                i, j, text, ha="center", va="center", fontsize=8,
                color="white" if luminance < 0.5 else "black",
            )
            if fail_grid[j, i]:
                ax.add_patch(
                    plt.Rectangle(
                        (i - 0.5, j - 0.5), 1, 1, fill=False, hatch="xx",
                        edgecolor="black", linewidth=0.0,
                    )
                )

    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(v) for v in xs])
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([str(v) for v in ys])
    ax.set_xlabel(x_label or xkey)
    ax.set_ylabel(y_label or ykey)
    if title:
        ax.set_title(title)
    if fail_grid.any():
        ax.plot(
            [], [], marker="s", linestyle="none", markerfacecolor="white",
            markeredgecolor="black", label="* did not learn (at the unigram baseline)",
        )
        ax.legend(fontsize=7, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.13))
    return _save(fig, out_path, dpi)


def row_steps(row: Dict[str, Any], n: Optional[int] = None) -> List[int]:
    """The training steps at which a row's traces were measured.

    `run_cell` writes ``steps_logged`` straight from ``History.step``, which is the list of
    evaluation steps, so that is used when it is a list of the right length. Older rows
    stored a scalar there, and some hand-built rows have no such field at all; in both
    cases the steps are reconstructed as multiples of ``cell.eval_every`` (the shared loop
    evaluates every ``eval_every`` steps and once more at the end), falling back to a plain
    1-based index. Reconstructing rather than refusing keeps a figure drawable from an
    older JSON, and the x-axis label always says "training step" so a reader is never
    misled about what the axis is.
    """
    if n is None:
        n = len(row.get("val_trace") or [])
    logged = row.get("steps_logged")
    if isinstance(logged, list) and len(logged) == n and all(_finite(v) for v in logged):
        return [int(v) for v in logged]
    every = get_path(row, "cell.eval_every")
    every = int(every) if _finite(every) and _as_float(every) > 0 else None
    if every:
        return [every * (k + 1) for k in range(n)]
    return list(range(1, n + 1))


def _row_label(row: Dict[str, Any], label: Union[None, str, Callable[[Dict[str, Any]], str]]) -> str:
    if callable(label):
        return str(label(row))
    if isinstance(label, str):
        return str(get_path(row, label, label))
    return str(get_path(row, "cell.name", "cell"))


def plot_ladder(
    rungs: Sequence[Tuple[int, str, float]],
    out_path: Union[str, os.PathLike],
    anchors: Optional[Tuple[float, float]] = None,
    y_label: str = "validation perplexity (per token, lower is better)",
) -> str:
    """The cumulative switch ladder: rung index against perplexity.

    Not a scaling plot, and it was drawn with the scaling helper, which forces a logarithmic
    x-axis. A *count* of switches from 1 to 10 on a log axis puts rungs 1 and 2 a third of the
    width apart and rungs 9 and 10 almost on top of each other, which misreads the shape of the
    curve the figure exists to show. Linear in the rung index, therefore, with the name of the
    switch flipped at each step under its tick -- the reader's question at every point of this
    figure is "which one was that", and the table is two pages away.

    The y-axis stays logarithmic: the ladder spans roughly 130 to 700 perplexity and the
    interesting structure is at the bottom.

    ``rungs`` is ``(index, switch name, value)``; ``anchors`` is the (transformer, PT) pair,
    drawn as horizontal references so that the cliff can be read against the span it is a
    fraction of.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rungs = sorted(rungs, key=lambda r: r[0])
    xs = [r[0] for r in rungs]
    ys = [r[2] for r in rungs]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))

    if anchors is not None:
        a_tr, a_pt = anchors
        for value, text, colour in ((a_tr, "all-transformer", "0.45"),
                                    (a_pt, "all-PT", "0.45")):
            ax.axhline(value, color=colour, linestyle=(0, (1, 2)), linewidth=1.1)
            ax.annotate(f"{text} {value:.0f}", xy=(0.99, value),
                        xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points",
                        ha="right", va="bottom", fontsize=8, color="0.3")

    ax.plot(xs, ys, "-o", color="#27496d", markersize=5,
            markerfacecolor="white", markeredgewidth=1.3, linewidth=1.4, zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[1].replace("_", "\n") for r in rungs], fontsize=7.5)
    ax.tick_params(axis="x", length=0)
    _log_ticks(ax.yaxis, ys + ([] if anchors is None else list(anchors)))
    ax.set_xlabel("switches flipped, cumulatively, from a causal transformer towards the PT")
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", which="both", linewidth=0.4, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return str(out_path)


def plot_traces(
    rows: Sequence[Dict[str, Any]],
    out_path: Union[str, os.PathLike],
    diag: Optional[str] = "msg_over_unary",
    trace: str = "val_trace",
    label: Union[None, str, Callable[[Dict[str, Any]], str]] = None,
    x_label: str = "training step (optimiser steps)",
    y_label: str = "validation perplexity (per token)",
    diag_label: Optional[str] = None,
    diag_band: Optional[Tuple[Optional[float], Optional[float]]] = None,
    logy: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7.2, 4.4),
    dpi: int = 200,
) -> Path:
    """Validation-perplexity traces against step, with a diagnostic on a second axis.

    This reproduces Fig. 1 of `REPORT_2026-08.md` from the new row schema. That figure is
    the evidence for the session's first mechanism: "the message/unary balance decides
    learning, and the transition is visible within a run" -- in run 940844 validation
    perplexity drops 703.6 -> 539.0 at the very evaluation where ``msg/unary`` first enters
    the healthy range, and in run 940848 an excursion out of the range coincides with
    perplexity reverting 516 -> 694. The claim is about the *alignment in step* of two
    series, so the two series have to share an x-axis; putting them in separate panels
    would destroy the only thing the figure shows.

    The perplexity axis is drawn with solid lines and filled markers, the diagnostic axis
    with dashed lines and open markers, and each row keeps one colour across both, so the
    pairing survives greyscale printing.

    ``diag`` names a key inside ``diag_trace`` (``msg_over_unary``, ``attn_entropy_frac``,
    ``rho``, ``max_abs_T``, ...); pass ``None`` to draw perplexity alone. ``diag_band``
    shades an acceptable range for the diagnostic. For ``msg_over_unary`` the honest band
    is a *ceiling* of about 5 and not an interval -- correction #10 of the 2026-08 report
    withdrew the two-sided reading after the then-best run (336.89) was rejected by it --
    so pass ``(None, 5.0)`` rather than a lower bound invented for symmetry.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx() if diag else None
    drew_diag = False
    all_y: List[float] = []

    for i, row in enumerate(rows):
        values = [_as_float(v) for v in (row.get(trace) or [])]
        if not values:
            continue
        all_y.extend(v for v in values if _finite(v) and v > 0)
        steps = row_steps(row, len(values))
        style = _style(i)
        name = _row_label(row, label)
        ax.plot(
            steps, values, label=name, linewidth=1.7, markersize=4.5,
            marker=style["marker"], linestyle="-", color=style["color"],
        )
        if not diag:
            continue
        dtrace = row.get("diag_trace") or []
        dvalues, dsteps = [], []
        for k, entry in enumerate(dtrace[: len(steps)]):
            if isinstance(entry, Mapping) and diag in entry and _finite(entry[diag]):
                dvalues.append(_as_float(entry[diag]))
                dsteps.append(steps[k])
        if dvalues:
            drew_diag = True
            ax2.plot(
                dsteps, dvalues, linewidth=1.3, markersize=4.5, alpha=0.9,
                marker=style["marker"], linestyle="--", color=style["color"],
                markerfacecolor="none", markeredgewidth=1.2,
            )

    if ax2 is not None and not drew_diag:
        # Nothing was drawn on the twin axis, so take it off the figure. Leaving it in
        # place puts an unlabelled 0.0-1.0 scale down the right-hand side, and that is the
        # *normal* case for a GPT or Looped row: the shared loop only fills `diag_trace`
        # for models that expose `content_stream`, so every baseline figure would carry a
        # second axis a reader can only read as a real quantity.
        ax2.remove()
        ax2 = None

    if diag and drew_diag and diag_band is not None:
        lo, hi = diag_band
        lo = ax2.get_ylim()[0] if lo is None else lo
        hi = ax2.get_ylim()[1] if hi is None else hi
        ax2.axhspan(lo, hi, color="0.6", alpha=0.15, zorder=0)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if logy:
        ax.set_yscale("log")
        _log_ticks(ax.yaxis, all_y)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    if diag and drew_diag:
        ax2.set_ylabel(
            (diag_label or "{} (dashed, right axis)".format(diag))
        )
    if title:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if drew_diag:
        # marker="" and not marker="none": the string "none" is only accepted as a marker
        # from matplotlib 3.4 onward and raises `ValueError: Unrecognized marker style
        # 'none'` on the 3.3.4 that ships with the cluster python 3.8.8, killing the figure
        # at the last line of the function. The empty string means "no marker" in every
        # version.
        handles.append(plt.Line2D([], [], color="0.3", linestyle="--", marker=""))
        # Prefer the human-readable label the caller gave for the axis over the raw column
        # name: "q_sharpness" is a dictionary key, not something a reader should have to parse.
        labels.append("{} (dashed, right axis)".format(diag_label or diag))
    if handles:
        # Below the axes rather than inside them. These traces fill the plotting area -- a
        # perplexity curve falling from the top-left and a diagnostic rising from the
        # bottom-left leave no interior corner free -- and a legend placed inside sits on the
        # data whichever corner it is given.
        ncol = 2 if len(labels) > 3 else 1
        ax.legend(handles, labels, fontsize=8, frameon=False, ncol=ncol,
                  loc="upper center", bbox_to_anchor=(0.5, -0.16))
    return _save(fig, out_path, dpi)
