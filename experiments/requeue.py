"""Find shards that are not finished and are not running, and put them back in the queue.

Why this is a script and not a habit. Jobs on this cluster die for reasons that have nothing
to do with the experiment: a node reports a GPU it cannot allocate (six deaths on 2026-09-08),
`/public` is unmounted on two nodes, an allocation expires. The sweep runner already makes each
of those recoverable --- rows are written per cell and a re-run resumes --- but recovery only
happens if somebody notices, and a grid with one silently missing shard looks exactly like a
grid with a hole in its design.

So: enumerate what each spec expects, count what is on disk, and re-queue the difference.
A shard is re-queued only when it is (a) incomplete and (b) not currently running under some
slurm job, so this is safe to run repeatedly, including from a loop.

    python -m experiments.requeue                 # report and append to experiments/pending.txt
    python -m experiments.requeue --dry-run       # report only
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent
PENDING = REPO / "experiments" / "pending.txt"

# spec module -> (number of shards, extra args passed to build_cells)
SPECS: Dict[str, Tuple[int, List[str]]] = {
    "experiments.exp2_scaling.spec_baselines": (4, []),
    "experiments.exp2_scaling.spec_pt": (6, ["qknorm", "1.0", "0.25", "", "1.0"]),
    "experiments.exp4_width_transfer.spec_width": (6, []),
    "experiments.exp4_width_transfer.spec_gain": (4, []),
    "experiments.exp4_width_transfer.spec_lr_transfer": (3, []),
    "experiments.exp4_width_transfer.spec_rank": (2, []),
    "experiments.exp5_init.spec_init": (4, []),
    "experiments.exp3_readout.spec_open": (5, []),
    "experiments.exp6_switches.spec_switches": (6, []),
}


def running_shards() -> Set[str]:
    """`spec --shard i/n` strings that a live slurm job is already working on.

    Read from the job's own log, because that is where the runner prints the arguments it was
    given; squeue does not carry them.
    """
    out: Set[str] = set()
    try:
        ids = subprocess.check_output(
            ["squeue", "-u", "belyack", "-h", "-o", "%i"], stderr=subprocess.DEVNULL
        ).decode().split()
    except Exception:
        return out
    for jid in ids:
        log = REPO / f"slurm-{jid}.out"
        if not log.exists():
            continue
        head = log.read_text(errors="replace")[:4000]
        m = re.search(r"^spec: (\S+)\s+args: (.*)$", head, re.M)
        if m:
            out.add(f"{m.group(1)} {m.group(2)}".strip())
    return out


def done_cells(spec: str, shard: int, n: int) -> int:
    mod = importlib.import_module(spec)
    path = Path(mod.__file__).with_suffix("")
    f = path.parent / f"{path.name}.shard{shard}of{n}.json"
    if not f.exists():
        return 0
    try:
        rows = json.loads(f.read_text())["rows"]
    except Exception:
        return 0  # a file being written mid-flight; treat as unknown and let the check retry
    return sum(1 for r in rows if "error" not in r)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    live = running_shards()
    pending = set(PENDING.read_text().splitlines()) if PENDING.exists() else set()
    to_add: List[str] = []

    for spec, (n, extra) in SPECS.items():
        try:
            mod = importlib.import_module(spec)
            cells = mod.build_cells(*extra)
        except Exception as exc:
            print(f"{spec}: cannot build cells ({type(exc).__name__}: {exc})")
            continue
        for i in range(n):
            expect = len([c for k, c in enumerate(cells) if k % n == i])
            have = done_cells(spec, i, n)
            line = f"{spec} --shard {i}/{n}"
            state = "running" if line in live else ("queued" if line in pending else "idle")
            flag = ""
            if have < expect and state == "idle":
                to_add.append(line)
                flag = "  <- REQUEUE"
            print(f"{spec.split('.')[-1]:>16} shard {i}/{n}: {have:>3}/{expect:<3} {state}{flag}")

    if not to_add:
        print("\nnothing to re-queue")
        return
    print(f"\n{len(to_add)} shard(s) to re-queue")
    if a.dry_run:
        return
    with open(PENDING, "a") as fh:
        for line in to_add:
            fh.write(line + "\n")
    print(f"appended to {PENDING}")


if __name__ == "__main__":
    main()
