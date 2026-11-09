"""Run cs336_systems.benchmark over a grid of configs and tabulate the results.

Each config runs in a fresh Python process, so one config's CUDA/cuBLAS
initialization, allocator state, or out-of-memory failure cannot leak into the
next one, and the "no warm-up" measurements really start cold.

Examples:
    # Part (b): every model size, full training step, 5 warm-up + 10 measured steps
    uv run python -m cs336_systems.sweep --out results/benchmark.jsonl

    # Part (c): effect of the number of warm-up steps
    uv run python -m cs336_systems.sweep --warmups 0 1 2 5 --out results/warmup.jsonl

    # Re-print the tables (plain text and LaTeX) from an earlier sweep without running anything
    uv run python -m cs336_systems.sweep --out results/benchmark.jsonl --tabulate-only --latex
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd

from cs336_systems.benchmark import MODEL_CONFIGS, MODES

PHASES = ("forward", "backward", "optimizer", "total")


def gpu_load() -> tuple[int, int] | None:
    """(utilization %, memory used in MiB) of GPU 0, or None when nvidia-smi is unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return None
    util, mem = (int(field) for field in out.stdout.strip().split(","))
    return util, mem


def wait_for_idle_gpu(max_util: int, timeout_s: float) -> tuple[int, int] | None:
    """Wait until other workloads have left the GPU, so they cannot inflate the timings. Returns the last reading."""
    deadline = time.monotonic() + timeout_s
    load = gpu_load()
    while load is not None and load[0] > max_util and time.monotonic() < deadline:
        time.sleep(2)
        load = gpu_load()
    return load


def read_records(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def tabulate(records: list[dict], latex: bool) -> None:
    pm = r" $\pm$ " if latex else " ± "
    rows = []
    for r in records:
        row = {"size": r["size"], "ctx": r["context_length"], "mode": r["mode"], "warmup": r["warmup"]}
        for phase in PHASES:
            mean, std = r.get(f"{phase}_mean_ms"), r.get(f"{phase}_std_ms")
            row[f"{phase} (ms)"] = "-" if mean is None else f"{mean:.2f}{pm}{std:.2f}"
        row["peak mem (GB)"] = "-" if r.get("peak_mem_gb") is None else f"{r['peak_mem_gb']:.1f}"
        row["status"] = r["status"] + (" (GPU shared)" if r.get("gpu_busy") else "")
        rows.append(row)
    df = pd.DataFrame(rows)
    # Drop columns that carry no information for this sweep.
    df = df[[c for c in df.columns if c in ("size", "status") or not (df[c] == "-").all()]]
    df = df[[c for c in df.columns if c in ("size", "status") or c.endswith(")") or df[c].nunique() > 1]]
    print(df.to_latex(index=False) if latex else df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", choices=list(MODEL_CONFIGS), default=list(MODEL_CONFIGS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["full"])
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[512])
    parser.add_argument("--warmups", nargs="+", type=int, default=[5])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--out", required=True, help="JSON-lines file holding one record per config")
    parser.add_argument("--tabulate-only", action="store_true", help="Only print tables from an existing --out file")
    parser.add_argument("--latex", action="store_true", help="Also print the table as LaTeX")
    parser.add_argument("--max-gpu-util", type=int, default=10, help="Wait for GPU utilization (%%) to fall to this before each config")
    parser.add_argument("--idle-timeout", type=float, default=120, help="Seconds to wait for an idle GPU before running anyway")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Arguments after `--` are passed through to cs336_systems.benchmark")
    args = parser.parse_args()
    extra = [a for a in args.extra if a != "--"]

    if not args.tabulate_only:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        open(args.out, "w").close()
        for size, context_length, mode, warmup in itertools.product(args.sizes, args.context_lengths, args.modes, args.warmups):
            load = wait_for_idle_gpu(args.max_gpu_util, args.idle_timeout)
            gpu_busy = load is not None and load[0] > args.max_gpu_util
            if gpu_busy:
                print(f"WARNING: GPU still at {load[0]}% utilization from another workload; timings will be inflated")
            n_before = len(read_records(args.out))
            command = [sys.executable, "-m", "cs336_systems.benchmark", "--size", size, "--mode", mode]
            command += ["--context-length", str(context_length), "--warmup", str(warmup), "--steps", str(args.steps)]
            command += ["--json-out", args.out, *extra]
            print(f"\n=== {size} ctx={context_length} mode={mode} warmup={warmup} (GPU before: {load}) ===", flush=True)
            returncode = subprocess.run(command).returncode
            records = read_records(args.out)
            if len(records) == n_before:
                # The benchmark died before it could report (anything other than a caught CUDA OOM).
                records.append(dict(size=size, context_length=context_length, mode=mode, warmup=warmup, status=f"crashed ({returncode})"))
            records[-1].update(gpu_before=load, gpu_busy=gpu_busy)
            with open(args.out, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in records)

    records = read_records(args.out)
    print()
    tabulate(records, latex=False)
    if args.latex:
        print()
        tabulate(records, latex=True)


if __name__ == "__main__":
    main()
