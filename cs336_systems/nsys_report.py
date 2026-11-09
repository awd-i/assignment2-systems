"""Collect Nsight Systems profiles of the benchmark and summarize them.

`collect` runs cs336_systems.benchmark under `nsys profile` once per (size,
context length, mode). Only the measured steps are captured: collection is
triggered by the benchmark's "measure" NVTX range, so warm-up is excluded.

`report` reads the profiles (exporting each .nsys-rep to SQLite on first use)
and prints the tables behind the nsys_profile answers: wall time per phase,
the top GPU kernels, the matmul share of GPU time, GPU time per module, and
the softmax-vs-matmul breakdown inside attention.

Kernels are attributed to NVTX ranges through the CUDA API call that launched
them (CUPTI correlation ids), which is what the Nsight Systems GUI does when it
projects NVTX ranges onto the GPU.

Examples:
    uv run python -m cs336_systems.nsys_report collect --sizes small large --context-lengths 256 512 1024
    uv run python -m cs336_systems.nsys_report report profiles/*.nsys-rep --baseline results/nsys_baseline.jsonl
"""

import argparse
import glob
import itertools
import os
import re
import shutil
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd

from cs336_systems.benchmark import MODEL_CONFIGS, MODES
from cs336_systems.sweep import read_records, wait_for_idle_gpu

PHASES = ("forward", "backward", "optimizer")

# Ordered by priority: the first pattern found in a demangled PyTorch kernel name gives the operation it applies.
KERNEL_OP_PATTERNS = [
    r"binary_internal::(\w+?)Functor",
    r"(where|pow_tensor_scalar)_kernel_impl",
    r"(\w+)_kernel_cuda",
    r"CUDAFunctor(?:OnSelf|OnOther)?_(\w+)",
    r"(Fill|Compare)Functor",
    r"(Max|Mean)Ops",
    r"(sum)_functor",
    r"(\w+)_cuda_out",
]


def kernel_label(short: str, demangled: str) -> str:
    """A readable kernel name: the cutlass GEMM variant, or the PyTorch kernel plus the operation it applies."""
    if match := re.search(r"cutlass_\w+", demangled):
        return match.group(0)
    for pattern in KERNEL_OP_PATTERNS:
        if match := re.search(pattern, demangled):
            scalar = ", scalar" if re.search(r"[AB]UnaryFunctor|CUDAFunctorOn", demangled) else ""
            op = match.group(1).lower()
            # PyTorch subtracts with its add kernel (alpha = -1), so the two cannot be told apart by name.
            return f"{short}<{'add/sub' if op == 'add' else op}{scalar}>"
    return short


def find_nsys(explicit: str | None) -> str:
    nsys = explicit or os.environ.get("NSYS") or shutil.which("nsys")
    if nsys is None:
        sys.exit("nsys not found: put it on PATH, set the NSYS environment variable, or pass --nsys")
    return nsys


def collect(args: argparse.Namespace) -> None:
    nsys = find_nsys(args.nsys)
    os.makedirs(args.out_dir, exist_ok=True)
    for size, context_length, mode in itertools.product(args.sizes, args.context_lengths, args.modes):
        stem = os.path.join(args.out_dir, f"{size}_ctx{context_length}_{mode}" + ("_wholerun" if args.whole_run else ""))
        for stale in (stem + ".json", stem + ".nsys-rep", stem + ".sqlite"):
            if os.path.exists(stale):
                os.remove(stale)
        load = wait_for_idle_gpu(max_util=10, timeout_s=120)
        print(f"\n=== {size} ctx={context_length} mode={mode} (GPU before: {load}) ===", flush=True)
        # Start collecting at the benchmark's "measure" NVTX range, i.e. after warm-up. PyTorch does not
        # register its NVTX strings, so nsys has to be told to match unregistered ones.
        capture = ["--capture-range=nvtx", "--nvtx-capture=measure", "--capture-range-end=stop", "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0"]
        command = [
            nsys, "profile", "--trace=cuda,nvtx",
            # CPU sampling and context-switch tracing need administrator rights on Windows and are not used by the report.
            "--sample=none", "--cpuctxsw=none",
            *([] if args.whole_run else capture),
            "--force-overwrite=true", "-o", stem,
            sys.executable, "-m", "cs336_systems.benchmark", "--size", size, "--context-length", str(context_length),
            "--mode", mode, "--warmup", str(args.warmup), "--steps", str(args.steps), "--nvtx", "--json-out", stem + ".json",
        ]  # fmt: skip
        subprocess.run(command)


def ensure_sqlite(path: str, nsys_arg: str | None) -> str:
    if path.endswith(".sqlite"):
        return path
    sqlite_path = os.path.splitext(path)[0] + ".sqlite"
    if not os.path.exists(sqlite_path) or os.path.getmtime(sqlite_path) < os.path.getmtime(path):
        subprocess.run(
            [find_nsys(nsys_arg), "export", "--type=sqlite", "--force-overwrite=true", "-o", sqlite_path, path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return sqlite_path


def containing(times: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Index of the interval containing each time (intervals sorted and disjoint), or -1."""
    idx = np.searchsorted(starts, times, side="right") - 1
    inside = (idx >= 0) & (times <= ends[np.clip(idx, 0, None)])
    return np.where(inside, idx, -1)


def innermost(times: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Index of the innermost of properly nested ranges (sorted by start, outer first) containing each time, or -1."""
    parent = np.full(len(starts), -1)
    stack: list[int] = []
    for i in range(len(starts)):
        while stack and ends[stack[-1]] < starts[i]:
            stack.pop()
        parent[i] = stack[-1] if stack else -1
        stack.append(i)
    idx = np.searchsorted(starts, times, side="right") - 1
    for j, time in enumerate(times):
        i = idx[j]
        while i >= 0 and ends[i] < time:
            i = parent[i]
        idx[j] = i
    return idx


def load_profile(sqlite_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (kernels, phases): one row per measured GPU kernel, and one row per measured forward/backward/optimizer range."""
    con = sqlite3.connect(sqlite_path)
    kernels = pd.read_sql(
        """
        select k.start, k.end, launches.launch, launches.tid, short.value as short, demangled.value as demangled
        from CUPTI_ACTIVITY_KIND_KERNEL k
        join (select correlationId, min(start) as launch, min(globalTid) as tid
              from CUPTI_ACTIVITY_KIND_RUNTIME group by correlationId) launches on launches.correlationId = k.correlationId
        join StringIds short on short.id = k.shortName
        join StringIds demangled on demangled.id = k.demangledName
        """,
        con,
    )
    ranges = pd.read_sql(
        """
        select n.start, n.end, n.globalTid as tid, coalesce(n.text, s.value) as name
        from NVTX_EVENTS n left join StringIds s on s.id = n.textId
        where n.end is not null and n.eventType = (select id from ENUM_NSYS_EVENT_TYPE where name = 'NvtxPushPopRange')
        """,
        con,
    )
    con.close()

    # Outer ranges first when two ranges open on the same timestamp.
    ranges = ranges.sort_values(["start", "end"], ascending=[True, False]).reset_index(drop=True)
    phases = ranges[ranges.name.isin(PHASES)]
    for warmup in ranges[ranges.name == "warmup"].itertuples():
        phases = phases[(phases.start < warmup.start) | (phases.start > warmup.end)]
    phases = phases.reset_index(drop=True)
    phases["step"] = phases.groupby("name").cumcount()
    phases["wall_ms"] = (phases.end - phases.start) / 1e6

    # Each phase ends with a synchronize, so a kernel belongs to the phase during which it was launched.
    launch = kernels.launch.to_numpy()
    phase_idx = containing(launch, phases.start.to_numpy(), phases.end.to_numpy())
    kernels = kernels[phase_idx >= 0].copy()
    phase_idx = phase_idx[phase_idx >= 0]
    kernels["phase"] = phases.name.to_numpy()[phase_idx]
    kernels["step"] = phases.step.to_numpy()[phase_idx]
    kernels["ms"] = (kernels.end - kernels.start) / 1e6
    labels = {(short, demangled): kernel_label(short, demangled) for short, demangled in set(zip(kernels.short, kernels.demangled))}
    kernels["kernel"] = [labels[key] for key in zip(kernels.short, kernels.demangled)]
    kernels["matmul"] = kernels.kernel.str.contains("gemm")

    # Innermost NVTX range (module or attention stage) around the launching API call. The forward pass and its
    # ranges live on the main thread; backward kernels are launched from autograd's thread and keep their phase name.
    kernels["scope"] = kernels.phase
    scopes = ranges[~ranges.name.isin(("warmup", "measure"))]
    for tid, thread_scopes in scopes.groupby("tid"):
        on_thread = (kernels.tid == tid).to_numpy()
        idx = innermost(kernels.launch.to_numpy()[on_thread], thread_scopes.start.to_numpy(), thread_scopes.end.to_numpy())
        names = thread_scopes.name.to_numpy()
        kernels.loc[on_thread, "scope"] = np.where(idx >= 0, names[np.clip(idx, 0, None)], kernels.phase.to_numpy()[on_thread])
    return kernels, phases


def step_by_step_table(sqlite_path: str) -> pd.DataFrame:
    """Every forward/backward/optimizer range in time order, warm-up included, with where its wall time went.

    Meant for whole-run profiles: it shows what the first (cold) steps pay for on the CPU side.
    """
    con = sqlite3.connect(sqlite_path)
    phases = pd.read_sql(
        "select n.start, n.end, coalesce(n.text, s.value) as name from NVTX_EVENTS n left join StringIds s on s.id = n.textId where n.end is not null order by n.start",
        con,
    )
    calls = pd.read_sql("select r.start, r.end - r.start as ns, s.value as name from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id = r.nameId", con)
    kernels = pd.read_sql(
        "select k.end - k.start as ns, r.start as launch from CUPTI_ACTIVITY_KIND_KERNEL k join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId = k.correlationId",
        con,
    )
    con.close()
    phases = phases[phases.name.isin(PHASES)].reset_index(drop=True)
    phases["step"] = phases.groupby("name").cumcount() + 1
    rows = []
    for phase in phases.itertuples():
        in_phase = calls[(calls.start >= phase.start) & (calls.start <= phase.end)]
        launched = kernels[(kernels.launch >= phase.start) & (kernels.launch <= phase.end)]
        mallocs = in_phase[in_phase.name.str.startswith("cudaMalloc")]
        rows.append(
            {
                "step": phase.step,
                "phase": phase.name,
                "wall ms": (phase.end - phase.start) / 1e6,
                "GPU kernel ms": launched.ns.sum() / 1e6,
                "kernels": len(launched),
                "CPU ms in kernel-launch calls": in_phase[in_phase.name.str.contains("LaunchKernel")].ns.sum() / 1e6,
                "CPU ms loading kernel libraries": in_phase[in_phase.name.str.contains("LibraryLoad|ModuleLoad")].ns.sum() / 1e6,
                "cudaMalloc calls": len(mallocs),
                "cudaMalloc ms": mallocs.ns.sum() / 1e6,
            }
        )
    return pd.DataFrame(rows)


def per_step_table(kernels: pd.DataFrame, by: str, n_steps: int, top: int | None = None) -> pd.DataFrame:
    """GPU time and launch count per step, grouped by `by`, sorted by GPU time."""
    table = kernels.groupby(by).agg(gpu_ms=("ms", "sum"), calls=("ms", "size")).sort_values("gpu_ms", ascending=False)
    table["share_%"] = 100 * table.gpu_ms / table.gpu_ms.sum()
    table[["gpu_ms", "calls"]] /= n_steps
    table = table.rename(columns={"gpu_ms": "GPU ms/step", "calls": "calls/step", "share_%": "% of GPU time"})
    return table.head(top).reset_index() if top else table.reset_index()


def show(title: str, table: pd.DataFrame, latex: bool) -> None:
    print(f"\n{title}")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if latex:
        print(table.to_latex(index=False, float_format="%.2f"))


def report_profile(path: str, args: argparse.Namespace, baseline: dict) -> dict | None:
    stem = os.path.splitext(path)[0]
    meta = read_records(stem + ".json")[-1] if os.path.exists(stem + ".json") else {}
    if meta.get("status", "ok") != "ok":
        print(f"\n######## {os.path.basename(stem)}: benchmark status {meta['status']}, skipped ########")
        return None
    sqlite_path = ensure_sqlite(path, args.nsys)
    kernels, phases = load_profile(sqlite_path)
    n_steps = int(phases[phases.name == "forward"].shape[0])
    print(f"\n######## {os.path.basename(stem)}: {n_steps} measured steps ########")
    if args.step_by_step:
        show("Step by step (warm-up included)", step_by_step_table(sqlite_path), args.latex)
        return None
    key = (meta.get("size"), meta.get("context_length"), meta.get("mode"))
    summary: dict[str, object] = {"size": key[0], "ctx": key[1], "mode": key[2]}

    # (a) Wall time per phase: NVTX range vs. the Python timer (inside this profiled run, and in an unprofiled run).
    rows = []
    for phase in PHASES:
        walls = phases[phases.name == phase].wall_ms
        if walls.empty:
            continue
        phase_kernels = kernels[kernels.phase == phase]
        gpu_ms = phase_kernels.ms.sum() / n_steps
        matmul_pct = 100 * phase_kernels[phase_kernels.matmul].ms.sum() / phase_kernels.ms.sum()
        rows.append(
            {
                "phase": phase,
                "nsys wall ms": walls.mean(),
                "std": walls.std(),
                "python ms (profiled)": meta.get(f"{phase}_mean_ms", np.nan),
                "python ms (unprofiled)": baseline.get(key, {}).get(f"{phase}_mean_ms", np.nan),
                "kernels/step": len(phase_kernels) / n_steps,
                "GPU ms/step": gpu_ms,
                "matmul % of GPU time": matmul_pct,
            }
        )
        summary.update({f"{phase} nsys ms": walls.mean(), f"{phase} matmul %": matmul_pct})
        if phase == "forward":
            summary.update({"forward python ms (profiled)": rows[-1]["python ms (profiled)"], "forward python ms (unprofiled)": rows[-1]["python ms (unprofiled)"]})
    show("Wall time and GPU time per phase", pd.DataFrame(rows), args.latex)

    # (b), (c), (d) Kernel summaries: forward only, and everything the run did per step.
    forward = kernels[kernels.phase == "forward"]
    scopes = {"forward pass": forward}
    if "backward" in set(kernels.phase):
        scopes["forward + backward"] = kernels[kernels.phase.isin(("forward", "backward"))]
    if "optimizer" in set(kernels.phase):
        scopes["full training step"] = kernels
    for title, scope_kernels in scopes.items():
        table = per_step_table(scope_kernels, "kernel", n_steps, args.top)
        show(f"Top kernels, {title} ({scope_kernels.ms.sum() / n_steps:.2f} ms of GPU time, {len(scope_kernels) / n_steps:.0f} kernels per step)", table, args.latex)
        top_kernel = table.iloc[0]
        matmul_pct = 100 * scope_kernels[scope_kernels.matmul].ms.sum() / scope_kernels.ms.sum()
        print(f"matmul (GEMM) kernels: {matmul_pct:.1f}% of GPU time; everything else: {100 - matmul_pct:.1f}%")
        summary.update(
            {
                f"{title} top kernel": top_kernel.kernel,
                f"{title} top calls/step": top_kernel["calls/step"],
                f"{title} top %": top_kernel["% of GPU time"],
                f"{title} matmul %": matmul_pct,
            }
        )

    # Which part of the model launched the forward-pass kernels.
    show("Forward pass GPU time by innermost NVTX range", per_step_table(forward, "scope", n_steps), args.latex)

    # (e) Softmax vs. matrix multiplication inside scaled dot-product attention.
    softmax_kernels = forward[forward.scope == "computing softmax"]
    attention = forward[forward.scope.isin(("computing attention scores", "computing softmax", "final matmul"))]
    attention_matmul = attention[attention.matmul]
    if not softmax_kernels.empty:
        softmax_ms, matmul_ms = softmax_kernels.ms.sum() / n_steps, attention_matmul.ms.sum() / n_steps
        other_ms = attention.ms.sum() / n_steps - softmax_ms - matmul_ms
        show("Softmax kernels inside attention (forward)", per_step_table(softmax_kernels, "kernel", n_steps), args.latex)
        print(f"attention forward GPU time per step: matmuls {matmul_ms:.2f} ms, softmax {softmax_ms:.2f} ms, scaling + masking {other_ms:.2f} ms")
        summary.update({"attn matmul ms": matmul_ms, "attn softmax ms": softmax_ms, "softmax / matmul time": softmax_ms / matmul_ms})
        if "num_heads" in meta:
            # Per layer: QK^T and PV are each batch*heads*T*T*d_k multiply-adds (2 FLOPs each); softmax is
            # about 5 elementwise operations (max, subtract, exp, sum, divide) on each of the batch*heads*T*T scores.
            cells = meta["batch_size"] * meta["num_heads"] * meta["context_length"] ** 2 * meta["num_layers"]
            matmul_flops, softmax_flops = 2 * 2 * cells * (meta["d_model"] // meta["num_heads"]), 5 * cells
            print(
                f"attention forward FLOPs per step: matmuls {matmul_flops:.3g}, softmax {softmax_flops:.3g} -> softmax/matmul = {softmax_flops / matmul_flops:.4f} in FLOPs vs. {softmax_ms / matmul_ms:.2f} in GPU time"
            )
            summary["softmax / matmul FLOPs"] = softmax_flops / matmul_flops
    return summary


def report(args: argparse.Namespace) -> None:
    baseline = {}
    for path in args.baseline:
        for record in read_records(path):
            baseline[(record["size"], record["context_length"], record["mode"])] = record
    paths = [path for pattern in args.profiles for path in sorted(glob.glob(pattern)) or [pattern]]
    summaries = [summary for path in paths if (summary := report_profile(path, args, baseline)) is not None]
    if summaries:
        order = {size: i for i, size in enumerate(MODEL_CONFIGS)}
        table = pd.DataFrame(summaries).sort_values(["mode", "size", "ctx"], key=lambda column: column.map(order) if column.name == "size" else column)
        print("\n######## Summary across profiles ########")
        with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 60):
            for start in range(3, len(table.columns), 6):
                print()
                print(table.iloc[:, [0, 1, 2, *range(start, min(start + 6, len(table.columns)))]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        if args.summary_csv:
            table.to_csv(args.summary_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Profile the benchmark under nsys")
    collect_parser.add_argument("--sizes", nargs="+", choices=list(MODEL_CONFIGS), default=["small", "large"])
    collect_parser.add_argument("--context-lengths", nargs="+", type=int, default=[256, 512, 1024])
    collect_parser.add_argument("--modes", nargs="+", choices=MODES, default=["forward", "full"])
    collect_parser.add_argument("--warmup", type=int, default=5)
    collect_parser.add_argument("--steps", type=int, default=10)
    collect_parser.add_argument("--out-dir", default="profiles")
    collect_parser.add_argument("--whole-run", action="store_true", help="Capture the whole process, including warm-up, instead of only the measured steps")
    collect_parser.add_argument("--nsys", default=None, help="Path to the nsys executable (default: $NSYS, then PATH)")
    collect_parser.set_defaults(run=collect)

    report_parser = subparsers.add_parser("report", help="Summarize collected profiles")
    report_parser.add_argument("profiles", nargs="+", help=".nsys-rep or .sqlite files (globs allowed)")
    report_parser.add_argument("--baseline", nargs="*", default=[], help="JSON-lines files from cs336_systems.sweep with unprofiled timings to compare against")
    report_parser.add_argument("--top", type=int, default=8, help="Number of kernels to list per table")
    report_parser.add_argument("--step-by-step", action="store_true", help="Instead of the summaries, list every step's wall, GPU, and CUDA-API time (for --whole-run profiles)")
    report_parser.add_argument("--latex", action="store_true", help="Also print each table as LaTeX")
    report_parser.add_argument("--summary-csv", default=None, help="Write the cross-profile summary to this CSV file")
    report_parser.add_argument("--nsys", default=None, help="Path to the nsys executable, needed to export .nsys-rep files (default: $NSYS, then PATH)")
    report_parser.set_defaults(run=report)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
