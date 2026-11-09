"""End-to-end benchmarking of the basics Transformer model.

Times the forward pass, backward pass, and optimizer step, with warm-up steps
and per-phase mean/std statistics. Every phase ends with a device synchronize,
so each timing includes the GPU work that the phase launched.

Examples:
    # Forward-only timing for the small model
    uv run python -m cs336_systems.benchmark --size small --mode forward

    # Full training step (forward + backward + optimizer) for several sizes,
    # 5 warm-up steps and 10 measured steps (the part (b) setting)
    uv run python -m cs336_systems.benchmark --size small medium large --mode full --warmup 5 --steps 10

    # Under Nsight Systems: NVTX ranges mark warm-up vs. measured steps, each
    # phase, each module type, and the pieces of scaled dot-product attention
    nsys profile --trace=cuda,nvtx -o small_512 python -m cs336_systems.benchmark --size small --nvtx
"""

import argparse
import gc
import json
import math
import re
import timeit
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.cuda.nvtx as nvtx
from einops import einsum

import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW

# Model configurations from Table 1 of the assignment handout (section 2.1.2).
MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

MODES = ("forward", "forward_backward", "full")


def get_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def cap_to_dedicated_vram(device: torch.device, headroom_bytes: int = 512 * 2**20) -> None:
    """Make PyTorch raise OOM once the GPU's currently free dedicated memory is used up.

    On Windows the NVIDIA driver otherwise spills allocations into shared system
    memory instead of failing, which silently makes every step orders of
    magnitude slower. The headroom covers memory outside PyTorch's allocator
    (CUDA context, cuBLAS workspaces).
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    torch.cuda.set_per_process_memory_fraction(max(free - headroom_bytes, 0) / total, index)


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """cs336_basics.model.scaled_dot_product_attention with an NVTX range around each stage."""
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)
    with nvtx.range("final matmul"):
        return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


def add_nvtx_module_ranges(model: torch.nn.Module) -> None:
    """Wrap every submodule's forward in an NVTX range such as ``Linear:layers.*.ffn.w1`` (layer indices are wildcarded)."""

    # Hooks must return None: any other return value replaces the module's inputs/output.
    def push(label: str):
        def hook(_module, _inputs) -> None:
            nvtx.range_push(label)

        return hook

    def pop(_module, _inputs, _output) -> None:
        nvtx.range_pop()

    for name, module in model.named_modules():
        if not name:
            continue
        module.register_forward_pre_hook(push(f"{type(module).__name__}:{re.sub(r'\.\d+(?=\.|$)', '.*', name)}"))
        module.register_forward_hook(pop)


def benchmark(
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    vocab_size: int,
    context_length: int,
    batch_size: int,
    mode: str,
    warmup: int,
    steps: int,
    device: torch.device,
    mixed_precision: bool = False,
    compile_model: bool = False,
    use_nvtx: bool = False,
) -> dict[str, np.ndarray]:
    """Run warm-up + timed steps; return per-phase arrays of per-step seconds."""
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    ).to(device)
    if use_nvtx:
        add_nvtx_module_ranges(model)
    if compile_model:
        model = torch.compile(model)

    optimizer = AdamW(model.parameters()) if mode == "full" else None

    inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    def autocast():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if mixed_precision else nullcontext()

    def nvtx_range(name: str):
        return nvtx.range(name) if use_nvtx else nullcontext()

    phases = {"forward": ["forward"], "forward_backward": ["forward", "backward"], "full": ["forward", "backward", "optimizer"]}[mode]
    timings: dict[str, list[float]] = {phase: [] for phase in phases}

    def run_step(record: bool) -> None:
        # Gradients are cleared outside the timed regions, so `backward` costs the same in every mode.
        model.zero_grad(set_to_none=True)
        elapsed = {}

        with nvtx_range("forward"):
            start = timeit.default_timer()
            # Forward-only mode is inference: no autograd graph, so no activations are kept for a backward pass.
            with torch.set_grad_enabled(mode != "forward"), autocast():
                logits = model(inputs)
                with nvtx_range("loss"):
                    loss = cross_entropy(logits, targets)
            synchronize(device)
            elapsed["forward"] = timeit.default_timer() - start

        if "backward" in timings:
            with nvtx_range("backward"):
                start = timeit.default_timer()
                loss.backward()
                synchronize(device)
                elapsed["backward"] = timeit.default_timer() - start

        if optimizer is not None:
            with nvtx_range("optimizer"):
                start = timeit.default_timer()
                optimizer.step()
                synchronize(device)
                elapsed["optimizer"] = timeit.default_timer() - start

        if record:
            for phase, seconds in elapsed.items():
                timings[phase].append(seconds)

    with nvtx_range("warmup"):
        for _ in range(warmup):
            run_step(record=False)
    with nvtx_range("measure"):
        for _ in range(steps):
            run_step(record=True)

    return {phase: np.array(times) for phase, times in timings.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--size",
        nargs="+",
        choices=list(MODEL_CONFIGS),
        default=None,
        help="One or more preset model sizes to benchmark. Omit to specify hyperparameters manually.",
    )
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="full",
        help="forward: forward pass only, as inference under no_grad; forward_backward: forward + backward; full: forward + backward + optimizer step",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Number of warm-up steps (not timed)")
    parser.add_argument("--steps", type=int, default=10, help="Number of timed steps")
    parser.add_argument("--device", default=None, help="Device to run on (default: cuda if available, else mps, else cpu)")
    parser.add_argument("--mixed-precision", action="store_true", help="Run under BF16 autocast")
    parser.add_argument("--compile", action="store_true", help="Wrap the model in torch.compile")
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Emit NVTX ranges for nsys: warmup/measure, forward/loss/backward/optimizer, one per module, and the stages of scaled dot-product attention",
    )
    parser.add_argument("--latex", action="store_true", help="Also print the results table as LaTeX")
    parser.add_argument("--json-out", default=None, help="Append one JSON line per benchmarked config (with per-step timings) to this file")
    args = parser.parse_args()

    device = get_device(args.device)
    if device.type == "cuda":
        cap_to_dedicated_vram(device)
    if args.nvtx:
        cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    print(
        f"device={device}, mode={args.mode}, warmup={args.warmup}, steps={args.steps}, "
        f"batch_size={args.batch_size}, context_length={args.context_length}, "
        f"mixed_precision={args.mixed_precision}, compile={args.compile}, nvtx={args.nvtx}"
    )

    if args.size is not None:
        configs = {name: MODEL_CONFIGS[name] for name in args.size}
    else:
        configs = {"custom": dict(d_model=args.d_model, d_ff=args.d_ff, num_layers=args.num_layers, num_heads=args.num_heads)}

    rows = []
    for name, config in configs.items():
        row: dict[str, object] = {"size": name, "context_length": args.context_length, "mode": args.mode, "warmup": args.warmup, "status": "ok"}
        record = dict(row, steps=args.steps, batch_size=args.batch_size, mixed_precision=args.mixed_precision, compile=args.compile, nvtx=args.nvtx, **config)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            timings = benchmark(
                **config,
                vocab_size=args.vocab_size,
                context_length=args.context_length,
                batch_size=args.batch_size,
                mode=args.mode,
                warmup=args.warmup,
                steps=args.steps,
                device=device,
                mixed_precision=args.mixed_precision,
                compile_model=args.compile,
                use_nvtx=args.nvtx,
            )
        except torch.OutOfMemoryError:
            timings = {}
            row["status"] = record["status"] = "OOM"
            print(f"[{name}] out of memory")
        else:
            # Sample standard deviation across the measured steps.
            ddof = 1 if args.steps > 1 else 0
            for phase, times in timings.items():
                row[f"{phase}_mean_ms"] = times.mean() * 1e3
                row[f"{phase}_std_ms"] = times.std(ddof=ddof) * 1e3
            total = np.sum(list(timings.values()), axis=0)
            row["total_mean_ms"] = total.mean() * 1e3
            row["total_std_ms"] = total.std(ddof=ddof) * 1e3
            print(f"[{name}] " + "  ".join(f"{phase}: {times.mean() * 1e3:.2f} ± {times.std(ddof=ddof) * 1e3:.2f} ms" for phase, times in timings.items()))
        if device.type == "cuda":
            row["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
        rows.append(row)

        if args.json_out is not None:
            record.update({key: value for key, value in row.items() if key not in record})
            record["step_ms"] = {phase: (times * 1e3).tolist() for phase, times in timings.items()}
            with open(args.json_out, "a") as f:
                f.write(json.dumps(record) + "\n")

        # Release this model's memory before building the next one.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows).round(2)
    print()
    print(df.to_string(index=False))
    if args.latex:
        print()
        print(df.to_latex(index=False))


if __name__ == "__main__":
    main()
