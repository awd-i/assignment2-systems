"""End-to-end benchmarking of the basics Transformer model.

Times the forward pass, backward pass, and optimizer step, with warm-up steps
and per-phase mean/std statistics.

Examples:
    # Forward-only timing for the small model
    uv run python -m cs336_systems.benchmark --size small --mode forward

    # Full training step (forward + backward + optimizer) for all sizes,
    # 5 warm-up steps and 10 measured steps (the part (b) setting)
    uv run python -m cs336_systems.benchmark --size small medium large xl 10B \
        --mode full --warmup 5 --steps 10
"""

import argparse
import timeit
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
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
    if compile_model:
        model = torch.compile(model)

    optimizer = AdamW(model.parameters()) if mode == "full" else None

    inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if mixed_precision
        else nullcontext()
    )

    timings: dict[str, list[float]] = {"forward": []}
    if mode in ("forward_backward", "full"):
        timings["backward"] = []
    if mode == "full":
        timings["optimizer"] = []

    for step in range(warmup + steps):
        measured = step >= warmup

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        start = timeit.default_timer()
        with autocast_ctx:
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
        synchronize(device)
        if measured:
            timings["forward"].append(timeit.default_timer() - start)

        if mode in ("forward_backward", "full"):
            start = timeit.default_timer()
            loss.backward()
            synchronize(device)
            if measured:
                timings["backward"].append(timeit.default_timer() - start)

        if mode == "full":
            start = timeit.default_timer()
            optimizer.step()
            synchronize(device)
            if measured:
                timings["optimizer"].append(timeit.default_timer() - start)

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
        help="forward: forward pass only; forward_backward: forward + backward; full: forward + backward + optimizer step",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Number of warm-up steps (not timed)")
    parser.add_argument("--steps", type=int, default=10, help="Number of timed steps")
    parser.add_argument("--device", default=None, help="Device to run on (default: cuda if available, else mps, else cpu)")
    parser.add_argument("--mixed-precision", action="store_true", help="Run under BF16 autocast")
    parser.add_argument("--compile", action="store_true", help="Wrap the model in torch.compile")
    parser.add_argument("--latex", action="store_true", help="Also print the results table as LaTeX")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"device={device}, mode={args.mode}, warmup={args.warmup}, steps={args.steps}, "
          f"batch_size={args.batch_size}, context_length={args.context_length}, "
          f"mixed_precision={args.mixed_precision}, compile={args.compile}")

    if args.size is not None:
        configs = {name: MODEL_CONFIGS[name] for name in args.size}
    else:
        configs = {
            "custom": dict(d_model=args.d_model, d_ff=args.d_ff, num_layers=args.num_layers, num_heads=args.num_heads)
        }

    rows = []
    for name, config in configs.items():
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
        )
        row: dict[str, object] = {"size": name}
        for phase, times in timings.items():
            row[f"{phase}_mean_ms"] = times.mean() * 1e3
            row[f"{phase}_std_ms"] = times.std() * 1e3
        total = np.sum([times for times in timings.values()], axis=0)
        row["total_mean_ms"] = total.mean() * 1e3
        row["total_std_ms"] = total.std() * 1e3
        rows.append(row)
        print(f"[{name}] " + "  ".join(
            f"{phase}: {times.mean() * 1e3:.2f} ± {times.std() * 1e3:.2f} ms" for phase, times in timings.items()
        ))

    df = pd.DataFrame(rows).round(2)
    print()
    print(df.to_string(index=False))
    if args.latex:
        print()
        print(df.to_latex(index=False))


if __name__ == "__main__":
    main()
