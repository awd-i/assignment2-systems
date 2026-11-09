# Assignment 2, sections 2.1.3 and 2.1.4: benchmarking and Nsight Systems profiling

Setup for every number below: RTX 5090 (32 GB), Windows 11, PyTorch 2.11.0+cu128, FP32,
vocabulary 10,000, batch size 4, context length 512 unless stated. Times are
mean ± sample standard deviation over 10 measured steps, in milliseconds.

## Problem (benchmarking_script)

### (a) Script

`cs336_systems/benchmark.py` builds a `BasicsTransformerLM` from a preset (`--size small|medium|large|xl|10B`)
or from explicit hyperparameters, makes one random batch, runs `--warmup` untimed steps and then times
`--steps` steps with `timeit.default_timer()`, calling `torch.cuda.synchronize()` at the end of every
phase. `--mode forward` times an inference forward pass (under `no_grad`), `--mode forward_backward`
adds the backward pass, and `--mode full` adds the AdamW step. It reports mean and standard deviation
per phase and the peak GPU memory, and records an out-of-memory failure instead of crashing.

`cs336_systems/sweep.py` runs that script over a grid, one fresh process per configuration, and prints
the tables below (add `--latex` for LaTeX).

### (b) Timings with 5 warm-up steps

| size   | forward, inference | forward, training | backward      | AdamW step   | peak memory, full step |
|--------|--------------------|-------------------|---------------|--------------|------------------------|
| small  | 19.29 ± 1.11       | 21.70 ± 3.13      | 38.54 ± 0.81  | 23.68 ± 5.06 | 5.0 GB                 |
| medium | 51.78 ± 0.36       | 52.25 ± 1.62      | 113.16 ± 1.40 | 44.43 ± 7.53 | 13.7 GB                |
| large  | 116.27 ± 0.39      | 115.69 ± 0.53     | 233.64 ± 1.57 | 71.22 ± 3.26 | 27.5 GB                |
| xl     | 333.68 ± 1.38      | out of memory     | out of memory | out of memory| does not fit           |
| 10B    | out of memory      | out of memory     | out of memory | out of memory| does not fit           |

**Response.** A forward pass takes about 20 / 52 / 116 ms and a backward pass about 39 / 113 / 234 ms
for small / medium / large, so backward costs roughly twice forward, and the AdamW step takes
24 / 44 / 71 ms; on this 32 GB card xl only fits as an inference forward pass (334 ms) and 10B does
not fit at all. Variability is small where the GPU is the bottleneck (standard deviation 0.3–3% of the
mean for every backward pass and for the medium, large and xl forward passes) and larger for the small
model's forward pass (6–14%) and for the optimizer step (5–21%), the phases where CPU-side kernel
launching matters most.

### (c) Fewer warm-up steps

Full training step, each row a fresh process:

| size   | warm-up steps | forward        | backward       | AdamW step   |
|--------|---------------|----------------|----------------|--------------|
| small  | 0             | 69.38 ± 153.44 | 50.81 ± 38.84  | 24.17 ± 4.50 |
| small  | 1             | 20.46 ± 2.29   | 39.12 ± 1.48   | 19.88 ± 2.27 |
| small  | 2             | 19.68 ± 0.84   | 38.65 ± 0.67   | 19.73 ± 1.25 |
| small  | 5             | 22.99 ± 2.11   | 39.27 ± 1.43   | 27.35 ± 4.03 |
| medium | 0             | 77.16 ± 76.58  | 120.40 ± 17.91 | 50.84 ± 7.42 |
| medium | 1             | 53.74 ± 3.37   | 113.81 ± 2.11  | 50.38 ± 7.53 |
| medium | 2             | 51.83 ± 0.71   | 112.85 ± 0.71  | 44.25 ± 5.18 |
| medium | 5             | 51.88 ± 1.05   | 112.76 ± 0.82  | 40.78 ± 3.56 |
| large  | 0             | 140.50 ± 77.11 | 239.34 ± 16.93 | 71.44 ± 3.39 |
| large  | 1             | 116.74 ± 1.82  | 234.23 ± 2.22  | 73.85 ± 4.46 |
| large  | 2             | 115.72 ± 1.14  | 233.54 ± 0.87  | 70.70 ± 3.59 |
| large  | 5             | 115.27 ± 0.57  | 233.71 ± 1.13  | 72.98 ± 5.08 |

**Response.** Without warm-up the very first step dominates the statistics: its forward pass takes
300–500 ms and its backward pass 160–290 ms whatever the model size, so the small model's mean forward
time becomes 69 ± 153 ms instead of about 20 ± 2 ms. A whole-run nsys trace shows the GPU does the same
work in that first step as in any other (16 ms of kernel time for small); the extra time is one-off
CPU-side setup: kernel-launch calls take 178 ms instead of 5 ms, consistent with CUDA loading each
kernel lazily on first use, plus 25 ms loading kernel libraries and 109 `cudaMalloc` calls while
PyTorch's allocator grows its pool. With one or two warm-up steps the averages are already within noise
of five, but the second step is still slightly slow (forward 23 vs 19 ms for small, 63 vs 52 ms for
medium), most likely because the allocator only settles once a complete step has also allocated the
gradients and optimizer state: the second forward pass still issues 24 `cudaMalloc` calls, and the
third none.

## Problem (nsys_profile)

Profiles: **small** and **large**, context lengths **256, 512, 1024**, each as an inference forward
pass and as a full training step, 10 measured steps after 5 warm-up steps, with warm-up excluded by an
NVTX capture range. For small, 1024 is the longest context whose training step fits in 32 GB. For large
the longest that fits is 512, so its 1024 profile is inference only.

| model | context | nsys forward range | Python timer, unprofiled | GPU kernel time | GEMM share, forward | GEMM share, full step | softmax / attention matmuls (GPU time) |
|-------|---------|--------------------|--------------------------|-----------------|---------------------|-----------------------|----------------------------------------|
| small | 256     | 16.35              | 15.13 ± 2.59             | 8.60            | 81.7%               | 59.6%                 | 0.34 / 0.37 ms = 0.9                   |
| small | 512     | 18.78              | 19.91 ± 0.80             | 15.87           | 76.4%               | 59.3%                 | 1.15 / 1.16 ms = 1.0                   |
| small | 1024    | 50.67              | 51.03 ± 0.64             | 47.50           | 52.4%               | 43.6%                 | 11.82 / 4.66 ms = 2.5                  |
| large | 256     | 56.81              | 56.88 ± 1.21             | 48.70           | 86.3%               | 59.3%                 | 1.61 / 1.48 ms = 1.1                   |
| large | 512     | 117.94             | 116.34 ± 1.53            | 110.34          | 74.4%               | 57.2%                 | 11.93 / 5.57 ms = 2.1                  |
| large | 1024    | 294.83             | 291.57 ± 1.35            | 287.52          | 59.4%               | does not fit          | 61.48 / 22.52 ms = 2.7                 |

Forward columns come from the inference profiles, the full-step column from the training profiles.

**(a)** The forward pass takes 16.4 / 18.8 / 50.7 ms for small and 56.8 / 117.9 / 294.8 ms for large at
context 256 / 512 / 1024. That matches the Python timer: inside the profiled run the two agree to
0.02 ms, and against separate unprofiled runs they agree to about 1% for large and for small at 1024,
and to 6–8% for small at 256 and 512, where unprofiled runs differ from each other by as much.

**(b)** The top forward kernel in every profile is cuBLAS's SGEMM
`cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>`, the Linear layers, at 69% of forward
GPU time for both models at context 512 (51–59% at 256, 43–52% at 1024). It runs once per Linear layer,
7 per block plus the LM head, so 85 times per forward pass for small and 253 for large; at context 256
cuBLAS moves the four attention projections per block to a `128x128` variant, leaving 37 and 109 calls.
It is still the top kernel for forward plus backward (23% at 512), followed by two further SGEMM
variants that backward adds.

**(c)** The elementwise and reduction kernels that run over the T×T attention scores: the softmax's
max, subtract, exp, sum and divide, the `1/sqrt(d_k)` scaling multiply, and the causal-mask `where`.
Together they are 12% of forward GPU time for small at 512 and 33–38% at context 1024; RoPE, the
cross-entropy loss, the SiLU gating in SwiGLU and RMSNorm add roughly 2–3% each.

**(d)** The GEMM share of GPU time falls from inference to a full training step: 76% → 59% for small
and 74% → 57% for large at context 512, and 52% → 44% for small at 1024. Backward is itself only about
60% GEMM at 512, and the AdamW step has no matrix multiplies at all: it is 16 small elementwise kernels
per parameter tensor (1,776 kernels for small, 5,232 for large, 10–11% of the step's GPU time), so
elementwise multiply, add, divide and copy kernels take up the difference.

**(e)** Softmax needs about 51 times fewer FLOPs than the two attention matmuls (roughly 5·T² against
4·d_k·T² per head, d_k = 64), yet it takes about as much GPU time as them while the score tensor is
small (ratio 0.9–1.1) and 2.1–2.7 times more for large at 512 and for both models at 1024. Softmax is
five separate memory-bound kernels that each stream the whole batch × heads × T × T score tensor, while
GEMM is compute-dense; for large at 1024 the softmax moves roughly 1.6 TB/s, close to the card's
1.79 TB/s memory bandwidth.

## Things a reader should know

- **xl and 10B do not fit in 32 GB at FP32.** xl needs 13.4 GB for inference but its training step
  runs out of memory; 10B's weights alone exceed the card.
- **The large model's full step sits at the edge of VRAM** (27.5 GB allocated, 29.5 GB reserved). In
  earlier runs, when other desktop apps held about 1.2 GB of VRAM, Windows showed the process at
  29.9 GiB dedicated plus 0.27 GiB in shared system memory (0.08 GiB when not under pressure), and its
  forward / backward slowed to 131–146 / 285–301 ms with about ten times the standard deviation. The
  table in (b) is from a run with about 0.8 GB held by other apps.
  `results/benchmark_vram_pressure.jsonl` keeps a slowed run.
- **Launch-bound phases.** The optimizer step for small is 1,776 kernels totalling 5.3 ms of GPU time
  inside a 21 ms step, which is why its timing is the noisiest.
- **Cache size is a likely factor in (e), not a measured one.** nsys reports a 96 MiB L2 cache for this
  GPU; per-score softmax cost roughly doubles between score tensors of 50 MB (small, 512) and 84 MB or
  more (large at 512, both models at 1024), which is consistent with the tensor no longer fitting.
- **Windows nsys.** Profiles were collected without administrator rights, which requires
  `--sample=none --cpuctxsw=none`; `--gpu-metrics-devices` needs elevation, and `--cudabacktrace`,
  `--python-backtrace` and `osrt` tracing do not exist in the Windows CLI. Module attribution uses NVTX
  ranges emitted by `benchmark.py --nvtx` (nsys's own `--pytorch=functions-trace` also works here).

## Reproducing

```sh
# (b) and (c)
uv run python -m cs336_systems.sweep --out results/benchmark.jsonl --modes forward forward_backward full
uv run python -m cs336_systems.sweep --out results/warmup.jsonl --sizes small medium large --warmups 0 1 2 5

# nsys profiles and their analysis (set NSYS to the nsys executable if it is not on PATH)
uv run python -m cs336_systems.sweep --out results/nsys_baseline.jsonl --sizes small medium large --modes forward full --context-lengths 256 512 1024
uv run python -m cs336_systems.nsys_report collect --sizes small medium large --context-lengths 256 512 1024
uv run python -m cs336_systems.nsys_report report "profiles/*.nsys-rep" --baseline results/nsys_baseline.jsonl

# cold-start trace behind (c)
uv run python -m cs336_systems.nsys_report collect --sizes small --context-lengths 512 --modes full --warmup 0 --steps 5 --whole-run
uv run python -m cs336_systems.nsys_report report "profiles/*_wholerun.nsys-rep" --step-by-step
```

Full per-profile tables are in `results/nsys_report.txt`; medium was profiled as well and is included there.
