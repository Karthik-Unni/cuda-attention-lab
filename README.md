# CUDA Attention Lab

A from-scratch CUDA kernel optimization laboratory focused on understanding
GPU architecture, memory hierarchy, parallel algorithms, and the techniques
used to build efficient Transformer attention kernels.

The project starts with simple CUDA kernels and progressively transforms them
into optimized implementations through measurement, profiling, and
hardware-aware optimization.

---

## Why this project?

Modern GPU performance is not achieved simply by writing a correct parallel
algorithm.

A kernel can be mathematically correct and still perform poorly because of:

- inefficient global-memory access
- insufficient data reuse
- poor memory coalescing
- excessive synchronization
- register pressure
- shared-memory usage
- low occupancy
- instruction dependencies
- memory bandwidth limitations
- compute throughput limitations

This project is an experimental study of these problems.

#
No optimization is accepted simply because it "looks faster".

---

## Project progression

### 1. CUDA Fundamentals

Learn the CUDA execution model:

- threads
- warps
- blocks
- grids
- SMs
- global memory
- shared memory
- registers
- kernel launches
- host/device memory transfers

### 2. Matrix Multiplication

Matrix multiplication is used as the first major optimization case study.

Implementations will progress from:

    Naive matmul
        ↓
    Tiled matmul
        ↓
    Tile-size tuning
        ↓
    Loop unrolling
        ↓
    Register reuse
        ↓
    Prefetching
        ↓
    PTX/instruction analysis
        ↓
    cuBLAS comparison

The purpose is to understand how memory hierarchy and data reuse affect
GPU performance.

### 3. GPU Reductions

Build the primitives required by many ML kernels:

- sum reduction
- max reduction
- warp shuffle operations
- block-level reductions
- synchronization

### 4. Softmax

Progress from a simple implementation toward:

- numerically stable softmax
- warp-level reductions
- online softmax
- block-level softmax
- optimized memory access

### 5. LayerNorm

Implement:

- naive mean/variance calculation
- Welford's algorithm
- warp-level Welford
- shared-memory reduction
- cross-warp reduction

### 6. Attention

Build attention from its fundamental operations:

    Q × Kᵀ
       ↓
     softmax
       ↓
      × V

Start with separate kernels and progressively investigate fusion and
data reuse.

### 7. Flash-Attention-style Optimization

The final stage investigates how attention can avoid unnecessary global-memory
traffic by processing attention in tiles and keeping intermediate values
on-chip.

This connects:

- shared-memory tiling
- register reuse
- online softmax
- reductions
- kernel fusion
- memory bandwidth
- arithmetic intensity

---

## Benchmarking

Performance will be measured using CUDA events and reported using metrics
such as:

- kernel latency
- GFLOP/s
- TFLOP/s
- effective memory bandwidth
- speedup relative to baseline
- percentage of theoretical peak

Different implementations will be tested using the same inputs and
benchmark methodology.

---

## Profiling

Nsight Compute will be used to investigate why kernels perform the way they do.

Important areas include:

- memory throughput
- compute throughput
- occupancy
- shared-memory usage
- register usage
- warp state
- instruction mix
- load/store behavior

PTX inspection will also be used to connect high-level CUDA code to generated
instructions.

---

## Roofline Analysis

The project will use the Roofline Model to reason about whether a kernel is
limited by computation or memory bandwidth.

For each major kernel we will investigate:

    FLOPs
    bytes transferred
    arithmetic intensity
    theoretical compute ceiling
    theoretical memory-bandwidth ceiling

The goal is to understand not only whether a kernel is fast, but why its
performance is limited.
## Status

This is an educational and experimental CUDA project.

The emphasis is on understanding the reasoning behind GPU optimization rather
than reproducing a particular production implementation.

---

## Matmul Size-Scaling Benchmark

### Objective

Measure how the naive and shared-memory tiled CUDA matrix multiplication
kernels scale as matrix dimensions increase.

The experiment evaluates four square matrix sizes on an NVIDIA Tesla T4 GPU.

### Experimental Setup

- **GPU:** NVIDIA Tesla T4
- **CUDA Toolkit:** 13.0
- **Compiler optimization:** `nvcc -O2`
- **Data type:** FP32
- **Tile size:** 16 × 16 for the tiled kernel
- **Warm-up launches:** 5 per kernel
- **Timed iterations:** 20 for 512 and 1024, 5 for 2048, and 2 for 4096
- **Timing:** CUDA events
- **Correctness:** Output compared against the expected result for the
  initialized inputs

The input matrices contain values of 1.0 for A and 2.0 for B. For an
N × N multiplication, every output element should equal 2N.

Both implementations passed this correctness check at all four sizes in
each of three benchmark runs.

### Results

The table reports the median latency across three runs and computes speedup
from the ratio of median naive latency to median tiled latency.

| Matrix size | Naive latency (ms) | Tiled latency (ms) | Speedup |
|---:|---:|---:|---:|
| 512 × 512 | 0.434 | 0.278 | 1.562× |
| 1024 × 1024 | 3.984 | 2.448 | 1.628× |
| 2048 × 2048 | 36.158 | 21.955 | 1.647× |
| 4096 × 4096 | 289.339 | 178.144 | 1.624× |

These are measurements from this experimental environment, not universal
performance guarantees.

### Observations

- The tiled implementation was faster than the naive implementation at all
  four tested sizes.
- The median-based speedup ranged from approximately 1.56× to 1.65×.
- Runtime increased substantially with matrix size, consistent with the
  cubic arithmetic complexity of square matrix multiplication.
- The speedup was not perfectly monotonic across sizes. Further profiling is
  needed to identify the relative effects of memory traffic, synchronization,
  register usage, occupancy, and instruction execution.
- The benchmark uses constant-valued inputs, so correctness testing with
  additional varied inputs remains a useful future improvement.

### Limitations

- Results were collected on a shared/cloud GPU environment and may vary with
  clock behavior, temperature, and system activity.
- The number of timed iterations varies by matrix size to limit benchmark
  duration. Naive and tiled kernels use the same iteration count for each
  size.
- The experiment does not yet include Nsight Compute profiling, a cuBLAS
  baseline, or a full Roofline analysis.

The next step is to use profiling and systematic tile-size experiments to
investigate why the measured speedup differs across workloads.
