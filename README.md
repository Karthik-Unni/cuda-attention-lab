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

The goal is not to copy optimized CUDA implementations, but to understand
why each optimization works, when it stops helping, and what hardware
resource becomes the next bottleneck.

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
