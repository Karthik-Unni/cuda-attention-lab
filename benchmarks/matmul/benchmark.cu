
#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "../../kernels/matmul/naive.cuh"
#include "../../kernels/matmul/tiled.cuh"

#define CHECK_CUDA(call)                                             \
do {                                                                 \
    cudaError_t err = call;                                          \
    if (err != cudaSuccess) {                                        \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)       \
                  << " at " << __FILE__ << ":" << __LINE__ << '\n'; \
        std::exit(EXIT_FAILURE);                                     \
    }                                                                \
} while (0)


float benchmark_naive(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int K,
    int N,
    dim3 blocks,
    dim3 threads,
    int warmup,
    int iterations
) {
    for (int i = 0; i < warmup; i++) {
        matmul_naive<<<blocks, threads>>>(
            d_A, d_B, d_C, M, K, N
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        matmul_naive<<<blocks, threads>>>(
            d_A, d_B, d_C, M, K, N
        );
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaGetLastError());

    float milliseconds = 0.0f;

    CHECK_CUDA(
        cudaEventElapsedTime(
            &milliseconds,
            start,
            stop
        )
    );

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return milliseconds / iterations;
}


float benchmark_tiled(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int K,
    int N,
    dim3 blocks,
    dim3 threads,
    int warmup,
    int iterations
) {
    for (int i = 0; i < warmup; i++) {
        matmul_tiled<<<blocks, threads>>>(
            d_A, d_B, d_C, M, K, N
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        matmul_tiled<<<blocks, threads>>>(
            d_A, d_B, d_C, M, K, N
        );
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaGetLastError());

    float milliseconds = 0.0f;

    CHECK_CUDA(
        cudaEventElapsedTime(
            &milliseconds,
            start,
            stop
        )
    );

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return milliseconds / iterations;
}


int main() {

    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    const int WARMUP = 10;
    const int ITERATIONS = 100;

    std::cout
        << "========================================\n"
        << "CUDA MATMUL BENCHMARK\n"
        << "========================================\n\n";

    std::cout
        << "Matrix dimensions:\n"
        << "M = " << M << '\n'
        << "K = " << K << '\n'
        << "N = " << N << "\n\n";

    size_t size_A =
        static_cast<size_t>(M) * K * sizeof(float);

    size_t size_B =
        static_cast<size_t>(K) * N * sizeof(float);

    size_t size_C =
        static_cast<size_t>(M) * N * sizeof(float);


    // -------------------------
    // Host memory
    // -------------------------

    float* h_A =
        new float[M * K];

    float* h_B =
        new float[K * N];

    float* h_C =
        new float[M * N];


    for (int i = 0; i < M * K; i++) {
        h_A[i] = 1.0f;
    }

    for (int i = 0; i < K * N; i++) {
        h_B[i] = 2.0f;
    }


    // -------------------------
    // Device memory
    // -------------------------

    float* d_A;
    float* d_B;
    float* d_C;

    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));


    CHECK_CUDA(
        cudaMemcpy(
            d_A,
            h_A,
            size_A,
            cudaMemcpyHostToDevice
        )
    );

    CHECK_CUDA(
        cudaMemcpy(
            d_B,
            h_B,
            size_B,
            cudaMemcpyHostToDevice
        )
    );


    // -------------------------
    // Launch configuration
    // -------------------------

    dim3 threads(16, 16);

    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );


    // -------------------------
    // Benchmark
    // -------------------------

    float naive_ms =
        benchmark_naive(
            d_A,
            d_B,
            d_C,
            M,
            K,
            N,
            blocks,
            threads,
            WARMUP,
            ITERATIONS
        );


    float tiled_ms =
        benchmark_tiled(
            d_A,
            d_B,
            d_C,
            M,
            K,
            N,
            blocks,
            threads,
            WARMUP,
            ITERATIONS
        );


    // -------------------------
    // Performance calculation
    // -------------------------

    double flops =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(K) *
        static_cast<double>(N);


    double naive_gflops =
        flops / (naive_ms * 1e6);

    double tiled_gflops =
        flops / (tiled_ms * 1e6);


    double speedup =
        naive_ms / tiled_ms;


    // -------------------------
    // Results
    // -------------------------

    std::cout << std::fixed
              << std::setprecision(3);

    std::cout
        << "\n========================================\n"
        << "RESULTS\n"
        << "========================================\n";

    std::cout
        << "Naive:\n"
        << "  Latency : "
        << naive_ms
        << " ms\n"
        << "  GFLOP/s : "
        << naive_gflops
        << "\n\n";

    std::cout
        << "Tiled:\n"
        << "  Latency : "
        << tiled_ms
        << " ms\n"
        << "  GFLOP/s : "
        << tiled_gflops
        << "\n\n";

    std::cout
        << "Tiled speedup: "
        << speedup
        << "x\n";

    std::cout
        << "========================================\n";


    // -------------------------
    // Cleanup
    // -------------------------

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
