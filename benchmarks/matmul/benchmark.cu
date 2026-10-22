#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../../kernels/matmul/naive.cuh"
#include "../../kernels/matmul/tiled.cuh"

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

float benchmark_naive(
    const float* A, const float* B, float* C,
    int M, int K, int N,
    dim3 blocks, dim3 threads, int warmup, int iterations
) {
    for (int i = 0; i < warmup; ++i)
        matmul_naive<<<blocks, threads>>>(A, B, C, M, K, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i)
        matmul_naive<<<blocks, threads>>>(A, B, C, M, K, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / iterations;
}

float benchmark_tiled(
    const float* A, const float* B, float* C,
    int M, int K, int N,
    dim3 blocks, dim3 threads, int warmup, int iterations
) {
    for (int i = 0; i < warmup; ++i)
        matmul_tiled<<<blocks, threads>>>(A, B, C, M, K, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i)
        matmul_tiled<<<blocks, threads>>>(A, B, C, M, K, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / iterations;
}

bool validate_output(
    const float* C, int M, int N, float expected
) {
    std::vector<float> result(static_cast<size_t>(M) * N);

    CHECK_CUDA(cudaMemcpy(
        result.data(), C, result.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    for (size_t i = 0; i < result.size(); ++i) {
        if (!std::isfinite(result[i]) ||
            std::fabs(result[i] - expected) > 1e-3f) {
            std::cerr << "Validation failed at index " << i
                      << ": expected " << expected
                      << ", got " << result[i] << '\n';
            return false;
        }
    }
    return true;
}

int main() {
    const int sizes[] = {512, 1024, 2048, 4096};
    const int WARMUP = 5;

    struct Result {
        int size;
        float naive_ms;
        float tiled_ms;
        double naive_gflops;
        double tiled_gflops;
        double speedup;
    };
    std::vector<Result> results;

    int device = 0;
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    std::cout << "CUDA MATMUL SIZE-SCALING BENCHMARK\n"
              << "GPU: " << prop.name << "\n\n";

    for (int n : sizes) {
        const int M = n, K = n, N = n;
        const size_t count = static_cast<size_t>(n) * n;
        const size_t bytes = count * sizeof(float);

        // Use fewer repetitions for larger workloads.
        const int iterations =
            (n <= 1024) ? 20 : (n == 2048 ? 5 : 2);

        std::cout << "Testing " << n << " x " << n
                  << " (warmup=" << WARMUP
                  << ", iterations=" << iterations << ")...\n";

        std::vector<float> h_A(count, 1.0f);
        std::vector<float> h_B(count, 2.0f);

        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        CHECK_CUDA(cudaMalloc(&d_A, bytes));
        CHECK_CUDA(cudaMalloc(&d_B, bytes));
        CHECK_CUDA(cudaMalloc(&d_C, bytes));

        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes,
                              cudaMemcpyHostToDevice));

        dim3 threads(16, 16);
        dim3 blocks((N + threads.x - 1) / threads.x,
                    (M + threads.y - 1) / threads.y);

        float naive_ms = benchmark_naive(
            d_A, d_B, d_C, M, K, N,
            blocks, threads, WARMUP, iterations
        );
        bool naive_ok = validate_output(d_C, M, N, 2.0f * K);

        float tiled_ms = benchmark_tiled(
            d_A, d_B, d_C, M, K, N,
            blocks, threads, WARMUP, iterations
        );
        bool tiled_ok = validate_output(d_C, M, N, 2.0f * K);

        if (!naive_ok || !tiled_ok) {
            std::cerr << "Correctness check failed for size "
                      << n << ".\n";
            CHECK_CUDA(cudaFree(d_A));
            CHECK_CUDA(cudaFree(d_B));
            CHECK_CUDA(cudaFree(d_C));
            return EXIT_FAILURE;
        }

        const double flops =
            2.0 * static_cast<double>(M) * K * N;
        const double naive_gflops = flops / (naive_ms * 1e6);
        const double tiled_gflops = flops / (tiled_ms * 1e6);

        results.push_back({
            n, naive_ms, tiled_ms, naive_gflops,
            tiled_gflops, naive_ms / tiled_ms
        });

        std::cout << "  Correctness: PASS (both kernels)\n"
                  << "  Naive: " << naive_ms << " ms, "
                  << naive_gflops << " GFLOP/s\n"
                  << "  Tiled: " << tiled_ms << " ms, "
                  << tiled_gflops << " GFLOP/s\n"
                  << "  Speedup: " << naive_ms / tiled_ms << "x\n\n";

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nSUMMARY TABLE\n"
              << "Size,Naive_ms,Tiled_ms,Naive_GFLOP_s,"
                 "Tiled_GFLOP_s,Speedup\n";

    for (const auto& r : results) {
        std::cout << r.size << ","
                  << r.naive_ms << ","
                  << r.tiled_ms << ","
                  << r.naive_gflops << ","
                  << r.tiled_gflops << ","
                  << r.speedup << "\n";
    }

    return EXIT_SUCCESS;
}
