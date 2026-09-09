
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// Each thread computes one element of C.
__global__ void matmul_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {

        float sum = 0.0f;

        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}


void matmul_cpu(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    for (int row = 0; row < M; row++) {

        for (int col = 0; col < N; col++) {

            float sum = 0.0f;

            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}


int main() {

    const int M = 128;
    const int K = 128;
    const int N = 128;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];
    float* h_C_cpu = new float[M * N];

    for (int i = 0; i < M * K; i++) {
        h_A[i] = 1.0f;
    }

    for (int i = 0; i < K * N; i++) {
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(
        d_A,
        h_A,
        sizeA,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        h_B,
        sizeB,
        cudaMemcpyHostToDevice
    );

    dim3 threads(16, 16);

    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    matmul_naive<<<blocks, threads>>>(
        d_A,
        d_B,
        d_C,
        M,
        K,
        N
    );

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_C,
        d_C,
        sizeC,
        cudaMemcpyDeviceToHost
    );

    matmul_cpu(
        h_A,
        h_B,
        h_C_cpu,
        M,
        K,
        N
    );

    bool correct = true;

    for (int i = 0; i < M * N; i++) {

        if (std::fabs(h_C[i] - h_C_cpu[i]) > 1e-5f) {
            std::cout
                << "Mismatch at index "
                << i
                << ": GPU = "
                << h_C[i]
                << ", CPU = "
                << h_C_cpu[i]
                << '\n';

            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "PASS: naive matmul correctness\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cpu;

    return 0;
}
