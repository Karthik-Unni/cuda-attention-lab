
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define TILE_SIZE 16

__global__ void matmul_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {

        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;

        if (row < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] =
                A[row * K + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] =
                B[b_row * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] *
                   tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
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
                sum += A[row * K + k] *
                       B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}

int main() {

    const int M = 127;
    const int K = 131;
    const int N = 129;

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

    dim3 threads(TILE_SIZE, TILE_SIZE);

    dim3 blocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    matmul_tiled<<<blocks, threads>>>(
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
        std::cout
            << "PASS: tiled matmul correctness\n";
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
