#include <cuda_runtime.h>
#include <iostream>

// This global keyword function is Cuda Kernal thats launched from the cpu to the gpu and then executed.

__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {

    const int N = 1024;

    float *h_a, *h_b, *h_c;

    cudaMallocHost(&h_a, N * sizeof(float));
    cudaMallocHost(&h_b, N * sizeof(float));
    cudaMallocHost(&h_c, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // cudaMemcpy is used for moving the data between the CPU and the GPU

    cudaMemcpy(
        d_a,
        h_a,
        N * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_b,
        h_b,
        N * sizeof(float),
        cudaMemcpyHostToDevice
    );

    int threads = 512;
    int blocks = (N + threads - 1) / threads;

    //This vector_add used to launch the kernal

    vector_add<<<blocks, threads>>>(
        d_a,
        d_b,
        d_c,
        N
    );

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_c,
        d_c,
        N * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    for (int i = 0; i < 10; i++) {
        std::cout
            << h_a[i]
            << " + "
            << h_b[i]
            << " = "
            << h_c[i]
            << '\n';
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    return 0;
}
