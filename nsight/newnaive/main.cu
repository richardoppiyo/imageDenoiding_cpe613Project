#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace kernels {

template <typename T>
__global__ void matrixMultiply(T *A, T *B, T *C, int M) {
    int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    int tidY = blockIdx.y * blockDim.y + threadIdx.y;
    int strideX = blockDim.x * gridDim.x;
    int strideY = blockDim.y * gridDim.y;

    for (int row = tidY; row < M; row += strideY) {
        for (int col = tidX; col < M; col += strideX) {
            T sum = 0.0;
            for (int i = 0; i < M; i++) {
                sum += A[row * M + i] * B[i * M + col];
            }
            C[row * M + col] = sum;
        }
    }
}

} // end namespace kernels

template <typename T>
void matrixMultiply(T *A, T *B, T *C, int M) {
    int blockSizeX = 32; // Standard block size for 2D CUDA matrix multiplication
    int blockSizeY = 32;

    dim3 dimBlock(blockSizeX, blockSizeY);
    dim3 dimGrid((M + blockSizeX - 1) / blockSizeX, (M + blockSizeY - 1) / blockSizeY);

    double cumulativeTime_ms = 0.0;
    int num_runs = 2;
    for(int run_idx = 0; run_idx < num_runs; ++run_idx) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        kernels::matrixMultiply<<<dimGrid, dimBlock>>>(A, B, C, M);
        cudaEventRecord(stop);

        float duration_ms = 0.0f;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&duration_ms, start, stop);
        cumulativeTime_ms += duration_ms;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("N = %d , Time in milliseconds: %20.16f\n", M, cumulativeTime_ms / num_runs);

    checkCudaErrors(cudaGetLastError());
#ifndef NDEBUG
    checkCudaErrors(cudaDeviceSynchronize());
#endif
}

template <typename T>
void printMatrix(T *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

template <typename T>
void initializeMatrix(T *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = static_cast<T>(rand()) / RAND_MAX;
    }
}

int main() {
    const int minN = 64;
    const int maxN = 128;

    for (int N = minN; N <= maxN; N *= 2) {
        float *h_A = (float *)malloc(N * N * sizeof(float));
        float *h_B = (float *)malloc(N * N * sizeof(float));
        float *h_C = (float *)malloc(N * N * sizeof(float));

        initializeMatrix(h_A, N * N);
        initializeMatrix(h_B, N * N);

        float *d_A, *d_B, *d_C;
        checkCudaErrors(cudaMalloc((void **)&d_A, N * N * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_B, N * N * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_C, N * N * sizeof(float)));

        checkCudaErrors(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice));

        printf("Time taken for %d input Single precision:\n", N);
        matrixMultiply(d_A, d_B, d_C, N);

        checkCudaErrors(cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

        // Optional: Print matrices
        // printf("Matrix A:\n");
        // printMatrix(h_A, N, N);
        // printf("Matrix B:\n");
        // printMatrix(h_B, N, N);
        // printf("Result Matrix (float):\n");
        // printMatrix(h_C, N, N);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        free(h_A);
        free(h_B);
        free(h_C);
    }
    return 0;
}

