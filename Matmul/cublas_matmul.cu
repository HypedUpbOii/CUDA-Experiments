#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

int main(int argc, char* argv[]) {
    int M, K, N;
    M = K = N = std::atoi(argv[1]);

    float* matrix1 = new float[M * K];
    float* matrix2 = new float[K * N];
    float* matrix3 = new float[M * N];

    // random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 9);

    for(int i=0;i<M*K;i++) matrix1[i] = dist(gen);
    for(int i=0;i<K*N;i++) matrix2[i] = dist(gen);

    float alpha = 1.0f;
    float beta = 0.0f;

    float* A;
    float* B;
    float* C;

    cudaMalloc((void**)&A, sizeof(float) * M * K);
    cudaMalloc((void**)&B, sizeof(float) * K * N);
    cudaMalloc((void**)&C, sizeof(float) * M * N);

    cudaMemcpy(A, matrix1, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(B, matrix2, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N
    );

    cudaMemcpy(matrix3, C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cublasDestroy(handle);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    delete[] matrix1;
    delete[] matrix2;
    delete[] matrix3;
    return 0;
}