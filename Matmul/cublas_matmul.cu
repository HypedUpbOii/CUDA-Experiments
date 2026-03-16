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
    float* matrix_ref = new float[M * N];

    // random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 9);

    for(int i=0;i<M*K;i++) matrix1[i] = dist(gen);
    for(int i=0;i<K*N;i++) matrix2[i] = dist(gen);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += matrix1[i*K + k] * matrix2[k*N + j];
            }
            matrix_ref[i * N + j] = sum;
        }
    }

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

    float max_diff = 0.0f;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            max_diff = std::max(max_diff, std::abs(matrix_ref[i * N + j] - matrix3[i * N + j]));
        }
    }

    std::cout << "Maximum difference between elements: " << max_diff << std::endl;

    delete[] matrix1;
    delete[] matrix2;
    delete[] matrix3;
    delete[] matrix_ref;
    return 0;
}