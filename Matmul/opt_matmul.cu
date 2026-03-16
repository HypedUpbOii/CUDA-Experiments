#include <cuda_runtime.h>
#include <random>
#include <iostream>

__global__ void matmul(float* A, float* B, float* C, int M, int K, int N) {
    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    int y = threadIdx.y + (blockIdx.y * blockDim.y);

    if (y < M && x < N) {
        float sum = 0;
        for (int i = 0; i < K; ++i) {
            sum += A[y * K + i] * B[i * N + x];
        }
        C[y * N + x] = sum;
    }
}

int main(int argc, char* argv[]){
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

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    float* A;
    float* B;
    float* C;
    cudaMalloc((void**)&A, sizeof(float) * M * K);
    cudaMalloc((void**)&B, sizeof(float) * K * N);
    cudaMalloc((void**)&C, sizeof(float) * M * N);
    cudaMemcpy(A, matrix1, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(B, matrix2, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    matmul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(matrix3, C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
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
