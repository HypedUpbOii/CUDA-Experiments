#include <cuda_runtime.h>
#include <random>
#include <iostream>

__global__ void matmul(float* A, float* B, float* C, int M, int K, int N) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < M && y < N) {
        float sum = 0;
        for (int i = 0; i < K; i++) {
            sum += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = sum;
    }
}

int main(int argc, char* argv[]){
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

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

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

    delete[] matrix1;
    delete[] matrix2;
    delete[] matrix3;
    return 0;
}
