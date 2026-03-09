#include <iostream>
#include <random>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
using namespace std;

constexpr int DIM = 1000;
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> dist(-50, 50);

bool check_same_pdt(int a[][DIM], int b[][DIM], int c[][DIM]) {
    // run probabilistic test 4 times
    for (int _ = 0; _ < 4; ++_) {
        int rv[DIM]; // random vector
        int pdt_b[DIM];
        int pdt_a[DIM];
        int pdt_c[DIM];
        for (int i = 0; i < DIM; ++i) {
            rv[i] = dist(gen);
        }

        // compute C * v and A * (B * v) and compare
        for (int i = 0; i < DIM; ++i) {
            pdt_b[i] = 0;
            for (int j = 0; j < DIM; ++j)
                pdt_b[i] += b[i][j] * rv[j];
        }

        for (int i = 0; i < DIM; ++i) {
            pdt_a[i] = 0;
            for (int j = 0; j < DIM; ++j)
                pdt_a[i] += a[i][j] * pdt_b[j];
        }

        for (int i = 0; i < DIM; ++i) {
            pdt_c[i] = 0;
            for (int j = 0; j < DIM; ++j)
                pdt_c[i] += c[i][j] * rv[j];
        }

        for (int i = 0; i < DIM; ++i) {
            if (pdt_c[i] != pdt_a[i]) {
                return false;
            }
        }
    }

    return true;
}

__global__ void matmul(int* A, int* B, int* C, int N) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < N && col < N) {
        int sum = 0;

        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

int main() {
    int a[1000][1000];
    int b[1000][1000];
    int c[1000][1000];

    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            a[i][j] = dist(gen);
            b[i][j] = dist(gen);
        }
    }

    // CPU - Correct implementation
    auto cpu_start = chrono::high_resolution_clock::now();
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            c[i][j] = 0;
            for (int k = 0; k < DIM; ++k)
                c[i][j] += a[i][k] * b[k][j];
        }
    }
    auto cpu_end = chrono::high_resolution_clock::now();
    cout << "CPU time: " << (chrono::duration_cast<chrono::milliseconds>(cpu_end - cpu_start)).count() << endl;

    auto gpu_start = chrono::high_resolution_clock::now();
    // send buffers to gpu
    int *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, DIM*DIM*sizeof(int));
    cudaMalloc(&d_b, DIM*DIM*sizeof(int));
    cudaMalloc(&d_c, DIM*DIM*sizeof(int));

    cudaMemcpy(d_a, a, DIM*DIM*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, DIM*DIM*sizeof(int), cudaMemcpyHostToDevice);
    // call kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((DIM + 15) / 16, (DIM + 15) / 16);

    matmul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, DIM);
    cudaDeviceSynchronize();
    // bring back buffer from gpu
    cudaMemcpy(c, d_c, DIM*DIM*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    auto gpu_end = chrono::high_resolution_clock::now();
    cout << "GPU time: " << (chrono::duration_cast<chrono::milliseconds>(gpu_end - gpu_start)).count() << endl;

    if (check_same_pdt(a, b, c)) {
        cout << "MATMUL was correct" << endl;
    } else {
        cout << "MATMUL was wrong" << endl;
    }
}