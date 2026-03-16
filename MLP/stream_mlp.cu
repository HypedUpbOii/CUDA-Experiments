#include <cuda_runtime.h>
#include <random>
#include <iostream>

#define NUM_STREAMS 4

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

__global__ void relu(float* B, int M, int N) {
    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    int y = threadIdx.y + (blockIdx.y * blockDim.y);

    if (y < M && x < N) {
        if (B[(y * N) + x] < 0.0f) {
            B[(y * N) + x] = 0.0f;
        }
    }
}

int main(int argc, char* argv[]) {
    int B, N;
    bool perform_check = false;
    B = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    if (argc > 3) {
        perform_check = true;
    }

    float* inputs = new float[B * N];
    float* weights1 = new float[N * N];
    float* weights2 = new float[N * N];
    float* answer = new float[B * N];

    // random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 9);

    for(int i=0;i<B * N;i++) inputs[i] = dist(gen);
    for(int i=0;i<N * N;i++) {
        weights1[i] = dist(gen);
        weights2[i] = dist(gen);
    }

    float* I;
    float* W1;
    float* res;
    float* W2;

    cudaMalloc((void**)&I, sizeof(float) * B * N);
    cudaMalloc((void**)&W1, sizeof(float) * N * N);
    cudaMalloc((void**)&res, sizeof(float) * B * N);
    cudaMalloc((void**)&W2, sizeof(float) * N * N);

    cudaMemcpy(I, inputs, sizeof(float) * B * N, cudaMemcpyHostToDevice);
    cudaMemcpy(W1, weights1, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(W2, weights2, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    int chunk = B / NUM_STREAMS;
    cudaStream_t streams[NUM_STREAMS];

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (chunk + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
        float* I_s = I + (i * chunk * N);
        float* res_s = res + (i * chunk * N);
        matmul<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(I_s, W1, res_s, chunk, N, N);
        relu<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(res_s, chunk, N);
        matmul<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(res_s, W2, I_s, chunk, N, N);
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(answer, I, sizeof(float) * B * N, cudaMemcpyDeviceToHost);

    cudaFree(I);
    cudaFree(W1);
    cudaFree(W2);
    cudaFree(res);

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    if (perform_check) {
        float* hidden = new float[B * N];
        float* answer_ref = new float[B * N];
    
        // hidden = inputs * weights1
        for (int y = 0; y < B; ++y) {
            for (int x = 0; x < N; ++x) {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    sum += inputs[y * N + k] * weights1[k * N + x];
                }
                hidden[y * N + x] = sum;
            }
        }
    
        // ReLU
        for (int i = 0; i < B * N; ++i) {
            if (hidden[i] < 0.0f) hidden[i] = 0.0f;
        }
    
        // answer_ref = hidden * weights2
        for (int y = 0; y < B; ++y) {
            for (int x = 0; x < N; ++x) {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    sum += hidden[y * N + k] * weights2[k * N + x];
                }
                answer_ref[y * N + x] = sum;
            }
        }
    
        float max_diff = 0.0f;
    
        for (int i = 0; i < B * N; ++i) {
            float diff = std::abs(answer[i] - answer_ref[i]);
            if (diff > max_diff) max_diff = diff;
        }
    
        std::cout << "Max difference: " << max_diff << std::endl;
        delete[] answer_ref;
        delete[] hidden;
    }

    std::cout << "Time elapsed: " << ms << std::endl;

    delete[] inputs;
    delete[] weights1;
    delete[] weights2;
    delete[] answer;
    return 0;
}
