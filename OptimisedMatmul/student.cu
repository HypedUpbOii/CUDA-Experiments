#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// ─── Error-checking macro ────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)


/* ============================================================
   Edit ONLY this section.
   ============================================================ */

/**
 * TODO: Implement your CUDA kernel(s) here.
 *
 * You may define multiple __global__ and __device__ functions.
 * You may use templates, #define constants, and helper structs.
 */

#define TILE_DIM 64
#define REG_DIM 4
#define BLOCK_THREADS (TILE_DIM / REG_DIM)
#define LOADS_PER_THREAD (TILE_DIM * TILE_DIM / (BLOCK_THREADS * BLOCK_THREADS))

// ─── Example: a bare naive kernel to get you started ─────────────────────────
__global__
void matmul_kernel_naive(const float* __restrict__ A,
                         const float* __restrict__ B,
                               float* __restrict__ C,
                         int N)
{
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BLOCK_THREADS + tx;

    const int block_row = blockIdx.y * TILE_DIM;
    const int block_col = blockIdx.x * TILE_DIM;

    const int row = block_row + ty * REG_DIM;
    const int col = block_col + tx * REG_DIM;

    float c[REG_DIM][REG_DIM] = {0.0f};

    for (int t = 0; t < N; t += TILE_DIM)
    {
        #pragma unroll
        for (int load = 0; load < LOADS_PER_THREAD; ++load)
        {
            int idx   = tid + load * (BLOCK_THREADS * BLOCK_THREADS);
            int s_row = idx / TILE_DIM;
            int s_col = idx % TILE_DIM;
            As[s_row][s_col] = A[(block_row + s_row) * N + (t + s_col)];
        }

        #pragma unroll
        for (int load = 0; load < LOADS_PER_THREAD; ++load)
        {
            int idx   = tid + load * (BLOCK_THREADS * BLOCK_THREADS);
            int s_row = idx / TILE_DIM;
            int s_col = idx % TILE_DIM;
            // Transpose on write: swap [s_row][s_col] -> [s_col][s_row]
            Bs[s_col][s_row] = B[(block_col + s_row) * N + (t + s_col)];
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k)
        {
            float a[REG_DIM], b[REG_DIM];

            #pragma unroll
            for (int i = 0; i < REG_DIM; ++i)
                a[i] = As[ty * REG_DIM + i][k];

            #pragma unroll
            for (int j = 0; j < REG_DIM; ++j)
                b[j] = Bs[k][tx * REG_DIM + j];  // row-stride, conflict-free

            #pragma unroll
            for (int i = 0; i < REG_DIM; ++i)
                #pragma unroll
                for (int j = 0; j < REG_DIM; ++j)
                    c[i][j] += a[i] * b[j];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < REG_DIM; ++i)
        #pragma unroll
        for (int j = 0; j < REG_DIM; ++j)
            C[(row + i) * N + (col + j)] = c[i][j];
}

/**
 * @brief Launch wrapper — allocate device memory, copy data,
 *        run your kernel(s), copy result back. You aren't allowed to change this function signature.
 *
 * @param N    Matrix dimension (N x N).  Always a power of 2.
 * @param A_h  Host pointer to matrix A (row-major, N*N floats).
 * @param B_h  Host pointer to matrix B (row-major, N*N floats).
 * @param C_h  Host pointer to output C (row-major, N*N floats).
 *             You must write the result here before returning.
 */
void matmul_gpu(int N,
                const float* A_h,
                const float* B_h,
                      float* C_h)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    std::vector<float> B_T(N * N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            B_T[(j * N) + i] = B_h[(i * N) + j];
    }

    // ── Allocate device buffers ───────────────────────────────
    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, bytes));
    CUDA_CHECK(cudaMalloc(&B_d, bytes));
    CUDA_CHECK(cudaMalloc(&C_d, bytes));

    // ── Transfer inputs to device ─────────────────────────────
    CUDA_CHECK(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_T.data(), bytes, cudaMemcpyHostToDevice));

    {
        dim3 block(16, 16);
        dim3 grid(N / 64,
                  N / 64);

        matmul_kernel_naive<<<grid, block>>>(A_d, B_d, C_d, N);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Copy result back to host ──────────────────────────────
    CUDA_CHECK(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

    // ── Free device memory ────────────────────────────────────
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

/* ============================================================
   END OF STUDENT CODE — do not modify below this line
   ============================================================ */


// ─── CPU reference ────────────────────────────────────────────────────────────
static void matmul_cpu(int N,
                       const float* A,
                       const float* B,
                             float* C)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < N; ++k)
                s += A[i*N+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

// ─── Element-wise verification ────────────────────────────────────────────────
static bool verify(int N, const float* ref, const float* gpu,
                   float tol = 1e-2f)
{
    for (int i = 0; i < N*N; ++i) {
        float diff = fabsf(ref[i] - gpu[i]);
        if (diff > tol) {
            int row = i / N, col = i % N;
            fprintf(stderr,
                    "MISMATCH at (%d,%d): ref=%.6f  gpu=%.6f  |diff|=%.2e\n",
                    row, col, ref[i], gpu[i], diff);
            return false;
        }
    }
    return true;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main()
{
    // ── Correctness tests (small sizes, CPU reference) ────────
    printf("=== Correctness Tests ===\n");
    {
        const std::vector<int> small_sizes = {64, 128, 256, 512};
        bool all_ok = true;

        for (int N : small_sizes) {

            std::vector<float> A(N*N), B(N*N),
                               C_cpu(N*N, 0.f),
                               C_gpu(N*N, 0.f);

            for (int i = 0; i < N*N; ++i) {
                A[i] = (float)(i % 97) / 97.f;
                B[i] = (float)((i * 7 + 3) % 97) / 97.f;
            }

            matmul_cpu(N, A.data(), B.data(), C_cpu.data());
            matmul_gpu(N, A.data(), B.data(), C_gpu.data());

            bool ok = verify(N, C_cpu.data(), C_gpu.data());
            printf("  N = %4d : %s\n", N, ok ? "PASSED" : "FAILED");
            all_ok &= ok;
        }

        if (!all_ok) {
            fprintf(stderr,
                    "\nCorrectness FAILED — fix your kernel before optimising.\n");
            return EXIT_FAILURE;
        }
        printf("All correctness tests PASSED.\n\n");
    }

    return EXIT_SUCCESS;
}
