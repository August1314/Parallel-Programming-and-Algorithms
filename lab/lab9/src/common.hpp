#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// CUDA error check macro
#define CHECK_CUDA(call) do {                                   \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",       \
                     __FILE__, __LINE__,                        \
                     cudaGetErrorString(err));                  \
        std::exit(1);                                           \
    }                                                           \
} while(0)

// Initialize matrix with random float values in [0, 1)
inline void initMatrix(float *mat, int N, unsigned int seed = 20250401) {
    std::srand(seed);
    for (int i = 0; i < N * N; ++i) {
        mat[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
}

// CPU transpose for verification
inline void cpuTranspose(const float *A, float *B, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            B[j * N + i] = A[i * N + j];
        }
    }
}

// Verify GPU transpose against CPU reference
inline bool verifyTranspose(const float *A, const float *B_gpu, int N,
                             float tolerance = 1e-5f) {
    float *B_cpu = new float[N * N];
    cpuTranspose(A, B_cpu, N);

    bool correct = true;
    int errors = 0;
    for (int i = 0; i < N * N; ++i) {
        float diff = std::fabs(B_gpu[i] - B_cpu[i]);
        if (diff > tolerance) {
            correct = false;
            errors++;
            if (errors <= 5) {
                std::fprintf(stderr, "Mismatch at [%d]: cpu=%.6f gpu=%.6f diff=%.6f\n",
                             i, B_cpu[i], B_gpu[i], diff);
            }
        }
    }
    if (errors > 0) {
        std::fprintf(stderr, "Total errors: %d / %d\n", errors, N * N);
    }

    delete[] B_cpu;
    return correct;
}

// Compute checksum for correctness comparison across kernels
inline float computeChecksum(const float *mat, int N) {
    double sum = 0.0;
    for (int i = 0; i < N * N; ++i) {
        sum += mat[i];
    }
    return static_cast<float>(sum);
}

// Print matrix (for small matrices in report)
inline void printMatrix(const float *mat, int N, const char *name) {
    if (N > 16) return;  // Only print small matrices
    std::printf("%s:\n", name);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::printf("%8.4f ", mat[i * N + j]);
        }
        std::printf("\n");
    }
}

// Get block dim helper: convert string like "32x32" to dim3
inline dim3 parseBlockDim(const char *s) {
    int bx = 32, by = 32;
    std::sscanf(s, "%dx%d", &bx, &by);
    return dim3(bx, by);
}

#endif // COMMON_HPP
