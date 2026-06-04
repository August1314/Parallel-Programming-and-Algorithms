#ifndef COMMON_HPP
#define COMMON_HPP

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
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
inline void initMatrix(float *mat, int rows, int cols, unsigned int seed = 20250401) {
    std::srand(seed);
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
}

// CPU GEMM for verification: C = A * B
inline void cpuGemm(const float *A, const float *B, float *C,
                    int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += static_cast<double>(A[i * K + k]) *
                       static_cast<double>(B[k * N + j]);
            }
            C[i * N + j] = static_cast<float>(sum);
        }
    }
}

// Verify GPU GEMM against CPU reference
inline bool verifyGemm(const float *C_gpu, const float *A, const float *B,
                       int M, int N, int K, float tolerance = 1e-4f) {
    float *C_cpu = new float[M * N];
    cpuGemm(A, B, C_cpu, M, N, K);

    bool correct = true;
    int errors = 0;
    // Use relative error for large values
    for (int i = 0; i < M * N; ++i) {
        float abs_diff = std::fabs(C_gpu[i] - C_cpu[i]);
        float rel_diff = abs_diff / std::max(1.0f, std::fabs(C_cpu[i]));
        if (rel_diff > tolerance && abs_diff > tolerance) {
            correct = false;
            errors++;
            if (errors <= 5) {
                std::fprintf(stderr, "Mismatch at [%d]: cpu=%.6f gpu=%.6f "
                             "abs_diff=%.6f rel_diff=%.6f\n",
                             i, C_cpu[i], C_gpu[i], abs_diff, rel_diff);
            }
        }
    }
    if (errors > 0) {
        std::fprintf(stderr, "Total errors: %d / %d\n", errors, M * N);
    }

    delete[] C_cpu;
    return correct;
}

// Compute checksum for correctness comparison across kernels
inline float computeChecksum(const float *mat, int rows, int cols) {
    double sum = 0.0;
    for (int i = 0; i < rows * cols; ++i) {
        sum += mat[i];
    }
    return static_cast<float>(sum);
}

// Print matrix (for small matrices in report)
inline void printMatrix(const float *mat, int rows, int cols, const char *name) {
    if (rows * cols > 100) return;  // Only print small matrices
    std::printf("%s:\n", name);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::printf("%8.4f ", mat[i * cols + j]);
        }
        std::printf("\n");
    }
}

// Parse block dimension from string e.g. "32x32" -> (32, 32)
inline void parseBlockDim(const char *s, int &bx, int &by) {
    bx = 32; by = 32;
    std::sscanf(s, "%dx%d", &bx, &by);
}

// Get wall clock time in seconds (for CPU timing)
inline double getTimeSec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#endif // COMMON_HPP
