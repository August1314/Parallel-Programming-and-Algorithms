#ifndef COMMON_HPP
#define COMMON_HPP

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>

// ============================================================
// CUDA error check
// ============================================================
#define CHECK_CUDA(call) do {                                   \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",       \
                     __FILE__, __LINE__,                        \
                     cudaGetErrorString(err));                  \
        std::exit(1);                                           \
    }                                                           \
} while(0)

// ============================================================
// cuDNN error check (only when USE_CUDNN is defined)
// ============================================================
#ifdef USE_CUDNN
#include <cudnn.h>
#define CHECK_CUDNN(call) do {                                  \
    cudnnStatus_t s = (call);                                   \
    if (s != CUDNN_STATUS_SUCCESS) {                            \
        std::fprintf(stderr, "cuDNN error at %s:%d: %s\n",      \
                     __FILE__, __LINE__,                        \
                     cudnnGetErrorString(s));                   \
        std::exit(1);                                           \
    }                                                           \
} while(0)
#endif

// ============================================================
// Initialize 4D tensor layout with random values in [0, 1)
// Layout: data[n * (C*H*W) + c * (H*W) + y * W + x]
// For this lab, typically N=1.
// ============================================================
inline void initTensor(float *data, int N, int C, int H, int W,
                       unsigned int seed = 20250401) {
    std::srand(seed);
    int total = N * C * H * W;
    for (int i = 0; i < total; ++i) {
        data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
}

// ============================================================
// Compute checksum (sum of all elements) for cross-kernel verification
// ============================================================
inline float computeChecksum(const float *data, int total) {
    double sum = 0.0;
    for (int i = 0; i < total; ++i) {
        sum += data[i];
    }
    return static_cast<float>(sum);
}

// ============================================================
// CPU 2D Convolution (reference implementation for verification)
//
// Layout:
//   input[ (c * H + y) * W + x ]
//   weight[ ((co * C_in + ci) * KH + ky) * KW + kx ]
//   output[ (co * H_out + y) * W_out + x ]
//
// No bias, no filter flip (CNN-style cross-correlation).
// ============================================================
inline void cpuConv2D(const float *input, const float *weight,
                      float *output,
                      int C_in, int C_out,
                      int H, int W, int KH, int KW,
                      int stride, int padding,
                      int H_out, int W_out) {
    for (int co = 0; co < C_out; ++co) {
        for (int y = 0; y < H_out; ++y) {
            for (int x = 0; x < W_out; ++x) {
                float sum = 0.0f;
                for (int ci = 0; ci < C_in; ++ci) {
                    for (int ky = 0; ky < KH; ++ky) {
                        for (int kx = 0; kx < KW; ++kx) {
                            int iy = y * stride + ky - padding;
                            int ix = x * stride + kx - padding;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                sum += input[(ci * H + iy) * W + ix]
                                     * weight[((co * C_in + ci) * KH + ky) * KW + kx];
                            }
                        }
                    }
                }
                output[(co * H_out + y) * W_out + x] = sum;
            }
        }
    }
}

// ============================================================
// Verify GPU conv result against CPU reference
// ============================================================
inline bool verifyConv(const float *gpu_out, const float *input,
                       const float *weight,
                       int C_in, int C_out,
                       int H, int W, int KH, int KW,
                       int stride, int padding,
                       int H_out, int W_out,
                       float tolerance = 1e-4f) {
    int total = C_out * H_out * W_out;
    float *cpu_out = new float[total];
    cpuConv2D(input, weight, cpu_out,
              C_in, C_out, H, W, KH, KW, stride, padding, H_out, W_out);

    bool correct = true;
    int errors = 0;
    for (int i = 0; i < total; ++i) {
        float abs_diff = std::fabs(gpu_out[i] - cpu_out[i]);
        float rel_diff = abs_diff / std::max(1.0f, std::fabs(cpu_out[i]));
        if (rel_diff > tolerance && abs_diff > tolerance) {
            correct = false;
            errors++;
            if (errors <= 5) {
                std::fprintf(stderr, "Mismatch at [%d]: cpu=%.6f gpu=%.6f "
                             "abs=%.6f rel=%.6f\n",
                             i, cpu_out[i], gpu_out[i], abs_diff, rel_diff);
            }
        }
    }
    if (errors > 0) {
        std::fprintf(stderr, "Total errors: %d / %d\n", errors, total);
    }

    delete[] cpu_out;
    return correct;
}

// ============================================================
// Get wall clock time in seconds
// ============================================================
inline double getTimeSec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#endif // COMMON_HPP
