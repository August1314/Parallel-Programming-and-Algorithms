/*
 * lab10 — CUDA 并行通用矩阵乘法 (GEMM)
 *
 * 实现多种 CUDA kernel 对 C = A × B 进行性能对比分析：
 *   Kernel 1: Naive（朴素全局内存）
 *   Kernel 2: Shared Memory Tiled（共享内存分块, TILE=16）
 *   Kernel 3: Shared Memory Tiled（共享内存分块, TILE=32）
 *
 * 分析维度：
 *   - 线程块大小 (block size): 8×8, 16×16, 32×32
 *   - 访存方式: global memory vs shared memory tiling
 *   - 分块大小: TILE=16 vs TILE=32
 *   - 矩阵规模: M, N, K ∈ [128, 2048]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// Kernel 1: Naive GEMM（朴素全局内存实现）
//
// 每个线程计算 C 的一个元素，直接从全局内存读取 A 和 B。
// 访存特征：
//   - A[row][k]: 同一 warp 内 row 相同的线程访问相同地址（冗余读），
//                 row 不同的线程访问步长为 K（非合并）
//   - B[k][col]: 同一 warp 内相邻线程访问相邻列（合并读）
// 问题：大量冗余的 A 读取 + 无数据复用 = 低效
// ============================================================
__global__ void gemmNaive(const float *A, const float *B, float *C,
                          int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================
// Kernel 2: Shared Memory Tiled GEMM（共享内存分块，TILE=16）
//
// 使用 16×16 共享内存分块：
//   1. 每个 block 内的线程协作将 A 和 B 的分块载入共享内存
//   2. 从共享内存读取数据进行局部乘累加
//   3. 遍历所有 K 维度的分块完成完整点积
//
// 优势：
//   - 共享内存访问延迟远低于全局内存（~20 vs ~400 cycles）
//   - 同一 block 内所有线程复用分块数据
//   - A 和 B 的全局内存读取都是合并的
// ============================================================
#define TILE_DIM_16 16

__global__ void gemmTiled16(const float *A, const float *B, float *C,
                            int M, int N, int K) {
    __shared__ float As[TILE_DIM_16][TILE_DIM_16];
    __shared__ float Bs[TILE_DIM_16][TILE_DIM_16];

    int row = blockIdx.y * TILE_DIM_16 + threadIdx.y;
    int col = blockIdx.x * TILE_DIM_16 + threadIdx.x;

    float sum = 0.0f;

    // 遍历 K 维度的所有分块
    int numTiles = (K + TILE_DIM_16 - 1) / TILE_DIM_16;
    for (int t = 0; t < numTiles; ++t) {
        // 协作加载 A 分块 (合并读)
        int a_col = t * TILE_DIM_16 + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 协作加载 B 分块 (合并读)
        int b_row = t * TILE_DIM_16 + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // 局部乘累加
        for (int k = 0; k < TILE_DIM_16; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================
// Kernel 3: Shared Memory Tiled GEMM（共享内存分块，TILE=32）
//
// 与 Kernel 2 相同算法，但使用更大的 32×32 分块。
// 优势：
//   - 更大的分块 → 更高的数据复用率
//   - 减少全局内存访问次数
// 劣势：
//   - 更多共享内存占用（2 × 32×32 × 4B = 8KB/block）
//   - 更少的活跃 block/SM → 占用率可能降低
// ============================================================
#define TILE_DIM_32 32

__global__ void gemmTiled32(const float *A, const float *B, float *C,
                            int M, int N, int K) {
    __shared__ float As[TILE_DIM_32][TILE_DIM_32];
    __shared__ float Bs[TILE_DIM_32][TILE_DIM_32];

    int row = blockIdx.y * TILE_DIM_32 + threadIdx.y;
    int col = blockIdx.x * TILE_DIM_32 + threadIdx.x;

    float sum = 0.0f;

    int numTiles = (K + TILE_DIM_32 - 1) / TILE_DIM_32;
    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE_DIM_32 + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_row = t * TILE_DIM_32 + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_DIM_32; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================
// Kernel 信息结构
// ============================================================
struct KernelInfo {
    const char *name;
    int tileDim;     // 分块大小（0 = 无分块）
    int defaultBlockX;
    int defaultBlockY;
};

// ============================================================
// CPU GEMM for verification
// ============================================================
void cpuGemm(const float *A, const float *B, float *C, int M, int N, int K) {
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

// ============================================================
// Verify GPU result against CPU reference
// ============================================================
bool verify(const float *C_gpu, const float *A, const float *B,
            int M, int N, int K, float tol = 1e-4f) {
    float *C_cpu = new float[M * N];
    cpuGemm(A, B, C_cpu, M, N, K);

    bool correct = true;
    int errors = 0;
    for (int i = 0; i < M * N && errors < 5; ++i) {
        float absDiff = std::fabs(C_gpu[i] - C_cpu[i]);
        float relDiff = absDiff / std::max(1.0f, std::fabs(C_cpu[i]));
        if (relDiff > tol && absDiff > tol) {
            correct = false;
            errors++;
            std::fprintf(stderr, "Mismatch[%d]: cpu=%.6f gpu=%.6f "
                         "abs=%.6f rel=%.6f\n",
                         i, C_cpu[i], C_gpu[i], absDiff, relDiff);
        }
    }
    if (errors > 0)
        std::fprintf(stderr, "Total errors: %d / %d\n", errors, M * N);

    delete[] C_cpu;
    return correct;
}

// ============================================================
// Compute checksum
// ============================================================
float checksum(const float *mat, int rows, int cols) {
    double s = 0.0;
    for (int i = 0; i < rows * cols; ++i) s += mat[i];
    return static_cast<float>(s);
}

// ============================================================
// Main
// ============================================================
int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::fprintf(stderr,
            "Usage: %s <M> <N> <K> <kernel_id> [block_x] [block_y]\n\n"
            "  M, N, K     : 矩阵维度 C(M×N) = A(M×K) × B(K×N), 范围 [128, 2048]\n"
            "  kernel_id   : 1=naive 2=tiled_16 3=tiled_32\n"
            "  block_x     : block x 维度 (默认: kernel 默认值)\n"
            "  block_y     : block y 维度 (默认: =block_x)\n\n"
            "Kernels:\n"
            "  1 - Naive global memory (无分块，直接全局内存)\n"
            "  2 - Shared memory tiled (TILE=16, 16×16 block)\n"
            "  3 - Shared memory tiled (TILE=32, 32×32 block)\n",
            argv[0]);
        return 1;
    }

    int M        = std::atoi(argv[1]);
    int N        = std::atoi(argv[2]);
    int K        = std::atoi(argv[3]);
    int kernelId = std::atoi(argv[4]);

    // 参数校验
    for (int dim : {M, N, K}) {
        if (dim < 128 || dim > 2048) {
            std::fprintf(stderr, "Error: 矩阵维度必须在 [128, 2048] 范围内\n");
            return 1;
        }
    }

    // 配置 kernel 信息
    KernelInfo info;
    int blockX, blockY;

    switch (kernelId) {
        case 1:
            info = {"naive", 0, 16, 16};
            break;
        case 2:
            info = {"tiled_16", TILE_DIM_16, TILE_DIM_16, TILE_DIM_16};
            break;
        case 3:
            info = {"tiled_32", TILE_DIM_32, TILE_DIM_32, TILE_DIM_32};
            break;
        default:
            std::fprintf(stderr, "Error: kernel_id 须为 1, 2, 或 3\n");
            return 1;
    }

    // 命令行覆盖 block 维度
    blockX = (argc >= 6) ? std::atoi(argv[5]) : info.defaultBlockX;
    blockY = (argc >= 7) ? std::atoi(argv[6]) : info.defaultBlockY;

    // 对于 tiled kernel，block 维度必须与 tile 维度匹配
    if (kernelId >= 2) {
        if (blockX != info.tileDim || blockY != info.tileDim) {
            std::fprintf(stderr,
                "Warning: tiled kernel 要求 block=(%d,%d)，已自动调整\n",
                info.tileDim, info.tileDim);
            blockX = info.tileDim;
            blockY = info.tileDim;
        }
    }

    size_t sizeA = static_cast<size_t>(M) * K * sizeof(float);
    size_t sizeB = static_cast<size_t>(K) * N * sizeof(float);
    size_t sizeC = static_cast<size_t>(M) * N * sizeof(float);

    // 分配主机内存
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    // 初始化随机矩阵（固定 seed 保证可复现）
    std::srand(20250401);
    for (size_t i = 0; i < static_cast<size_t>(M) * K; ++i)
        h_A[i] = static_cast<float>(std::rand()) / RAND_MAX;
    for (size_t i = 0; i < static_cast<size_t>(K) * N; ++i)
        h_B[i] = static_cast<float>(std::rand()) / RAND_MAX;

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // 配置 grid/block 维度
    dim3 blockDim(blockX, blockY);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 启动 kernel（取决于 kernel_id）
    auto launch = [&]() {
        switch (kernelId) {
            case 1:
                gemmNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 2:
                gemmTiled16<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 3:
                gemmTiled32<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
        }
    };

    // Warm-up（不计时）
    launch();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Kernel launch error (warm-up): %s\n",
                     cudaGetErrorString(err));
        return 1;
    }

    // 计时运行
    cudaEventRecord(start);
    launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Kernel launch error: %s\n",
                     cudaGetErrorString(err));
        return 1;
    }

    float kernelTimeMs = 0.0f;
    cudaEventElapsedTime(&kernelTimeMs, start, stop);

    // 拷贝结果回主机
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // 验证正确性
    bool correct = verify(h_C, h_A, h_B, M, N, K);

    // 计算 GFLOPS: 2 * M * N * K 次浮点运算
    double flops = 2.0 * M * N * K;
    double gflops = flops / (kernelTimeMs / 1000.0) / 1e9;

    // 计算有效带宽: 读 A + 读 B + 写 C
    double totalBytes = static_cast<double>(M * K + K * N + M * N) * sizeof(float);
    double bwGBs = totalBytes / (kernelTimeMs / 1000.0) / 1e9;

    // 输出 key=value 格式
    std::printf("experiment=gemm\n");
    std::printf("M=%d\n", M);
    std::printf("N=%d\n", N);
    std::printf("K=%d\n", K);
    std::printf("kernel=%s\n", info.name);
    std::printf("tile_dim=%d\n", info.tileDim);
    std::printf("block_dim=%dx%d\n", blockDim.x, blockDim.y);
    std::printf("grid_dim=%dx%d\n", gridDim.x, gridDim.y);
    std::printf("kernel_time_ms=%.6f\n", kernelTimeMs);
    std::printf("gflops=%.4f\n", gflops);
    std::printf("bandwidth_gb_s=%.4f\n", bwGBs);
    std::printf("checksum=%.4f\n", checksum(h_C, M, N));
    std::printf("correct=%s\n", correct ? "true" : "false");

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return correct ? 0 : 1;
}
