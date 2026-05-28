#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// ============================================================
// Kernel 1: Naive transpose (uncoalesced read)
// Each thread reads A[col][row] and writes B[row][col]
// Read is uncoalesced (stride-N access) → poor performance
// ============================================================
__global__ void transposeNaive(const float *A, float *B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        B[row * N + col] = A[col * N + row];
    }
}

// ============================================================
// Kernel 2: Coalesced read + uncoalesced write
// Each thread reads A[row][col] (coalesced) writes B[col][row]
// Better than naive because reads are coalesced
// ============================================================
__global__ void transposeCoalescedRead(const float *A, float *B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        B[col * N + row] = A[row * N + col];
    }
}

// ============================================================
// Kernel 3: Shared memory tiling (coalesced read + write)
// Uses shared memory tile to achieve coalesced reads AND writes
// TILE_DIM is a template parameter for compiler optimization
// ============================================================
#define TILE_DIM 32

__global__ void transposeSharedMem(const float *A, float *B, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 padding to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Coalesced read from global memory into shared memory
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    }

    __syncthreads();

    // Compute transposed indices for coalesced write
    int x_out = blockIdx.y * TILE_DIM + threadIdx.x;
    int y_out = blockIdx.x * TILE_DIM + threadIdx.y;

    // Coalesced write from shared memory to global memory
    if (x_out < N && y_out < N) {
        B[y_out * N + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}

// ============================================================
// Helper: get kernel by id
// ============================================================
typedef void (*TransposeKernel)(const float*, float*, int);

struct KernelInfo {
    const char *name;
    TransposeKernel kernel;
    int tileDim;  // 0 = not tiled
};

KernelInfo getKernelInfo(int kernelId) {
    switch (kernelId) {
        case 1: return {"naive",            transposeNaive,         0};
        case 2: return {"coalesced_read",   transposeCoalescedRead, 0};
        case 3: return {"shared_mem",       transposeSharedMem,     TILE_DIM};
        default:
            std::fprintf(stderr, "Invalid kernel_id=%d. Use 1=naive 2=coalesced_read 3=shared_mem\n",
                         kernelId);
            std::exit(1);
    }
}

// ============================================================
// CPU transpose for verification
// ============================================================
void cpuTranspose(const float *A, float *B, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            B[j * N + i] = A[i * N + j];
}

// ============================================================
// Verify GPU result against CPU reference
// ============================================================
bool verify(const float *A, const float *B_gpu, int N, float tol = 1e-5f) {
    float *B_cpu = new float[N * N];
    cpuTranspose(A, B_cpu, N);
    int errors = 0;
    for (int i = 0; i < N * N && errors < 5; ++i) {
        if (std::fabs(B_gpu[i] - B_cpu[i]) > tol) {
            std::fprintf(stderr, "Mismatch[%d]: cpu=%.6f gpu=%.6f\n", i, B_cpu[i], B_gpu[i]);
            errors++;
        }
    }
    delete[] B_cpu;
    return errors == 0;
}

// ============================================================
// Compute checksum
// ============================================================
float checksum(const float *mat, int N) {
    double s = 0.0;
    for (int i = 0; i < N * N; ++i) s += mat[i];
    return static_cast<float>(s);
}

// ============================================================
// Main
// ============================================================
int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <N> <kernel_id> [block_dim_x] [block_dim_y]\n", argv[0]);
        std::fprintf(stderr, "  N           : matrix size [512, 2048]\n");
        std::fprintf(stderr, "  kernel_id   : 1=naive 2=coalesced_read 3=shared_mem\n");
        std::fprintf(stderr, "  block_dim_x : block x dimension (default: 32)\n");
        std::fprintf(stderr, "  block_dim_y : block y dimension (default: 32)\n");
        return 1;
    }

    int N               = std::atoi(argv[1]);
    int kernelId        = std::atoi(argv[2]);
    int blockDimX       = (argc >= 4) ? std::atoi(argv[3]) : 32;
    int blockDimY       = (argc >= 5) ? std::atoi(argv[4]) : blockDimX;

    if (N < 512 || N > 2048) {
        std::fprintf(stderr, "Error: N must be in [512, 2048]\n");
        return 1;
    }

    KernelInfo info = getKernelInfo(kernelId);

    // For shared_mem kernel, override block dims to match TILE_DIM
    if (kernelId == 3) {
        blockDimX = TILE_DIM;
        blockDimY = TILE_DIM;
    }

    size_t matBytes = static_cast<size_t>(N) * N * sizeof(float);

    // Allocate host memory
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];

    // Initialize with random values
    std::srand(20250401);
    for (int i = 0; i < N * N; ++i)
        h_A[i] = static_cast<float>(std::rand()) / RAND_MAX;

    // Allocate device memory
    float *d_A, *d_B;
    cudaMalloc(&d_A, matBytes);
    cudaMalloc(&d_B, matBytes);
    cudaMemcpy(d_A, h_A, matBytes, cudaMemcpyHostToDevice);

    // Setup grid/block dimensions
    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run (not timed)
    info.kernel<<<gridDim, blockDim>>>(d_A, d_B, N);
    cudaDeviceSynchronize();

    // Timed run
    cudaEventRecord(start);
    info.kernel<<<gridDim, blockDim>>>(d_A, d_B, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernelTimeMs = 0.0f;
    cudaEventElapsedTime(&kernelTimeMs, start, stop);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back
    cudaMemcpy(h_B, d_B, matBytes, cudaMemcpyDeviceToHost);

    // Verify correctness
    bool correct = verify(h_A, h_B, N);

    // Compute bandwidth: 2 * N^2 * sizeof(float) bytes (read A + write B)
    float bwGBs = (2.0f * N * N * sizeof(float)) / (kernelTimeMs / 1000.0f) / 1e9f;

    // Output key=value format
    std::printf("experiment=transpose\n");
    std::printf("N=%d\n", N);
    std::printf("kernel=%s\n", info.name);
    std::printf("block_dim=%dx%d\n", blockDimX, blockDimY);
    std::printf("grid_dim=%dx%d\n", gridDim.x, gridDim.y);
    std::printf("kernel_time_ms=%.6f\n", kernelTimeMs);
    std::printf("bandwidth_gb_s=%.4f\n", bwGBs);
    std::printf("checksum=%.4f\n", checksum(h_B, N));
    std::printf("correct=%s\n", correct ? "true" : "false");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    delete[] h_A;
    delete[] h_B;

    return correct ? 0 : 1;
}
