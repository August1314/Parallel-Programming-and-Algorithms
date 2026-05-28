#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA kernel: each thread prints its block ID, thread ID, and "Hello World!"
__global__ void helloKernel() {
    int blockId = blockIdx.x;
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    printf("Hello World from Thread (%d, %d) in Block %d!\n",
           threadX, threadY, blockId);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <grid_x> <block_x> <block_y>\n", argv[0]);
        fprintf(stderr, "  All values must be in range [1, 32]\n");
        return 1;
    }

    int grid_x  = atoi(argv[1]);
    int block_x = atoi(argv[2]);
    int block_y = atoi(argv[3]);

    // Validate input ranges
    if (grid_x < 1 || grid_x > 32 ||
        block_x < 1 || block_x > 32 ||
        block_y < 1 || block_y > 32) {
        fprintf(stderr, "Error: all arguments must be in range [1, 32]\n");
        return 1;
    }

    int total_threads = grid_x * block_x * block_y;
    printf("Launching %d block(s) × (%d × %d) threads = %d total threads\n",
           grid_x, block_x, block_y, total_threads);
    printf("--- GPU Output ---\n");

    // Launch kernel
    dim3 gridDim(grid_x);
    dim3 blockDim(block_x, block_y);
    helloKernel<<<gridDim, blockDim>>>();

    // Wait for all GPU threads to finish printing
    cudaDeviceSynchronize();

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("--- GPU Output End ---\n");
    printf("Hello World from the host!\n");

    // Output key=value for benchmark compatibility
    printf("experiment=hello_world\n");
    printf("grid_x=%d\n", grid_x);
    printf("block_x=%d\n", block_x);
    printf("block_y=%d\n", block_y);
    printf("total_threads=%d\n", total_threads);

    return 0;
}
