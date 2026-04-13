#include "matmul_common.hpp"

void matmul(const double* a, const double* b, double* c, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            double sum = 0.0;
            for (int p = 0; p < n; ++p) {
                sum += a[static_cast<std::size_t>(i) * n + p] * b[static_cast<std::size_t>(p) * k + j];
            }
            c[static_cast<std::size_t>(i) * k + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    return run_program("v2_cpp_baseline", argc, argv, matmul);
}
