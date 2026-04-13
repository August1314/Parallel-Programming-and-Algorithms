#include "matmul_common.hpp"

void matmul(const double* a, const double* b, double* c, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        const std::size_t c_row = static_cast<std::size_t>(i) * k;
        for (int p = 0; p < n; ++p) {
            const double a_ip = a[static_cast<std::size_t>(i) * n + p];
            const std::size_t b_row = static_cast<std::size_t>(p) * k;

            int j = 0;
            for (; j + 3 < k; j += 4) {
                c[c_row + j] += a_ip * b[b_row + j];
                c[c_row + j + 1] += a_ip * b[b_row + j + 1];
                c[c_row + j + 2] += a_ip * b[b_row + j + 2];
                c[c_row + j + 3] += a_ip * b[b_row + j + 3];
            }
            for (; j < k; ++j) {
                c[c_row + j] += a_ip * b[b_row + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    return run_program("v5_cpp_unrolled", argc, argv, matmul);
}
