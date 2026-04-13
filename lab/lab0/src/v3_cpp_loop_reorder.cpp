#include "matmul_common.hpp"

void matmul(const double* a, const double* b, double* c, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < n; ++p) {
            const double a_ip = a[static_cast<std::size_t>(i) * n + p];
            const std::size_t b_row = static_cast<std::size_t>(p) * k;
            const std::size_t c_row = static_cast<std::size_t>(i) * k;
            for (int j = 0; j < k; ++j) {
                c[c_row + j] += a_ip * b[b_row + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    return run_program("v3_cpp_loop_reorder", argc, argv, matmul);
}
