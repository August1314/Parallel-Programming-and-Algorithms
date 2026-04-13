#include "matmul_common.hpp"

#include <mkl_cblas.h>

void matmul(const double* a, const double* b, double* c, int m, int n, int k) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, n, 1.0, a, n, b, k, 0.0, c, k);
}

int main(int argc, char** argv) {
    return run_program("v6_mkl", argc, argv, matmul);
}
