#include "matmul_common.hpp"

#if __has_include(<mkl_cblas.h>)
#include <mkl_cblas.h>
#define LAB1_HAS_MKL 1
#elif __has_include(<cblas.h>)
#include <cblas.h>
#define LAB1_HAS_MKL 1
#else
#define LAB1_HAS_MKL 0
#endif

void matmul(const double* a, const double* b, double* c, int m, int n, int k) {
#if LAB1_HAS_MKL
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, n, 1.0, a, n, b, k, 0.0, c, k);
#else
    (void)a;
    (void)b;
    (void)c;
    (void)m;
    (void)n;
    (void)k;
    throw std::runtime_error("MKL headers not found. Build and run this target on Ubuntu/Intel with Intel MKL.");
#endif
}

int main(int argc, char** argv) {
    return run_program("v6_mkl", argc, argv, matmul);
}
