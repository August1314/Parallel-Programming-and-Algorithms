#include "matmul_common.hpp"

#include <omp.h>

#include <vector>

#ifndef OPENMP_SCHEDULE_KIND
#define OPENMP_SCHEDULE_KIND 0
#endif

static const char* version_name() {
#if OPENMP_SCHEDULE_KIND == 0
    return "openmp_default";
#elif OPENMP_SCHEDULE_KIND == 1
    return "openmp_static1";
#else
    return "openmp_dynamic1";
#endif
}

static void compute_openmp(const MatrixRunConfig& config, const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c) {
#if OPENMP_SCHEDULE_KIND == 0
#pragma omp parallel for num_threads(config.threads)
#elif OPENMP_SCHEDULE_KIND == 1
#pragma omp parallel for schedule(static, 1) num_threads(config.threads)
#else
#pragma omp parallel for schedule(dynamic, 1) num_threads(config.threads)
#endif
    for (int row = 0; row < config.m; ++row) {
        for (int col = 0; col < config.k; ++col) {
            double sum = 0.0;
            for (int p = 0; p < config.n; ++p) {
                sum += a[static_cast<std::size_t>(row) * config.n + p] * b[static_cast<std::size_t>(p) * config.k + col];
            }
            c[static_cast<std::size_t>(row) * config.k + col] = sum;
        }
    }
}

int main(int argc, char** argv) {
    return run_matmul_program("openmp", version_name(), argc, argv, compute_openmp);
}
