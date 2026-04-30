#include "matmul_common.hpp"
#include "parallel_for.hpp"

#include <vector>

struct MatmulFunctorArgs {
    int n;
    int k;
    const double* a;
    const double* b;
    double* c;
};

static void* compute_row(int row, void* raw_args) {
    auto* args = static_cast<MatmulFunctorArgs*>(raw_args);
    for (int col = 0; col < args->k; ++col) {
        double sum = 0.0;
        for (int p = 0; p < args->n; ++p) {
            sum += args->a[static_cast<std::size_t>(row) * args->n + p] * args->b[static_cast<std::size_t>(p) * args->k + col];
        }
        args->c[static_cast<std::size_t>(row) * args->k + col] = sum;
    }
    return nullptr;
}

static void compute_parallel_for(const MatrixRunConfig& config, const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c) {
    MatmulFunctorArgs args{
        config.n,
        config.k,
        a.data(),
        b.data(),
        c.data(),
    };
    parallel_for(0, config.m, 1, compute_row, &args, config.threads);
}

int main(int argc, char** argv) {
    return run_matmul_program("pthreads_parallel_for", "parallel_for_row_block", argc, argv, compute_parallel_for);
}
