#include "pthreads_common.hpp"

#include <chrono>
#include <pthread.h>

#include <exception>
#include <vector>

#ifndef PTHREADS_MATMUL_VERSION
#define PTHREADS_MATMUL_VERSION 1
#endif

struct MatmulThreadContext {
    int thread_id;
    int begin_row;
    int end_row;
    int total_threads;
    int m;
    int n;
    int k;
    const double* a;
    const double* b;
    double* c;
};

void compute_row(const MatmulThreadContext* ctx, int row) {
    for (int col = 0; col < ctx->k; ++col) {
        double sum = 0.0;
        for (int p = 0; p < ctx->n; ++p) {
            sum += ctx->a[static_cast<std::size_t>(row) * ctx->n + p] * ctx->b[static_cast<std::size_t>(p) * ctx->k + col];
        }
        ctx->c[static_cast<std::size_t>(row) * ctx->k + col] = sum;
    }
}

void* matmul_worker(void* arg) {
    auto* ctx = static_cast<MatmulThreadContext*>(arg);
#if PTHREADS_MATMUL_VERSION == 1
    for (int row = ctx->begin_row; row < ctx->end_row; ++row) {
        compute_row(ctx, row);
    }
#else
    for (int row = ctx->thread_id; row < ctx->m; row += ctx->total_threads) {
        compute_row(ctx, row);
    }
#endif
    return nullptr;
}

int main(int argc, char** argv) {
    try {
        const MatrixRunConfig config = parse_matrix_args(argc, argv);
        const std::vector<double> a = generate_matrix(config.m, config.n, config.seed, 1);
        const std::vector<double> b = generate_matrix(config.n, config.k, config.seed, 2);
        std::vector<double> c(static_cast<std::size_t>(config.m) * static_cast<std::size_t>(config.k), 0.0);

        std::vector<pthread_t> threads(static_cast<std::size_t>(config.threads));
        std::vector<MatmulThreadContext> contexts(static_cast<std::size_t>(config.threads));
        const auto ranges = make_block_ranges(config.m, config.threads);

        const auto start = std::chrono::steady_clock::now();
        for (int tid = 0; tid < config.threads; ++tid) {
            contexts[static_cast<std::size_t>(tid)] = MatmulThreadContext{
                tid,
                ranges[static_cast<std::size_t>(tid)].first,
                ranges[static_cast<std::size_t>(tid)].second,
                config.threads,
                config.m,
                config.n,
                config.k,
                a.data(),
                b.data(),
                c.data(),
            };
            pthread_create(&threads[static_cast<std::size_t>(tid)], nullptr, matmul_worker, &contexts[static_cast<std::size_t>(tid)]);
        }
        for (pthread_t& thread : threads) {
            pthread_join(thread, nullptr);
        }
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end - start;

#if PTHREADS_MATMUL_VERSION == 1
        const char* version = "v1_row_block";
#else
        const char* version = "v2_cyclic_rows";
#endif

        print_kv("experiment", "pthreads_matmul");
        print_kv("version", version);
        print_kv("m", config.m);
        print_kv("n", config.n);
        print_kv("k", config.k);
        print_kv("seed", config.seed);
        print_kv("threads", config.threads);
        print_kv("time_sec", elapsed.count());
        print_kv("checksum", checksum_matrix(c));
        print_kv("max_abs", max_abs_matrix(c));

        if (should_dump_matrix(config)) {
            dump_matrix(a, config.m, config.n, "A");
            dump_matrix(b, config.n, config.k, "B");
            dump_matrix(c, config.m, config.k, "C");
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
