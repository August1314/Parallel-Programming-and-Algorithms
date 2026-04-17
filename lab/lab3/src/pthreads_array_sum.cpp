#include "pthreads_common.hpp"

#include <chrono>
#include <pthread.h>

#include <exception>
#include <vector>

#ifndef PTHREADS_ARRAY_SUM_VERSION
#define PTHREADS_ARRAY_SUM_VERSION 1
#endif

struct ArrayThreadContext {
    int begin_index;
    int end_index;
    const double* values;
    double partial_sum;
    double* shared_sum;
    pthread_mutex_t* mutex;
};

void* array_sum_worker(void* arg) {
    auto* ctx = static_cast<ArrayThreadContext*>(arg);
    double local = 0.0;
    for (int index = ctx->begin_index; index < ctx->end_index; ++index) {
        local += ctx->values[index];
    }
#if PTHREADS_ARRAY_SUM_VERSION == 1
    ctx->partial_sum = local;
#else
    pthread_mutex_lock(ctx->mutex);
    *ctx->shared_sum += local;
    pthread_mutex_unlock(ctx->mutex);
    ctx->partial_sum = local;
#endif
    return nullptr;
}

int main(int argc, char** argv) {
    try {
        const ArrayRunConfig config = parse_array_args(argc, argv);
        const std::vector<double> values = generate_array(config.length, config.seed, 3);

        std::vector<pthread_t> threads(static_cast<std::size_t>(config.threads));
        std::vector<ArrayThreadContext> contexts(static_cast<std::size_t>(config.threads));
        const auto ranges = make_block_ranges(static_cast<int>(config.length), config.threads);
        double shared_sum = 0.0;
        pthread_mutex_t mutex;
        pthread_mutex_init(&mutex, nullptr);

        const auto start = std::chrono::steady_clock::now();
        for (int tid = 0; tid < config.threads; ++tid) {
            contexts[static_cast<std::size_t>(tid)] = ArrayThreadContext{
                ranges[static_cast<std::size_t>(tid)].first,
                ranges[static_cast<std::size_t>(tid)].second,
                values.data(),
                0.0,
                &shared_sum,
                &mutex,
            };
            pthread_create(&threads[static_cast<std::size_t>(tid)], nullptr, array_sum_worker, &contexts[static_cast<std::size_t>(tid)]);
        }
        for (pthread_t& thread : threads) {
            pthread_join(thread, nullptr);
        }
#if PTHREADS_ARRAY_SUM_VERSION == 1
        for (const ArrayThreadContext& ctx : contexts) {
            shared_sum += ctx.partial_sum;
        }
#endif
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end - start;
        pthread_mutex_destroy(&mutex);

#if PTHREADS_ARRAY_SUM_VERSION == 1
        const char* version = "v1_local_sum";
#else
        const char* version = "v2_mutex_shared_sum";
#endif

        print_kv("experiment", "pthreads_array_sum");
        print_kv("version", version);
        print_kv("length", static_cast<unsigned long long>(config.length));
        print_kv("seed", config.seed);
        print_kv("threads", config.threads);
        print_kv("time_sec", elapsed.count());
        print_kv("sum", shared_sum);

        if (should_dump_array(config)) {
            dump_array(values, "array");
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
