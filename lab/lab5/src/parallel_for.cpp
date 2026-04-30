#include "parallel_for.hpp"

#include <pthread.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

struct ParallelForWorkerContext {
    int start;
    int inc;
    int iter_begin;
    int iter_end;
    parallel_for_functor functor;
    void* arg;
};

static void* parallel_for_worker(void* raw) {
    auto* ctx = static_cast<ParallelForWorkerContext*>(raw);
    for (int iter = ctx->iter_begin; iter < ctx->iter_end; ++iter) {
        const int idx = ctx->start + iter * ctx->inc;
        ctx->functor(idx, ctx->arg);
    }
    return nullptr;
}

extern "C" void parallel_for(int start, int end, int inc, parallel_for_functor functor, void* arg, int num_threads) {
    if (functor == nullptr) {
        throw std::runtime_error("functor must not be null");
    }
    if (inc <= 0) {
        throw std::runtime_error("inc must be positive");
    }
    if (num_threads <= 0) {
        throw std::runtime_error("num_threads must be positive");
    }
    if (end <= start) {
        return;
    }

    const int total_iters = (end - start + inc - 1) / inc;
    const int worker_count = total_iters < num_threads ? total_iters : num_threads;
    if (worker_count <= 0) {
        return;
    }

    std::vector<pthread_t> threads(static_cast<std::size_t>(worker_count));
    std::vector<ParallelForWorkerContext> contexts(static_cast<std::size_t>(worker_count));

    const int base = total_iters / worker_count;
    const int extra = total_iters % worker_count;
    int iter_begin = 0;

    for (int tid = 0; tid < worker_count; ++tid) {
        const int count = base + (tid < extra ? 1 : 0);
        contexts[static_cast<std::size_t>(tid)] = ParallelForWorkerContext{
            start,
            inc,
            iter_begin,
            iter_begin + count,
            functor,
            arg,
        };
        iter_begin += count;
        pthread_create(&threads[static_cast<std::size_t>(tid)], nullptr, parallel_for_worker, &contexts[static_cast<std::size_t>(tid)]);
    }

    for (pthread_t& thread : threads) {
        pthread_join(thread, nullptr);
    }
}
