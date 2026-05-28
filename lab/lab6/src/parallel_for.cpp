#include "parallel_for.hpp"

#include <pthread.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

struct ParallelForShared {
    int start;
    int inc;
    int total_iters;
    int chunk_size;
    ParallelForSchedule schedule;
    parallel_for_functor functor;
    void* arg;
    int next_iter;
    pthread_mutex_t mutex;
};

struct ParallelForWorkerContext {
    ParallelForShared* shared;
    int worker_id;
    int worker_count;
    int iter_begin;
    int iter_end;
};

static void execute_range(ParallelForShared* shared, int iter_begin, int iter_end) {
    for (int iter = iter_begin; iter < iter_end; ++iter) {
        const int idx = shared->start + iter * shared->inc;
        shared->functor(idx, shared->arg);
    }
}

static void* parallel_for_worker(void* raw) {
    auto* ctx = static_cast<ParallelForWorkerContext*>(raw);
    ParallelForShared* shared = ctx->shared;

    if (shared->schedule == PARALLEL_FOR_BLOCK) {
        execute_range(shared, ctx->iter_begin, ctx->iter_end);
        return nullptr;
    }

    if (shared->schedule == PARALLEL_FOR_CYCLIC) {
        for (int chunk_begin = ctx->worker_id * shared->chunk_size;
             chunk_begin < shared->total_iters;
             chunk_begin += ctx->worker_count * shared->chunk_size) {
            const int chunk_end = std::min(chunk_begin + shared->chunk_size, shared->total_iters);
            execute_range(shared, chunk_begin, chunk_end);
        }
        return nullptr;
    }

    while (true) {
        int iter_begin = 0;
        int iter_end = 0;

        // POSIX mutexes serialize access to the shared dynamic-work cursor.
        // Source: https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_mutex_lock.html
        pthread_mutex_lock(&shared->mutex);
        iter_begin = shared->next_iter;
        shared->next_iter += shared->chunk_size;
        pthread_mutex_unlock(&shared->mutex);

        if (iter_begin >= shared->total_iters) {
            break;
        }
        iter_end = std::min(iter_begin + shared->chunk_size, shared->total_iters);
        execute_range(shared, iter_begin, iter_end);
    }

    return nullptr;
}

extern "C" void parallel_for_with_schedule(
    int start,
    int end,
    int inc,
    parallel_for_functor functor,
    void* arg,
    int num_threads,
    ParallelForSchedule schedule,
    int chunk_size
) {
    if (functor == nullptr) {
        throw std::runtime_error("functor must not be null");
    }
    if (inc <= 0) {
        throw std::runtime_error("inc must be positive");
    }
    if (num_threads <= 0) {
        throw std::runtime_error("num_threads must be positive");
    }
    if (chunk_size <= 0) {
        throw std::runtime_error("chunk_size must be positive");
    }
    if (end <= start) {
        return;
    }

    const int total_iters = (end - start + inc - 1) / inc;
    const int worker_count = std::min(total_iters, num_threads);
    if (worker_count <= 0) {
        return;
    }

    ParallelForShared shared{
        start,
        inc,
        total_iters,
        chunk_size,
        schedule,
        functor,
        arg,
        0,
        PTHREAD_MUTEX_INITIALIZER,
    };

    if (pthread_mutex_init(&shared.mutex, nullptr) != 0) {
        throw std::runtime_error("pthread_mutex_init failed");
    }

    std::vector<pthread_t> threads(static_cast<std::size_t>(worker_count));
    std::vector<ParallelForWorkerContext> contexts(static_cast<std::size_t>(worker_count));
    int created_threads = 0;

    const int base = total_iters / worker_count;
    const int extra = total_iters % worker_count;
    int iter_begin = 0;

    for (int tid = 0; tid < worker_count; ++tid) {
        const int count = base + (tid < extra ? 1 : 0);
        contexts[static_cast<std::size_t>(tid)] = ParallelForWorkerContext{
            &shared,
            tid,
            worker_count,
            iter_begin,
            iter_begin + count,
        };
        iter_begin += count;

        // pthread_create starts a worker at the supplied start routine; pthread_join waits for it.
        // Sources:
        // https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_create.html
        // https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_join.html
        const int rc = pthread_create(
            &threads[static_cast<std::size_t>(tid)],
            nullptr,
            parallel_for_worker,
            &contexts[static_cast<std::size_t>(tid)]
        );
        if (rc != 0) {
            for (int join_tid = 0; join_tid < created_threads; ++join_tid) {
                pthread_join(threads[static_cast<std::size_t>(join_tid)], nullptr);
            }
            pthread_mutex_destroy(&shared.mutex);
            throw std::runtime_error("pthread_create failed");
        }
        ++created_threads;
    }

    for (pthread_t& thread : threads) {
        pthread_join(thread, nullptr);
    }

    pthread_mutex_destroy(&shared.mutex);
}

extern "C" void parallel_for(int start, int end, int inc, parallel_for_functor functor, void* arg, int num_threads) {
    parallel_for_with_schedule(start, end, inc, functor, arg, num_threads, PARALLEL_FOR_BLOCK, 1);
}

ParallelForSchedule parse_parallel_for_schedule(const char* raw) {
    if (raw == nullptr || std::strcmp(raw, "block") == 0) {
        return PARALLEL_FOR_BLOCK;
    }
    if (std::strcmp(raw, "cyclic") == 0) {
        return PARALLEL_FOR_CYCLIC;
    }
    if (std::strcmp(raw, "dynamic") == 0) {
        return PARALLEL_FOR_DYNAMIC;
    }
    throw std::runtime_error("schedule must be one of: block, cyclic, dynamic");
}

const char* parallel_for_schedule_name(ParallelForSchedule schedule) {
    switch (schedule) {
        case PARALLEL_FOR_BLOCK:
            return "block";
        case PARALLEL_FOR_CYCLIC:
            return "cyclic";
        case PARALLEL_FOR_DYNAMIC:
            return "dynamic";
    }
    return "unknown";
}
