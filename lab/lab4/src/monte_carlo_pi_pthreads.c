#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    uint64_t samples;
    uint64_t inside;
    unsigned int seed;
} MonteCarloTask;

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int default_thread_count(void) {
    const long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpus < 1) {
        return 4;
    }
    return cpus > 64 ? 64 : (int)cpus;
}

static int parse_u64(const char *text, uint64_t *out) {
    char *end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }
    *out = (uint64_t)value;
    return 1;
}

static int parse_int(const char *text, int *out) {
    char *end = NULL;
    errno = 0;
    const long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 1 || value > 1024) {
        return 0;
    }
    *out = (int)value;
    return 1;
}

static void print_usage(const char *program) {
    fprintf(stderr, "Usage: %s <n> [threads] [seed]\n", program);
    fprintf(stderr, "Example: %s 10000000 8 20250401\n", program);
}

static double unit_random(unsigned int *seed) {
    return (double)rand_r(seed) / (double)RAND_MAX;
}

static void *monte_carlo_worker(void *arg) {
    MonteCarloTask *task = (MonteCarloTask *)arg;
    unsigned int seed = task->seed;
    uint64_t inside = 0;

    for (uint64_t i = 0; i < task->samples; ++i) {
        const double x = unit_random(&seed);
        const double y = unit_random(&seed);
        if (x * x + y * y <= 1.0) {
            ++inside;
        }
    }

    task->inside = inside;
    task->seed = seed;
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        print_usage(argv[0]);
        return 1;
    }

    uint64_t n = 0;
    if (!parse_u64(argv[1], &n) || n == 0) {
        fprintf(stderr, "Error: n must be a positive integer.\n");
        return 1;
    }

    int threads = default_thread_count();
    if (argc >= 3 && !parse_int(argv[2], &threads)) {
        fprintf(stderr, "Error: threads must be an integer in [1, 1024].\n");
        return 1;
    }

    uint64_t seed_value = 20250401;
    if (argc >= 4 && !parse_u64(argv[3], &seed_value)) {
        fprintf(stderr, "Error: seed must be an unsigned integer.\n");
        return 1;
    }

    pthread_t *thread_ids = (pthread_t *)calloc((size_t)threads, sizeof(pthread_t));
    MonteCarloTask *tasks = (MonteCarloTask *)calloc((size_t)threads, sizeof(MonteCarloTask));
    if (thread_ids == NULL || tasks == NULL) {
        fprintf(stderr, "Error: failed to allocate thread state.\n");
        free(thread_ids);
        free(tasks);
        return 1;
    }

    const uint64_t base = n / (uint64_t)threads;
    const uint64_t rem = n % (uint64_t)threads;
    const double start = monotonic_seconds();

    for (int tid = 0; tid < threads; ++tid) {
        tasks[tid].samples = base + ((uint64_t)tid < rem ? 1ULL : 0ULL);
        tasks[tid].inside = 0;
        tasks[tid].seed = (unsigned int)(seed_value + 2654435761ULL * (uint64_t)(tid + 1));
        if (pthread_create(&thread_ids[tid], NULL, monte_carlo_worker, &tasks[tid]) != 0) {
            fprintf(stderr, "Error: failed to create pthread %d.\n", tid);
            for (int joined = 0; joined < tid; ++joined) {
                pthread_join(thread_ids[joined], NULL);
            }
            free(thread_ids);
            free(tasks);
            return 1;
        }
    }

    uint64_t m = 0;
    for (int tid = 0; tid < threads; ++tid) {
        pthread_join(thread_ids[tid], NULL);
        m += tasks[tid].inside;
    }
    const double elapsed = monotonic_seconds() - start;
    const double pi_estimate = 4.0 * (double)m / (double)n;

    printf("experiment=monte_carlo_pi_pthreads\n");
    printf("n=%" PRIu64 "\n", n);
    printf("m=%" PRIu64 "\n", m);
    printf("pi=%.12f\n", pi_estimate);
    printf("threads=%d\n", threads);
    printf("seed=%" PRIu64 "\n", seed_value);
    printf("time_sec=%.9f\n", elapsed);

    free(thread_ids);
    free(tasks);
    return 0;
}
