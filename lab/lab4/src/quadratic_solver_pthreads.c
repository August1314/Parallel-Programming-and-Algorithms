#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    double a;
    double b;
    double c;
    double delta;
    bool delta_ready;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} SharedState;

typedef struct {
    SharedState *state;
    int sign;
    double real;
    double imag;
} RootTask;

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static bool parse_double(const char *text, double *out) {
    char *end = NULL;
    errno = 0;
    const double value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    *out = value;
    return true;
}

static void print_usage(const char *program) {
    fprintf(stderr, "Usage: %s <a> <b> <c>\n", program);
    fprintf(stderr, "Example: %s 1 -3 2\n", program);
}

static void *discriminant_worker(void *arg) {
    SharedState *state = (SharedState *)arg;
    const double delta = state->b * state->b - 4.0 * state->a * state->c;

    pthread_mutex_lock(&state->mutex);
    state->delta = delta;
    state->delta_ready = true;
    pthread_cond_broadcast(&state->cond);
    pthread_mutex_unlock(&state->mutex);
    return NULL;
}

static void *root_worker(void *arg) {
    RootTask *task = (RootTask *)arg;
    SharedState *state = task->state;

    pthread_mutex_lock(&state->mutex);
    while (!state->delta_ready) {
        pthread_cond_wait(&state->cond, &state->mutex);
    }
    const double a = state->a;
    const double b = state->b;
    const double delta = state->delta;
    pthread_mutex_unlock(&state->mutex);

    const double denom = 2.0 * a;
    if (delta >= 0.0) {
        task->real = (-b + (double)task->sign * sqrt(delta)) / denom;
        task->imag = 0.0;
    } else {
        task->real = -b / denom;
        task->imag = (double)task->sign * sqrt(-delta) / fabs(denom);
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    if (!parse_double(argv[1], &a) || !parse_double(argv[2], &b) || !parse_double(argv[3], &c)) {
        fprintf(stderr, "Error: a, b and c must be valid floating-point numbers.\n");
        return 1;
    }
    if (a == 0.0) {
        fprintf(stderr, "Error: a must be non-zero for a quadratic equation.\n");
        return 1;
    }

    SharedState state = {
        .a = a,
        .b = b,
        .c = c,
        .delta = 0.0,
        .delta_ready = false,
        .mutex = PTHREAD_MUTEX_INITIALIZER,
        .cond = PTHREAD_COND_INITIALIZER,
    };
    RootTask x1 = {.state = &state, .sign = 1, .real = 0.0, .imag = 0.0};
    RootTask x2 = {.state = &state, .sign = -1, .real = 0.0, .imag = 0.0};

    pthread_t delta_thread;
    pthread_t x1_thread;
    pthread_t x2_thread;

    const double start = monotonic_seconds();
    if (pthread_create(&x1_thread, NULL, root_worker, &x1) != 0 ||
        pthread_create(&x2_thread, NULL, root_worker, &x2) != 0 ||
        pthread_create(&delta_thread, NULL, discriminant_worker, &state) != 0) {
        fprintf(stderr, "Error: failed to create pthreads.\n");
        return 1;
    }

    pthread_join(delta_thread, NULL);
    pthread_join(x1_thread, NULL);
    pthread_join(x2_thread, NULL);
    const double elapsed = monotonic_seconds() - start;

    pthread_cond_destroy(&state.cond);
    pthread_mutex_destroy(&state.mutex);

    printf("experiment=quadratic_solver_pthreads\n");
    printf("a=%.17g\n", a);
    printf("b=%.17g\n", b);
    printf("c=%.17g\n", c);
    printf("delta=%.17g\n", state.delta);
    if (state.delta >= 0.0) {
        printf("root_type=real\n");
        printf("x1=%.17g\n", x1.real);
        printf("x2=%.17g\n", x2.real);
    } else {
        printf("root_type=complex\n");
        printf("x1=%.17g%+.17gi\n", x1.real, x1.imag);
        printf("x2=%.17g%+.17gi\n", x2.real, x2.imag);
    }
    printf("time_sec=%.9f\n", elapsed);
    return 0;
}
