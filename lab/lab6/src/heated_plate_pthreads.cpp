#include "heated_plate_common.hpp"
#include "parallel_for.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>

struct FillInteriorArgs {
    std::vector<double>* w;
    int n;
    double mean;
};

struct CopyArgs {
    const std::vector<double>* w;
    std::vector<double>* u;
    int n;
};

struct UpdateArgs {
    const std::vector<double>* u;
    std::vector<double>* w;
    int n;
};

struct DiffArgs {
    const std::vector<double>* u;
    const std::vector<double>* w;
    std::vector<double>* row_diff;
    int n;
};

static void* fill_interior_row(int i, void* raw) {
    auto* args = static_cast<FillInteriorArgs*>(raw);
    for (int j = 1; j < args->n - 1; ++j) {
        at(*args->w, args->n, i, j) = args->mean;
    }
    return nullptr;
}

static void* copy_row(int i, void* raw) {
    auto* args = static_cast<CopyArgs*>(raw);
    for (int j = 0; j < args->n; ++j) {
        at(*args->u, args->n, i, j) = at_const(*args->w, args->n, i, j);
    }
    return nullptr;
}

static void* update_row(int i, void* raw) {
    auto* args = static_cast<UpdateArgs*>(raw);
    for (int j = 1; j < args->n - 1; ++j) {
        at(*args->w, args->n, i, j) = (
            at_const(*args->u, args->n, i - 1, j) +
            at_const(*args->u, args->n, i + 1, j) +
            at_const(*args->u, args->n, i, j - 1) +
            at_const(*args->u, args->n, i, j + 1)
        ) / 4.0;
    }
    return nullptr;
}

static void* diff_row(int i, void* raw) {
    auto* args = static_cast<DiffArgs*>(raw);
    double local = 0.0;
    for (int j = 1; j < args->n - 1; ++j) {
        local = std::max(local, std::fabs(at_const(*args->w, args->n, i, j) - at_const(*args->u, args->n, i, j)));
    }
    (*args->row_diff)[static_cast<std::size_t>(i)] = local;
    return nullptr;
}

static HeatedPlateConfig parse_config(int argc, char* argv[], ParallelForSchedule& schedule) {
    if (argc < 6) {
        throw std::runtime_error("usage: heated_plate_pthreads <m> <n> <epsilon> <threads> <block|cyclic|dynamic> [chunk_size] [--dump]");
    }

    HeatedPlateConfig config;
    config.m = std::stoi(argv[1]);
    config.n = std::stoi(argv[2]);
    config.epsilon = std::stod(argv[3]);
    config.threads = std::stoi(argv[4]);
    schedule = parse_parallel_for_schedule(argv[5]);

    for (int index = 6; index < argc; ++index) {
        if (std::strcmp(argv[index], "--dump") == 0) {
            config.dump = true;
        } else {
            config.chunk_size = std::stoi(argv[index]);
        }
    }

    validate_config(config);
    return config;
}

int main(int argc, char* argv[]) {
    try {
        ParallelForSchedule schedule = PARALLEL_FOR_BLOCK;
        const HeatedPlateConfig config = parse_config(argc, argv, schedule);

        std::vector<double> u(static_cast<std::size_t>(config.m * config.n), 0.0);
        std::vector<double> w(static_cast<std::size_t>(config.m * config.n), 0.0);
        std::vector<double> row_diff(static_cast<std::size_t>(config.m), 0.0);

        const double mean = initialize_boundary_and_mean(w, config.m, config.n);
        FillInteriorArgs fill_args{&w, config.n, mean};
        parallel_for_with_schedule(1, config.m - 1, 1, fill_interior_row, &fill_args, config.threads, schedule, config.chunk_size);

        int iterations = 0;
        double diff = config.epsilon;

        const auto start = std::chrono::steady_clock::now();
        while (config.epsilon <= diff) {
            CopyArgs copy_args{&w, &u, config.n};
            parallel_for_with_schedule(0, config.m, 1, copy_row, &copy_args, config.threads, schedule, config.chunk_size);

            UpdateArgs update_args{&u, &w, config.n};
            parallel_for_with_schedule(1, config.m - 1, 1, update_row, &update_args, config.threads, schedule, config.chunk_size);

            std::fill(row_diff.begin(), row_diff.end(), 0.0);
            DiffArgs diff_args{&u, &w, &row_diff, config.n};
            parallel_for_with_schedule(1, config.m - 1, 1, diff_row, &diff_args, config.threads, schedule, config.chunk_size);

            diff = 0.0;
            for (double value : row_diff) {
                diff = std::max(diff, value);
            }
            ++iterations;
        }
        const auto end = std::chrono::steady_clock::now();
        const double time_sec = std::chrono::duration<double>(end - start).count();

        const char* schedule_name = parallel_for_schedule_name(schedule);
        std::string version = std::string("pthreads_parallel_for_") + schedule_name;
        print_common_output("pthreads_parallel_for", version.c_str(), config, schedule_name, iterations, time_sec, diff, w);
        if (config.dump) {
            dump_matrix("W", w, config.m, config.n);
        }
    } catch (const std::exception& exc) {
        std::cerr << "error=" << exc.what() << "\n";
        return 1;
    }

    return 0;
}
