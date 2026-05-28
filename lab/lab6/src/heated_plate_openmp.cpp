#include "heated_plate_common.hpp"

#include <cstring>
#include <iostream>
#include <omp.h>
#include <stdexcept>

static HeatedPlateConfig parse_config(int argc, char* argv[]) {
    if (argc < 5) {
        throw std::runtime_error("usage: heated_plate_openmp <m> <n> <epsilon> <threads> [--dump]");
    }

    HeatedPlateConfig config;
    config.m = std::stoi(argv[1]);
    config.n = std::stoi(argv[2]);
    config.epsilon = std::stod(argv[3]);
    config.threads = std::stoi(argv[4]);
    for (int index = 5; index < argc; ++index) {
        if (std::strcmp(argv[index], "--dump") == 0) {
            config.dump = true;
        }
    }
    validate_config(config);
    return config;
}

int main(int argc, char* argv[]) {
    try {
        const HeatedPlateConfig config = parse_config(argc, argv);
        std::vector<double> u(static_cast<std::size_t>(config.m * config.n), 0.0);
        std::vector<double> w(static_cast<std::size_t>(config.m * config.n), 0.0);
        std::vector<double> row_diff(static_cast<std::size_t>(config.m), 0.0);

        const double mean = initialize_boundary_and_mean(w, config.m, config.n);

#pragma omp parallel for num_threads(config.threads)
        for (int i = 1; i < config.m - 1; ++i) {
            for (int j = 1; j < config.n - 1; ++j) {
                at(w, config.n, i, j) = mean;
            }
        }

        int iterations = 0;
        double diff = config.epsilon;

        // omp_get_wtime returns elapsed wall-clock time in seconds for OpenMP regions.
        // Source: https://www.openmp.org/spec-html/5.0/openmpsu160.html
        const double start = omp_get_wtime();
        while (config.epsilon <= diff) {
#pragma omp parallel for num_threads(config.threads)
            for (int i = 0; i < config.m; ++i) {
                for (int j = 0; j < config.n; ++j) {
                    at(u, config.n, i, j) = at_const(w, config.n, i, j);
                }
            }

#pragma omp parallel for num_threads(config.threads)
            for (int i = 1; i < config.m - 1; ++i) {
                for (int j = 1; j < config.n - 1; ++j) {
                    at(w, config.n, i, j) = (
                        at_const(u, config.n, i - 1, j) +
                        at_const(u, config.n, i + 1, j) +
                        at_const(u, config.n, i, j - 1) +
                        at_const(u, config.n, i, j + 1)
                    ) / 4.0;
                }
            }

            std::fill(row_diff.begin(), row_diff.end(), 0.0);
#pragma omp parallel for num_threads(config.threads)
            for (int i = 1; i < config.m - 1; ++i) {
                double local = 0.0;
                for (int j = 1; j < config.n - 1; ++j) {
                    local = std::max(local, std::fabs(at_const(w, config.n, i, j) - at_const(u, config.n, i, j)));
                }
                row_diff[static_cast<std::size_t>(i)] = local;
            }

            diff = 0.0;
            for (double value : row_diff) {
                diff = std::max(diff, value);
            }
            ++iterations;
        }
        const double time_sec = omp_get_wtime() - start;

        print_common_output("openmp", "openmp_original", config, "openmp_default", iterations, time_sec, diff, w);
        if (config.dump) {
            dump_matrix("W", w, config.m, config.n);
        }
    } catch (const std::exception& exc) {
        std::cerr << "error=" << exc.what() << "\n";
        return 1;
    }

    return 0;
}
