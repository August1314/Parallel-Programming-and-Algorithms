#ifndef LAB6_HEATED_PLATE_COMMON_HPP
#define LAB6_HEATED_PLATE_COMMON_HPP

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct HeatedPlateConfig {
    int m = 500;
    int n = 500;
    double epsilon = 0.001;
    int threads = 1;
    int chunk_size = 1;
    bool dump = false;
};

inline void validate_config(const HeatedPlateConfig& config) {
    if (config.m < 3 || config.n < 3) {
        throw std::runtime_error("m and n must be at least 3");
    }
    if (config.epsilon <= 0.0) {
        throw std::runtime_error("epsilon must be positive");
    }
    if (config.threads <= 0) {
        throw std::runtime_error("threads must be positive");
    }
    if (config.chunk_size <= 0) {
        throw std::runtime_error("chunk_size must be positive");
    }
}

inline double& at(std::vector<double>& values, int n, int i, int j) {
    return values[static_cast<std::size_t>(i * n + j)];
}

inline double at_const(const std::vector<double>& values, int n, int i, int j) {
    return values[static_cast<std::size_t>(i * n + j)];
}

inline double initialize_boundary_and_mean(std::vector<double>& w, int m, int n) {
    std::fill(w.begin(), w.end(), 0.0);
    for (int i = 1; i < m - 1; ++i) {
        at(w, n, i, 0) = 100.0;
        at(w, n, i, n - 1) = 100.0;
    }
    for (int j = 0; j < n; ++j) {
        at(w, n, m - 1, j) = 100.0;
        at(w, n, 0, j) = 0.0;
    }

    double mean = 0.0;
    for (int i = 1; i < m - 1; ++i) {
        mean += at_const(w, n, i, 0) + at_const(w, n, i, n - 1);
    }
    for (int j = 0; j < n; ++j) {
        mean += at_const(w, n, m - 1, j) + at_const(w, n, 0, j);
    }
    return mean / static_cast<double>(2 * m + 2 * n - 4);
}

inline double checksum(const std::vector<double>& values) {
    double total = 0.0;
    for (double value : values) {
        total += value;
    }
    return total;
}

inline double max_abs(const std::vector<double>& values) {
    double result = 0.0;
    for (double value : values) {
        result = std::max(result, std::fabs(value));
    }
    return result;
}

inline void dump_matrix(const char* label, const std::vector<double>& values, int m, int n) {
    std::cout << label << ":\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setprecision(3) << at_const(values, n, i, j);
            if (j + 1 < n) {
                std::cout << ' ';
            }
        }
        std::cout << '\n';
    }
}

inline void print_common_output(
    const char* backend,
    const char* version,
    const HeatedPlateConfig& config,
    const char* schedule,
    int iterations,
    double time_sec,
    double diff,
    const std::vector<double>& w
) {
    std::cout << std::setprecision(12);
    std::cout << "experiment=heated_plate\n";
    std::cout << "backend=" << backend << "\n";
    std::cout << "version=" << version << "\n";
    std::cout << "m=" << config.m << "\n";
    std::cout << "n=" << config.n << "\n";
    std::cout << "epsilon=" << config.epsilon << "\n";
    std::cout << "threads=" << config.threads << "\n";
    std::cout << "schedule=" << schedule << "\n";
    std::cout << "chunk_size=" << config.chunk_size << "\n";
    std::cout << "iterations=" << iterations << "\n";
    std::cout << "time_sec=" << time_sec << "\n";
    std::cout << "diff=" << diff << "\n";
    std::cout << "checksum=" << checksum(w) << "\n";
    std::cout << "max_abs=" << max_abs(w) << "\n";
}

#endif
