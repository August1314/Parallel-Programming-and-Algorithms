#ifndef LAB1_MATMUL_COMMON_HPP
#define LAB1_MATMUL_COMMON_HPP

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct MatrixRunConfig {
    int m;
    int n;
    int k;
    std::uint64_t seed;
    bool dump_matrix;
};

struct MatrixRunResult {
    double seconds;
    double checksum;
    double max_abs_value;
};

inline bool parse_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return true;
        }
    }
    return false;
}

inline MatrixRunConfig parse_args(int argc, char** argv) {
    if (argc < 4) {
        throw std::runtime_error("usage: <m> <n> <k> [seed] [--dump]");
    }

    MatrixRunConfig config{};
    config.m = std::stoi(argv[1]);
    config.n = std::stoi(argv[2]);
    config.k = std::stoi(argv[3]);
    config.seed = argc >= 5 && std::string(argv[4]).rfind("--", 0) != 0
        ? static_cast<std::uint64_t>(std::stoull(argv[4]))
        : 20250401ULL;
    config.dump_matrix = parse_flag(argc, argv, "--dump");

    if (config.m <= 0 || config.n <= 0 || config.k <= 0) {
        throw std::runtime_error("matrix dimensions must be positive");
    }
    return config;
}

inline std::uint64_t splitmix64_next(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

inline double next_unit_value(std::uint64_t& state) {
    const std::uint64_t raw = splitmix64_next(state);
    const double unit = static_cast<double>((raw >> 11U) & ((1ULL << 53U) - 1ULL)) / static_cast<double>(1ULL << 53U);
    return unit * 2.0 - 1.0;
}

inline std::vector<double> generate_matrix(int rows, int cols, std::uint64_t seed, std::uint64_t salt) {
    std::uint64_t state = seed ^ (salt * 0x9E3779B97F4A7C15ULL);
    std::vector<double> matrix(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    for (double& value : matrix) {
        value = next_unit_value(state);
    }
    return matrix;
}

inline double checksum_matrix(const std::vector<double>& matrix) {
    double checksum = 0.0;
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        checksum += matrix[i] * (static_cast<double>(i % 17U) + 1.0);
    }
    return checksum;
}

inline double max_abs_matrix(const std::vector<double>& matrix) {
    double max_abs = 0.0;
    for (double value : matrix) {
        max_abs = std::max(max_abs, std::abs(value));
    }
    return max_abs;
}

inline void dump_matrix(const std::vector<double>& matrix, int rows, int cols, const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (j > 0) {
                std::cout << ' ';
            }
            std::cout << std::fixed << std::setprecision(6) << matrix[static_cast<std::size_t>(i) * cols + j];
        }
        std::cout << '\n';
    }
}

template <typename Kernel>
int run_program(const std::string& version, int argc, char** argv, Kernel kernel) {
    try {
        const MatrixRunConfig config = parse_args(argc, argv);
        std::vector<double> a = generate_matrix(config.m, config.n, config.seed, 1);
        std::vector<double> b = generate_matrix(config.n, config.k, config.seed, 2);
        std::vector<double> c(static_cast<std::size_t>(config.m) * static_cast<std::size_t>(config.k), 0.0);

        const auto start = std::chrono::steady_clock::now();
        kernel(a.data(), b.data(), c.data(), config.m, config.n, config.k);
        const auto end = std::chrono::steady_clock::now();

        const std::chrono::duration<double> elapsed = end - start;
        const MatrixRunResult result{elapsed.count(), checksum_matrix(c), max_abs_matrix(c)};

        std::cout << std::fixed << std::setprecision(9);
        std::cout << "version=" << version << '\n';
        std::cout << "m=" << config.m << " n=" << config.n << " k=" << config.k << '\n';
        std::cout << "seed=" << config.seed << '\n';
        std::cout << "time_sec=" << result.seconds << '\n';
        std::cout << "checksum=" << result.checksum << '\n';
        std::cout << "max_abs=" << result.max_abs_value << '\n';

        const bool allow_dump = config.dump_matrix && config.m <= 8 && config.n <= 8 && config.k <= 8;
        if (allow_dump) {
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

#endif
