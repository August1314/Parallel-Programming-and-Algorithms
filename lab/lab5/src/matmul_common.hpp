#ifndef LAB5_MATMUL_COMMON_HPP
#define LAB5_MATMUL_COMMON_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

inline constexpr std::uint64_t kDefaultSeed = 20250401ULL;
inline constexpr int kMaxDumpElements = 64;

struct MatrixRunConfig {
    int m;
    int n;
    int k;
    std::uint64_t seed;
    int threads;
    bool dump;
};

inline bool parse_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return true;
        }
    }
    return false;
}

inline MatrixRunConfig parse_matrix_args(int argc, char** argv) {
    if (argc < 6) {
        throw std::runtime_error("usage: <m> <n> <k> <seed> <threads> [--dump]");
    }

    MatrixRunConfig config{};
    config.m = std::stoi(argv[1]);
    config.n = std::stoi(argv[2]);
    config.k = std::stoi(argv[3]);
    config.seed = static_cast<std::uint64_t>(std::stoull(argv[4]));
    config.threads = std::stoi(argv[5]);
    config.dump = parse_flag(argc, argv, "--dump");

    if (config.m <= 0 || config.n <= 0 || config.k <= 0) {
        throw std::runtime_error("matrix dimensions must be positive");
    }
    if (config.threads <= 0) {
        throw std::runtime_error("threads must be positive");
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

inline bool should_dump_matrix(const MatrixRunConfig& config) {
    return config.dump &&
           static_cast<long long>(config.m) * config.n <= kMaxDumpElements &&
           static_cast<long long>(config.n) * config.k <= kMaxDumpElements &&
           static_cast<long long>(config.m) * config.k <= kMaxDumpElements;
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

inline void print_kv(const std::string& key, const std::string& value) {
    std::cout << key << '=' << value << '\n';
}

inline void print_kv(const std::string& key, const char* value) {
    std::cout << key << '=' << value << '\n';
}

template <typename T>
inline void print_kv(const std::string& key, T value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(9) << value;
    std::cout << key << '=' << oss.str() << '\n';
}

template <typename Kernel>
inline int run_matmul_program(const std::string& backend, const std::string& version, int argc, char** argv, Kernel kernel) {
    try {
        const MatrixRunConfig config = parse_matrix_args(argc, argv);
        const std::vector<double> a = generate_matrix(config.m, config.n, config.seed, 1);
        const std::vector<double> b = generate_matrix(config.n, config.k, config.seed, 2);
        std::vector<double> c(static_cast<std::size_t>(config.m) * static_cast<std::size_t>(config.k), 0.0);

        const auto start = std::chrono::steady_clock::now();
        kernel(config, a, b, c);
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end - start;

        print_kv("experiment", "matmul");
        print_kv("backend", backend);
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

#endif
