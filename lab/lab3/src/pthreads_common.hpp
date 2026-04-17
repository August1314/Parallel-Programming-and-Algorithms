#ifndef LAB3_PTHREADS_COMMON_HPP
#define LAB3_PTHREADS_COMMON_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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

struct ArrayRunConfig {
    std::size_t length;
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
        throw std::runtime_error("usage: <m> <n> <k> [seed] <threads> [--dump]");
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

inline ArrayRunConfig parse_array_args(int argc, char** argv) {
    if (argc < 4) {
        throw std::runtime_error("usage: <length> [seed] <threads> [--dump]");
    }
    ArrayRunConfig config{};
    config.length = static_cast<std::size_t>(std::stoull(argv[1]));
    config.seed = static_cast<std::uint64_t>(std::stoull(argv[2]));
    config.threads = std::stoi(argv[3]);
    config.dump = parse_flag(argc, argv, "--dump");
    if (config.length == 0) {
        throw std::runtime_error("length must be positive");
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

inline std::vector<double> generate_array(std::size_t length, std::uint64_t seed, std::uint64_t salt) {
    std::uint64_t state = seed ^ (salt * 0x9E3779B97F4A7C15ULL);
    std::vector<double> values(length);
    for (double& value : values) {
        value = next_unit_value(state);
    }
    return values;
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

inline double sum_array(const std::vector<double>& values) {
    double total = 0.0;
    for (double value : values) {
        total += value;
    }
    return total;
}

inline bool should_dump_matrix(const MatrixRunConfig& config) {
    return config.dump && static_cast<long long>(config.m) * config.n <= kMaxDumpElements && static_cast<long long>(config.n) * config.k <= kMaxDumpElements && static_cast<long long>(config.m) * config.k <= kMaxDumpElements;
}

inline bool should_dump_array(const ArrayRunConfig& config) {
    return config.dump && config.length <= static_cast<std::size_t>(kMaxDumpElements);
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

inline void dump_array(const std::vector<double>& values, const std::string& name) {
    std::cout << name << ":\n";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            std::cout << ' ';
        }
        std::cout << std::fixed << std::setprecision(6) << values[i];
    }
    std::cout << '\n';
}

inline std::vector<std::pair<int, int>> make_block_ranges(int total, int workers) {
    std::vector<std::pair<int, int>> ranges;
    ranges.reserve(static_cast<std::size_t>(workers));
    const int base = total / workers;
    const int extra = total % workers;
    int begin = 0;
    for (int worker = 0; worker < workers; ++worker) {
        const int count = base + (worker < extra ? 1 : 0);
        ranges.emplace_back(begin, begin + count);
        begin += count;
    }
    return ranges;
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

#endif
