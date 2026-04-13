#ifndef LAB2_MATMUL_MPI_HPP
#define LAB2_MATMUL_MPI_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

struct MatrixRunConfig {
    int m;
    int n;
    int k;
    std::uint64_t seed;
    int dump_matrix;
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
        throw std::runtime_error("usage: mpirun -np <procs> ./mpi_matmul <m> <n> <k> [seed] [--dump]");
    }

    MatrixRunConfig config{};
    config.m = std::stoi(argv[1]);
    config.n = std::stoi(argv[2]);
    config.k = std::stoi(argv[3]);
    config.seed = argc >= 5 && std::string(argv[4]).rfind("--", 0) != 0
        ? static_cast<std::uint64_t>(std::stoull(argv[4]))
        : 20250401ULL;
    config.dump_matrix = parse_flag(argc, argv, "--dump") ? 1 : 0;

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

inline void local_matmul(const double* a, const double* b, double* c, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            double sum = 0.0;
            for (int p = 0; p < n; ++p) {
                sum += a[static_cast<std::size_t>(i) * n + p] * b[static_cast<std::size_t>(p) * k + j];
            }
            c[static_cast<std::size_t>(i) * k + j] = sum;
        }
    }
}

inline void compute_row_partition(int m, int size, std::vector<int>& row_counts, std::vector<int>& row_displs) {
    row_counts.assign(size, 0);
    row_displs.assign(size, 0);
    const int base = m / size;
    const int remainder = m % size;
    for (int p = 0; p < size; ++p) {
        row_counts[p] = base + (p < remainder ? 1 : 0);
        row_displs[p] = (p == 0) ? 0 : row_displs[p - 1] + row_counts[p - 1];
    }
}

inline MPI_Datatype build_config_type() {
    MatrixRunConfig sample{};
    int block_lengths[5] = {1, 1, 1, 1, 1};
    MPI_Aint displacements[5];
    MPI_Datatype types[5] = {MPI_INT, MPI_INT, MPI_INT, MPI_UINT64_T, MPI_INT};
    MPI_Aint base_address;
    MPI_Get_address(&sample, &base_address);
    MPI_Get_address(&sample.m, &displacements[0]);
    MPI_Get_address(&sample.n, &displacements[1]);
    MPI_Get_address(&sample.k, &displacements[2]);
    MPI_Get_address(&sample.seed, &displacements[3]);
    MPI_Get_address(&sample.dump_matrix, &displacements[4]);
    for (MPI_Aint& displacement : displacements) {
        displacement -= base_address;
    }
    MPI_Datatype config_type;
    MPI_Type_create_struct(5, block_lengths, displacements, types, &config_type);
    MPI_Type_commit(&config_type);
    return config_type;
}

inline void print_result(const std::string& version, const MatrixRunConfig& config, int processes, double elapsed, const std::vector<double>& c) {
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "version=" << version << '\n';
    std::cout << "m=" << config.m << " n=" << config.n << " k=" << config.k << '\n';
    std::cout << "seed=" << config.seed << '\n';
    std::cout << "processes=" << processes << '\n';
    std::cout << "time_sec=" << elapsed << '\n';
    std::cout << "checksum=" << checksum_matrix(c) << '\n';
    std::cout << "max_abs=" << max_abs_matrix(c) << '\n';
}

#endif
