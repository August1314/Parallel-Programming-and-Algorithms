#include "matmul_mpi.hpp"
#include <mpi.h>

void mpi_matmul_v1(const MatrixRunConfig& config, int rank, int size) {
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> c;

    std::vector<int> row_counts;
    std::vector<int> row_displs;
    compute_row_partition(config.m, size, row_counts, row_displs);

    const int my_rows = row_counts[rank];
    const int my_row_start = row_displs[rank];

    std::vector<double> local_a(static_cast<std::size_t>(my_rows) * config.n);
    std::vector<double> local_c(static_cast<std::size_t>(my_rows) * config.k, 0.0);

    if (rank == 0) {
        a = generate_matrix(config.m, config.n, config.seed, 1);
        b = generate_matrix(config.n, config.k, config.seed, 2);
        c.resize(static_cast<std::size_t>(config.m) * config.k, 0.0);
    } else {
        b.resize(static_cast<std::size_t>(config.n) * config.k);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();

    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            MPI_Send(&a[static_cast<std::size_t>(row_displs[p]) * config.n], row_counts[p] * config.n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }
        std::copy(a.begin(), a.begin() + static_cast<std::size_t>(my_rows) * config.n, local_a.begin());
        for (int p = 1; p < size; ++p) {
            MPI_Send(b.data(), config.n * config.k, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(local_a.data(), my_rows * config.n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b.data(), config.n * config.k, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    local_matmul(local_a.data(), b.data(), local_c.data(), my_rows, config.n, config.k);

    if (rank == 0) {
        std::copy(local_c.begin(), local_c.end(), c.begin() + static_cast<std::size_t>(my_row_start) * config.k);
        for (int p = 1; p < size; ++p) {
            MPI_Recv(&c[static_cast<std::size_t>(row_displs[p]) * config.k], row_counts[p] * config.k, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_c.data(), my_rows * config.k, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    const double elapsed = MPI_Wtime() - start_time;

    if (rank == 0) {
        print_result("mpi_v1_p2p", config, size, elapsed, c);
        const bool allow_dump = config.dump_matrix && config.m <= 8 && config.n <= 8 && config.k <= 8;
        if (allow_dump) {
            dump_matrix(a, config.m, config.n, "A");
            dump_matrix(b, config.n, config.k, "B");
            dump_matrix(c, config.m, config.k, "C");
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        MatrixRunConfig config = parse_args(argc, argv);
        mpi_matmul_v1(config, rank, size);
    } catch (const std::exception& ex) {
        if (rank == 0) {
            std::cerr << "Error: " << ex.what() << '\n';
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
