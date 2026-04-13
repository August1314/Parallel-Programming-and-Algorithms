#include "matmul_mpi.hpp"
#include <mpi.h>

void mpi_matmul_v2_collective(MatrixRunConfig config, int rank, int size) {
    MPI_Datatype config_type = build_config_type();
    MPI_Bcast(&config, 1, config_type, 0, MPI_COMM_WORLD);

    std::vector<int> row_counts;
    std::vector<int> row_displs;
    compute_row_partition(config.m, size, row_counts, row_displs);

    std::vector<int> a_counts(size);
    std::vector<int> a_displs(size);
    std::vector<int> c_counts(size);
    std::vector<int> c_displs(size);
    for (int p = 0; p < size; ++p) {
        a_counts[p] = row_counts[p] * config.n;
        a_displs[p] = row_displs[p] * config.n;
        c_counts[p] = row_counts[p] * config.k;
        c_displs[p] = row_displs[p] * config.k;
    }

    std::vector<double> a;
    std::vector<double> b(static_cast<std::size_t>(config.n) * config.k);
    std::vector<double> c;
    std::vector<double> local_a(static_cast<std::size_t>(row_counts[rank]) * config.n);
    std::vector<double> local_c(static_cast<std::size_t>(row_counts[rank]) * config.k, 0.0);

    if (rank == 0) {
        a = generate_matrix(config.m, config.n, config.seed, 1);
        b = generate_matrix(config.n, config.k, config.seed, 2);
        c.resize(static_cast<std::size_t>(config.m) * config.k, 0.0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();

    MPI_Scatterv(rank == 0 ? a.data() : nullptr, a_counts.data(), a_displs.data(), MPI_DOUBLE, local_a.data(), a_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b.data(), config.n * config.k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    local_matmul(local_a.data(), b.data(), local_c.data(), row_counts[rank], config.n, config.k);

    MPI_Gatherv(local_c.data(), c_counts[rank], MPI_DOUBLE, rank == 0 ? c.data() : nullptr, c_counts.data(), c_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const double elapsed = MPI_Wtime() - start_time;

    if (rank == 0) {
        print_result("mpi_v2_collective", config, size, elapsed, c);
        const bool allow_dump = config.dump_matrix && config.m <= 8 && config.n <= 8 && config.k <= 8;
        if (allow_dump) {
            dump_matrix(a, config.m, config.n, "A");
            dump_matrix(b, config.n, config.k, "B");
            dump_matrix(c, config.m, config.k, "C");
        }
    }

    MPI_Type_free(&config_type);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        MatrixRunConfig config{};
        if (rank == 0) {
            config = parse_args(argc, argv);
        }
        mpi_matmul_v2_collective(config, rank, size);
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
