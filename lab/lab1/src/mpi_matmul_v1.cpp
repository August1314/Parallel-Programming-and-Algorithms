#include "matmul_mpi.hpp"
#include <mpi.h>

void mpi_matmul_v1(const MatrixRunConfig& config, int rank, int size) {
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> c;
    
    std::vector<int> row_counts(size);
    std::vector<int> row_displs(size);
    
    int local_m = config.m / size;
    int remainder = config.m % size;
    
    for (int p = 0; p < size; ++p) {
        row_counts[p] = local_m + (p < remainder ? 1 : 0);
        row_displs[p] = (p == 0) ? 0 : row_displs[p - 1] + row_counts[p - 1];
    }
    
    int my_rows = row_counts[rank];
    int my_row_start = row_displs[rank];
    
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
    double start_time = MPI_Wtime();
    
    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            int p_rows = row_counts[p];
            int p_start = row_displs[p];
            
            MPI_Send(&a[static_cast<std::size_t>(p_start) * config.n],
                     p_rows * config.n, MPI_DOUBLE,
                     p, 0, MPI_COMM_WORLD);
        }
        
        std::copy(a.begin(), a.begin() + static_cast<std::size_t>(my_rows) * config.n,
                  local_a.begin());
    } else {
        MPI_Recv(local_a.data(), my_rows * config.n, MPI_DOUBLE,
                 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            MPI_Send(b.data(), config.n * config.k, MPI_DOUBLE,
                     p, 1, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(b.data(), config.n * config.k, MPI_DOUBLE,
                 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    local_matmul(local_a.data(), b.data(), local_c.data(), my_rows, config.n, config.k);
    
    if (rank == 0) {
        std::copy(local_c.begin(), local_c.end(),
                  c.begin() + static_cast<std::size_t>(my_row_start) * config.k);
        
        for (int p = 1; p < size; ++p) {
            int p_rows = row_counts[p];
            int p_start = row_displs[p];
            
            MPI_Recv(&c[static_cast<std::size_t>(p_start) * config.k],
                     p_rows * config.k, MPI_DOUBLE,
                     p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_c.data(), my_rows * config.k, MPI_DOUBLE,
                 0, 2, MPI_COMM_WORLD);
    }
    
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;
    
    if (rank == 0) {
        double checksum = checksum_matrix(c);
        double max_abs = max_abs_matrix(c);
        
        std::cout << std::fixed << std::setprecision(9);
        std::cout << "version=mpi_v1_p2p\n";
        std::cout << "m=" << config.m << " n=" << config.n << " k=" << config.k << '\n';
        std::cout << "seed=" << config.seed << '\n';
        std::cout << "processes=" << size << '\n';
        std::cout << "time_sec=" << elapsed << '\n';
        std::cout << "checksum=" << checksum << '\n';
        std::cout << "max_abs=" << max_abs << '\n';
        
        bool allow_dump = config.dump_matrix && config.m <= 8 && config.n <= 8 && config.k <= 8;
        if (allow_dump) {
            dump_matrix(a, config.m, config.n, "A");
            dump_matrix(b, config.n, config.k, "B");
            dump_matrix(c, config.m, config.k, "C");
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
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
