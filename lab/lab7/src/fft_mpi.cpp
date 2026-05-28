#include "fft_common.hpp"
#include <cstring>
#include <mpi.h>
#include <vector>

using namespace std;

// Parallel version of step(): each MPI process computes a subset of
// the j-loop groups. Results are accumulated in a local temp array
// and combined via Allreduce into the global output array.
// a, b: input halves (from serial cfft2 convention)
// out: output array — c values at out[group*mj2 + k],
//      d values at out[group*mj2 + mj + k] (interleaved)
void step_parallel(int n, int mj, const double* a, const double* b,
                   double* out, const double* w, double sgn,
                   int rank, int P) {
    int mj2 = 2 * mj;
    int lj = n / mj2;

    // Local temp: each process writes only its groups
    vector<double> local_out(2 * n, 0.0);

    int groups_per = lj / P;
    int rem = lj % P;
    int start = rank * groups_per + min(rank, rem);
    int end = start + groups_per + (rank < rem ? 1 : 0);

    for (int j = start; j < end; ++j) {
        int jw = j * mj;
        int jg = j * mj2;

        double w_re = w[jw * 2 + 0];
        double w_im = w[jw * 2 + 1];
        if (sgn < 0.0) w_im = -w_im;

        for (int k = 0; k < mj; ++k) {
            int ia = (jw + k) * 2;
            int ic = (jg + k) * 2;       // c position
            int id = (jg + mj + k) * 2;  // d position

            double ar = a[ia + 0], ai = a[ia + 1];
            double br = b[ia + 0], bi = b[ia + 1];

            local_out[ic + 0] = ar + br;
            local_out[ic + 1] = ai + bi;

            double ambr = ar - br;
            double ambu = ai - bi;
            local_out[id + 0] = w_re * ambr - w_im * ambu;
            local_out[id + 1] = w_im * ambr + w_re * ambu;
        }
    }

    // Allreduce: sum local contributions into global out
    MPI_Allreduce(local_out.data(), out, 2 * n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

// Parallel cfft2: same algorithm as serial but step is parallelized.
// x, y, w are replicated on all processes.
void cfft2_parallel(int n, double* x, double* y, const double* w,
                    double sgn, int rank, int P) {
    int m = static_cast<int>(log(static_cast<double>(n)) / log(1.99));
    int mj = 1;
    int tgle = 1;

    // Stage 1: mj=1, from x to y
    step_parallel(n, mj, x, &x[n], y, w, sgn, rank, P);

    if (n == 2) return;

    for (int stage = 0; stage < m - 2; ++stage) {
        mj *= 2;
        if (tgle) {
            step_parallel(n, mj, y, &y[n], x, w, sgn, rank, P);
            tgle = 0;
        } else {
            step_parallel(n, mj, x, &x[n], y, w, sgn, rank, P);
            tgle = 1;
        }
    }

    // If result ended up in y but caller expects it in y (works either way
    // since we always end by writing to y for the final step)
    if (tgle) {
        for (int i = 0; i < n; ++i) {
            x[i * 2 + 0] = y[i * 2 + 0];
            x[i * 2 + 1] = y[i * 2 + 1];
        }
    }

    // Final stage: mj = n/2
    mj = n / 2;
    step_parallel(n, mj, x, &x[n], y, w, sgn, rank, P);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int N = 1024;
    double seed_val = 331.0;
    bool dump = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dump") == 0) dump = true;
        else if (i == 1) N = atoi(argv[1]);
        else if (i == 2) seed_val = atof(argv[2]);
    }

    if (!is_pow2(N) || N < 2) {
        if (rank == 0) cerr << "Error: N must be a power of 2\n";
        MPI_Finalize(); return 1;
    }

    int nits = 10000;
    int ln2 = static_cast<int>(log2(N));
    int reduced = max(1, ln2 / 4);
    for (int i = 0; i < reduced; ++i) nits = max(1, nits / 10);

    // Replicated data on all processes
    vector<double> w_arr(N);
    vector<double> x_arr(2 * N);
    vector<double> y_arr(2 * N);
    vector<double> z_arr(2 * N);

    double* w = w_arr.data();
    double* x = x_arr.data();
    double* y = y_arr.data();
    double* z = z_arr.data();

    if (rank == 0) {
        double seed = seed_val;
        for (int i = 0; i < 2 * N; i += 2) {
            z[i] = ggl(&seed);
            z[i + 1] = ggl(&seed);
            x[i] = z[i];
            x[i + 1] = z[i + 1];
        }
    }
    MPI_Bcast(x, 2 * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(z, 2 * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    cffti(N, w);

    // --- Accuracy check ---
    cfft2_parallel(N, x, y, w, +1.0, rank, P);
    cfft2_parallel(N, y, x, w, -1.0, rank, P);

    double error = 0.0;
    {
        double fnm1 = 1.0 / static_cast<double>(N);
        for (int i = 0; i < 2 * N; i += 2) {
            error += pow(z[i] - fnm1 * x[i], 2)
                   + pow(z[i + 1] - fnm1 * x[i + 1], 2);
        }
        error = sqrt(fnm1 * error);
    }

    // --- Timing ---
    double t1 = MPI_Wtime();
    for (int it = 0; it < nits; ++it) {
        for (int i = 0; i < 2 * N; ++i) x[i] = z[i];
        cfft2_parallel(N, x, y, w, +1.0, rank, P);
        cfft2_parallel(N, y, x, w, -1.0, rank, P);
    }
    double t2 = MPI_Wtime();
    double ctime = t2 - t1;

    double flops = 2.0 * static_cast<double>(nits)
                 * (5.0 * static_cast<double>(N) * static_cast<double>(ln2));
    double mflops = flops / 1.0E+06 / ctime;
    double cs = fft_checksum(N, x);

    if (rank == 0) {
        cout << "\n";
        print_fft_output("mpi", N, nits, error, ctime, mflops, cs, P);
        cout << "\n";
    }

    if (dump && rank == 0) {
        cout << "# Inverse vs N*original (first 8)\n";
        for (int i = 0; i < min(8, 2 * N); i += 2) {
            cout << "#   [" << i/2 << "] got=" << x[i] << " expect=" << N*z[i]
                 << " | imag got=" << x[i+1] << " expect=" << N*z[i+1] << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
