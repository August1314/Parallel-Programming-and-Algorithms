#include "fft_common.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char* argv[]) {
    // Parse arguments: n [seed] [--dump]
    int n = 1024;
    double seed_val = 331.0;
    bool dump = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dump") == 0) {
            dump = true;
        } else if (i == 1) {
            n = atoi(argv[1]);
        } else if (i == 2) {
            seed_val = atof(argv[2]);
        }
    }

    if (!is_pow2(n) || n < 2) {
        cerr << "Error: N must be a power of 2 and >= 2\n";
        return 1;
    }

    int nits = 10000;
    int ln2 = static_cast<int>(log2(n));
    int reduced = max(1, ln2 / 4);
    for (int i = 0; i < reduced; ++i) nits = max(1, nits / 10);

    double seed = seed_val;

    // Allocate
    vector<double> w_arr(n);
    vector<double> x_arr(2 * n);
    vector<double> y_arr(2 * n);
    vector<double> z_arr(2 * n);

    double* w = w_arr.data();
    double* x = x_arr.data();
    double* y = y_arr.data();
    double* z = z_arr.data();

    // Generate random complex data
    for (int i = 0; i < 2 * n; i += 2) {
        z[i] = ggl(&seed);
        z[i + 1] = ggl(&seed);
        x[i] = z[i];
        x[i + 1] = z[i + 1];
    }

    if (dump) {
        cout << "# Input X (first 16 values)\n";
        for (int i = 0; i < min(16, 2 * n); i += 2) {
            cout << "#   [" << i / 2 << "] = " << x[i] << " + " << x[i + 1] << "i\n";
        }
    }

    // Initialize sine/cosine tables
    cffti(n, w);

    // Forward + backward FFT accuracy check
    double sgn = +1.0;
    cfft2(n, x, y, w, sgn);
    sgn = -1.0;
    cfft2(n, y, x, w, sgn);

    double fnm1 = 1.0 / static_cast<double>(n);
    double error = 0.0;
    for (int i = 0; i < 2 * n; i += 2) {
        error += pow(z[i] - fnm1 * x[i], 2)
               + pow(z[i + 1] - fnm1 * x[i + 1], 2);
    }
    error = sqrt(fnm1 * error);

    // Timing runs — restore x from z each iteration to avoid overflow
    double ctime1 = cpu_time();
    for (int it = 0; it < nits; ++it) {
        for (int i = 0; i < 2 * n; ++i) x[i] = z[i];
        sgn = +1.0;
        cfft2(n, x, y, w, sgn);
        sgn = -1.0;
        cfft2(n, y, x, w, sgn);
    }
    double ctime2 = cpu_time();
    double ctime = ctime2 - ctime1;

    double flops = 2.0 * static_cast<double>(nits)
                 * (5.0 * static_cast<double>(n) * static_cast<double>(ln2));
    double mflops = flops / 1.0E+06 / ctime;

    double cs = fft_checksum(n, x);

    cout << "\n";
    print_fft_output("serial", n, nits, error, ctime, mflops, cs, 1);
    cout << "\n";

    if (dump) {
        cout << "# Result X after FFT/Inverse-FFT (first 16 values)\n";
        for (int i = 0; i < min(16, 2 * n); i += 2) {
            cout << "#   [" << i / 2 << "] = " << x[i] << " + " << x[i + 1] << "i\n";
        }
        cout << "# Expected (N * original)\n";
        for (int i = 0; i < min(16, 2 * n); i += 2) {
            cout << "#   [" << i / 2 << "] = " << n * z[i] << " + " << n * z[i + 1] << "i\n";
        }
    }

    return 0;
}
