#ifndef LAB7_FFT_COMMON_HPP
#define LAB7_FFT_COMMON_HPP

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

const double PI = 3.141592653589793;

// --- Pseudorandom number generator (uniform) ---
inline double ggl(double* seed) {
    double d2 = 0.2147483647e10;
    double t = *seed;
    t = fmod(16807.0 * t, d2);
    *seed = t;
    return (t - 1.0) / (d2 - 1.0);
}

// --- CPU time in seconds ---
inline double cpu_time() {
    return static_cast<double>(clock()) / static_cast<double>(CLOCKS_PER_SEC);
}

// --- Timestamp ---
inline void timestamp() {
    char buf[40];
    time_t now = time(nullptr);
    const struct tm* tm = localtime(&now);
    strftime(buf, sizeof(buf), "%d %B %Y %I:%M:%S %p", tm);
    std::cout << buf << "\n";
}

// --- Sine/cosine table for FFT ---
inline void cffti(int n, double w[]) {
    int n2 = n / 2;
    double aw = 2.0 * PI / static_cast<double>(n);
    for (int i = 0; i < n2; ++i) {
        double arg = aw * static_cast<double>(i);
        w[i * 2 + 0] = cos(arg);
        w[i * 2 + 1] = sin(arg);
    }
}

// --- Copy complex vector ---
inline void ccopy(int n, double x[], double y[]) {
    for (int i = 0; i < n; ++i) {
        y[i * 2 + 0] = x[i * 2 + 0];
        y[i * 2 + 1] = x[i * 2 + 1];
    }
}

// --- One butterfly stage of FFT ---
inline void step(int n, int mj, double a[], double b[],
                 double c[], double d[], double w[], double sgn) {
    int mj2 = 2 * mj;
    int lj = n / mj2;
    double wjw[2];

    for (int j = 0; j < lj; ++j) {
        int jw = j * mj;
        int ja = jw;
        int jb = ja;
        int jc = j * mj2;
        int jd = jc;

        wjw[0] = w[jw * 2 + 0];
        wjw[1] = w[jw * 2 + 1];
        if (sgn < 0.0) {
            wjw[1] = -wjw[1];
        }

        for (int k = 0; k < mj; ++k) {
            c[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0];
            c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1];

            double ambr = a[(ja + k) * 2 + 0] - b[(jb + k) * 2 + 0];
            double ambu = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1];

            d[(jd + k) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu;
            d[(jd + k) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu;
        }
    }
}

// --- Complex FFT (Cooley-Tukey, in-place via workspace toggling) ---
inline void cfft2(int n, double x[], double y[], double w[], double sgn) {
    int m = static_cast<int>(log(static_cast<double>(n)) / log(1.99));
    int mj = 1;
    int tgle = 1;

    step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0],
         &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);

    if (n == 2) return;

    for (int j = 0; j < m - 2; ++j) {
        mj = mj * 2;
        if (tgle) {
            step(n, mj, &y[0 * 2 + 0], &y[(n / 2) * 2 + 0],
                 &x[0 * 2 + 0], &x[mj * 2 + 0], w, sgn);
            tgle = 0;
        } else {
            step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0],
                 &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);
            tgle = 1;
        }
    }

    if (tgle) {
        ccopy(n, y, x);
    }

    mj = n / 2;
    step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0],
         &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);
}

// --- Checksum for verification ---
inline double fft_checksum(int n, double x[]) {
    double total = 0.0;
    for (int i = 0; i < 2 * n; ++i) {
        total += x[i];
    }
    return total;
}

// --- Max absolute error ---
inline double fft_max_abs_error(int n, double expected[], double actual[]) {
    double err = 0.0;
    for (int i = 0; i < 2 * n; ++i) {
        err = std::max(err, std::fabs(expected[i] - actual[i]));
    }
    return err;
}

// --- Print key=value output ---
inline void print_fft_output(const char* backend, int n, int nits,
                             double error, double time_sec, double mflops,
                             double checksum, int num_procs) {
    std::cout << std::setprecision(12);
    std::cout << "experiment=fft\n";
    std::cout << "backend=" << backend << "\n";
    std::cout << "n=" << n << "\n";
    std::cout << "nits=" << nits << "\n";
    std::cout << "error=" << error << "\n";
    std::cout << "time_sec=" << time_sec << "\n";
    std::cout << "mflops=" << mflops << "\n";
    std::cout << "checksum=" << checksum << "\n";
    std::cout << "num_procs=" << num_procs << "\n";
}

// --- Next power of 2 ---
inline int next_pow2(int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

// --- Check if n is power of 2 ---
inline bool is_pow2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

#endif // LAB7_FFT_COMMON_HPP
