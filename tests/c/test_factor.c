// =============================================================================
//    Copyright (C) 2026  Liam McAllister Group
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.
// =============================================================================
//
// Checks the int128 fast factorization (fpk_factor_fast) against the GMP
// reference (fpk_factor_exact) on random symmetric integer matrices:
// identical positive-definiteness decisions, and U / M0 data equal up to a
// few ulps. Exits nonzero on any disagreement.
//
//     make -C tests/c test_factor && tests/c/test_factor

#include <stdio.h>
#include <stdlib.h>

#define FP_KERNEL_IMPLEMENTATION
#include "fp_kernel.h"

static uint64_t rng_state = 88172645463325252ULL;
static int64_t rnd(int64_t lo, int64_t hi)          // uniform in [lo, hi]
{
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return lo + (int64_t)(rng_state % (uint64_t)(hi - lo + 1));
}

static double rel(double a, double b, double scale)
{
    return fabs(a - b) / (scale > 0 ? scale : 1.0);
}

int main(void)
{
    enum { MAXD = 12 };
    long n_cases = 0, n_fast = 0, n_fail = 0, n_notpd = 0;
    double worst = 0.0;
    for (int trial = 0; trial < 200000; ++trial) {
        int dim = (int)rnd(1, MAXD);
        int bits = (int)rnd(1, 20);
        int kind = (int)rnd(0, 3);          // 0: B^T B + I, 1: B^T B (PSD), 2: random sym, 3: sheared
        int64_t B[MAXD * MAXD], mat[MAXD * MAXD], lin[MAXD];
        for (int k = 0; k < dim * dim; ++k) B[k] = rnd(-(1LL << bits), 1LL << bits) >> (bits > 3 ? 3 : 0);
        if (kind == 3) {                    // unimodular shears: ill-conditioned
            for (int k = 0; k < dim * dim; ++k) B[k] = (k % (dim + 1) == 0);
            for (int t = 0; t < 3 && dim > 1; ++t) {
                int a = (int)rnd(0, dim - 1), b = (int)rnd(0, dim - 1);
                if (a == b) continue;
                int64_t m = rnd(1, 1LL << (bits / 2 + 1));
                for (int c = 0; c < dim; ++c) B[a * dim + c] += m * B[b * dim + c];
            }
        }
        int overflow = 0;
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j) {
                fpk_i128 s = 0;
                if (kind == 2) {
                    s = i <= j ? rnd(-(1LL << bits), 1LL << bits) : mat[j * dim + i];
                } else {
                    for (int k = 0; k < dim; ++k) s += (fpk_i128)B[k * dim + i] * B[k * dim + j];
                    if (kind == 0 && i == j) s += 1;
                }
                if (s > INT64_MAX / 4 || s < -(INT64_MAX / 4)) overflow = 1;
                mat[i * dim + j] = (int64_t)s;
            }
        if (overflow) continue;
        int use_lin = (int)rnd(0, 1);
        for (int i = 0; i < dim; ++i) lin[i] = rnd(-5, 5);

        double U1[MAXD * MAXD], U2[MAXD * MAXD];
        double N1[MAXD], N2[MAXD], V1[MAXD * MAXD] = {0}, V2[MAXD * MAXD] = {0};
        double Va1[MAXD * MAXD] = {0};
        int s1 = fpk_factor_fast(mat, dim, U1, use_lin ? lin : NULL, N1, V1, Va1);
        int s2 = fpk_factor_exact(mat, dim, U2, use_lin ? lin : NULL, N2, V2);
        n_cases++;
        if (s1 == -1) continue;             // fast path declined: GMP is used
        n_fast++;
        if (s1 != s2) {
            printf("STATUS MISMATCH dim=%d kind=%d fast=%d gmp=%d\n", dim, kind, s1, s2);
            n_fail++;
            continue;
        }
        if (s1 == -9) { n_notpd++; continue; }
        for (int i = 0; i < dim; ++i)
            for (int j = i; j < dim; ++j) {
                double e = rel(U1[i * dim + j], U2[i * dim + j], fabs(U2[i * dim + i]) + fabs(U2[i * dim + j]));
                if (e > worst) worst = e;
            }
        if (use_lin)
            for (int i = 0; i < dim; ++i) {
                double e = rel(N1[i], N2[i], fabs(N2[i]));
                if (e > worst) worst = e;
                for (int j = i + 1; j < dim; ++j) {
                    // V may cancel: the kernel's margin uses Va, so compare to it
                    double ev = rel(V1[i * dim + j], V2[i * dim + j], Va1[i * dim + j] + fabs(V2[i * dim + j]));
                    if (ev > worst) worst = ev;
                }
            }
    }
    double ulps = worst / 0x1p-53;
    printf("cases=%ld fast-path=%ld (not PD: %ld) mismatches=%ld worst error=%.1f ulps (relative to margins)\n",
           n_cases, n_fast, n_notpd, n_fail, ulps);
    // the kernel's error bound allows 2*(dim+8) unit roundoffs; require far less
    return (n_fail == 0 && ulps < 64.0) ? 0 : 1;
}
