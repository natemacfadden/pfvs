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
// Standalone driver for fp_kernel.h (no Python), for profiling and debugging.
// Reads problems from stdin, one after another, each as whitespace-separated
// integers:
//
//     dim Q qmax strict nrows has_linvec linmin
//     mat        (dim*dim)
//     H          (nrows*dim, arbitrary precision)
//     linvec     (dim, only if has_linvec)
//
// and prints, per problem: status, #outputs, nodes, candidates, leaves, time.
// Build with -DFPK_STATS to also print rejected candidates by reason/level.
//
//     make -C tests/c && tests/c/fp_kernel_cli < problems.txt

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define FP_KERNEL_IMPLEMENTATION
#include "fp_kernel.h"

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char **argv)
{
    int reps = argc > 1 ? atoi(argv[1]) : 1;
    int dim, strict, nrows, has_lin;
    long long Q, qmax, linmin;
    long total_out = 0, total_nodes = 0, total_cand = 0;
    double total_t = 0;
    int n_prob = 0;

    while (scanf("%d %lld %lld %d %d %d %lld", &dim, &Q, &qmax, &strict, &nrows,
                 &has_lin, &linmin) == 7) {
        int64_t *mat = malloc((size_t)dim * dim * sizeof(int64_t));
        int64_t *lin = has_lin ? malloc(dim * sizeof(int64_t)) : NULL;
        mpz_t *H = malloc((size_t)(nrows ? nrows : 1) * dim * sizeof(mpz_t));
        for (int k = 0; k < dim * dim; ++k) {
            long long x;
            if (scanf("%lld", &x) != 1) return 2;
            mat[k] = x;
        }
        for (int k = 0; k < nrows * dim; ++k) {
            mpz_init(H[k]);
            if (gmp_scanf("%Zd", H[k]) != 1) return 2;
        }
        for (int k = 0; has_lin && k < dim; ++k) {
            long long x;
            if (scanf("%lld", &x) != 1) return 2;
            lin[k] = x;
        }

        fpk_problem P = {
            .dim = dim, .mat = mat, .qmax = qmax, .Q = Q, .nrows = nrows,
            .H = H, .strict = strict, .linvec = lin, .linmin = linmin,
            .max_N_out = 100000000, .eps = 0.0,
        };
        fpk_output out = {0};
        double best = 1e300;
        int st = 0;
        for (int r = 0; r < reps; ++r) {
            fpk_output_free(&out);
            out = (fpk_output){0};
            double t = now();
            st = fpk_enumerate(&P, &out);
            t = now() - t;
            if (t < best) best = t;
        }
        printf("status=%d n_out=%ld nodes=%ld cand=%ld leaf=%ld time_ms=%.4f\n",
               st, out.n, out.n_nodes, out.n_cand, out.n_leaf, best * 1e3);
        total_out += out.n; total_nodes += out.n_nodes; total_cand += out.n_cand;
        total_t += best;
        n_prob++;

        fpk_output_free(&out);
        for (int k = 0; k < nrows * dim; ++k) mpz_clear(H[k]);
        free(H); free(mat); free(lin);
    }
    printf("TOTAL problems=%d n_out=%ld nodes=%ld cand=%ld time_ms=%.3f\n",
           n_prob, total_out, total_nodes, total_cand, total_t * 1e3);
#ifdef FPK_STATS
    const char *names[FPK_N_REJ] = {"ellipsoid", "M0", "gcd", "leaf"};
    for (int r = 0; r < FPK_N_REJ; ++r) {
        printf("rejected[%s]:", names[r]);
        for (int l = 0; l < FPK_MAX_DIM; ++l)
            if (fpk_stats[r][l]) printf(" L%d=%ld", l, fpk_stats[r][l]);
        printf("\n");
    }
#endif
    return 0;
}
