#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>

#define CONIPFV_KERNEL_IMPLEMENTATION
#include "coni_kernel.h"

int main(int argc, char *argv[])
{
    // Manwe:
    // ------
    int dim = 7;
    double U[] = {19.131126469708992, -2.50900019274872, -12.022292590254283, 9.408750722807701,  10.97687584327565,  8.363333975829066,   24.77637690339361,
                  0.0,                20.8255832579255,  30.243381795036953,  31.864968637769284, 12.462603346669052, 6.193517125542706,   6.826408301055712,
                  0.0,                0.0,               31.73014873072524,   -6.636894774358689, 15.791162726449791, 12.771246313546088,  -22.268563128240686,
                  0.0,                0.0,               0.0,                 10.772688211590271, -2.932867573104269, -3.2752582214981767, -13.592907165071438,
                  0.0,                0.0,               0.0,                 0.0,                27.133543485699047, 2.9384531625551986,  1.5013990104854835,
                  0.0,                0.0,               0.0,                 0.0,                0.0,                15.786970406543043,  -15.837576931579958,
                  0.0,                0.0,               0.0,                 0.0,                0.0,                0.0,                 36.49372858726784};
    int Q = 162;
    int linvec[] = {0,0,0,4,-4,-2,-2};
    double linmin = 13;
    int max_N_out = 100000000;
    double eps = 1e-4;

    // initialize H as mpz_t array
    int H_flat[] = {2, 0, 0, 0, 62,  58974,  -5086,
                    0, 2, 0, 0, 84,  224666, -19686,
                    0, 0, 2, 0, 12,  234014, -20736,
                    0, 0, 0, 4, 52,  78376,  -6916,
                    0, 0, 0, 0, 120, 161172, -14052,
                    0, 0, 0, 0, 0,   262692, -23292,
                    0, 0, 0, 0, 0,   0,      0};
    mpz_t H[49];
    for (int i = 0; i < 49; i++) {
        mpz_init_set_si(H[i], H_flat[i]);
    }

    // read dilation
    int dilation = 1;

    if (argc > 1) {
        char *end;
        long val = strtol(argv[1], &end, 10);
        if (*end != '\0' || val <= 0) {
            fprintf(stderr, "Invalid dilation\n");
            return 1;
        }
        dilation = (int)val;
    }

    // initialize output array
    int32_t *out = malloc((size_t)max_N_out * dim * sizeof(int32_t));
    float *Qs    = malloc((size_t)max_N_out * sizeof(float));
    if (!out || !Qs) {
        perror("malloc out and Qs");
        return 1;
    }

    // do the enumeration
    int N_out = 0;

    clock_t start = clock();
    int rc = _coni_kernel_c(
        out,
        Qs,
        &N_out,
        dim,
        U,
        Q,
        1.0*dilation,
        linvec,
        linmin,
        H,
        max_N_out,
        eps);
    clock_t end = clock();
    double eval_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    if (rc != 0) {
        fprintf(stderr, "_coni_kernel_c failed (%d)\n", rc);
    } else {
        printf("Generated %d vectors in %fs\n", N_out, eval_time);
        for (int i=0; i<N_out; ++i) {
            for (int j=0; j<dim; ++j) {
                printf("%d,", out[i*dim + j]);
            }
            printf("\n");
        }
    }

    // free memory
    free(out);
    free(Qs);
    for (int i = 0; i < 49; i++) mpz_clear(H[i]);
    return 0;
}
