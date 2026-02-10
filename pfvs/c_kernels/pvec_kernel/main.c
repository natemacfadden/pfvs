#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>

#include "pvec_kernel.h"

int main(int argc, char *argv[])
{
    // read dimension
    int dim = argc - 1;
    if (dim <= 0) {
        fprintf(stderr, "Usage: %s b0 b1 ... b{d-1}\n", argv[0]);
        return 1;
    }

    return 0;
    // read bounds
    int32_t *bounds = malloc(dim * sizeof(int32_t));
    if (!bounds) {
        perror("malloc bounds");
        return 1;
    }

    for (int i = 0; i < dim; ++i) {
        char *endptr;
        errno = 0;

        long val = strtol(argv[i + 1], &endptr, 10);
        if (errno || *endptr != '\0' || val < 0 || val > INT32_MAX) {
            fprintf(stderr, "Invalid bound: %s\n", argv[i + 1]);
            free(bounds);
            return 1;
        }

        bounds[i] = (int32_t)val;
    }

    // initialize output array
    int max_N = 1;
    for (int i =0; i < dim; ++i) {
        max_N *= 2*bounds[i] + 1;
    }

    int32_t *out = malloc((size_t)max_N * dim * sizeof(int32_t));
    if (!out) {
        perror("malloc out");
        free(bounds);
        return 1;
    }

    // do the enumeration
    int N_out = 0;

    clock_t start = clock();
    //int rc = enumerate_box_c(bounds, dim, max_N, out, &N_out);
    int rc = 0;
    clock_t end = clock();
    float eval_time = (float)(end - start) / CLOCKS_PER_SEC;
    
    if (rc != 0) {
        fprintf(stderr, "enumerate_box_c failed (%d)\n", rc);
    } else {
        printf("Generated %d vectors if %fs\n", N_out, eval_time);
    }

    // free memory
    free(out);
    free(bounds);
    return 0;
}