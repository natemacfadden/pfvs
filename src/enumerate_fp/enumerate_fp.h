#ifndef ENUMERATE_FP_H
#define ENUMERATE_FP_H

#include <stdint.h>

int enumerate_fp_c(
    int32_t * restrict out, // shape: [max_N_out, dim]
    float * restrict Qs,    // shape: [max_N_out,]
    int * restrict N_out,   // number of rows written
    int max_N_out,
    int dim,
    float * restrict L,
    int Q,
    int * restrict linvec,
    float linmin,
    float eps,
    int COORD_BUFF_SIZE
);

#endif
