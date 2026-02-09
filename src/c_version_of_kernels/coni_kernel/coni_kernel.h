#ifndef CONI_KERNEL_H
#define CONI_KERNEL_H

#include <stdint.h>

int coni_kernel(
    int32_t * restrict out, // shape: [max_N_out, dim]
    double * restrict Qs,    // shape: [max_N_out,]
    int * restrict N_out,   // number of rows written
    int max_N_out,
    int dim,
    double * restrict L,
    int Q,
    double dilation,
    int * restrict linvec,
    double linmin,
    int * restrict H,
    double eps
);

#endif
