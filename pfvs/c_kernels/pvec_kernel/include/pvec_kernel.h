#ifndef ENUMERATE_BOX_H
#define ENUMERATE_BOX_H

#include <stdint.h>

int enumerate_box_c(
    const int32_t * restrict bounds,
    int dim,
    int max_N_out,
    int32_t * restrict out,   // shape: [max_N_out, dim]
    int * restrict N_out      // number of rows written
);

#endif
