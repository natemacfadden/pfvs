#ifndef PVEC_KERNEL_H
#define PVEC_KERNEL_H

#include <stdint.h>

/*
**Description:**
Enumerate lattice points x obeying linmat@x >= linmin and |x_i| <= B using
Kannan's algorithm.

VERY preferable that you the columns of linmat so stricter components come
first.

**Arguments:**
// output objects
- `out`:        A container for the lattice points vec.
- `N_out`:      An integer we write to, indicating the number of outputs.
// box definition
- `dim`:        The dimension of the problem.
- `B`:          The upper triangular matrix such that mat = U.T@L
// cone definition cuts
- `linmat`:     The matrix defining the cone.
- `linmin`:     The closest permitted distance to a hyperplane.
- `numhyps`:    The number of hyperplane constraints.
// misc specs
- `max_N_out`:  The maximum number of output allowed.
- `max_N_iter`: The maximum number of iterations allowed.
- `eps`:        A small number used for correctly setting bounds despite
                floating point errors.

**Returns:**
A status code.
*/
int _pvec_kernel_c(
    int32_t * restrict out,
    int * restrict N_out,
    int dim,
    int B,
    int * restrict linmat,
    int linmin,
    int numhyps,
    int max_N_out,
    int max_N_iter,
    double eps
);

#endif
