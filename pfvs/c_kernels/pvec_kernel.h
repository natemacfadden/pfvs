#ifndef PVEC_KERNEL_H
#define PVEC_KERNEL_H

// HEADER
// ======
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
- `B`:          The bounds |x_i| <= B
// cone definition cuts
- `linmat`:     The matrix defining the cone.
- `linmin`:     The closest permitted distance to a hyperplane.
- `numhyps`:    The number of hyperplane constraints.
// misc specs
- `max_N_out`:  The maximum number of output allowed.
- `max_N_iter`: The maximum number of iterations allowed.

**Returns:**
A status code according to following list:
        0: success
        -6: problem dimension too high (currently >256)
        -5: no vectors
        -2: exceed max_N_out outputs
*/
int _pvec_kernel_c(
    int32_t * restrict out,
    int * restrict N_out,
    int dim,
    int B,
    int * restrict linmat,
    int linmin,
    int numhyps,
    long max_N_out,
    long max_N_iter
);


// IMPLEMENTATION
// ==============
#ifdef PVEC_KERNEL_IMPLEMENTATION

#include "pvec_kernel.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline int max_int(int a, int b) {
    return a > b ? a : b;
}
static inline int min_int(int a, int b) {
    return a < b ? a : b;
}

// Kannan vec[i] bound setting helper
static inline int set_bounds(
    int sp,
    int i,
    int dim,
    int numhyps,
    int B,
    int * restrict linmat,
    int linmin,
    int32_t * restrict stack_partial_sum,
    int * restrict abssum,
    int32_t * restrict stack_val_min,
    int32_t * restrict stack_val_len)
{
    /*
    **Description:**
    Defines the bounds to iterate vec[i] over in the next Kannan iteration.

    Most of the work is in writing to `stack_val_min` and `stack_val_len`.

    **Arguments:**
    - `sp`:            A pointer to the current stack element.
    ...
    - `stack_val_min`: The minimum value to try for vec[i].
    - `stack_val_len`: The number of candidates to try for vec[i].
    ...

    **Returns:**
    The number of candidates to try, `stack_val_len[sp]`.
    */
    int lo = -B;
    int hi =  B;

    // cut by each hyperplane
    for (int j=0; j<numhyps; ++j) {
        int h = linmat[j*dim + i];
        if (h == 0){
            continue;
        }

        int numer = linmin - stack_partial_sum[sp*numhyps + j] - B*abssum[j*(dim+1) + i];

        if (h>0){
            lo = max_int(lo, (int)ceil(1.0*numer/h));
        } else {
            hi = min_int(hi, (int)floor(1.0*numer/h));
        }
    }

    // store the data to recreate the interval
    int num = 0;
    if (hi >= lo) {
        num = hi - lo + 1;
    }
    stack_val_min[sp] = lo;
    stack_val_len[sp] = num;

    return num;
}

// custom Kannan code for p-vector generation
int _pvec_kernel_c(
    int32_t * restrict out,
    int * restrict N_out,
    int dim,
    int B,
    int * restrict linmat,
    int linmin,
    int numhyps,
    long max_N_out,
    long max_N_iter)
{
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
    - `B`:          The bounds |x_i| <= B.
    // cone definition cuts
    - `linmat`:     The matrix defining the cone.
    - `linmin`:     The closest permitted distance to a hyperplane.
    - `numhyps`:    The number of hyperplane constraints.
    // misc specs
    - `max_N_out`:  The maximum number of output allowed.
    - `max_N_iter`: The maximum number of iterations allowed.

    **Returns:**
    A status code according to following list:
        0: success
        -6: problem dimension too high (currently >256)
        -5: no vectors
        -2: exceed max_N_out outputs
    */
    // define variables
    // ----------------
    int status = 0;

    // define arrays
    #define MAX_DIM dim
    #define MAX_DEPTH (MAX_DIM + 1)

    int32_t vec[MAX_DIM];

    int32_t stack_i[MAX_DEPTH];
    int32_t stack_pos[MAX_DEPTH];

    int32_t stack_val_len[MAX_DEPTH];
    int32_t stack_val_min[MAX_DEPTH];

    int32_t stack_partial_sum[numhyps*MAX_DEPTH];
    memset(stack_partial_sum, 0, sizeof(stack_partial_sum));

    // output/stack pointer
    int op = 0;
    int sp = 0;

    // misc helpers
    int abssum[numhyps*(dim+1)];

    // check dimensions are reasonable
    // -------------------------------
    #define MAX_SUPPORTED_DIM 256
    if (dim > MAX_SUPPORTED_DIM) {
        status = -6;
        goto end;
    }

    // define variables
    // ----------------
    // compute helper variable
    for (int j=0; j<numhyps; ++j) {
        abssum[j*(dim+1) + 0] = 0;

        for (int k=0; k<dim; ++k) {
            abssum[j*(dim+1) + k+1] = abssum[j*(dim+1) + k] + abs(linmat[j*dim + k]);
        }
    }

    // initialize stack
    // ----------------
    stack_i[sp]   = dim-1;
    stack_pos[sp] = 0;

    int k = set_bounds(
            sp,
            dim-1,
            dim,
            numhyps,
            B,
            linmat,
            linmin,
            stack_partial_sum,
            abssum,
            stack_val_min,
            stack_val_len);
    if (k == 0) {
        printf("ERROR NO VECTORS");
        status = -5;
        goto end;
    }

    // iterate over the stack
    // ----------------------
    int i;
    int pos;

    int Niter = 0;
    while (sp >= 0) {
        // quit it too many iterations
        Niter += 1;
        if (Niter >= max_N_iter) {
            break;
        }

        // read from the stack
        i    = stack_i[sp];
        pos  = stack_pos[sp];

        // save if node is complete
        // if i==-1, then we have fully written vec
        if (i == -1) {
            if (op >= max_N_out) {
                status = -2;
                goto end;
            }

            int32_t *dst = &out[op * dim];

            #pragma unroll
            for (int j = 0; j < dim; ++j)
                dst[j] = vec[j];
            //memcpy(&out[op * dim], vec, dim * sizeof(int32_t));
            
            op ++;

            // kill node
            sp --;
            continue;
        }

        // check if we exhausted values for this component
        if (pos == stack_val_len[sp]) {
            sp--;
            continue;
        }

        // set vec[sp]
        int veci = stack_val_min[sp] + pos;
        vec[i] = veci;

        // advance pos for next iteration
        stack_pos[sp] += 1;

        // passes cuts -> push next depth :)
        sp += 1;
        stack_i[sp]       = i-1;
        stack_pos[sp]     = 0;

        // update the partial sums
        for (int j = 0; j<numhyps; ++j) {
            stack_partial_sum[sp*numhyps+j] = stack_partial_sum[(sp-1)*numhyps + j] + linmat[j*dim + i]*veci;
        }

        if (i >= 1) {
            set_bounds(
                sp,
                i-1,
                dim,
                numhyps,
                B,
                linmat,
                linmin,
                stack_partial_sum,
                abssum,
                stack_val_min,
                stack_val_len);
        }
    }

    end:
        *N_out = op;
        return status;
}

#endif // PVEC_KERNEL_IMPL

#endif // PVEC_KERNEL_H
