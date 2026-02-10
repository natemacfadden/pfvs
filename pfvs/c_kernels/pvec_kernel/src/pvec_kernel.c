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
    int B,
    int * restrict linmat,
    int linmin,
    int32_t (* restrict stack_partial_sum)[dim],
    int    (* restrict abssum)[dim],
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
    for (int j=0; j<dim; ++j) {
        if (linmat[j*dim + i] == 0){
            continue;
        }

        int numer = linmin - stack_partial_sum[sp][j] - B*abssum[j][i];
        int h     = linmat[j*dim + i];

        if (h>0){
            lo = max_int(lo, (int)ceil(1.0*numer/h));
        } else {
            hi = min_int(hi, (int)floor(1.0*numer/h));
        }

    }

    // store the data to recreate the interval
    int num = hi - lo + 1;
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
    int max_N_out,
    int max_N_iter,
    double eps)
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
    // define arrays
    #define MAX_DIM dim
    #define MAX_DEPTH (MAX_DIM + 1)

    int32_t vec[MAX_DIM];

    int32_t stack_i[MAX_DEPTH];
    int32_t stack_pos[MAX_DEPTH];

    int32_t stack_val_len[MAX_DEPTH];
    int32_t stack_val_min[MAX_DEPTH];

    int32_t stack_partial_sum[MAX_DEPTH][MAX_DIM];
    memset(stack_partial_sum, 0, sizeof(stack_partial_sum));

    // define variables
    // ----------------
    // output/stack pointer
    int op = 0;
    int sp = 0;

    // compute helper variable
    int abssum[numhyps][MAX_DIM+1];
    for (int j=0; j<numhyps; ++j) {
        abssum[j][0] = 0;

        for (int i=0; i<dim; ++i) {
            abssum[j][i+1] = abssum[j][i] + abs(linmat[j*dim + i]);
        }
    }       

    // initialize stack
    stack_i[sp]   = dim-1;
    stack_pos[sp] = 0;

    int k = set_bounds(
            sp,
            dim-1,
            dim,
            B,
            linmat,
            linmin,
            stack_partial_sum,
            abssum,
            stack_val_min,
            stack_val_len);
    if (k == 0) {
        printf("ERROR NO VECTORS");
        return -5;
    }

    // iterate over the stack
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
            if (op >= max_N_out)
                return -2;

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
            stack_partial_sum[sp][j] = stack_partial_sum[sp-1][j] + linmat[j*dim + i]*veci;
        }

        if (i >= 1) {
            set_bounds(
                sp,
                i-1,
                dim,
                B,
                linmat,
                linmin,
                stack_partial_sum,
                abssum,
                stack_val_min,
                stack_val_len);
        }
    }

    *N_out = op;

    return 0;
}
