#ifndef CONI_KERNEL_H
#define CONI_KERNEL_H

// HEADER
// ======
#include <stdint.h>

/*
**Description:**
Adaptation of the (iterative) Fincke-Pohst algorithm for utility in
constructing coni-PFVs. I.e., solves
    0 <= vec^T @ mat     @ vec <= dilation*Q.
    0 <= vec^T @ (U.T@U) @ vec <= dilation*Q
as well as (M[0] cuts)
    linmin <= dot(linvec, vec)
as well as (K'>0 cuts)
    (vec^T @ mat @ vec)//Q <= gcd(Kperp)
                            = gcd([0, 1]@Z@Binter@vec)
                            = gcd(H@vec)
for H the row-HNF of [0, 1]@Z@Binter (this matrix computes Kperp from vec).

Any `vec` satisfying all of the above can generate a coni-PFV, as long as
det(N) != 0.

Most of the work is in writing to `out`, `Qs`, and `N_out`.

**Arguments:**
// output objects
- `out`:       A container for the lattice points vec.
- `Qs`:        A container for the valuations of vec^T @ mat @ vec of the
               outputs.
- `N_out`:     An integer we write to, indicating the number of outputs.
// ellipsoid def
- `dim`      : The dimension of the problem.
- `U`:         The upper triangular matrix such that mat = U.T@L
- `Q`:         The ellipsoid bound.
- `dilation`:  The maximum allowed dilation to allow... As long as
               gcd(Kperp) >= (vec^T @ mat @ vec)//Q, the vector vec can
               still define coni-PFV.
// M0 cuts
- `linvec`:    Binter[0,:]. The vector such that dot(linvec,vec)=M0. BEST TO
               ORDER COLUMNS OF BINTER SUCH THAT linvec HAS A LARGE NUMBER
               OF LEADING 0s.
- `linmin`:    The minimum value of M0 permitted. Inclusive.
// Kprime cuts
- `H`:         Let G be the matrix such that Kperp = G@vec. Then H = HNF(G).
// misc specs
- `max_N_out`: The maximum number of output allowed.
- `eps`:       A small number used for correctly setting bounds despite
               floating point errors.

**Returns:**
A status code.
*/
int _coni_kernel_c(
    int32_t * restrict out,
    double * restrict Qs,
    int * restrict N_out,
    int dim,
    double * restrict U,
    int Q,
    double dilation,
    int * restrict linvec,
    double linmin,
    int * restrict H,
    int max_N_out,
    double eps
);


// IMPLEMENTATION
// ==============
#ifdef CONI_KERNEL_IMPLEMENTATION

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// GCD helper
static inline uint32_t gcd(uint32_t u, uint32_t v, uint32_t min_allowed_gcd)
{
    /*
    **Description:**
    Modification of Stein's binary GCD algorithm gcd(u,v) with an explicit
    minum allowed GCD. If gcd(u,v) < min_allowed_gcd, then quit early, returning
    1.

    **Arguments:**
    - `u`:               One integer.
    - `v`:               The other integer.
    - `min_allowed_gcd`: The minimum allowed GCD to return.

    **Returns:**
    gcd(u,v) if it is >=min_allowed_gcd. Otherwise, 1.
    */
    if (u == 0) return v;
    if (v == 0) return u;

    // remove factors of 2
    int shift = __builtin_ctz(u | v);
    u >>= __builtin_ctz(u);

    // min allowed gcd,shifted
    uint32_t min_allowed_gcd_shifted = min_allowed_gcd >> shift;

    // cut if current bound on gcd is below allowed value
    // current upper bound is u << shift
    if (u < min_allowed_gcd_shifted) {
        return 1;
    }

    // gcd(u,v) = gcd(|u-v|, min(u,v))
    // just keep u < v so this is gcd(v-u, u)
    do {
        // remove factors of 2 from v
        v >>= __builtin_ctz(v);
        
        // ensure u <= v
        if (u > v) { int t = v; v = u; u = t; }
        /*
        uint32_t mask = -(u > v);
        uint32_t t = mask & (u ^ v);
        u ^= t;
        v ^= t;
        */

        // cut if current bound on gcd is below allowed value
        // current upper bound is u << shift
        if (u < min_allowed_gcd_shifted) {
            return 1;
        }

        // v -> v-u
        v -= u;
    } while (v);

    return u << shift;
}

// FP vec[i] bound setting helper
static inline int set_bounds(
    int sp,
    double remQ,
    double ci_offset,
    double U_diag_inv,
    int32_t * restrict stack_val_min,
    int32_t * restrict stack_val_len,
    double eps)
{
    /*
    **Description:**
    Defines the bounds to iterate vec[i] over in the next FP iteration.

    Most of the work is in writing to `stack_val_min` and `stack_val_len`.

    **Arguments:**
    - `sp`:            A pointer to the current stack element.
    - `remQ`:          The remaining slack for the norm squared.
    - `ci_offset`:     A constant shift in the value vec[i] when computing norm.
    - `U_diag_inv`:    1/diag(U) for U the upper-triangular matrix (from
                       Cholesky).
    - `stack_val_min`: The minimum value to try for vec[i].
    - `stack_val_len`: The number of candidates to try for vec[i].
    - `eps`:           A small number eps used for ensuring the bounds contain
                       all possible values of vec[i].

    **Returns:**
    The number of candidates to try, `stack_val_len[sp]`.
    */
    if (remQ < 0)
        remQ = 0;
    double R = sqrt(remQ);

    int lo = (int)ceil(( -R - ci_offset) * U_diag_inv - eps);
    int hi = (int)floor(( R - ci_offset) * U_diag_inv + eps);

    // store the data to recreate the interval
    int num = hi - lo + 1;
    stack_val_min[sp] = lo;
    stack_val_len[sp] = num;

    return num;
}

// custom FP code for coniPFVs
int _coni_kernel_c(
    int32_t * restrict out,
    double * restrict Qs,
    int * restrict N_out,
    int dim,
    double * restrict U,
    int Q,
    double dilation,
    int * restrict linvec,
    double linmin,
    int * restrict H,
    int max_N_out,
    double eps)
{
    /*
    **Description:**
    Adaptation of the (iterative) Fincke-Pohst algorithm for utility in
    constructing coni-PFVs. I.e., solves
        0 <= vec^T @ mat     @ vec <= dilation*Q.
        0 <= vec^T @ (U.T@U) @ vec <= dilation*Q
    as well as (M[0] cuts)
        linmin <= dot(linvec, vec)
    as well as (K'>0 cuts)
        (vec^T @ mat @ vec)//Q <= gcd(Kperp)
                                = gcd([0, 1]@Z@Binter@vec)
                                = gcd(H@vec)
    for H the row-HNF of [0, 1]@Z@Binter (this matrix computes Kperp from vec).

    Any `vec` satisfying all of the above can generate a coni-PFV, as long as
    det(N) != 0.

    Most of the work is in writing to `out`, `Qs`, and `N_out`.

    **Arguments:**
    // output objects
    - `out`:       A container for the lattice points vec.
    - `Qs`:        A container for the valuations of vec^T @ mat @ vec of the
                   outputs.
    - `N_out`:     An integer we write to, indicating the number of outputs.
    // ellipsoid def
    - `dim`      : The dimension of the problem.
    - `U`:         The upper triangular matrix such that mat = U.T@L
    - `Q`:         The ellipsoid bound.
    - `dilation`:  The maximum allowed dilation to allow... As long as
                   gcd(Kperp) >= (vec^T @ mat @ vec)//Q, the vector vec can
                   still define coni-PFV.
    // M0 cuts
    - `linvec`:    Binter[0,:]. The vector such that dot(linvec,vec)=M0. BEST TO
                   ORDER COLUMNS OF BINTER SUCH THAT linvec HAS A LARGE NUMBER
                   OF LEADING 0s.
    - `linmin`:    The minimum value of M0 permitted. Inclusive.
    // Kprime cuts
    - `H`:         Let G be the matrix such that Kperp = G@vec. Then H = HNF(G).
    // misc specs
    - `max_N_out`: The maximum number of output allowed.
    - `eps`:       A small number used for correctly setting bounds despite
                   floating point errors.

    **Returns:**
    A status code.
    */
    // define arrays
    #define MAX_DIM dim
    #define MAX_DEPTH (MAX_DIM + 1)

    double  U_diag_inv[MAX_DIM];
    int32_t vec[MAX_DIM];

    int32_t stack_i[MAX_DEPTH];
    int32_t stack_pos[MAX_DEPTH];
    double  stack_remQ[MAX_DEPTH];
    int32_t stack_M0[MAX_DEPTH];
    uint32_t stack_gcd[MAX_DEPTH];

    int32_t stack_val_len[MAX_DEPTH];
    int32_t stack_val_min[MAX_DEPTH];

    double  stack_ci_offset[MAX_DEPTH];
    int     stack_Hveci[MAX_DEPTH];


    // compute  useful variables
    double Q_upper = Q*dilation;
    for (int j=0; j<dim; ++j)
        U_diag_inv[j] = 1.0 / U[j*dim + j];

    // number of zeros in linvec
    int num_zeros = 0;
    bool zeros = true;
    for (int j=0; j<dim; ++j){
        if (linvec[j] == 0) {
            if (zeros == false)
                return -4;
            num_zeros += 1;
        } else {
            zeros = false;
        }
    }

    // define variables
    // ----------------
    // output/stack pointer
    int op = 0;
    int sp = 0;

    // initialize stack
    stack_i[sp]         = dim-1;
    stack_pos[sp]       = 0;
    stack_remQ[sp]      = Q_upper;
    stack_M0[sp]        = 0;
    stack_gcd[sp]       = 0;

    stack_ci_offset[sp] = 0;
    stack_Hveci[sp]     = 0;
    
    int k = set_bounds(
            sp,
            Q_upper,
            0,
            U_diag_inv[dim-1],
            stack_val_min,
            stack_val_len,
            eps);
    if (k == 0) {
        printf("ERROR NO VECTORS");
        return -5;
    }

    // iterate over the stack
    int i;
    int pos;
    double remQ;
    int M0;

    while (sp >= 0) {
        // read from the stack
        i    = stack_i[sp];
        pos  = stack_pos[sp];
        remQ = stack_remQ[sp];
        M0   = stack_M0[sp];

        // save if node is complete
        // if i==-1, then we have fully written vec
        if (i == -1) {
            if (op >= max_N_out)
                return -2;

            int Qsave = Q_upper-remQ;
            if (Qsave > eps) {
                int32_t *dst = &out[op * dim];

                #pragma unroll
                for (int j = 0; j < dim; ++j)
                    dst[j] = vec[j];
                //memcpy(&out[op * dim], vec, dim * sizeof(int32_t));
                
                Qs[op] = Qsave;
                op ++;
            }
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

        // cut on M0 >= M0min
        // ------------------
        M0 = M0 + linvec[i]*veci;
        if ((i == num_zeros) & (M0 < linmin)) {
            continue;
        }

        // get ci, the new amount of remaining Q
        double ci = U[dim*i + i]*veci + stack_ci_offset[sp];
        double new_rem = remQ - ci*ci;

        // cut of no more Q left...
        if (new_rem < 0 - eps) {
            continue;
        }

        // check if we violated K'>0 constraints
        // -------------------------------------
        uint32_t required_dilation = (uint32_t)floor((Q_upper-new_rem)/Q - eps);

        // first try a simple upper bound... compute new_gcd only if needed
        uint32_t new_gcd = stack_gcd[sp]; // upper bound
        if (new_gcd == 0) {
            // gcd was 0... update it to whatever abs(Hvec_i) is....
            uint32_t Hvec_i  = abs(stack_Hveci[sp] + H[i*dim+i]*veci);
            new_gcd          = Hvec_i;

            if (new_gcd == 0) {
                // still 0... can't do jack with this
                goto write_stack;
            }

        } else if (new_gcd < required_dilation) {
            // bad! we can't get back under tadpole...
            // (we check here to avoid a gcd call...)
            continue;

        } else if (new_gcd != 1) {
            // only other case where we can nontrivially change the gcd
            uint32_t Hvec_i  = abs(stack_Hveci[sp] + H[i*dim+i]*veci);
            new_gcd          = gcd(new_gcd, Hvec_i, required_dilation);
        }

        // here, the gcd is nonzero and may newly violate tadpole
        if (new_gcd < required_dilation) {
            continue;
        }

        // passes cuts -> push next depth :)
        write_stack:
            sp += 1;
            stack_i[sp]       = i-1;
            stack_pos[sp]     = 0;
            stack_remQ[sp]    = new_rem;
            stack_M0[sp]      = M0;
            stack_gcd[sp]     = new_gcd;

        // compute the new ci, Hvec_i offset value for i-1 using this vector
        double ci_offset = 0.0;
        int Hvec_i = 0;
        for (int j = i; j<dim; ++j) {
            ci_offset += U[(i-1)*dim + j] * vec[j];
            Hvec_i += H[(i-1)*dim + j] * vec[j];
        }

        stack_ci_offset[sp] = ci_offset;
        stack_Hveci[sp] = Hvec_i;
        
        set_bounds(
            sp,
            new_rem,
            ci_offset,
            U_diag_inv[i-1],
            stack_val_min,
            stack_val_len,
            eps);
    }

    *N_out = op;

    return 0;
}

#endif // CONI_KERNEL_IMPL

#endif // CONI_KERNEL_H
