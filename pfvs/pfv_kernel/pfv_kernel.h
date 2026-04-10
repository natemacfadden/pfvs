#ifndef PFV_KERNEL_H
#define PFV_KERNEL_H

// HEADER
// ======
#include <stdint.h>
#include <gmp.h>

/*
**Description:**
Adaptation of the (iterative) Fincke-Pohst algorithm for utility in
constructing PFVs. I.e., solves
    0 <= vec^T @ mat     @ vec <= dilation*Q.
    0 <= vec^T @ (U.T@U) @ vec <= dilation*Q
as well as (GCD cut)
    (vec^T @ mat @ vec)//Q <= gcd(K)
                            = gcd(Z@Binter@vec)
                            = gcd(H@vec)
for H the row-HNF of Z@Binter (this matrix computes K from vec).

H has shape (dim+1) x dim, reflecting that Z@Binter has one more row than
column (no K[0] is dropped, unlike the coni case). The zero row of the HNF
sits at row 0; the upper-triangular non-zero block occupies rows 1..dim.
Consequently, H indexing is shifted by one relative to conipfv_kernel:
the diagonal element for column i is at H[(i+1)*dim + i], and the
off-diagonal elements for column j >= i are at H[i*dim + j].

Any `vec` satisfying all of the above can generate a PFV, as long as
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
- `U`:         The upper triangular matrix such that mat = U.T@U
- `Q`:         The ellipsoid bound.
- `dilation`:  The maximum allowed dilation to allow... As long as
               gcd(K) >= (vec^T @ mat @ vec)//Q, the vector vec can
               still define a PFV.
// GCD cut
- `H`:         Let G be the matrix such that K = G@vec. Then H = HNF(G).
               Shape (dim+1) x dim, stored row-major with stride dim.
// misc specs
- `max_N_out`: The maximum number of output allowed.
- `eps`:       A small number used for correctly setting bounds despite
               floating point errors.

**Returns:**
A status code according to following list:
    0: success
    -6: problem dimension too high (currently >256)
    -100: dilation overflows uint32_t
    -5: no vectors
    -2: exceed max_N_out outputs
*/
int _pfv_kernel_c(
    int32_t * restrict out,
    float * restrict Qs,
    int * restrict N_out,
    int dim,
    double * restrict U,
    int Q,
    double dilation,
    mpz_t * restrict H,
    long max_N_out,
    double eps
);


// IMPLEMENTATION
// ==============
#ifdef PFV_KERNEL_IMPLEMENTATION

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#define DEBUG
#ifdef DEBUG
    #define DEBUG_LOG(...) fprintf(stderr, __VA_ARGS__)
#else
    #define DEBUG_LOG(...) ((void)0)
#endif

// GCD helper
static inline uint64_t gcd(uint64_t u, uint64_t v, uint64_t min_allowed_gcd)
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
    uint64_t min_allowed_gcd_shifted = min_allowed_gcd >> shift;

    // cut if current bound on gcd is below allowed value
    // current upper bound is u << shift
    if ((u < min_allowed_gcd_shifted) || (v < min_allowed_gcd_shifted)) {
        return 1;
    }

    // gcd(u,v) = gcd(|u-v|, min(u,v))
    // just keep u < v so this is gcd(v-u, u)
    do {
        // remove factors of 2 from v
        v >>= __builtin_ctz(v);

        // ensure u <= v
        if (u > v) { uint64_t t = v; v = u; u = t; }
        /*
        // branchless but slower
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

static inline void gcd_gmp(mpz_t result, mpz_t u, mpz_t v, const mpz_t min_allowed_gcd)
{
    // take absolute values
    mpz_abs(u, u);
    mpz_abs(v, v);

    // check if values fit in 64 bits
    if (mpz_sizeinbase(u, 2) <= 64 &&
        mpz_sizeinbase(v, 2) <= 64 &&
        mpz_sizeinbase(min_allowed_gcd, 2) <= 64) {

        uint64_t u_u64 = mpz_get_ui(u);
        uint64_t v_u64 = mpz_get_ui(v);
        uint64_t min_u64 = mpz_get_ui(min_allowed_gcd);

        uint64_t res = gcd(u_u64, v_u64, min_u64);
        mpz_set_ui(result, res);
        return;
    }

    // fallback to GMP for large values
    if ((mpz_cmp(u, min_allowed_gcd) < 0) || (mpz_cmp(v, min_allowed_gcd) < 0)) {
        mpz_set_ui(result, 1);
        return;
    }

    mpz_gcd(result, u, v);

    if (mpz_cmp(result, min_allowed_gcd) < 0) {
        mpz_set_ui(result, 1);
    }
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

    // debug print statement
    DEBUG_LOG("R=%f, ci_offset=%f, U_diag_inv=%f, eps=%f\n", R, ci_offset, U_diag_inv, eps);
    DEBUG_LOG("lo =  ceil(%f)\n", ( -R - ci_offset) * U_diag_inv - eps);
    DEBUG_LOG("hi = floor(%f)\n", ( +R - ci_offset) * U_diag_inv + eps);
    DEBUG_LOG("Set bounds for %d to %d->%d\n", sp, lo, lo+num-1);

    return num;
}

// custom FP code for PFVs
int _pfv_kernel_c(
    int32_t * restrict out,
    float * restrict Qs,
    int * restrict N_out,
    int dim,
    double * restrict U,
    int Q,
    double dilation,
    mpz_t * restrict H,
    long max_N_out,
    double eps)
{
    /*
    **Description:**
    Adaptation of the (iterative) Fincke-Pohst algorithm for utility in
    constructing PFVs. I.e., solves
        0 <= vec^T @ mat     @ vec <= dilation*Q.
        0 <= vec^T @ (U.T@U) @ vec <= dilation*Q
    as well as (GCD cut)
        (vec^T @ mat @ vec)//Q <= gcd(K)
                                = gcd(Z@Binter@vec)
                                = gcd(H@vec)
    for H the row-HNF of Z@Binter (this matrix computes K from vec).

    H has shape (dim+1) x dim with a zero row at row 0. The diagonal element
    for column i is at H[(i+1)*dim + i], and off-diagonal elements for column
    j >= i are at H[i*dim + j].

    Any `vec` satisfying all of the above can generate a PFV, as long as
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
    - `U`:         The upper triangular matrix such that mat = U.T@U
    - `Q`:         The ellipsoid bound.
    - `dilation`:  The maximum allowed dilation to allow... As long as
                   gcd(K) >= (vec^T @ mat @ vec)//Q, the vector vec can
                   still define a PFV.
    // GCD cut
    - `H`:         Let G be the matrix such that K = G@vec. Then H = HNF(G).
                   Shape (dim+1) x dim, stored row-major with stride dim.
    // misc specs
    - `max_N_out`: The maximum number of output allowed.
    - `eps`:       A small number used for correctly setting bounds despite
                   floating point errors.

    **Returns:**
    A status code according to following list:
        0: success
        -6: problem dimension too high (currently >256)
        -100: dilation overflows uint32_t
        -5: no vectors
        -2: exceed max_N_out outputs
    */
    // define variables
    // ----------------
    int status = 0;

    // define arrays
    #define MAX_DIM dim
    #define MAX_DEPTH (MAX_DIM + 1)

    double  U_diag_inv[MAX_DIM];
    int32_t vec[MAX_DIM];

    int32_t stack_i[MAX_DEPTH];
    int32_t stack_pos[MAX_DEPTH];
    double  stack_remQ[MAX_DEPTH];
    mpz_t   stack_gcd[MAX_DEPTH];
    mpz_t   veci_gmp, temp_gcd, temp_hvec, temp_required, temp_hvec_abs;

    int32_t stack_val_len[MAX_DEPTH];
    int32_t stack_val_min[MAX_DEPTH];

    double  stack_ci_offset[MAX_DEPTH];
    mpz_t   stack_Hveci[MAX_DEPTH];

    // output/stack pointer
    int op = 0;
    int sp = 0;

    // initialize GMP variables
    for (int i = 0; i < MAX_DEPTH; i++) {
        mpz_init(stack_gcd[i]);
        mpz_init(stack_Hveci[i]);
    }
    mpz_init(veci_gmp);
    mpz_init(temp_gcd);
    mpz_init(temp_hvec);
    mpz_init(temp_required);
    mpz_init(temp_hvec_abs);

    // check dimensions are reasonable
    // -------------------------------
    #define MAX_SUPPORTED_DIM 256
    if (dim > MAX_SUPPORTED_DIM) {
        status = -6;
        goto end;
    }

    // compute useful variables
    // ------------------------
    double Q_upper = Q*dilation;
    for (int j=0; j<dim; ++j)
        U_diag_inv[j] = 1.0 / U[j*dim + j];

    // check the dilation is reasonable
    // --------------------------------
    double required_dilation_dbl = Q_upper/Q;
    if (required_dilation_dbl > UINT32_MAX) {
        status = -100;
        goto end;
    }

    // initialize stack
    // ----------------
    stack_i[sp]         = dim-1;
    stack_pos[sp]       = 0;
    stack_remQ[sp]      = Q_upper;
    mpz_set_ui(stack_gcd[sp], 0);

    stack_ci_offset[sp] = 0;
    mpz_set_ui(stack_Hveci[sp], 0);

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
        status = -5;
        goto end;
    }

    // iterate over the stack
    // ----------------------
    int i;
    int pos;
    double remQ;

    while (sp >= 0) {
        // read from the stack
        i    = stack_i[sp];
        pos  = stack_pos[sp];
        remQ = stack_remQ[sp];

        DEBUG_LOG("Setting component-%d for op=%d, sp=%d, pos=%d, remQ=%f\n", i, op, sp, pos, remQ);

        // save if node is complete
        // if i==-1, then we have fully written vec
        if (i == -1) {
            if (op >= max_N_out) {
                status = -2;
                goto end;
            }

            float Qsave = Q_upper-remQ;
            if (Qsave > -eps) {
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

        // set vec[i]
        int veci = stack_val_min[sp] + pos;
        vec[i] = veci;

        DEBUG_LOG("Set   component-%d for op=%d, sp=%d, pos=%d, remQ=%f to %d\n", i, op, sp, pos, remQ, veci);

        // advance pos for next iteration
        stack_pos[sp] += 1;

        // get ci, the new amount of remaining Q
        double ci = U[dim*i + i]*veci + stack_ci_offset[sp];
        double new_rem = remQ - ci*ci;

        // cut if no more Q left...
        if (new_rem < 0 - eps) {
            DEBUG_LOG("SKIPPED SINCE TADPOLE %f < %f\n",new_rem,-eps);
            continue;
        }

        // GCD cut
        // -------
        // H is (dim+1) x dim with a zero row at row 0. The diagonal element
        // for column i is at row i+1: H[(i+1)*dim + i].
        double required_dilation_dbl = floor((Q_upper - new_rem) / Q - eps);
        mpz_set_d(temp_required, required_dilation_dbl);

        // convert veci to GMP for H multiplication
        mpz_set_si(veci_gmp, veci);

        // first try a simple upper bound... compute new_gcd only if needed
        mpz_set(temp_gcd, stack_gcd[sp]); // upper bound
        if (mpz_cmp_ui(temp_gcd, 0) == 0) {
            // gcd was 0... update it to whatever abs(Hvec_i) is....
            mpz_set(temp_hvec, stack_Hveci[sp]);
            mpz_addmul(temp_hvec, H[(i+1)*dim + i], veci_gmp);
            mpz_abs(temp_hvec_abs, temp_hvec);
            mpz_set(temp_gcd, temp_hvec_abs);

            if (mpz_cmp_ui(temp_gcd, 0) == 0) {
                // still 0... can't do jack with this
                goto write_stack;
            }

        } else if (mpz_cmp(temp_gcd, temp_required) < 0) {
            // bad! we can't get back under tadpole...
            // (we check here to avoid a gcd call...)
            DEBUG_LOG("1SKIPPED BAD GCD\n");
            //gmp_fprintf(stderr, "  temp_gcd = %Zd, temp_required = %Zd\n", temp_gcd, temp_required);
            continue;

        } else if (mpz_cmp_ui(temp_gcd, 1) != 0) {
            // only other case where we can nontrivially change the gcd
            mpz_set(temp_hvec, stack_Hveci[sp]);
            mpz_addmul(temp_hvec, H[(i+1)*dim + i], veci_gmp);
            mpz_abs(temp_hvec_abs, temp_hvec);
            gcd_gmp(temp_gcd, temp_gcd, temp_hvec_abs, temp_required);
        }

        // here, the gcd is nonzero and may newly violate tadpole
        if (mpz_cmp(temp_gcd, temp_required) < 0) {
            DEBUG_LOG("2SKIPPED BAD GCD\n");
            //gmp_fprintf(stderr, "  temp_gcd = %Zd, temp_required = %Zd\n", temp_gcd, temp_required);
            continue;
        }

        // passes cuts -> push next depth :)
        write_stack:
            sp += 1;
            stack_i[sp]       = i-1;
            stack_pos[sp]     = 0;
            stack_remQ[sp]    = new_rem;
            mpz_set(stack_gcd[sp], temp_gcd);

        // compute the new ci, Hvec_i offset value for i-1 using this vector
        // H[(i-1)*dim + j] in coni → H[i*dim + j] here (zero row shifts index by 1)
        double ci_offset = 0.0;
        mpz_set_ui(stack_Hveci[sp], 0);
        if (i > 0) {
            for (int j = i; j<dim; ++j) {
                DEBUG_LOG("%d %d %f %d\n",i,j,U[(i-1)*dim + j],vec[j]);
                ci_offset += U[(i-1)*dim + j] * vec[j];

                // Hvec_i += H[i*dim + j] * vec[j]
                mpz_set_si(veci_gmp, vec[j]);
                mpz_addmul(stack_Hveci[sp], H[i*dim + j], veci_gmp);
            }

            DEBUG_LOG("%d %d %f\n", i, vec[i], ci_offset);

            stack_ci_offset[sp] = ci_offset;
        }

        set_bounds(
            sp,
            new_rem,
            ci_offset,
            U_diag_inv[i-1],
            stack_val_min,
            stack_val_len,
            eps);
    }

    DEBUG_LOG("DONE\n");

    end:
        // clean up GMP variables
        for (int i = 0; i < MAX_DEPTH; i++) {
            mpz_clear(stack_gcd[i]);
            mpz_clear(stack_Hveci[i]);
        }
        mpz_clear(veci_gmp);
        mpz_clear(temp_gcd);
        mpz_clear(temp_hvec);
        mpz_clear(temp_required);
        mpz_clear(temp_hvec_abs);

        // return
        *N_out = op;
        return status;
}

#endif // PFV_KERNEL_IMPL

#endif // PFV_KERNEL_H
