#include "enumerate_fp.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

long long gcd(long long a, long long b)
{
    a = llabs(a);
    b = llabs(b);

    while (b != 0) {
        long long t = b;
        b = a % b;
        a = t;
    }
    return a;
}

int enumerate_fp_c(
    int32_t * restrict out,
    float * restrict Qs,
    int * restrict N_out,
    int max_N_out,
    int dim,
    float * restrict L,
    int Q,
    float dilation,
    int * restrict linvec,
    float linmin,
    int * restrict H,
    float eps,
    int COORD_BUFF_SIZE)
{
    // misc
    float Q_upper = Q*dilation;
    float *L_diag_inv = malloc(dim * sizeof(float));
    for (int j=0; j<dim; ++j)
        L_diag_inv[j] = 1.0 / L[j*dim + j];

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

    // allocate memory for stack variables
    int32_t *vec        = malloc(dim * sizeof(int32_t));
    float *ci_offsets   = malloc(dim * sizeof(float));

    int32_t *stack_i    = malloc(dim * sizeof(int32_t));
    int32_t *stack_pos  = malloc(dim * sizeof(int32_t));
    float *stack_remQ   = malloc(dim * sizeof(float));
    int32_t *stack_gcd  = malloc(dim * sizeof(int32_t));
    bool *stack_nz      = malloc(dim * sizeof(bool));

    int32_t *stack_val_len = malloc(dim * sizeof(int32_t));

    if (!vec || !ci_offsets || !stack_i || !stack_pos || !stack_remQ || !stack_nz || !stack_val_len)
        return -1;

    // define candidate array
    int **candidates = malloc((size_t)(dim) * sizeof(int*));
    for (int i = 0; i < dim; ++i) {
        candidates[i] = malloc((size_t)COORD_BUFF_SIZE * sizeof(int32_t));
    }

    // initialize stack
    int sp = 0;

    stack_i[sp]       = dim-1;
    stack_pos[sp]     = 0;
    stack_remQ[sp]    = Q_upper;
    stack_gcd[sp]     = 0;
    stack_nz[sp]      = false;
    stack_val_len[sp] = -1; // will fill below

    // iterate over the stack
    int op = 0;

    int i;
    int pos;
    float remQ;
    int _gcd;
    bool nz;

    while (sp >= 0) {
        // read from the stack
        i    = stack_i[sp];
        pos  = stack_pos[sp];
        remQ = stack_remQ[sp];
        _gcd = stack_gcd[sp];
        nz   = stack_nz[sp];

        // save if node is complete
        // if i==-1, then we have fully written vec
        if (i == -1) {
            if (nz) {
                if (op >= max_N_out)
                    return -2;
                memcpy(&out[op * dim], vec, dim * sizeof(int32_t));
                Qs[op] = Q_upper-remQ;
                op ++;
            }
            // kill node
            sp --;
            continue;
        }

        // check if we exhausted values for this component
        if (stack_pos[sp] == stack_val_len[sp]) {
            sp--;
            for (int j=0; j<i; ++j)
                ci_offsets[j] += L[i*dim + j] * vec[i];
            continue;
        }

        // current depth incomplete...
        // ---------------------------
        // set candidate values of vec[i] if first time to depth
        if (stack_val_len[sp] == -1) {
            /*
            feasible integer bounds for vec[i]
            -R                      <= c[i]          <= R
            -R - ci_offset          <= L[i,i]*vec[i] <= R - ci_offset
            (-R - ci_offset)/L[i,i] <= vec[i]        <= (R - ci_offset)/L[i,i]
            where we used that the diagonal is positive
            */
            if (remQ<0)
                remQ = 0;
            float R = sqrt(remQ);
            int lo = (int)ceil(( -R - ci_offsets[i]) * L_diag_inv[i] - eps);
            int hi = (int)floor(( R - ci_offsets[i]) * L_diag_inv[i] + eps);

            // values of veci to iterate over
            int numvals = 0;
            for (int v=lo; v<=hi; ++v) {
                candidates[sp][numvals] = v;
                numvals += 1;
            } 

            // kill node if no valid veci values
            if (numvals == 0) {
                sp -= 1;
                continue;
            }
            // kill execution if there are too many values
            else if (numvals>COORD_BUFF_SIZE) {
                printf("Assumed |hi-lo| <= {COORD_BUFF_SIZE}, but got %d", numvals);
                return -3;
            }

            // yes valid veci values
            stack_val_len[sp] = numvals;
            stack_pos[sp] = 0;
            pos = 0;

            for (int j=0; j<i; ++j)
                ci_offsets[j] += L[i*dim + j] * (candidates[sp][pos]-1);
        }

        // set vec[sp]
        int veci = candidates[sp][pos];
        vec[i] = veci;

        // advance pos for next iteration
        stack_pos[sp] += 1;

        // update ci_offsets for descendents
        for (int k=0; k<i; ++k)
            ci_offsets[k] += L[i*dim + k]; // +1

        // get ci, the new amount of remaining Q
        float ci = L[dim*i + i]*veci + ci_offsets[i];
        float new_rem = remQ - ci*ci;

        // cut of no more Q left...
        if (new_rem < 0 - eps)
            continue;

        // cut if dot product violates bounds
        if (i == num_zeros) {
            float val = 0;
            for (int j=i; j<dim; ++j)
                val += linvec[j]*vec[j];

            if (val < linmin)
                continue;
        }

        // check if we violated K'>0 constraints
        // -------------------------------------
        int Hvec_i = 0;
        for (int k=i; k<dim; ++k)
            Hvec_i += H[i*dim + k] * vec[k];
        
        float required_dilation = (Q_upper-new_rem)/Q - eps;

        // first try a simpler-to-compute upper
        int new_gcd_upper_bound = _gcd;
        if ((new_gcd_upper_bound > 0) && (new_gcd_upper_bound < required_dilation))
            continue;

        int new_gcd = gcd(_gcd, Hvec_i);
        if ((new_gcd > 0) && (new_gcd < required_dilation)) 
            continue;

        // passes cuts -> push next depth :)
        sp += 1;
        stack_i[sp]       = i-1;
        stack_pos[sp]     = 0;
        stack_remQ[sp]    = new_rem;
        stack_gcd[sp]     = new_gcd;
        stack_nz[sp]      = nz || (veci != 0);
        stack_val_len[sp] = -1; // will fill when we visit
        // candidate array for this depth is stack_vals[sp,:]
        // else do not push (prune)
    }

    *N_out = op;

    // free variabls
    free(L_diag_inv);
    free(vec);
    free(ci_offsets);
    free(stack_i);
    free(stack_pos);
    free(stack_remQ);
    free(stack_nz);
    free(stack_val_len);
    for (int i=0; i<dim; ++i)
        free(candidates[i]);
    free(candidates);
    return 0;
}
