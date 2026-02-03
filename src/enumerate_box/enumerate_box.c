#include "enumerate_box.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int enumerate_box_c(
    const int32_t * restrict bounds,
    int dim,
    int max_N_out,
    int32_t * restrict out,
    int * restrict N_out)
{
    // allocate memory for stack variables
    int32_t *vec       = calloc(dim, sizeof(int32_t));
    int32_t *stack_pos = calloc(dim, sizeof(int32_t));
    int32_t *stack_len = calloc(dim, sizeof(int32_t));

    if (!vec || !stack_pos || !stack_len)
        return -1;

    // set # of components per dimension
    for (int i = 0; i < dim; ++i)
        stack_len[i] = 2 * bounds[i] + 1;

    // define candidate array
    int **candidates = malloc(dim * sizeof(int*));
    for (int i = 0; i < dim; ++i) {
        candidates[i] = malloc(stack_len[i] * sizeof(int32_t));
        for (int k = 0; k < stack_len[i]; ++k)
            candidates[i][k] = -bounds[i] + k;
    }

    // iterate over the stack
    int stack_i = 0;
    int op = 0;

    while (stack_i >= 0) {
        // check if we exhausted values for this component
        if (stack_pos[stack_i] == stack_len[stack_i]) {
            stack_i--;
            if (stack_i >= 0)
                // increment position of prior element
                stack_pos[stack_i]++;
            continue;
        }

        // set vec[stack_i]
        vec[stack_i] = candidates[stack_i][stack_pos[stack_i]];

        // check if done
        if (stack_i == dim - 1) {
            if (op >= max_N_out)
                return -2;

            //for (int j = 0; j < dim; ++j)
            //    out[op * dim + j] = vec[j];
            memcpy(&out[op * dim], vec, dim * sizeof(int32_t));

            op++;
            stack_pos[stack_i]++;
        } else {
            // not done -move on to next component
            stack_i++;
            stack_pos[stack_i] = 0;
        }
    }

    *N_out = op;

    // free variabls
    free(vec);
    free(stack_pos);
    free(stack_len);
    return 0;
}
