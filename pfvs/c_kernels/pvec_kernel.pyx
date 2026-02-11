# coni_kernel.pyx
# Cython wrapper for coni_kernel

# import C types
# --------------
from libc.stdint cimport int32_t, uint32_t
from libc.stdlib cimport malloc, free

# declare the external C function
# -------------------------------
cdef extern from "pvec_kernel.h":
    int _pvec_kernel_c(
        int32_t * out,
        int * N_out,
        int dim,
        int B,
        int * linmat,
        int linmin,
        int numhyps,
        int max_N_out,
        int max_N_iter,
        double eps
    )

# --- Python-exposed wrapper ---
def pvec_kernel(B: int,
                int[:] linmat,
                int linmin,
                int max_N_out,
                int max_N_iter = -1,
                double eps = 1e-12):
    import numpy as np

    cdef int numhyps = linmat.shape[0]
    cdef int dim = linmat.shape[1]
    cdef int status

    # Allocate output arrays
    cdef int32_t *c_out = <int32_t *>malloc(max_N_out * dim * sizeof(int32_t))
    if c_out == NULL:
        raise MemoryError("Failed to allocate c_out")

    # ensure linmat is sorted
    col_l1_norm = np.sum(np.abs(linmat), axis=0)
    sort_inds   = np.argsort(col_l1_norm)
    undo_sort   = np.argsort(sort_inds)

    linmat = np.transpose(linmat,sort_inds).ravel()

    if max_N_iter == -1:
        max_N_iter = 1000*max_N_out

    # call the C function
    status = _pvec_kernel_c(
        c_out
        &N_out,
        dim,
        B,
        &linmat[0],
        linmin,
        numhyps,
        max_N_iter,
        max_N_out,
        eps
    );

    # convert outputs to Python arrays
    import numpy as np
    out = np.empty((N_out, dim), dtype=np.int32)

    # copy results
    for i in range(N_out):
        for j in range(dim):
            out[i, j] = c_out[i*dim + j]

    # free C memory
    free(c_out)

    return out, status
