# coni_kernel.pyx
# Cython wrapper for coni_kernel

# --- Import C types ---
from libc.stdint cimport int32_t
from libc.stdlib cimport malloc, free

# --- Declare the external C function ---
cdef extern from "coni_kernel.h":
    int _coni_kernel_c(
        int32_t *out,       # shape [max_N_out, dim]
        double *Qs,         # shape [max_N_out]
        int *N_out,         # number of rows written
        int max_N_out,
        int dim,
        double *U,
        int Q,
        double dilation,
        int *linvec,
        double linmin,
        int *H,
        double eps
    )

# --- Python-exposed wrapper ---
def coni_kernel(double[:] U,
                int Q,
                double dilation,
                int[:] linvec,
                double linmin,
                int[:] H,
                int max_N_out,
                double eps = 1e-12):
    """
    Python wrapper for coni_kernel.
    U: square 1D array of length dim*dim (row-major)
    linvec: 1D int array of length dim
    H: 1D int array of length dim*dim (row-major)
    Returns:
        out: 2D int array [N_out, dim]
        Qs: 1D double array [N_out]
        N_out: int
        status: int (return code from C function)
    """
    cdef int dim = linvec.shape[0]
    cdef int N_out = 0
    cdef int status

    # Allocate output arrays
    cdef int32_t *c_out = <int32_t *>malloc(max_N_out * dim * sizeof(int32_t))
    if c_out == NULL:
        raise MemoryError("Failed to allocate c_out")
    cdef double *c_Qs = <double *>malloc(max_N_out * sizeof(double))
    if c_Qs == NULL:
        free(c_out)
        raise MemoryError("Failed to allocate c_Qs")

    # Call the C function
    status = _coni_kernel_c(
        c_out,
        c_Qs,
        &N_out,
        max_N_out,
        dim,
        &U[0],
        Q,
        dilation,
        &linvec[0],
        linmin,
        &H[0],
        eps
    )

    # Convert outputs to Python arrays
    import numpy as np
    out = np.empty((N_out, dim), dtype=np.int32)
    Qs = np.empty(N_out, dtype=np.float64)

    # Copy results
    for i in range(N_out):
        for j in range(dim):
            out[i, j] = c_out[i*dim + j]
        Qs[i] = c_Qs[i]

    # Free C memory
    free(c_out)
    free(c_Qs)

    return out, Qs, N_out, status
