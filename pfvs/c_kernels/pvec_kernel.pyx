# pvec_kernel.pyx
# Cython wrapper for pvec_kernel

# import C types
# --------------
from libc.stdint cimport int32_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
import numpy as np

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
        long max_N_out,
        long max_N_iter
    )

# Python-exposed wrapper
# ----------------------
def pvec_kernel(B: int,
                int[:, ::1] linmat,
                int linmin,
                long max_N_out,
                long max_N_iter = -1):
    """
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
    The vectors `vec` in the ellipsoid and obeying the extra constraints.
    A status code according to following list:
        0: success
        -6: problem dimension too high (currently >256)
        -5: no vectors
        -2: exceed max_N_out outputs
    """
    # read some inputs
    cdef int dim     = linmat.shape[1]
    cdef int numhyps = linmat.shape[0]
    cdef int N_out = 0
    cdef int status

    # allocate output arrays
    cdef int32_t *c_out = <int32_t *>malloc(max_N_out * dim * sizeof(int32_t))
    if c_out == NULL:
        raise MemoryError("Failed to allocate c_out")

    # ensure linmat is sorted
    linmat_np   = np.asarray(linmat)
    col_l1_norm = np.sum(np.abs(linmat_np), axis=0)
    sort_inds   = np.argsort(col_l1_norm)
    undo_sort_np= np.argsort(sort_inds).astype(np.int32)
    linmat_np   = linmat_np[:, sort_inds]
    linmat_np   = np.ascontiguousarray(linmat_np, dtype=np.int32)
    
    cdef int[:, ::1] linmat_view = linmat_np
    cdef int *linmat_ptr = &linmat_view[0, 0]

    cdef int32_t[::1] undo_sort_view = undo_sort_np

    if max_N_iter == -1:
        max_N_iter = 1000*max_N_out

    # call the C function
    status = _pvec_kernel_c(
        c_out,
        &N_out,
        dim,
        B,
        linmat_ptr,
        linmin,
        numhyps,
        max_N_out,
        max_N_iter
    );

    if N_out == 0:
        free(c_out)
        return np.empty((0, dim), dtype=np.int32), status

    # copy results
    out_sorted = np.empty((N_out, dim), dtype=np.int32)
    cdef int32_t[:, ::1] out_sorted_view = out_sorted
    memcpy(&out_sorted_view[0, 0], c_out, N_out * dim * sizeof(int32_t))

    # unsort
    out = np.empty((N_out, dim), dtype=np.int32)
    cdef int32_t[:, ::1] out_view = out
    cdef int i, j

    for i in range(N_out):
        for j in range(dim):
            out_view[i, undo_sort_view[j]] = out_sorted_view[i, j]

    # free C memory
    free(c_out)

    return out, status
