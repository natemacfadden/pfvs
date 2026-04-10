# pfv_kernel.pyx
# Cython wrapper for pfv_kernel

# import C types
# --------------
from libc.stdint cimport int32_t
from libc.stdlib cimport malloc, free
import numpy as np

# declare GMP types
cdef extern from "gmp.h":
    ctypedef struct __mpz_struct:
        pass
    ctypedef __mpz_struct mpz_t[1]

    void mpz_init(mpz_t)
    void mpz_clear(mpz_t)
    void mpz_set_si(mpz_t, long)
    void mpz_import(mpz_t, size_t, int, size_t, int, size_t, const void*)
    void mpz_neg(mpz_t, mpz_t)

# declare the external C function
# -------------------------------
cdef extern from "pfv_kernel.h":
    int _pfv_kernel_c(
        int32_t *out,
        float *Qs,
        int *N_out,
        int dim,
        double *U,
        int Q,
        double dilation,
        mpz_t *H,
        long max_N_out,
        double eps
    )

# Python-exposed wrapper
# ----------------------
def pfv_kernel(U,
               int Q,
               double dilation,
               H,
               long max_N_out,
               double eps = 1e-12):
    """
    Adaptation of the (iterative) Fincke-Pohst algorithm for constructing
    PFVs. Finds integer vectors ``vec`` satisfying:

    - Ellipsoid: ``0 <= vec^T @ (U.T@U) @ vec <= dilation * Q``
    - GCD cut:   ``(vec^T @ (U.T@U) @ vec) // Q <= gcd(H @ vec)``

    where ``H`` is the row-HNF of the matrix ``G`` such that ``K = G @ vec``.
    Any ``vec`` passing both cuts can generate a PFV (provided ``det(N) != 0``).

    Parameters
    ----------
    U : array-like of shape (dim, dim), dtype float64
        Upper-triangular Cholesky factor: ``mat = U.T @ U``.
    Q : int
        Tadpole charge bound.
    dilation : float
        Maximum dilation factor. Vectors with
        ``vec^T @ mat @ vec <= dilation * Q`` are accepted.
    H : array-like of shape (dim+1, dim), dtype int64
        Row-HNF of the matrix mapping ``vec`` to ``K``. Has a zero row at
        row 0; the upper-triangular non-zero block occupies rows 1..dim.
    max_N_out : long
        Maximum number of output vectors allowed.
    eps : float, optional
        Small tolerance for floating-point bound corrections. Default 1e-12.

    Returns
    -------
    out : ndarray of shape (N, dim), dtype int32
        Lattice points satisfying all constraints.
    Qs : ndarray of shape (N,), dtype float
        Valuations ``vec^T @ mat @ vec`` for each output vector.
    status : int
        Status code:
             0: success
            -6: problem dimension too high (currently >256)
          -100: dilation overflows uint32_t
            -5: no vectors found
            -2: exceeded max_N_out outputs
    """
    # convert inputs to C-contiguous arrays with correct dtype
    cdef double[:, ::1] U_c = np.ascontiguousarray(U, dtype=np.float64)

    H_obj = np.ascontiguousarray(H, dtype=object)

    # read some inputs
    # dim from U (square); H has dim+1 rows and dim cols
    cdef int dim = U_c.shape[0]
    cdef int h_rows = dim + 1
    cdef int N_out = 0
    cdef int status
    cdef int i, j

    # allocate GMP array for H: (dim+1) x dim elements
    cdef mpz_t *H_gmp = <mpz_t *>malloc(h_rows * dim * sizeof(mpz_t))
    if H_gmp == NULL:
        raise MemoryError("Failed to allocate H_gmp")

    # initialize and set GMP integers
    for i in range(h_rows):
        for j in range(dim):
            mpz_init(H_gmp[i * dim + j])

            # Get Python int
            val = H_obj[i, j]

            # Convert to GMP
            if abs(val) < 2**63:
                mpz_set_si(H_gmp[i * dim + j], <long> val)
            else:
                # For larger values, convert via bytes
                is_negative = (val < 0)
                abs_val = abs(val)

                val_bytes = abs_val.to_bytes((abs_val.bit_length() + 7) // 8, 'little')
                mpz_import(H_gmp[i * dim + j], len(val_bytes), -1, 1, 0, 0,
                          <const void*><char*>val_bytes)

                if is_negative:
                    mpz_neg(H_gmp[i * dim + j], H_gmp[i * dim + j])

    # allocate output arrays
    cdef int32_t *c_out = <int32_t *>malloc(max_N_out * dim * sizeof(int32_t))
    if c_out == NULL:
        for i in range(h_rows * dim):
            mpz_clear(H_gmp[i])
        free(H_gmp)
        raise MemoryError("Failed to allocate c_out")
    cdef float *c_Qs = <float *>malloc(max_N_out * sizeof(float))
    if c_Qs == NULL:
        for i in range(h_rows * dim):
            mpz_clear(H_gmp[i])
        free(H_gmp)
        free(c_out)
        raise MemoryError("Failed to allocate c_Qs")

    # call the C function
    status = _pfv_kernel_c(
        c_out, c_Qs, &N_out, dim, &U_c[0, 0], Q, dilation,
        H_gmp, max_N_out, eps
    )

    # convert outputs to Python arrays
    out = np.empty((N_out, dim), dtype=np.int32)
    Qs  = np.empty(N_out, dtype=float)

    # copy results
    for i in range(N_out):
        for j in range(dim):
            out[i, j] = c_out[i*dim + j]
        Qs[i] = c_Qs[i]

    # free C memory
    for i in range(h_rows * dim):
        mpz_clear(H_gmp[i])
    free(H_gmp)
    free(c_out)
    free(c_Qs)

    return out, Qs, status
