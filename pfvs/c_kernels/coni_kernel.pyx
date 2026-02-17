# coni_kernel.pyx
# Cython wrapper for coni_kernel

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
cdef extern from "coni_kernel.h":
    int _coni_kernel_c(
        int32_t *out,
        float *Qs,
        int *N_out,
        int dim,
        double *U,
        int Q,
        double dilation,
        int *linvec,
        double linmin,
        mpz_t *H,
        long max_N_out,
        double eps
    )

# Python-exposed wrapper
# ----------------------
def coni_kernel(U,
                int Q,
                double dilation,
                linvec,
                double linmin,
                H,
                long max_N_out,
                double eps = 1e-12):
    """
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
    - `U`:         The upper triangular matrix such that mat = U.T@U
    - `Q`:         The ellipsoid bound.
    - `dilation`:  The maximum allowed dilation to allow... As long as
                   gcd(Kperp) >= (vec^T @ mat @ vec)//Q, the vector vec can
                   still define coni-PFV.
    // M0 cuts
    - `linvec`:    Binter[0,:]. The vector such that dot(linvec,vec)=M0.
                   REQUIRED THAT COLUMNS OF BINTER ARE ORDERED S.T. ALL 0s OF
                   linvec ARE LEADING
    - `linmin`:    The minimum value of M0 permitted. Inclusive.
    // Kprime cuts
    - `H`:         Let G be the matrix such that Kperp = G@vec. Then H = HNF(G).
    // misc specs
    - `max_N_out`: The maximum number of output allowed.
    - `eps`:       A small number used for correctly setting bounds despite
                   floating point errors.

    **Returns:**
    The vectors `vec` in the ellipsoid and obeying the extra constraints.
    The valuation `vec^T @ mat @ vec`.
    A status code according to following list:
        0: success
        -6: problem dimension too high (currently >256)
        -4: not all 0s of linvec are leading
        -100: dilation overflows uint32_t
        -5: no vectors
        -2: exceed max_N_out outputs
    """
    # convert inputs to C-contiguous arrays with correct dtype
    # this ensures we always have the right memory layout
    cdef double[:, ::1] U_c = np.ascontiguousarray(U, dtype=np.float64)
    cdef int[::1] linvec_c = np.ascontiguousarray(linvec, dtype=np.int32)

    H_obj = np.ascontiguousarray(H, dtype=object)

    # read some inputs
    cdef int dim = linvec_c.shape[0]
    cdef int N_out = 0
    cdef int status
    cdef int i, j

    # allocate GMP array for H
    cdef mpz_t *H_gmp = <mpz_t *>malloc(dim * dim * sizeof(mpz_t))
    if H_gmp == NULL:
        raise MemoryError("Failed to allocate H_gmp")

    # initialize and set GMP integers
    for i in range(dim):
        for j in range(dim):
            mpz_init(H_gmp[i * dim + j])
            
            # Get Python int
            val = H_obj[i, j]
            
            # Convert to GMP
            # For values that fit in unsigned long
            if abs(val) < 2**63:
                mpz_set_si(H_gmp[i * dim + j], <long> val)
            else:
                # For larger values, convert via bytes
                # Handle sign separately
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
        for i in range(dim * dim):
            mpz_clear(H_gmp[i])
        free(H_gmp)
        raise MemoryError("Failed to allocate c_out")
    cdef float *c_Qs = <float *>malloc(max_N_out * sizeof(int))
    if c_Qs == NULL:
        for i in range(dim * dim):
            mpz_clear(H_gmp[i])
        free(H_gmp)
        free(c_out)
        raise MemoryError("Failed to allocate c_Qs")

    # call the C function
    status = _coni_kernel_c(
        c_out, c_Qs, &N_out, dim, &U_c[0, 0], Q, dilation,
        &linvec_c[0], linmin, H_gmp, max_N_out, eps
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
    for i in range(dim * dim):
        mpz_clear(H_gmp[i])
    free(H_gmp)
    free(c_out)
    free(c_Qs)

    return out, Qs, status
