# =============================================================================
#    Copyright (C) 2026  Liam McAllister Group
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# Cython wrapper for fp_kernel.h: the Fincke-Pohst kernel shared by the coni
# (`conipfv_kernel`) and non-coni (`pfv_kernel`) pipelines.

import math

import numpy as np

from libc.stdint cimport int32_t, int64_t
from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free

# GMP
# ---
cdef extern from "gmp.h":
    ctypedef struct __mpz_struct:
        pass
    ctypedef __mpz_struct mpz_t[1]

    void mpz_init(mpz_t)
    void mpz_clear(mpz_t)
    void mpz_set_si(mpz_t, long)
    void mpz_import(mpz_t, size_t, int, size_t, int, size_t, const void*)
    void mpz_neg(mpz_t, mpz_t)

# the C kernel
# ------------
cdef extern from "fp_kernel.h":
    ctypedef struct fpk_problem:
        int dim
        const int64_t *mat
        int64_t qmax
        int64_t Q
        int nrows
        mpz_t *H
        int strict
        const int64_t *linvec
        int64_t linmin
        long max_N_out
        double eps

    ctypedef struct fpk_output:
        int32_t *pts
        int64_t *qs
        long n
        long cap
        long n_nodes
        long n_cand
        long n_leaf

    int  fpk_enumerate(const fpk_problem *P, fpk_output *out) nogil
    void fpk_output_free(fpk_output *out)


# helpers
# -------
def _exact_mat(U, mat):
    """
    The exact integer ellipsoid matrix. If not given, recover it from its
    Cholesky factor as round(U^T U), checking that U^T U is integral.
    """
    if mat is not None:
        m = np.asarray(mat)
        if m.dtype == object:
            if any(abs(int(x)) >= 2**63 for x in m.ravel()):
                raise OverflowError("mat entries must fit in int64")
        return np.ascontiguousarray(m, dtype=np.int64)
    UU = U.T @ U
    m = np.rint(UU)
    if np.max(np.abs(m)) >= 2.0**53:
        raise OverflowError("cannot recover mat from U exactly; pass mat=")
    tol = 1e-6 * max(1.0, float(np.max(np.abs(UU))))
    if np.max(np.abs(UU - m)) > tol:
        raise ValueError(
            "U^T @ U is not an integer matrix (the kernels only support "
            "integral ellipsoids); pass the exact matrix via mat=")
    return np.ascontiguousarray(m, dtype=np.int64)


def _qmax(Q, dilation):
    """floor(dilation * Q), tolerant of float noise in dilation."""
    return int(math.floor(dilation * Q + 1e-7))


cdef int _load_H(mpz_t *H_gmp, H_obj, int nrows, int dim) except -1:
    cdef int i, j
    for i in range(nrows):
        for j in range(dim):
            val = int(H_obj[i, j])
            if -2**62 < val < 2**62:
                mpz_set_si(H_gmp[i * dim + j], <long> val)
            else:
                # |val| via bytes, then negate
                a = -val if val < 0 else val
                b = a.to_bytes((a.bit_length() + 7) // 8, 'little')
                mpz_import(H_gmp[i * dim + j], len(b), -1, 1, 0, 0,
                           <const void*><char*>b)
                if val < 0:
                    mpz_neg(H_gmp[i * dim + j], H_gmp[i * dim + j])
    return 0


def _enumerate(U, mat, long long Q, long long qmax, H, linvec, long long linmin,
               bint strict, long max_N_out, double eps, int return_n_nodes):
    cdef double[:, ::1] U_c = np.ascontiguousarray(U, dtype=np.float64)
    cdef int dim = U_c.shape[0]
    if U_c.shape[1] != dim:
        raise ValueError("U must be square")
    cdef int64_t[:, ::1] mat_c = _exact_mat(np.asarray(U_c), mat)
    if mat_c.shape[0] != dim or mat_c.shape[1] != dim:
        raise ValueError("mat must have the same shape as U")
    if Q <= 0:
        raise ValueError(f"Q must be positive, got {Q}")
    if max_N_out < 0:
        raise ValueError(f"max_N_out must be >= 0, got {max_N_out}")

    cdef int64_t[::1] lin_c
    cdef const int64_t *lin_ptr = NULL
    if linvec is not None:
        lin_c = np.ascontiguousarray(linvec, dtype=np.int64)
        if lin_c.shape[0] != dim:
            raise ValueError("linvec must have length dim")
        lin_ptr = &lin_c[0]

    H_obj = np.asarray(H, dtype=object)
    if H_obj.ndim != 2 or (H_obj.shape[0] > 0 and H_obj.shape[1] != dim):
        raise ValueError("H must have shape (nrows, dim)")
    cdef int nrows = H_obj.shape[0]
    cdef int k
    cdef mpz_t *H_gmp = NULL
    if nrows > 0:
        H_gmp = <mpz_t *> malloc(nrows * dim * sizeof(mpz_t))
        if H_gmp == NULL:
            raise MemoryError("failed to allocate H")
        for k in range(nrows * dim):
            mpz_init(H_gmp[k])

    cdef fpk_problem P
    cdef fpk_output out
    cdef int32_t[:, ::1] pts_c
    cdef int64_t[::1] qs_c
    cdef int status
    memset(&out, 0, sizeof(out))
    P.dim = dim
    P.mat = &mat_c[0, 0]
    P.qmax = qmax
    P.Q = Q
    P.nrows = nrows
    P.H = H_gmp
    P.strict = strict
    P.linvec = lin_ptr
    P.linmin = linmin
    P.max_N_out = max_N_out
    P.eps = eps

    try:
        if nrows > 0:
            _load_H(H_gmp, H_obj, nrows, dim)
        with nogil:
            status = fpk_enumerate(&P, &out)

        pts = np.empty((out.n, dim), dtype=np.int32)
        qs = np.empty(out.n, dtype=np.int64)
        if out.n > 0:
            pts_c = pts
            qs_c = qs
            memcpy(&pts_c[0, 0], out.pts, out.n * dim * sizeof(int32_t))
            memcpy(&qs_c[0], out.qs, out.n * sizeof(int64_t))
        if return_n_nodes == 2:
            n_nodes = (out.n_nodes, out.n_cand, out.n_leaf)
        else:
            n_nodes = out.n_nodes
    finally:
        fpk_output_free(&out)
        if H_gmp != NULL:
            for k in range(nrows * dim):
                mpz_clear(H_gmp[k])
            free(H_gmp)

    if status == -7:
        raise MemoryError("fp_kernel ran out of memory")
    if status == -9:
        raise ValueError("mat is not positive definite")
    if return_n_nodes:
        return pts, qs, status, n_nodes
    return pts, qs, status


# public API
# ----------
def conipfv_kernel(U,
                   int Q,
                   double dilation,
                   linvec,
                   double linmin,
                   H,
                   long max_N_out,
                   double eps = 1e-12,
                   *,
                   mat = None,
                   return_n_nodes = False):
    """
    Fincke-Pohst enumeration for constructing coni-PFVs. Returns exactly the
    integer vectors ``vec`` with

    - Ellipsoid:  ``vec^T @ mat @ vec <= floor(dilation * Q)``
    - M0 cut:     ``dot(linvec, vec) >= linmin``
    - K' cut:     ``g == 0`` or ``Q * g > vec^T @ mat @ vec``, ``g = gcd(H @ vec)``

    where ``mat = U.T @ U`` is the (integral) ellipsoid matrix and ``H``
    computes ``Kperp`` (up to a unimodular transform, e.g. its row-HNF). Any
    ``vec`` passing all three can generate a coni-PFV (given ``det(N) != 0``).
    All accept/reject decisions are exact. ``U`` is only used to recover
    ``mat`` when ``mat=`` is not given; the kernel factors ``mat`` itself.

    Parameters
    ----------
    U : array-like of shape (dim, dim), dtype float64
        Upper-triangular Cholesky factor: ``mat = U.T @ U``. Only used to
        recover ``mat`` if ``mat=`` is not given.
    Q : int
        Tadpole charge bound (exact equality for coni).
    dilation : float
        Dilation of the ellipsoid (see above).
    linvec : array-like of shape (dim,), int
        First row of ``Binter`` (``dot(linvec, vec) = M0``).
    linmin : float
        Minimum value of ``M0 = dot(linvec, vec)``. Inclusive.
    H : array-like of shape (nrows, dim), int or object (big ints)
        Matrix with ``gcd(H @ vec) = gcd(Kperp)``. Row-echelon form (e.g.
        the row-HNF) gives the earliest pruning.
    max_N_out : int
        Maximum number of output vectors (status -2 if exceeded). Memory is
        allocated as needed, not up front.
    eps : float, optional
        Extra absolute slack for the floating-point pruning. The kernel
        already uses a safe relative slack; results do not depend on it.
    mat : array-like of shape (dim, dim), int, optional (keyword-only)
        The exact ellipsoid matrix. If omitted, recovered as
        ``round(U.T @ U)``, which must be integral with entries < 2^53
        (else pass ``mat=``; ``coniZpM``/``ZpM`` always do).
    return_n_nodes : bool, optional (keyword-only)
        Also return the number of search-tree nodes visited.

    Returns
    -------
    out : ndarray of shape (N, dim), dtype int32
        Lattice points satisfying all constraints, in search order.
    Qs : ndarray of shape (N,), dtype float64
        Exact values ``vec^T @ mat @ vec`` (integers).
    status : int
        0 success, -2 exceeded max_N_out (first max_N_out returned),
        -6 dim > 256, -8 ellipsoid too large for int32 coordinates.
    """
    res = _enumerate(U, mat, Q, _qmax(Q, dilation), H, linvec,
                     math.ceil(linmin - 1e-9), True, max_N_out, eps, return_n_nodes)
    return (res[0], res[1].astype(np.float64)) + tuple(res[2:])


def pfv_kernel(U,
               int Q,
               double dilation,
               H,
               long max_N_out,
               double eps = 1e-12,
               *,
               mat = None,
               return_n_nodes = False):
    """
    Fincke-Pohst enumeration for constructing (non-coni) PFVs. Returns exactly
    the integer vectors ``vec`` with

    - Ellipsoid:  ``vec^T @ mat @ vec <= floor(dilation * Q)``
    - GCD cut:    ``g == 0`` or ``Q * g >= vec^T @ mat @ vec``, ``g = gcd(H @ vec)``

    where ``mat = U.T @ U`` is the (integral) ellipsoid matrix and ``H``
    computes ``K`` (up to a unimodular transform, e.g. its row-HNF, which may
    have more rows than columns). See ``conipfv_kernel`` for the parameters;
    this is the same kernel without the M0 cut and with a non-strict GCD cut.

    Returns
    -------
    out : ndarray of shape (N, dim), dtype int32
    Qs : ndarray of shape (N,), dtype float64
        Exact values ``vec^T @ mat @ vec`` (integers).
    status : int
        As for ``conipfv_kernel``.
    """
    res = _enumerate(U, mat, Q, _qmax(Q, dilation), H, None, 0, False,
                     max_N_out, eps, return_n_nodes)
    return (res[0], res[1].astype(np.float64)) + tuple(res[2:])
