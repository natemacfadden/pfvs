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
from libc.stdlib cimport malloc, free, realloc

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
        const int64_t *H64
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
    void fpk_output_free(fpk_output *out) nogil


cdef extern from "pfv_lattice.h":
    ctypedef long long pfl_i128_t "pfl_i128"
    ctypedef struct pfl_setup:
        int h11
        const int64_t *kappa
        const int64_t *Mbasis
        int coni
        int extra_lll
        int m0_basis

    ctypedef struct pfl_result:
        int64_t Z[64 * 64]
        int64_t Binter[64 * 63]
        int64_t ZB[64 * 63]
        int64_t mat[63 * 63]
        pfl_i128_t H[64 * 63]
        int nrows

    int pfl_build(const pfl_setup *S, const int64_t *p, pfl_result *R) nogil


cdef extern from *:
    """
    #define INT64_MAX_128 ((pfl_i128)INT64_MAX)
    """
    pfl_i128_t INT64_MAX_128


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
    P.H64 = NULL
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


# lattice setup (pfv_lattice.h)
# -----------------------------
def _lattice_build(kappa, Mbasis, p, bint coni, bint extra_lll=True, bint m0_basis=True):
    """
    The per-p lattice setup in C (see pfv_lattice.h). Returns
    (status, Z, Binter, ZB, mat, H). status 0: all valid. status -15: all but
    H valid (H is None; compute the HNF of ZB[r0:] instead). Any other
    nonzero status: use the exact Python path (arrays are None).
    """
    cdef int64_t[::1] k_c = np.ascontiguousarray(kappa, dtype=np.int64).reshape(-1)
    cdef int64_t[:, ::1] M_c = np.ascontiguousarray(Mbasis, dtype=np.int64)
    cdef int64_t[::1] p_c = np.ascontiguousarray(p, dtype=np.int64)
    cdef int h = M_c.shape[0]
    if h > 64 or p_c.shape[0] != h or k_c.shape[0] != h * h * h:
        raise ValueError("inconsistent shapes (or h11 > 64)")
    cdef pfl_setup S
    S.h11 = h
    S.kappa = &k_c[0]
    S.Mbasis = &M_c[0, 0]
    S.coni = coni
    S.extra_lll = extra_lll
    S.m0_basis = m0_basis
    cdef pfl_result *R = <pfl_result *> malloc(sizeof(pfl_result))
    if R == NULL:
        raise MemoryError()
    cdef int st
    try:
        with nogil:
            st = pfl_build(&S, &p_c[0], R)
        if st != 0 and st != -15:
            return st, None, None, None, None, None
        d = h - 1
        Z = np.array(<int64_t[:h * h]> R.Z).reshape(h, h)
        B = np.array(<int64_t[:h * d]> R.Binter).reshape(h, d)
        ZB = np.array(<int64_t[:h * d]> R.ZB).reshape(h, d)
        mat = np.array(<int64_t[:d * d]> R.mat).reshape(d, d)
        if st == -15:      # all but H valid: caller computes the HNF
            return st, Z, B, ZB, mat, None
        # H is int128 in C: two int64 words per entry (little-endian)
        w = np.array(<int64_t[:2 * R.nrows * d]> <int64_t *> &R.H[0]).reshape(-1, 2)
        lo = w[:, 0].view(np.uint64).astype(object)
        hi = w[:, 1].astype(object)
        Hv = (hi * (1 << 64) + lo)
        H = Hv.reshape(R.nrows, d)
        if all(-2**63 < int(x) < 2**63 for x in Hv):
            H = H.astype(np.int64)
        return 0, Z, B, ZB, mat, H
    finally:
        free(R)


# batched coni pipeline: lattice setup + kernel, no Python per p-vector
# --------------------------------------------------------------------
def _coni_batch(kappa, Mbasis, ps, long long Q, double dilation, double M0min,
                long max_N_out, bint extra_lll=True, double eps=1e-4,
                bint m0_basis=True):
    """
    For each coni p-vector (rows of ps, full length h11 with p[0] = 0): build
    the lattice data (pfv_lattice.h) and run the kernel, returning for every
    lattice point c found

        M = Binter c,   Kn = (Z Binter) c,   q = c^T mat c,   pidx

    (everything coniZpM's post-processing needs), plus a per-p status:
        0: done (its points are included)
        1: not handled here (lattice setup overflowed, H or M/Kn beyond
           int64): process this p with the exact per-p path
       <0: the kernel's status (e.g. -2 too many outputs, -8 coordinates
           beyond int32, -9 not positive definite)
    """
    cdef int64_t[::1] k_c = np.ascontiguousarray(kappa, dtype=np.int64).reshape(-1)
    cdef int64_t[:, ::1] M_c = np.ascontiguousarray(Mbasis, dtype=np.int64)
    cdef int64_t[:, ::1] p_c = np.ascontiguousarray(ps, dtype=np.int64).reshape(-1, M_c.shape[0])
    cdef int h = M_c.shape[0]
    cdef int d = h - 1
    cdef Py_ssize_t n = p_c.shape[0]
    if h > 64 or h < 2 or k_c.shape[0] != h * h * h:
        raise ValueError("inconsistent shapes (or h11 > 64)")
    status_np = np.zeros(n, dtype=np.int32)
    cdef int[::1] status = status_np
    if n == 0:
        z = np.empty((0, h), dtype=np.int64)
        return z, z.copy(), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), status_np

    cdef pfl_setup S
    S.h11 = h
    S.kappa = &k_c[0]
    S.Mbasis = &M_c[0, 0]
    S.coni = 1
    S.extra_lll = extra_lll
    S.m0_basis = m0_basis
    cdef fpk_problem P
    memset(&P, 0, sizeof(P))
    P.dim = d
    P.qmax = _qmax(Q, dilation)
    P.Q = Q
    P.strict = 1
    P.linmin = math.ceil(M0min - 1e-9)
    P.max_N_out = max_N_out
    P.eps = eps

    cdef pfl_result *R = <pfl_result *> malloc(sizeof(pfl_result))
    cdef int64_t *H64 = <int64_t *> malloc(h * d * sizeof(int64_t))
    cdef int64_t *Mb = NULL
    cdef int64_t *Kb = NULL
    cdef int64_t *qb = NULL
    cdef int64_t *pb = NULL
    cdef Py_ssize_t cnt = 0, cap = 0, start, t, newcap
    cdef Py_ssize_t ip
    cdef int st, a, i, bad, k
    cdef fpk_output out
    cdef pfl_i128_t acc_m, acc_k, hv
    cdef void *tmp
    if R == NULL or H64 == NULL:
        free(R); free(H64)
        raise MemoryError()
    try:
        with nogil:
            for ip in range(n):
                st = pfl_build(&S, &p_c[ip, 0], R)
                if st != 0:
                    status[ip] = 1
                    continue
                bad = 0
                for k in range(R.nrows * d):
                    hv = R.H[k]
                    if hv > INT64_MAX_128 or hv < -INT64_MAX_128:
                        bad = 1
                        break
                    H64[k] = <int64_t> hv
                if bad:
                    status[ip] = 1
                    continue
                P.mat = &R.mat[0]
                P.nrows = R.nrows
                P.H64 = H64
                P.linvec = &R.Binter[0]          # row 0 of Binter: M0 = linvec . c
                memset(&out, 0, sizeof(out))
                st = fpk_enumerate(&P, &out)
                if st != 0:
                    status[ip] = st
                    fpk_output_free(&out)
                    continue
                if cnt + out.n > cap:
                    newcap = 2 * cap if 2 * cap > cnt + out.n else cnt + out.n + 64
                    tmp = realloc(Mb, newcap * h * sizeof(int64_t))
                    if tmp == NULL:
                        status[ip] = -7; fpk_output_free(&out); continue
                    Mb = <int64_t *> tmp
                    tmp = realloc(Kb, newcap * h * sizeof(int64_t))
                    if tmp == NULL:
                        status[ip] = -7; fpk_output_free(&out); continue
                    Kb = <int64_t *> tmp
                    tmp = realloc(qb, newcap * sizeof(int64_t))
                    if tmp == NULL:
                        status[ip] = -7; fpk_output_free(&out); continue
                    qb = <int64_t *> tmp
                    tmp = realloc(pb, newcap * sizeof(int64_t))
                    if tmp == NULL:
                        status[ip] = -7; fpk_output_free(&out); continue
                    pb = <int64_t *> tmp
                    cap = newcap
                start = cnt
                bad = 0
                for t in range(out.n):
                    for i in range(h):
                        acc_m = 0
                        acc_k = 0
                        for a in range(d):
                            acc_m = acc_m + <pfl_i128_t> R.Binter[i * d + a] * out.pts[t * d + a]
                            acc_k = acc_k + <pfl_i128_t> R.ZB[i * d + a] * out.pts[t * d + a]
                        if (acc_m > INT64_MAX_128 or acc_m < -INT64_MAX_128 or
                                acc_k > INT64_MAX_128 or acc_k < -INT64_MAX_128):
                            bad = 1
                        Mb[cnt * h + i] = <int64_t> acc_m
                        Kb[cnt * h + i] = <int64_t> acc_k
                    qb[cnt] = out.qs[t]
                    pb[cnt] = ip
                    cnt += 1
                fpk_output_free(&out)
                if bad:                          # hand this p to the exact path
                    cnt = start
                    status[ip] = 1

        Mv = np.empty((cnt, h), dtype=np.int64)
        Kv = np.empty((cnt, h), dtype=np.int64)
        qv = np.empty(cnt, dtype=np.int64)
        pv = np.empty(cnt, dtype=np.int64)
        if cnt:
            _copy_i64(Mv.reshape(-1), Mb, cnt * h)
            _copy_i64(Kv.reshape(-1), Kb, cnt * h)
            _copy_i64(qv, qb, cnt)
            _copy_i64(pv, pb, cnt)
        return Mv, Kv, qv, pv, status_np
    finally:
        free(R); free(H64); free(Mb); free(Kb); free(qb); free(pb)


cdef void _copy_i64(int64_t[::1] dst, const int64_t *src, Py_ssize_t n):
    memcpy(&dst[0], src, n * sizeof(int64_t))
