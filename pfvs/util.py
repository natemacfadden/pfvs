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
# -----------------------------------------------------------------------------
# Description:  This module contains lattice-utilities for PFV construction.
# -----------------------------------------------------------------------------

# external imports
import flint
import functools
import math
from numba import njit
import numpy as np
import scipy as sp

from numpy.typing import ArrayLike

# basic helpers
# =============
def lcm(a: int, b: int) -> int:
    """Least common multiple of integers a and b."""
    return abs(a*b) // math.gcd(a, b)

# misc lattice
# ============
# LLL-reduction
def lll_reduce(B: ArrayLike) -> np.ndarray:
    """
    Apply LLL-reduction to the input matrix, representing a *columnwise* basis
    of some lattice.

    N.B.: Flint's LLL-transformation works on row-bases. This is because, for an
    integral matrix M, it solves for
        - a unimodular T and
        - an integral L
    obeying T@M = L. I.e., L[i,:] = T[i,k] M[k,:] so the *rows* of L are
    integral combinations of the *rows* of M.

    This is why we transpose.

    Parameters
    ----------
    B : ArrayLike
        A basis of a lattice, as column vectors.

    Returns
    -------
    ArrayLike
        The reduced basis, also as column vectors.
    """
    # transpose since Flint assumes a row-basis
    B_list = np.array(B.T).tolist()

    # lll-reduction
    # given input B, this solves for a T,L such that T@B = L
    # with T unimodular
    B = flint.fmpz_mat(B_list).lll(transform=False)

    # convert to numpy and transpose back to a column-basis
    B = np.array(B.tolist(), dtype=int).T
    return B

# orthogonal lattice
@njit
def extended_euclidean(a: int, b: int) -> tuple[int, int, int]:
    """
    Extended Euclidean algorithm. Computes GCD of a and b, as well as Bezout
    coefficients s, t such that s*a + t*b = gcd(a, b).

    Parameters
    ----------
    a, b : int

    Returns
    -------
    s, t, gcd : int
        Bezout coefficients and gcd, satisfying s*a + t*b = gcd(a, b).
    """
    old_r, r = (a,b)
    old_s, s = (1,0)
    old_t, t = (0,1)

    while r!=0:
        q = old_r//r

        old_r, r = (r, old_r - q*r)
        old_s, s = (s, old_s - q*s)
        old_t, t = (t, old_t - q*t)

    return old_s, old_t, old_r

@njit
def orthogonal_lattice(p: ArrayLike) -> np.ndarray:
    """
    Computes a basis of the lattice orthogonal to p via iterated Bezout
    reduction. Columns are basis vectors.

    Parameters
    ----------
    p : ArrayLike
        The orthogonal vector. Assumed to be integral.

    Returns
    -------
    ArrayLike
        A basis of the lattice orthogonal to p, as column vectors.
    """
    n = p.shape[0]
    U = np.eye(n, dtype=p.dtype)

    w = p.copy()
    # Invariant: after step k, w[0] = gcd(p[0],...,p[k]) and w[1:k] = 0.
    # U is kept unimodular throughout, satisfying U @ p_original = w.
    # At the end, w = (gcd(p), 0, ..., 0), so U[1:] @ p_original = 0,
    # meaning the rows of U[1:] span the orthogonal complement.
    for k in range(1,n):
        # w[k] is already zero, cleared by a prior step
        if w[k] == 0:
            continue

        # Use Bezout to find a unimodular 2x2 matrix M such that
        # M @ (w[0], w[k]) = (gcd(w[0], w[k]), 0).
        # This replaces (w[0], w[k]) with (gcd, 0) while preserving
        # the unimodularity of U.
        a,b   = w[0], w[k]
        s,t,g = extended_euclidean(a,b)

        M = [[s,t],[-b//g, a//g]]
        # det(M) = s*(a//g) + t*(b//g) = +/- 1  (Bezout identity)
        # M @ (a, b) = (s*a + t*b, 0) = (g, 0)

        # update w
        w[0] = g
        w[k] = 0

        # Apply M to rows 0 and k of U (i.e., U <- M_extended @ U),
        # maintaining U @ p_original = w.
        for r in range(n):
            tmp1   = M[0][0]*U[0,r] + M[0][1]*U[k,r]
            tmp2   = M[1][0]*U[0,r] + M[1][1]*U[k,r]
            U[0,r] = tmp1
            U[k,r] = tmp2

    # U[0] satisfies U[0] @ p = gcd(p); U[1:] satisfies U[1:] @ p = 0.
    # Return U[1:].T so that the orthogonal basis vectors are columns.
    return U[1:].T

# dual lattice
def dual_lattice(B: ArrayLike) -> tuple[np.ndarray, int]:
    """
    Computes a basis of the lattice dual to L(B).

    Uses the convention that basis vectors are the **columns** of B.
    See https://en.wikipedia.org/wiki/Dual_lattice

    Parameters
    ----------
    B : ArrayLike
        A basis of the primal lattice, as column vectors.

    Returns
    -------
    D : np.ndarray
        A basis of the dual lattice, as column vectors (numerator).
    denom : int
        The denominator, so the true dual basis is D / denom.
    """
    B = flint.fmpz_mat(B.tolist())
    D, denom = (B*( (B.transpose()*B).inv() )).numer_denom()

    return np.array(D.tolist()).astype(int), int(denom)

# integer 'inverse' of matrix (i.e., adjugate)
def inv_scaled(
    A_in: ArrayLike,
    as_flint: bool = False
) -> tuple[np.ndarray, int]:
    """
    Compute a scaled integer inverse of A, i.e. (B, s) such that B @ A = s*I.

    The minimal integer scaling factor s is the lcm of the Smith normal form
    (SNF) diagonal entries d_1, ..., d_n. To see why: write A = U @ D @ V with
    U, V unimodular and D = diag(d_1,...,d_n). Then

        A^{-1} = V^{-1} @ D^{-1} @ U^{-1}

    and D^{-1} = diag(1/d_1,...,1/d_n). Multiplying by s = lcm(d_i) clears
    all denominators, so s*A^{-1} = s * V^{-1} D^{-1} U^{-1} is integral.
    No smaller integer works because d_i | s is required for each i.

    B is found by solving A @ x = s*e_i column-by-column via exact integer
    linear solves.

    Parameters
    ----------
    A_in : ArrayLike
        A square integer matrix. Must be nonsingular over Q.
    as_flint : bool, optional
        If True, return B as a flint.fmpz_mat instead of a numpy array.

    Returns
    -------
    B : np.ndarray or flint.fmpz_mat
        Integer matrix satisfying B @ A = s * I.
    s : int
        The scaling factor (lcm of the Smith normal form diagonal entries).

    Raises
    ------
    ValueError
        If A_in is singular.
    """
    dim = A_in.shape[0]
    A   = flint.fmpz_mat(A_in.tolist())
    n   = A.nrows()

    # Smith normal form diagonal
    D = A.snf()

    diag = [int(D[i, i]) for i in range(n)]
    if any(d == 0 for d in diag):
        raise ValueError("Matrix is singular over Q")

    # Scaling factor
    s = functools.reduce(lcm, diag, 1)

    # Build scaled inverse column-by-column
    Ainv = flint.fmpz_mat(n, n)
    for i in range(n):
        rhs = flint.fmpz_mat(n, 1)
        rhs[i, 0] = s
        x = A.solve(rhs)   # exact integer solve
        for j, xj in enumerate(x):
            Ainv[j, i] = int(xj)

    # test the inverse
    test = A*Ainv
    for i in range(dim):
        for j in range(dim):
            if i==j:
                assert test[i,j]==s
            else:
                assert test[i,j]==0

    # return
    if as_flint:
        return Ainv, s
    else:
        # cast to numpy...
        Ainv_list = [[np.int64(Ainv[i, j]) for j in range(n)] for i in range(n)]
        return np.array(Ainv_list), s


# NJIT branch and bound algorithms
# (not recommended... left as reference)
# ======================================
# Fincke-Pohst (FP)
# -----------------
@njit
def fp_iterative_njit(
        L: ArrayLike,
        Q: float,
        linvec: ArrayLike | None = None,
        linmin: int | None = None,
        max_N_out: int = 10_000_000,
        eps: float = 1e-4,
        COORD_BUFF_SIZE: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """
    Enumerate all nonzero integer vectors vec such that
        0 <= vec^T @ mat @ vec <= Q.
    Also allows imposing that dot(linvec, vec) >= linmin.

    The 'Fincke-Pohst' algorithm (FP) from
        Improved Methods for Calculating Vectors of Short Length in a Lattice,
        Including a Complexity Analysis by Fincke, Pohst
    can be viewed as doing exactly this.

    Roughly, FP operates via:
        1) Cholesky-decompose mat = L@L^T for L.T upper triangular
        2) define c = L^T@vec, so the quadratic form becomes 0 <= |c|^2 <= Q
        3) observe that, since L^T is upper triangular, c[i] depends only on
           vec[i:]. E.g., c[0] depends on vec[0], ..., vec[dim-1]
                          c[1] depends on vec[1], ..., vec[dim-1]
                          c[dim-1] depends on vec[dim-1]
        4) fix vec[dim-1], which reduces the norm-bound on c from Q to
           Q-c[dim-1]^2 and effectively reduces the dimension of the problem,
           at the cost of adding a shift-vector to c[:dim-1]
        5) recurse

    This is an iterative (DFS) implementation using an explicit stack.

    Parameters
    ----------
    L : ArrayLike
        Lower triangular matrix such that mat = L @ L.T.
    Q : float
        The ellipsoid bound.
    linvec : ArrayLike, optional
        Linear constraint vector. If provided, only vectors with
        dot(linvec, vec) >= linmin are returned.
    linmin : int, optional
        Minimum value for the linear constraint.
    max_N_out : int, optional
        Maximum number of output vectors allowed.
    eps : float, optional
        Small tolerance for floating-point bound computations.
    COORD_BUFF_SIZE : int, optional
        Size of the per-depth candidate value buffer.

    Returns
    -------
    out : np.ndarray, shape (N, dim)
        Vectors in the ellipsoid.
    Qs : np.ndarray, shape (N,)
        Quadratic form value vec^T @ mat @ vec for each output vector.
    """
    dim        = L.shape[0]
    L_diag_inv = 1.0 / np.diag(L)

    # linear constraint
    if linvec is None:
        linvec    = np.zeros(dim)
        linmin    = 0
        num_zeros = -1
    else:
        num_zeros = 0
        zeros     = True
        for i in range(dim):
            if linvec[i] == 0:
                if zeros == False:
                    raise ValueError("linvec is not sorted so 0s are first...")
                num_zeros += 1
            else:
                zeros = False

    # output object
    # -------------
    out = np.empty((max_N_out, dim), dtype=np.int64)
    Qs  = np.empty((max_N_out,), dtype=np.float32)

    # output pointer
    op  = 0

    # internal vector that gets built/written to output
    vec = np.zeros(dim, dtype=np.int64)

    # stack variables
    # ---------------
    # stack pointer
    sp = 0

    # max stack depth
    MAX_DEPTH = dim

    # stack arrays: i, pos, remaining_Q, nonzero, candidate values
    stack_i      = np.empty(MAX_DEPTH, np.int64)
    stack_pos    = np.empty(MAX_DEPTH, np.int64)
    stack_remQ   = np.empty(MAX_DEPTH, np.float64)
    stack_nz     = np.zeros(MAX_DEPTH, np.bool_)

    # vec[i] candidate arrays per depth (preallocate maximum possible size)
    stack_val_len= np.zeros(MAX_DEPTH, np.int64) # number of candidates
    stack_vals   = np.empty((MAX_DEPTH, COORD_BUFF_SIZE), np.int64) # candidates

    # offsets for ci
    # c[i] = L[i,i]*vec[i] + sum_{j>i} L[j,i]*vec[j]
    ci_offsets = np.zeros(dim, dtype=np.float64)

    # initialize stack
    # ----------------
    stack_i[sp]    = dim-1
    stack_pos[sp]  = 0
    stack_remQ[sp] = Q
    stack_nz[sp]   = False

    stack_val_len[sp] = -1  # will fill below
    #stack_vals unset here

    # process stack until empty
    # -------------------------
    while sp >= 0:
        # read values
        i    = stack_i[sp]
        pos  = stack_pos[sp]
        remQ = stack_remQ[sp]
        nz   = stack_nz[sp]

        # check if node is completed
        # --------------------------
        # if i==-1, then we have fully written vec
        if i == -1:
            if nz:
                if op >= max_N_out:
                    break
                out[op, :] = vec
                Qs[op]      = Q - remQ
                op += 1
            # kill node
            sp -= 1
            continue

        # check if current depth is completed
        # -----------------------------------
        if pos == stack_val_len[sp]:
            # kill node
            sp -= 1
            for k in range(i):
                ci_offsets[k] -= L[i,k] * vec[i]
            continue

        # current depth incomplete...
        # ---------------------------
        # set candidate values of vec[i] if first time to depth
        if stack_val_len[sp] == -1:
            # feasible integer bounds for vec[i]
            # -R                      <= c[i]          <= R
            # -R - ci_offset          <= L[i,i]*vec[i] <= R - ci_offset
            # (-R - ci_offset)/L[i,i] <= vec[i]        <= (R - ci_offset)/L[i,i]
            # where we used that the diagonal is positive
            if remQ<0:
                remQ = 0
            R  = np.sqrt(remQ)
            lo = int(np.ceil(( -R - ci_offsets[i]) * L_diag_inv[i] - eps))
            hi = int(np.floor(( R - ci_offsets[i]) * L_diag_inv[i] + eps))

            # values of veci to iterate over
            k = 0
            for v in range(lo,hi+1):
                stack_vals[sp,k] = v
                k += 1

            # kill node if no valid veci values
            if k == 0:
                sp -= 1
                continue
            # kill execution if there are too many values
            elif k>COORD_BUFF_SIZE:
                msg = f"Assumed |hi-lo| <= {COORD_BUFF_SIZE}, but got {k}"
                raise ValueError(msg)

            # yes valid veci values
            stack_val_len[sp] = k
            stack_pos[sp] = 0
            pos = 0

            for k in range(i):
                ci_offsets[k] += L[i,k] * (stack_vals[sp, pos]-1)

        # pick candidate veci
        # -------------------
        veci   = stack_vals[sp, pos]
        vec[i] = veci

        # advance pos for next iteration
        stack_pos[sp] += 1

        # update ci_offsets for descendents
        for k in range(i):
            ci_offsets[k] += L[i,k]# * 1

        # get ci, the new amount of remaining Q
        ci      = L[i,i]*veci + ci_offsets[i]
        new_rem = remQ - ci*ci

        # cut of no more Q left...
        if new_rem < 0 - eps:
            continue

        # cut if dot product violates bounds
        if i == num_zeros:
            val = 0
            for j in range(i,dim):
                val += linvec[j]*vec[j]

            if val < linmin:
                continue

        # passes cuts -> push next depth :)
        sp += 1
        stack_i[sp]       = i-1
        stack_pos[sp]     = 0
        stack_remQ[sp]    = new_rem
        stack_nz[sp]      = nz or (veci != 0)
        stack_val_len[sp] = -1  # will fill when we visit
        # candidate array for this depth is stack_vals[sp,:]
        # else do not push (prune)

    return out[:op, :], Qs[:op]

# FP-style methods but tailored to coniZpM
# ----------------------------------------
@njit
def _conipfv_kernel_set_bounds(
    sp,
    remQ,
    ci_offset,
    L_diag_inv,
    stack_val_min,
    stack_val_len,
    eps) -> int:
    """
    ***For use only in conipfv_kernel_njit.***

    Sets the candidate integer range for vec[i] from the remaining quadratic
    budget remQ. Feasible bounds (with R = sqrt(remQ)):
        (-R - ci_offset) / L[i,i]  <=  vec[i]  <=  (R - ci_offset) / L[i,i]

    Parameters
    ----------
    sp : int
        Stack pointer, used to index into stack_val_min and stack_val_len.
    remQ : float
        Remaining quadratic budget Q - sum_{j>i} c[j]^2.
    ci_offset : float
        Offset to c[i] from contributions of vec[j] for j > i.
    L_diag_inv : float
        Reciprocal of the diagonal L[i,i].
    stack_val_min : ArrayLike
        Output buffer for the minimum candidate value of vec[i].
    stack_val_len : ArrayLike
        Output buffer for the number of candidate values.
    eps : float
        Floating-point tolerance for bound computation.

    Returns
    -------
    int
        The number of candidate values (stack_val_len[sp]).
    """
    if remQ < 0:
        remQ = 0
    R = np.sqrt(remQ)

    lo = int(np.ceil(( -R - ci_offset) * L_diag_inv - eps))
    hi = int(np.floor(( R - ci_offset) * L_diag_inv + eps))

    # store the data to recreate the interval
    num = hi - lo + 1
    stack_val_min[sp] = lo
    stack_val_len[sp] = num

    return num

@njit
def conipfv_kernel_njit(
        L: ArrayLike,
        Q: int,
        dilation: int,
        # M0 cuts:
        Binter0: ArrayLike,
        M0min: int,
        # K' cuts:
        H: ArrayLike,
        # misc:
        max_N_out: int,
        eps: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """
    Adaptation of the iterative Fincke-Pohst algorithm for constructing
    coni-PFVs. Enumerates integer vectors vec satisfying:

        0 <= vec^T @ (L@L^T) @ vec <= dilation * Q   (ellipsoid)
        dot(Binter0, vec) >= M0min                    (M[0] cut)
        (vec^T @ mat @ vec) // Q <= gcd(H @ vec)      (K' > 0 cut)

    Any vec passing all constraints can generate a coni-PFV as long as
    det(N) != 0.

    NOT recommended for production use. Use `conipfv_kernel` instead.

    Parameters
    ----------
    L : ArrayLike
        Lower triangular matrix such that mat = L @ L.T.
    Q : int
        Base ellipsoid bound (tadpole).
    dilation : int
        Maximum allowed dilation factor for the ellipsoid.
    Binter0 : ArrayLike
        First row of Binter; satisfies dot(Binter0, vec) = M[0]. Columns
        should be ordered so that Binter0 has as many leading zeros as
        possible to enable early M[0] cuts.
    M0min : int
        Minimum permitted value of M[0]. Inclusive.
    H : ArrayLike
        Row-HNF of the matrix G such that Kperp = G @ vec.
    max_N_out : int
        Maximum number of output vectors.
    eps : float, optional
        Floating-point tolerance for bound computation.

    Returns
    -------
    out : np.ndarray, shape (N, dim)
        Vectors satisfying all constraints.
    Qs : np.ndarray, shape (N,)
        Quadratic form value vec^T @ mat @ vec for each output vector.
    """
    # compute  useful variables
    Q_upper    = Q*dilation
    dim        = L.shape[0]
    L_diag_inv = 1.0 / np.diag(L)

    # linear constraint
    num_zeros = 0
    zeros     = True
    for i in range(dim):
        if Binter0[i] == 0:
            if zeros == False:
                raise ValueError("Binter0 is not sorted so 0s are first...")
            num_zeros += 1
        else:
            zeros = False

    # output object
    # -------------
    out = np.empty((max_N_out, dim), dtype=np.int64)
    Qs  = np.empty((max_N_out,), dtype=np.float64)

    op  = 0 # output pointer

    # internal vector that gets built/written to output
    vec = np.zeros(dim, dtype=np.int64)

    # stack variables
    # ---------------
    # stack pointer
    sp = 0

    # max stack depth
    MAX_DEPTH = dim+1

    # stack arrays: i, pos, remaining_Q, nonzero, candidate values
    stack_i      = np.empty(MAX_DEPTH, np.int64)
    stack_pos    = np.empty(MAX_DEPTH, np.int64)
    stack_remQ   = np.empty(MAX_DEPTH, np.float64)
    stack_M0     = np.empty(MAX_DEPTH, np.int64)
    stack_gcd    = np.empty(MAX_DEPTH, np.int64)

    # vec[i] candidate arrays per depth (preallocate maximum possible size)
    stack_val_len= np.zeros(MAX_DEPTH, np.int64) # number of candidates
    stack_val_min= np.empty(MAX_DEPTH, np.int64) # minimum value for vec[stack_i[sp]]

    # offsets for ci
    # c[i] = L[i,i]*vec[i] + sum_{j>i} L[j,i]*vec[j]
    stack_ci_offset = np.zeros(MAX_DEPTH, np.float64)# offset c[i]-L[i,i]*vec[i]
    stack_Hveci     = np.zeros(MAX_DEPTH, np.int64)

    # initialize stack
    # ----------------
    stack_i[sp]    = dim-1
    stack_pos[sp]  = 0
    stack_remQ[sp] = Q_upper
    stack_M0[sp]   = 0
    stack_gcd[sp]  = 0

    k = _conipfv_kernel_set_bounds(
        sp,
        Q_upper,
        0,
        L_diag_inv[dim-1],
        stack_val_min,
        stack_val_len,
        eps)
    if k == 0:
        print("ERROR NO VECTORS")
        return out[:op, :], Qs[:op]

    # process stack until empty
    # -------------------------
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # HOT LOOP HOT LOOP HOT LOOP HOT LOOP
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    while sp >= 0:
        # read values
        i    = stack_i[sp]
        pos  = stack_pos[sp]
        remQ = stack_remQ[sp]
        M0   = stack_M0[sp]

        # check if node is leaf
        # ---------------------
        # if i==-1, then we have fully written vec
        if i == -1:
            if op >= max_N_out:
                break

            # don't save the 0-vector
            Qsave = Q_upper - remQ
            if Qsave>eps:
                out[op, :] = vec
                Qs[op]     = Qsave
                op += 1

            # kill node
            sp -= 1
            continue

        # check if current depth is completed
        # -----------------------------------
        if pos == stack_val_len[sp]:
            # kill node
            sp -= 1
            continue

        # pick candidate veci
        # -------------------
        veci   = stack_val_min[sp] + pos
        vec[i] = veci

        stack_pos[sp] += 1 # advance pos for next iteration

        # cut on M0 >= M0min
        # ------------------
        M0 += Binter0[i]*veci
        if (i == num_zeros) and (M0 < M0min):
            continue

        # cut on Q > 0
        # ------------
        ci      = L[i,i]*veci + stack_ci_offset[sp]
        new_rem = remQ - ci*ci

        # cut of no more Q left...
        # (TBH unless there is a floating point error in the vec[i] range
        #  computation, this check should never be hit...)
        if new_rem < 0 - eps:
            continue

        # cut on K' > 0
        # -------------
        required_dilation = (Q_upper-new_rem)/Q - eps

        # first try the cut with old GCD but new tadpole
        new_gcd = stack_gcd[sp]
        if (new_gcd > 0) and (new_gcd < required_dilation):
            continue

        # try with the actual GCD
        if new_gcd != 1: # if the old GCD was 1, no need to recompute...
            # compute the GCD
            Hvec_i  = stack_Hveci[sp] + H[i,i]*veci
            new_gcd = math.gcd(new_gcd, Hvec_i)

            # check it
            if (new_gcd > 0) and (new_gcd < required_dilation):
                continue

        # passes cuts -> push next depth :)
        # ---------------------------------
        sp += 1
        stack_i[sp]       = i-1
        stack_pos[sp]     = 0
        stack_remQ[sp]    = new_rem
        stack_M0[sp]      = M0
        stack_gcd[sp]     = new_gcd

        # compute the new ci, Hvec_i offset value for i-1 using this vector
        ci_offset = 0.0
        Hvec_i    = 0
        for j in range(i, dim):
            ci_offset += L[j,i-1] * vec[j]
            Hvec_i += H[i-1,j] * vec[j]

        stack_ci_offset[sp] = ci_offset
        stack_Hveci[sp] = Hvec_i

        # set candidate values of vec[i-1]
        _conipfv_kernel_set_bounds(
            sp,
            new_rem,
            ci_offset,
            L_diag_inv[i-1],
            stack_val_min,
            stack_val_len,
            eps)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # HOT LOOP HOT LOOP HOT LOOP HOT LOOP
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    return out[:op, :], Qs[:op]
