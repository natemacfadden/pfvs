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

from numba import types
from numba.typed import Dict

# basic helpers
# =============
def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)

# misc lattice
# ============
# LLL-reduction
def lll_reduce(B: "ArrayLike") -> "ArrayLike":
    """
    Apply lll-reduction to the input matrix, representing a columnwise basis of
    some lattice

    N.B.: Flint's LLL-transformation effectively works on row-bases. This is
    because, for an integral matrix M, it solves for
        - a unimodular T and
        - an integral L
    obeying T@M = L. I.e., L[i,:] = T[i,k] M[k,:] so the *rows* of L are
    integral combinations of the *rows* of M.

    **Arguments:**
    - `B`: A basis of a lattice, as column vectors.

    **Returns:**
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
def extended_euclidean(a,b):
    old_r, r = (a,b)
    old_s, s = (1,0)
    old_t, t = (0,1)

    while r!=0:
        q = old_r//r

        old_r, r = (r, old_r - q*r)
        old_s, s = (s, old_s - q*s)
        old_t, t = (t, old_t - q*t)

    return old_s, old_t, old_r

def orthogonal_lattice(p: "ArrayLike") -> "ArrayLike":
    """
    **Description:**
    Computes a basis of the orthogonal lattice to some vector v, with columns
    as basis vectors.

    **Arguments:**
    - `v`: The orthogonal vector. Assumed to be integral

    **Returns:**
    A basis of the lattice orthogonal to v, as column vectors.
    """
    p = np.array(p).reshape(1,-1)
    dim = p.shape[1]

    # speeds up following computations if p is primitive
    #p = p//np.gcd.reduce(p,axis=1)

    # compute null space
    mat_fl = flint.fmpz_mat(p.tolist())
    null, nullity = mat_fl.nullspace()
    null = null.transpose().hnf().transpose()

    gcds = np.array([np.gcd.reduce([null[i,j] for i in range(dim)]) for j in range(nullity)])

    #print(dim, nullity, null)

    out = np.zeros((dim,nullity), dtype=int)
    for j in range(nullity):
        for i in range(dim):
            out[i,j] =  null[i,j]

    # reduce by gcd
    out = out // gcds

    return out

@njit
def orthogonal_lattice_custom(p: "ArrayLike") -> "ArrayLike":
    """
    **Description:**
    Computes a basis of the orthogonal lattice to some vector v, with columns
    as basis vectors.

    **Arguments:**
    - `v`: The orthogonal vector. Assumed to be integral

    **Returns:**
    A basis of the lattice orthogonal to v, as column vectors.
    """
    n = p.shape[0]
    U = np.eye(n, dtype=p.dtype)

    w = p.copy()
    for k in range(1,n):
        # clear out w[k]

        # already done :)
        if w[k] == 0:
            continue

        # not done - compute the bezout coefficients
        a,b   = w[0], w[k]
        s,t,g = extended_euclidean(a,b)

        M = [[s,t],[-b//g, a//g]]
        # interpretation:
        # det(M) = +/- 1
        # M@(w[0],w[k]) = (g, 0)

        # update w
        w[0] = g
        w[k] = 0

        # update U
        # think: U->\Tilde{M}@U
        # for \Tilde{M} an extended version of M
        # (specifically,
        #                \Tilde{M}<-1
        #                \Tilde{M}[[0,k],[0,k]] <- M)
        # you can track the indices but this has the effect
        #     U[[0,k],r] <- M@U[[0,k],r]
        for r in range(n):
            tmp1   = M[0][0]*U[0,r] + M[0][1]*U[k,r]
            tmp2   = M[1][0]*U[0,r] + M[1][1]*U[k,r]
            U[0,r] = tmp1
            U[k,r] = tmp2

    return U[1:].T

# dual lattice
def dual_lattice(B: "ArrayLike") -> "ArrayLike":
    """
    **Description:**
    Computes a basis of the lattice dual to L(B).

    Use the convention that the basis vectors are the **columns** of B.

    See https://en.wikipedia.org/wiki/Dual_lattice

    **Arguments:**
    - `B`: A basis of the primal lattice.

    **Returns:**
    A basis of the dual lattice.
    """
    B = flint.fmpz_mat(B.tolist())
    D, denom = (B*( (B.transpose()*B).inv() )).numer_denom()

    return np.array(D.tolist()).astype(int), denom

# integer 'inverse' of matrix (i.e., adjugate)
def inv_scaled(A_in, as_flint: bool = False):
    """
    Return (B, s) such that B*A = s*I
    """
    dim = A_in.shape[0]
    A = flint.fmpz_mat(A_in.tolist())
    n = A.nrows()

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


# branch and bound algorithms
# ===========================
# Fincke-Pohst (FP)
# -----------------
@njit
def fp_iterative_njit(
        L: "ArrayLike",
        Q: float,
        linvec: "ArrayLike" = None,
        linmin: int = None,
        max_N_out: int = 10_000_000,
        eps: float = 1e-4,
        COORD_BUFF_SIZE: int = 2048) -> "ArrayLike":
    """
    **Description:**
    Enumerate all nonzero integer vectors vec such that
        0 <= vec^T @ mat @ vec <= Q.

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

    **Note:**
    This is an iterative (DFS) implementation using an explicit stack.

    **Arguments:**
    - `L`:               The lower triangular matrix such that mat = L@L.T
    - `Q`:               The ellipsoid bound.
    - `max_N_out`:       The maximum number of output allowed.
    - `eps`:             A small number used for correctly setting bounds
                         despite floating point errors.
    - `COORD_BUFF_SIZE`: The size of the buffer that holds the possible values
                         of vec[i].

    **Returns:**
    The vectors `vec` in the ellipsoid.
    """
    dim        = L.shape[0]
    L_diag_inv = 1.0 / np.diag(L)

    # linear constraint
    if linvec is None:
        linvec = np.zeros(dim)
        linmin = 0
        num_zeros = -1
    else:
        num_zeros = 0
        zeros = True
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
            R = np.sqrt(remQ)
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
        ci = L[i,i]*veci + ci_offsets[i]
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

# Kannan box method (points in cones)
# -----------------------------------
@njit
def _kannan_box_mat_set_coord_candidates(
    sp,
    i,
    B,
    linmat,
    linmin,
    stack_partial_sum,
    abssum,
    stack_vals,
    stack_val_len,
    COORD_BUFF_SIZE) -> int:
    """
    ***For use only in kannan_box_mat_njit.***
    ***For use only in kannan_box_mat_njit.***
    ***For use only in kannan_box_mat_njit.***
    ***For use only in kannan_box_mat_njit.***
    ***For use only in kannan_box_mat_njit.***

    **Description:**
    Sets the candidate values for vec[i], bound by constraints on ...

    **Arguments:**
    - `sp`:              Stack pointer. For writing our outputs.

    - `stack_vals`:      The output array to store candidate vec[i] values to.
    - `stack_val_len`:   How many candidate vec[i] values exist.
    - `COORD_BUFF_SIZE`: The size of the buffer that holds the possible values
                         of vec[i].

    **Returns:**
    The number of candidate values, stack_val_len[sp].
    """
    lo = -B
    hi = B

    for j in range(linmat.shape[0]):
        if linmat[j,i] == 0:
            continue

        numer = linmin - stack_partial_sum[sp,j] - B*abssum[j,i]

        h = linmat[j,i]
        if h>0:
            lo = max(lo, int(np.ceil(numer/h)))
        else:
            hi = min(hi, int(np.floor(numer/h)))

    # compute the number of veci to iterate over
    k = hi - lo + 1
    if k <= 0:
        stack_val_len[sp] = 0
        return 0
    if k > COORD_BUFF_SIZE:
        msg = f"Assumed |hi-lo| <= {COORD_BUFF_SIZE}, but got {k}"
        raise ValueError(msg)

    for t in range(k):
        stack_vals[sp, t] = lo + t
    stack_val_len[sp] = k

    return k

@njit
def kannan_box_mat_njit(
        B: int,
        linmat: "ArrayLike",
        linmin: int,
        max_N_out: int = 1_000_000,
        max_N_iter: int = 1_000_000_000,
        eps: float = 1e-4,
        COORD_BUFF_SIZE: int = 2048) -> "ArrayLike":
    """
    **Description:**
    Enumerate all nonzero integer vectors vec such that
        -B <= vec[i] <= B
    and
        linmat@vec >= linmin.

    **Note:**
    This is an iterative (DFS) implementation using an explicit stack.

    **Arguments:**
    - `V`:               ...
    - `linmat`:          ...
    - `linmin`:          ...
    - `max_N_out`:       The maximum number of output allowed.
    - `eps`:             A small number used for correctly setting bounds
                         despite floating point errors.
    - `COORD_BUFF_SIZE`: The size of the buffer that holds the possible values
                         of vec[i].

    **Returns:**
    The vectors `vec` in the ellipsoid.
    """
    dim        = linmat.shape[1]

    # sort the columns of linmat so stricter components come first
    # ------------------------------------------------------------
    col_l1_norm = np.sum(np.abs(linmat), axis=0)
    sort_inds   = np.argsort(col_l1_norm)
    undo_sort   = np.argsort(sort_inds)

    linmat = linmat[:,sort_inds]

    # output object
    # -------------
    out = np.empty((max_N_out, dim), dtype=np.int64)

    # output pointer
    op  = 0

    # internal vector that gets built/written to output
    vec = np.zeros(dim, dtype=np.int64)

    # compute helper variable
    # -----------------------
    abssum = np.empty((linmat.shape[0],linmat.shape[1]+1), np.int64)
    for j in range(abssum.shape[0]):
        abssum[j,0] = 0#abs(linmat[j,0])

        for i in range(dim):
            abssum[j,i+1] = abssum[j,i] + abs(abs(linmat[j,i]))

    # stack variables
    # ---------------
    # stack pointer
    sp = 0

    # max stack depth
    MAX_DEPTH = dim+1

    # stack arrays: i, pos, remaining_Q, nonzero, candidate values
    stack_i      = np.empty(MAX_DEPTH, np.int64)
    stack_pos    = np.empty(MAX_DEPTH, np.int64)

    # vec[i] candidate arrays per depth (preallocate maximum possible size)
    stack_val_len= np.zeros(MAX_DEPTH, np.int64) # number of candidates
    stack_vals   = np.empty((MAX_DEPTH, COORD_BUFF_SIZE), np.int64) # candidates

    # misc helper
    # stack_partial_sum[sp][j] = \sum_{ k > stack_i[sp] } linmat[j][k] * vec[k]
    stack_partial_sum = np.zeros((MAX_DEPTH, linmat.shape[0]), np.int64)

    # initialize stack
    # ----------------
    stack_i[sp]    = dim-1
    stack_pos[sp]  = 0

    _kannan_box_mat_set_coord_candidates(
        sp,
        dim-1,
        B,
        linmat,
        linmin,
        stack_partial_sum,
        abssum,
        stack_vals,
        stack_val_len,
        COORD_BUFF_SIZE
    )

    # process stack until empty
    # -------------------------
    Niter = 0
    while sp >= 0:
        Niter += 1
        if Niter >= max_N_iter:
            break
        # read values
        i    = stack_i[sp]
        pos  = stack_pos[sp]

        # check if node is completed
        # --------------------------
        # if i==-1, then we have fully written vec
        if i == -1:
            if op >= max_N_out:
                break
            out[op, :] = vec
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
        veci   = stack_vals[sp, pos]
        vec[i] = veci

        # advance pos for next iteration
        stack_pos[sp] += 1

        # passes cuts -> push next depth :)
        sp += 1
        stack_i[sp]       = i-1
        stack_pos[sp]     = 0

        for j in range(linmat.shape[0]):
            stack_partial_sum[sp,j] = stack_partial_sum[sp-1,j] + linmat[j,i]*vec[i]

        if i >= 1:
            _kannan_box_mat_set_coord_candidates(
                sp,
                i-1,
                B,
                linmat,
                linmin,
                stack_partial_sum,
                abssum,
                stack_vals,
                stack_val_len,
                COORD_BUFF_SIZE
            )

    return out[:op, undo_sort], Niter

# FP-style methods but tailored to coniZpM
# ----------------------------------------
@njit
def _coni_kernel_set_bounds(
    sp,
    remQ,
    ci_offset,
    L_diag_inv,
    stack_val_min,
    stack_val_len,
    eps) -> int:
    """
    ***For use only in coni_kernel_njit.***
    ***For use only in coni_kernel_njit.***
    ***For use only in coni_kernel_njit.***
    ***For use only in coni_kernel_njit.***
    ***For use only in coni_kernel_njit.***

    **Description:**
    Sets the candidate values for vec[i], bound by constraints on Q. Operates as
    follows.

    Feasible integer bounds for vec[i] (let R = sqrt(remQ)):
        -R                      <= c[i]          <= R
        -R - ci_offset          <= L[i,i]*vec[i] <= R - ci_offset
        (-R - ci_offset)/L[i,i] <= vec[i]        <= (R - ci_offset)/L[i,i]
    where we used that the diagonal is positive.

    **Arguments:**
    - `sp`:              Stack pointer. For writing our outputs.
    - `remQ`:            The remaining amount of Q that we have to work with.
    - `ci_offset`:       The offset to the vector vec[i] from vec[j>i]
                         contributions.
    - `L_diag_inv`:      The value 1/L[i,i] for setting bounds.
    - `stack_val_min`:   Container for the minimum value of vec[i] permissible.
    - `stack_val_len`:   Container for how many candidate vec[i] values exist.
    - `eps`:             A small number used for correctly setting bounds
                         despite floating point errors.

    **Returns:**
    The number of candidate values, stack_val_len[sp].
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
def coni_kernel_njit(
        L: "ArrayLike",
        Q: int,
        dilation: int,
        # M0 cuts:
        Binter0: "ArrayLike",
        M0min: int,
        # K' cuts:
        H: "ArrayLike",
        # misc:
        max_N_out: int,
        eps: float = 1e-4) -> ("ArrayLike", "ArrayLike"):
    """
    **Description:**
    Adaptation of the (iterative) Fincke-Pohst algorithm for utility in
    constructing coni-PFVs. I.e., solves
        0 <= vec^T @ mat     @ vec <= dilation*Q.
        0 <= vec^T @ (L@L^T) @ vec <= dilation*Q
    as well as (M[0] cuts)
        M0min <= dot(Binter0, vec)
    as well as (K'>0 cuts)
        (vec^T @ mat @ vec)//Q <= gcd(Kperp)
                                = gcd([0, 1]@Z@Binter@vec)
                                = gcd(H@vec)
    for H the row-HNF of [0, 1]@Z@Binter (this matrix computes Kperp from vec).

    Any `vec` satisfying all of the above can generate a coni-PFV, as long as
    det(N) != 0.

    **Arguments:**
    - `L`:               The lower triangular matrix such that mat = L@L.T
    - `Q`:               The ellipsoid bound.
    - `dilation`:        The maximum allowed dilation to allow... As long as
                         gcd(Kperp) >= (vec^T @ mat @ vec)//Q, the vector vec
                         can still define coni-PFV.
    - `Binter0`:         Binter[0,:]. The vector such that dot(Binter0,vec)=M0.
                         BEST TO ORDER COLUMNS SUCH THAT Binter0 HAS A LARGE
                         NUMBER OF LEADING 0s.
    - `M0min`:           The minimum value of M0 permitted. Inclusive.
    - `H`:               Let G be the matrix such that Kperp = G@vec. Then
                         H = HNF(G).
    - `max_N_out`:       The maximum number of output allowed.
    - `eps`:             A small number used for correctly setting bounds
                         despite floating point errors.

    **Returns:**
    The vectors `vec` in the ellipsoid and obeying the extra constraints.
    The valuation `vec^T @ mat @ vec`.
    """
    # compute  useful variables
    Q_upper    = Q*dilation
    dim        = L.shape[0]
    L_diag_inv = 1.0 / np.diag(L)

    # linear constraint
    num_zeros = 0
    zeros = True
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

    k = _coni_kernel_set_bounds(
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
        Hvec_i = 0
        for j in range(i, dim):
            ci_offset += L[j,i-1] * vec[j]
            Hvec_i += H[i-1,j] * vec[j]

        stack_ci_offset[sp] = ci_offset
        stack_Hveci[sp] = Hvec_i

        # set candidate values of vec[i-1]
        _coni_kernel_set_bounds(
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
