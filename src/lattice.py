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
def orthogonal_lattice(p: "ArrayLike") -> "ArrayLike":
    """
    **Description:**
    Computes a basis of the orthogonal lattice to some vector p, with columns
    as basis vectors.

    Do so using the HNF (via Flint):
    for a matrix A, the HNF is a matrix H and U such that H = U@A such that
        - U is unimodular and square
        - H upper triangular, with 'leading row coefficients' to the right of
          those above it
    in the case that A = p.T (as a column), then
        - H.shape = [len(p),1]
        - H[0, 0]!=0 (since U is unimodular and hence full-rank), so rank(H) = 1
        - H[1:,0]=0 since there is nothing 'to the right' of H[0,0]
    thus U[1:,:] is a basis of the null-space
    
    **Arguments:**
    - `p`: The orthogonal vector. Assumed to be integral

    **Returns:**
    A basis of the lattice orthogonal to p, as column vectors.
    """
    n = len(p)

    # A = p as a column
    A = flint.fmpz_mat([np.array(p).tolist()]).transpose()

    # H = U * A
    U = A.hnf(transform=True)[1]

    # Extract bottom n-1 rows of U
    B = flint.fmpz_mat(n-1, n)
    for i in range(1, n):
        for j in range(n):
            B[i-1, j] = U[i, j]

    return np.array(B.lll().tolist()).astype(int).T

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
def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)
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
        return (
            np.array([[np.int64(Ainv[i, j]) for j in range(n)] for i in range(n)]),
            s
        )

# lattice points in ellipsoid
# ===========================
def fp_ellipsoid(
    mat: "ArrayLike",
    Q: float,
    Q_lower: float = 0,
    max_N_out: int = 1_000_000_000,
    eps: float = 1e-4,
    recursive: bool = False,
    verbosity: int = 0) -> "ArrayLike":
    """
    **Description:**
    Enumerate all nonzero integer vectors vec such that
        0 <= vec^T @ mat @ vec <= Q.

    Does so via the Fincke-Pohst algorithm. This is, roughly,
        1) Cholesky-decompose mat = L@L^T for L lower triangular
        2) define c = L^T@vec, so the quadratic form becomes 0 <= |c|^2 <= Q
        3) observe that, since L^T is upper triangular, c[i] depends only on
           vec[j>=i]. E.g., c[0] depends on vec[0], ..., vec[dim-1]
                            c[1] depends on vec[1], ..., vec[dim-1]
                            c[dim-1] depends on vec[dim-1]
        4) fix vec[dim-1], which reduces the norm-bound on c from Q to
           Q-c[dim-1]^2 and effectively reduces the dimension of the problem,
           at the cost of adding a shift-vector to c[:dim-1]
        5) recurse

    **Arguments:**
    - `mat`:       The matrix defining the ellipsoid via
                       0 <= vec^T @ mat @ vec <= Q.
    - `Q`:         A positive parameter defining the size of the ellipsoid.
    - `Q_lower`:   Allow searching of lattice points in the slice
                       Q_lower <= vec^T @ mat @ vec <= Q.
    - `max_N_out`: The maximum number of output allowed. We construct an array
                   with this length.
    - `eps`:       A small number used for correctly setting bounds despite
                   floating point errors.
    - `recursive`: Whether to use a recursive implementation of the
                   Fincke-Pohst algorithm.
    - `verbosity`: The verbosity level. Higher is more verbose.


    **Returns:**
    Nothing (fills the array out[:out_count])
    """
    mat = np.asarray(mat)
    dim = mat.shape[0]

    # inflate Q to accomodate floating-point mat
    if not np.isdtype(mat.dtype, 'integral'):
        Q = 1.1*Q

    if verbosity >= 1:
        prefactor = np.pi**(dim/2) / sp.special.gamma(dim/2+1)
        scaling   = Q**(dim/2)     / np.sqrt(np.linalg.det(mat))
        print(f"Expected number of lattice points is {prefactor*scaling}")

    # cholesky decomposition
    try:
        L = np.linalg.cholesky(mat)
    except np.linalg.LinAlgError as e:
        raise e

    # solve it!
    if recursive:
        # container for outputs
        vec = np.zeros(dim, dtype=np.int32)
        out = np.empty((max_N_out, dim), dtype=np.int32)
        out_count = np.zeros(1, np.int32)

        # recurse
        fp_recurse(
            i=dim-1, vec=vec,
            nonzero=False,
            remaining_Q=Q, Q_lower=Q_lower,
            L=L,
            out_count=out_count, out=out, max_N_out=max_N_out,
            eps=eps)
        if out_count[0] == max_N_out:
            print("SATURATED MAXIMUM ALLOWED OUTPUTS")
        out = out[:out_count[0], :]
    else:
        # iterative
        out = fp_iterative(
            L=L,
            Q_upper=Q, Q_lower=Q_lower,
            max_N_out=max_N_out,
            eps=eps)
        if out.shape[0] == max_N_out:
            print("SATURATED MAXIMUM ALLOWED OUTPUTS")

    if verbosity >= 1:
        print(f"Actual number of lattice points is {out.shape[0]}")

    return out

@njit
def fp_recurse(
        i: int,
        vec: "ArrayLike",
        nonzero: bool,
        remaining_Q: float,
        Q_lower: float,
        L: "ArrayLike",
        out_count: list[int],
        out: "ArrayLike",
        max_N_out: int,
        eps: float) -> None:
    """
    **Description:**
    The 'Fincke-Pohst' algorithm (FP) from
        Improved Methods for Calculating Vectors of Short Length in a Lattice,
        Including a Complexity Analysis by Fincke, Pohst
    can be viewed as enumerating the lattice points in an ellipsoid
        0 <= vec^T M vec <= Q.
    We take this perspective, using FP to compute such nonzero lattice points.

    FP operates by Cholesky-decomposing the matrix M = L L^T (L lower
    triangular) to enable rewriting of the ellipsoid condition as
        0 <= |L^T vec|^2 <= Q.
    Since L^T is upper triangular, the bounds on permissible values of vec[-1]
    (ignoring all other components) are easy to compute. Once vec[-1] is fixed,
    one can bound the component vec[-2] and so on.

    This method follows a DFS-style iteration, building and saving each lattice
    vector x to an output array `out`.

    **Arguments:**
    - `i`:           For this call, vec[i+1:] has been set. Iterate over
                     possible values of vec[i].
    - `vec`:         The vector being built.
    - `nonzero`:     Whether the vector is already nonzero (we only want to
                     output nonzero vectors).
    - `remaining_Q`: Imagine 'using-up' Q by setting vec[i+1:]. This is the
                     remaining Q allowed for setting vec[i].
    - `Q_lower`:     Allow user-set lower bounds on |L^T vec|^2. I.e.,
                     enumeration of lattice points in the shell
                        Q_lower <= vec^T M vec <= Q.
    - `L`:           The Cholesky-decomposition of M.
    - `out_count`:   The count of output lattice vectors. Saved as a list of
                     length-1 so it is mutable.
    - `out`:         A container for the output lattice vectors. Will contain
                     padding in case out_count < out.shape[0]
    - `max_N_out`:   The maximum number of output allowed. Should equal
                     out.shape[0]
    - `eps`:         A small number used for correctly setting bounds despite
                    floating point errors.

    **Returns:**
    Nothing (fills the array out[:out_count])
    """
    # check if there is room for more outputs
    if out_count[0] == max_N_out:
        return

    # check if vec is fully built
    if i == -1:
        if nonzero:
            out[out_count[0], :] = vec
            out_count[0] += 1
        return

    # compute the offset
    # c[i] = L[i,i]*vec[i] + sum_{j>i} L[j,i]*vec[j]
    ci_offset = 0.0
    for j in range(i+1, vec.size):
        ci_offset += L[j, i] * vec[j]

    # enumerate vec[i] such that |c[i]|^2 <= remaining_Q
    # where c[i] = L[i,i]*vec[i] + ci_offset

    # feasible integer bounds for vec[i]
    # -R                      <= c[i]          <= R
    # -R - ci_offset          <= L[i,i]*vec[i] <= R - ci_offset
    # (-R - ci_offset)/L[i,i] <= vec[i]        <= (R - ci_offset)/L[i,i]
    # where we used that the diagonal is positive
    R = np.sqrt(max(0.0, remaining_Q))
    lo = int(np.ceil((-R - ci_offset)/L[i,i] - eps))
    hi = int(np.floor(( R - ci_offset)/L[i,i] + eps))

    # define the range to iterate in increasing L1 norm
    # (the same as veci_values = range(lo,hi+1) but just ordered)
    if True:
        # split by case
        if lo >= 0:
            # 0 <= lo (<= hi)
            veci_values = np.arange(lo, hi+1, dtype=np.int32)
        elif hi <= 0:
            # (lo <=) hi <= 0
            veci_values = np.arange(hi, lo-1,-1, dtype=np.int32)
        else:
            # lo < 0 < hi
            veci_values = np.zeros(hi - lo + 1, dtype=np.int32)
            k = 1

            # positive/negative pairs in increasing abs value
            for v in range(1, max(-lo, hi) + 1):
                if v <= hi:
                    veci_values[k] = v
                    k += 1
                if -v >= lo:
                    veci_values[k] = -v
                    k += 1
    else:
        veci_values = np.arange(lo,hi+1, dtype=np.int32)

    # iterate over possible values of vec[i]
    for veci in veci_values:
        vec[i] = veci
        ci = L[i,i]*veci + ci_offset
        new_rem = remaining_Q - ci*ci
        if new_rem >= Q_lower-eps:
            fp_recurse(
                i=i-1, vec=vec,
                nonzero=nonzero or veci != 0,
                remaining_Q=new_rem, Q_lower=Q_lower,
                L=L,
                out_count=out_count, out=out, max_N_out=max_N_out,
                eps=eps)

@njit
def fp_iterative(
        L: "ArrayLike",
        Q_upper: float,
        Q_lower: float,
        max_N_out: int,
        eps: float) -> None:
    """
    Iterative DFS implementation of the Fincke-Pohst algorithm
    """
    dim = L.shape[0]

    # output object
    # -------------
    out = np.empty((max_N_out, dim), dtype=np.int32)
    
    # output pointer
    op  = 0

    # internal vector that gets built/written to output
    vec = np.zeros(dim, dtype=np.int32)
    
    # stack variables
    # ---------------
    # stack pointer
    sp = 0

    # max stack depth
    MAX_DEPTH = dim + 1

    # stack arrays: i, pos, remaining_Q, nonzero, candidate values
    stack_i      = np.empty(MAX_DEPTH, np.int32)
    stack_pos    = np.empty(MAX_DEPTH, np.int32)
    stack_remQ   = np.empty(MAX_DEPTH, np.float64)
    stack_nz     = np.zeros(MAX_DEPTH, np.uint8)  # bool not njit-friendly
    
    # candidate arrays per depth (preallocate maximum possible size)
    stack_val_len= np.zeros(MAX_DEPTH, np.int32)  # number of veci candidates
    stack_vals   = np.empty((MAX_DEPTH, 128), np.int32) # veci candidates

    # initialize stack
    # ----------------
    stack_i[sp]    = dim-1
    stack_pos[sp]  = 0
    stack_remQ[sp] = Q_upper
    stack_nz[sp]   = 0

    stack_val_len[sp] = -1  # will fill below
    #stack_vals unset here

    # process stack until empty
    # -------------------------
    while sp >= 0:
        # read values
        i    = stack_i[sp]
        pos  = stack_pos[sp]
        remQ = stack_remQ[sp]
        nz   = stack_nz[sp] != 0

        # check if node is completed
        # --------------------------
        # if i==-1, then we have fully written vec
        if i == -1:
            if nz:
                out[op, :] = vec
                op += 1
                if op >= max_N_out:
                    break
            # kill node
            sp -= 1
            continue

        # check if current depth is completed
        # -----------------------------------
        if pos == stack_val_len[sp]:
            # kill node
            sp -= 1
            continue

        # current depth incomplete...
        # ---------------------------
        # compute the offset
        # c[i] = L[i,i]*vec[i] + sum_{j>i} L[j,i]*vec[j]
        ci_offset = 0.0
        for j in range(i+1, dim):
            ci_offset += L[j, i] * vec[j]


        # set candidate values of vec[i] if first time to depth
        if stack_val_len[sp] == -1:
            # feasible integer bounds for vec[i]
            # -R                      <= c[i]          <= R
            # -R - ci_offset          <= L[i,i]*vec[i] <= R - ci_offset
            # (-R - ci_offset)/L[i,i] <= vec[i]        <= (R - ci_offset)/L[i,i]
            # where we used that the diagonal is positive
            R = np.sqrt(max(0.0, remQ))
            lo = int(np.ceil((-R - ci_offset)/L[i,i] - eps))
            hi = int(np.floor(( R - ci_offset)/L[i,i] + eps))

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
            elif k>128:
                raise ValueError(f"Assumed |hi-lo| <= 128, but got {k}")

            # yes valid veci values
            stack_val_len[sp] = k
            stack_pos[sp] = 0
            pos = 0

        # pick candidate veci
        veci = stack_vals[sp, pos]
        stack_pos[sp] += 1  # advance pos for next iteration
        vec[i] = veci

        ci = L[i,i]*veci + ci_offset
        new_rem = remQ - ci*ci

        if new_rem >= Q_lower - eps:
            # push next depth
            sp += 1
            stack_i[sp]      = i-1
            stack_pos[sp]    = 0
            stack_remQ[sp]   = new_rem
            stack_nz[sp]     = nz or (veci != 0)
            stack_val_len[sp] = -1  # will fill when we visit
            # candidate array for this depth is stack_vals[sp,:]
        # else do not push (prune)

    return out[:op, :]
