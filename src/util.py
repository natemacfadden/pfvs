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
# Description:  This module contains misc utilities for PFV construction and
#               verification.
# -----------------------------------------------------------------------------

# external imports
import itertools
import numpy as np

# local imports
from . import lattice

# read data from CYTools
# ----------------------
def read_cy_data(cy: "cytools.CalabiYau") -> tuple["ArrayLike", "ArrayLike", "ArrayLike"]:
    """
    **Description:**
    Reads the intersection numbers, second chern class, and (inwards-facing)
    hyperplane normals of the Kahler cone from a CalabiYau object.

    **Arguments:**
    - `cy`: The CalabiYau object
    
    **Returns:**
    The triple intersection numbers.
    The second chern class.
    The (inwards-facing) hyperplane normals of the Kahler cone.
    """
    try:
        import cytools
    except ImportError as e:
        raise ImportError(
            "cytools is required reading data from a CalabiYau object..."
        ) from e

    # read the data
    # -------------
    kappa = cy.intersection_numbers(in_basis=True, format='dense')
    c2    = cy.second_chern_class(in_basis=True)
    # (hyperplanes of the Kahler cone)
    H     = cy.mori_cone_cap(in_basis=True).rays()

    # return
    # ------
    return kappa, c2, H

# basis for M-vectors
# -------------------
# helpers
def a_mat(kappa: "ArrayLike") -> "ArrayLike":
    """
    **Description:**
    Returns the a-matrix, used for finding (Coni) PFVs. This matrix is defined
    componentwise as
    \\begin{equation}
        \\tilde{a}_{ij} = \\begin{cases}
            kappa_{ijj} & i\\geq j\\
            kappa_{iij} & i < j.
        \\end{cases}
    \\end{equation}
    See, e.g., eq 2.52 from https://arxiv.org/pdf/2406.13751.

    **Arguments:**
    - `kappa`: The intersection numbers.
    
    **Returns:**
    The a-matrix.
    """
    h11 = kappa.shape[0]
    a   = np.zeros((h11, h11), dtype=int)

    # fill the matrix
    for x,y in itertools.product(range(h11), range(h11)):
        if x>=y:
            a[x,y] = kappa[x,y,y]
        else:
            a[x,y] = kappa[x,x,y]

    # return
    return a

def b_vec(
    c2: "ArrayLike",
    coni: bool = False,
    coni_normal: "ArrayLike" = None) -> "ArrayLike":
    """
    **Description:**
    Returns the b-vector, used for finding (Coni-)PFVs. This vector is defined
    differently for non-Coni and Coni PFVs. For non-Coni PFVs, it is defined as
    \\begin{equation}
        \\tilde{b} = c_2.
    \\end{equation}
    For Coni PFVs, it is defined as
    \\begin{equation}
        \\tilde{b} = c_2 + n_{cf} q_{coni}
    \\end{equation}
    where $n_{cf} = 2$ and $q_{coni} = `coni_normal`$ (see below eq 3.5 of 
    https://arxiv.org/pdf/2406.13751).

    **Arguments:**
    - `c2`:          The second chern class.
    - `coni`:        Whether to output the Coni b. This is (b + 2*coni_normal)/24.
    - `coni_normal`: The normal defining the conifold facet.
    
    **Returns:**
    The b vector.
    """
    out = np.array(c2)

    # for non-Coni, this is just c2...
    if not coni:
        return out

    # adjust for Coni
    if coni_normal is None:
        # assume in coni basis
        coni_normal = np.zeros(shape=len(out), dtype=int)
        coni_normal[0] = 1
    else:
        coni_normal = np.array(coni_normal)

    out += 2*coni_normal

    return out

# actual M-lattice basis
def M_lattice(
    kappa: "ArrayLike",
    c2: "ArrayLike",
    coni: bool = False,
    coni_normal: "ArrayLike" = None) -> "ArrayLike":
    """
    **Description:**
    Computes a basis of the sublattice of all vectors, M, such that
        (b/24).M    and    (a/2)@M    and    M
    are all integral. This is equivalent to
        [b/24; a/2; 1]@M
    being integral.
    
    The set of all such M is just the dual to the lattice spanned by the
    **rows** of
        [b/24; a/2; 1].

    Work with column bases throughout.

    **Arguments:**
    - `kappa`:  The intersection numbers.
    - `c2`:     The second chern class.
    - `coni`:   Whether this is done in the Coni context (changes b).

    **Returns:**
    A basis for M satisfying the integrality constraints. Basis vectors are
    columns.
    """ 
    h11 = kappa.shape[0]

    # construct the primal lattice
    a   = a_mat(kappa=kappa)
    b   = b_vec(c2=c2, coni=coni, coni_normal=coni_normal)
    to_stack = [b,
                12*a,
                24*np.identity(h11,dtype=int)]
    _primal = np.vstack(to_stack, dtype=int).T

    # LLL-reduce
    primal = lattice.lll_reduce(_primal)
    primal = primal[:,-h11:]
    
    # construct the dual lattice
    dual, denom = lattice.dual_lattice(primal)
    assert 24%denom == 0 # the dual lattice should be integral
    out = dual * (24//denom)
    
    # LLL-reduce the dual to be extra nice
    return lattice.lll_reduce(out)
