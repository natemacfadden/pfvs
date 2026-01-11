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
import flint
import itertools
import numpy as np

# local imports
from . import lattice

# class object to hold the CY data
# --------------------------------
class CYData:
    """
    Simple class to hold the relevant data of a CY for constructing PFVs.

    **Arguments:**
    - `kappa`:      The intersection numbers.
    - `c2`:         The second chern class.
    - `H`:          Inwards-facing hyperplaness defining the Kahler cone.
    - `coni_curve`: The conifold curve. If not provided, then non-Coni PFVs are
                    assumed.
    """
    def __init__(
        self,
        kappa: "ArrayLike",
        c2: "ArrayLike",
        H: "ArrayLike",
        coni_curve: "ArrayLike" = None):
        """
        **Description:**
        Initializes an instance of a simple class to hold the relevant data of
        a CY for constructing PFVs.

        **Arguments:**
        - `kappa`:      The intersection numbers.
        - `c2`:         The second chern class.
        - `H`:          Inwards-facing hyperplaness defining the Kahler cone.
        - `coni_curve`: The conifold curve. If not provided, then non-Coni PFVs
                        are assumed.
        """
        self._kappa = np.array(kappa)
        self._c2    = np.array(c2)
        self._H     = np.array(H)
        self._coni  = (coni_curve is not None)

        # variables for the M-lattice
        self._a     = None
        self._b     = None

        # check if non-Coni...
        if not self._coni:
            return

        # coni stuff below...
        # -------------------
        # store information on the conifold curve if it is set
        self._coni_curve = np.array(coni_curve)
        type(self).coni_curve = property(lambda self: self._coni_curve)

        # compute the change of basis via the HNF
        q = np.array(self._coni_curve).reshape(-1,1)
        q = flint.fmpz_mat(q.tolist())
        self._cob = q.hnf(transform=True)[1]
        self._cob = np.array(self._cob.tolist()).astype(int)
        type(self).cob = property(lambda self: self._cob)

        # map kappa, c2, and H via this change of basis
        self._kappa_cob = np.einsum('ia,jb,kc,ijk->abc',
            self._cob, self._cob, self._cob, self._kappa)
        self._c2_cob = (self._cob@self._c2).reshape(-1)
        self._H_cob  = (self._H@self._cob.T)[:,1:]

        type(self).kappa_cob = property(lambda self: self._kappa_cob)
        type(self).c2_cob    = property(lambda self: self._c2_cob)
        type(self).H_cob     = property(lambda self: self._H_cob)
    
    @classmethod
    def from_cy(cls, cy, coni_curve: "ArrayLike" = None) -> "CYData":
        """
        **Description:**
        Construct a CYData object from a cytools.CalabiYau object.
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
        return cls(kappa=kappa, c2=c2, H=H, coni_curve=coni_curve)

    # getters
    @property
    def kappa(self):
        return self._kappa
    
    @property
    def c2(self):
        return self._c2

    @property
    def H(self):
        return self._H

    @property
    def coni(self):
        return self._coni

    # a-matrix, b-vector
    # ------------------
    @property
    def a(self):
        """
        **Description:**
        Returns the a-matrix, used for finding (Coni) PFVs. This matrix is
        defined componentwise as
        \\begin{equation}
            \\tilde{a}_{ij} = \\begin{cases}
                kappa_{ijj} & i\\geq j\\
                kappa_{iij} & i < j.
            \\end{cases}
        \\end{equation}
        See, e.g., eq 2.52 from https://arxiv.org/pdf/2406.13751.

        **Arguments:**
        Nothing.
        
        **Returns:**
        The a-matrix.
        """
        if self._a is not None:
            return self._a

        # compute the a-matrix
        kappa = self.kappa
        h11 = kappa.shape[0]
        a   = np.zeros((h11, h11), dtype=int)

        # fill the matrix
        for x,y in itertools.product(range(h11), range(h11)):
            if x>=y:
                a[x,y] = kappa[x,y,y]
            else:
                a[x,y] = kappa[x,x,y]

        # return
        self._a = a
        return self._a

    @property
    def b(self):
        """
        **Description:**
        Returns the b-vector, used for finding (Coni-)PFVs. This vector is
        defined differently for non-Coni and Coni PFVs. For non-Coni PFVs, it
        is defined as
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
        Nothing
        
        **Returns:**
        The b vector.
        """
        if self._b is not None:
            return self._b

        # for non-Coni, this is just c2...
        self._b = self.c2
        if not self.coni:
            return self._b

        # adjust for Coni
        self._b[0] = self._b[0]+2
        return self._b

    # basis for M-vectors
    # -------------------
    # actual M-lattice basis
    def M_lattice(self)-> "ArrayLike":
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
        h11 = self.kappa.shape[0]

        # construct the primal lattice
        to_stack = [self.b,
                    12*self.a,
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

# compute possible p-vectors
# --------------------------
def pvecs(
    H: "ArrayLike",
    coni: bool = False,
    coni_normal: "ArrayLike" = None) -> "ArrayLike":
    """
    **Description:**
    Computes possible p-vectors.

    For non-Coni PFVs, these are integral vectors p such that
        H @ p > 0
    for H the hyperplanes of the Kahler cone. I.e., p is in the interior of the
    Kahler cone.

    For Coni PFVs, these are integral vectors p such that
        (H\\coni_normal) @ p > 0
        dot(coni_normal, p)  = 0
    for H the hyperplanes of the Kahler cone. I.e., p is in the interior of the
    facet of the Kahler cone defined by coni_normal.

    **Arguments:**
    - `H`:           Hyperplane normals defining the Kahler cone.
    - `coni`:        Whether this is done in the Coni context.
    - `coni_normal`: The normal of the conifold facet.

    **Returns:**
    A basis for M satisfying the integrality constraints. Basis vectors are
    columns.
    """
    if not coni:
        # solve H @ p >= 0.5
        rhs = [0.5] * len(H)
    else:
        pass

    if coni_normal is None:
        # assume in coni basis
        coni_normal = np.zeros(shape=len(out), dtype=int)
        coni_normal[0] = 1
    else:
        coni_normal = np.array(coni_normal)
