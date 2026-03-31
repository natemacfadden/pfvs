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
# Description:  This module contains a class to store/compute certain CY data
#               needed for PFV computations.
# -----------------------------------------------------------------------------

# external imports
import flint
import itertools
import numpy as np

from numpy.typing import ArrayLike

# local imports
from . import lattice

class CYData:
    """
    Holds the CY data needed for constructing PFVs.

    Parameters
    ----------
    h21 : int
        The Hodge number h^{2,1}.
    kappa : ArrayLike
        The triple intersection numbers.
    c2 : ArrayLike
        The second Chern class.
    H : ArrayLike
        Inwards-facing hyperplanes defining the Kahler cone.
    coni_curve : ArrayLike, optional
        The conifold curve. If not provided, non-coni PFVs are assumed.
    coni_cob : ArrayLike, optional
        Change-of-basis matrix mapping the coni curve to (1,0,...,0).
    vertices : ArrayLike, optional
        Polytope vertices. Defaults to a placeholder string if not provided.
    heights : ArrayLike, optional
        Triangulation heights. Defaults to a placeholder string if not
        provided.
    """
    def __init__(
        self,
        h21: int,
        kappa: ArrayLike,
        c2: ArrayLike,
        H: ArrayLike,
        coni_curve: ArrayLike | None = None,
        coni_cob: ArrayLike | None = None,
        vertices: ArrayLike | str = "unknown (set when constructing CYData)",
        heights: ArrayLike | str = "unknown (set when constructing CYData)"):
        self._kappa = np.array(kappa)
        self._c2    = np.array(c2)
        self._H     = np.array(H)
        self._coni  = (coni_curve is not None)

        self._vertices = vertices
        self._heights  = heights

        # variables for the M-lattice
        self._a     = None
        self._b     = None
        self._M_lattice = None

        # misc
        self._h21   = h21
        self._h11   = self._kappa.shape[0]

        # check if non-Coni...
        if not self._coni:
            return

        # coni stuff below...
        # ===================
        # store information on the conifold curve if it is set
        self._coni_curve = np.array(coni_curve)
        type(self).coni_curve = property(lambda self: self._coni_curve.copy())

        # change of basis matrix to basis where coni_curve = (1,0,...,0)
        # --------------------------------------------------------------
        if coni_cob is None:
            # Use the HNF transform to find a unimodular integer matrix U
            # such that U @ coni_curve = (gcd, 0, ..., 0). Since coni_curve
            # is primitive (gcd = 1), this gives U @ coni_curve = (1, 0,...,0),
            # i.e. U is the desired change-of-basis matrix.
            q = np.array(self._coni_curve).reshape(-1,1)
            q = flint.fmpz_mat(q.tolist())
            self._cob = q.hnf(transform=True)[1]
            self._cob = np.array(self._cob.tolist()).astype(int)
        else:
            # user-input change of basis
            self._cob = np.array(coni_cob)

        # check the change of basis
        if not np.all(self._cob@self._coni_curve == [1]+[0]*(self.h11-1)):
            raise ValueError(
                "coni_cob is not a valid change of basis: "
                "cob @ coni_curve must equal (1, 0, ..., 0)."
            )
        type(self).cob = property(lambda self: self._cob.copy())


        # map kappa, c2, and H via this change of basis
        self._kappa_cob = np.einsum('ai,bj,ck,ijk->abc',
            self._cob, self._cob, self._cob, self._kappa)
        self._c2_cob = (self._cob@self._c2).reshape(-1)
        self._H_cob  = (self._H@self._cob.T)[:,1:]

        # delete rows of 0 from the hyperplanes
        nonzero_mask = self._H_cob.any(axis=1)
        self._H_cob = self._H_cob[nonzero_mask]

        type(self).kappa_cob = property(lambda self: self._kappa_cob.copy())
        type(self).c2_cob    = property(lambda self: self._c2_cob.copy())
        type(self).H_cob     = property(lambda self: self._H_cob.copy())

    # alternative constructor
    # -----------------------
    @classmethod
    def from_cy(cls,
        cy: "cytools.CalabiYau",
        coni_curve: "ArrayLike" = None,
        coni_cob: "ArrayLike" = None) -> "CYData":
        """
        Construct a CYData object from a cytools.CalabiYau object.

        Parameters
        ----------
        cy : cytools.CalabiYau
            The CYTools CalabiYau object of interest.
        coni_curve : ArrayLike, optional
            The conifold curve. If not provided, non-coni PFVs are assumed.
        coni_cob : ArrayLike, optional
            Change-of-basis matrix mapping the coni curve to (1,0,...,0).

        Returns
        -------
        CYData
            A fully initialized CYData object populated from the given
            CalabiYau.
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
        return cls(
            h21=cy.h21(),
            kappa=kappa,
            c2=c2,
            H=H,
            coni_curve=coni_curve,
            coni_cob=coni_cob,
            vertices=cy.polytope().vertices().tolist(),
            heights=cy.triangulation().heights().tolist())

    # standard class methods
    # ----------------------
    def __repr__(self):
        return str(self)

    def __str__(self):
        msg = f"Data defining an h11={self.h11} CY, for utility in making "
        if self.coni:
            msg += "Coni PFVs (original basis conifold curve = "
            msg += f"{self._coni_curve.tolist()})"
        else:
            msg += "PFVs"

        return msg

    # getters
    # -------
    @property
    def vertices(self) -> ArrayLike | str:
        return self._vertices
    verts = vertices

    @property
    def heights(self) -> ArrayLike | str:
        return self._heights

    @property
    def kappa(self) -> np.ndarray:
        return self._kappa.copy()

    @property
    def c2(self) -> np.ndarray:
        return self._c2.copy()

    @property
    def H(self) -> np.ndarray:
        return self._H.copy()

    @property
    def coni(self) -> bool:
        return self._coni

    @property
    def h11(self) -> int:
        return self._h11

    @property
    def h21(self) -> int:
        return self._h21

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
        if not self.coni:
            kappa = self._kappa
        else:
            kappa = self._kappa_cob
        h11 = kappa.shape[0]
        a   = np.zeros((h11, h11), dtype=int)

        # fill the matrix
        for x,y in itertools.product(range(h11), range(h11)):
            if x>=y:
                a[x,y] = kappa[x,x,y]
            else:
                a[x,y] = kappa[x,y,y]

        # trim if coni
        # (project out coni index)
        if self.coni:
            a = a[1:]

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
        if not self.coni:
            self._b = self.c2
            return self._b.copy()
        else:
            self._b = self.c2_cob

        # adjust for Coni
        self._b[0] = self._b[0]+2
        return self._b

    # basis for M-vectors
    # -------------------
    def M_lattice(self, verify: bool = True) -> np.ndarray:
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
        - `verify`: Whether to verify the computation by checking dot products
                    with a and b.

        **Returns:**
        A basis for M satisfying the integrality constraints. Basis vectors are
        columns.
        """
        # already known
        if self._M_lattice is not None:
            return self._M_lattice.copy()

        # compute from scratch
        h11 = self.kappa.shape[0]

        # define the constraints
        # ----------------------
        multiplier = 24

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
        if multiplier % denom != 0:
            raise RuntimeError(
                f"Dual lattice denominator {denom} does not divide "
                f"multiplier {multiplier}. This indicates an internal error."
            )
        out = dual * (multiplier//denom)

        # LLL-reduce the dual to be extra nice
        out = lattice.lll_reduce(out)
        if verify:
            if not np.all((self.a@out)%2 == 0):
                raise RuntimeError("M_lattice verification failed: a @ M is not even.")
            if not np.all((self.b.reshape(1,-1)@out)%24 == 0):
                raise RuntimeError("M_lattice verification failed: dot(b, M) is not divisible by 24.")

        self._M_lattice = out.copy()
        return out
