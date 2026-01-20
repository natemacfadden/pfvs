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
# Description:  This module contains utilities for validating PFVs.
# -----------------------------------------------------------------------------

# external imports
import flint
import functools
import math
import numpy as np

# local imports
from . import lattice

class PFV():
    """
    Class to stores/verifies (Coni-)PFVs.

    **Arguments:**
    - `cy`: A CYData object defining the CY.
    - `K`: K flux vector. Must be integral.
    - `M`: M flux vector. Must be integral.
    - `silent`: Whether to suppress messages.
    """
    def __init__(self,
        cy: "CYData",
        K: "ArrayLike",
        M: "ArrayLike",
        silent: bool = False):
        """
        Class to stores/verifies (Coni-)PFVs.

        **Arguments:**
        - `cy`: A CYData object defining the CY.
        - `K`: K flux vector. Must be integral.
        - `M`: M flux vector. Must be integral.
        - `silent`: Whether to suppress messages.
        """
        # read inputs
        self._cy    = cy
        self._K     = np.array(K)
        self._M     = np.array(M)
        self.silent = silent

        # initialize other variables
        self._p        = None
        self._pgrading = None

        self._N           = None
        self._Ninvertible = None
        self._Ninv        = None

        self._gvs         = None

        # coni-specific variables
        if self.coni:
            # Kprime computation/check (must be positive)
            type(self).Kprime = property(lambda self: 
                -self.K[0] + (self.M@self._cy.kappa_cob@self.p)[0] )
            type(self).check_Kprime = lambda self: self.Kprime > 0

    # basic
    # =====
    def __repr__(self):
        verts   = self._cy.vertices
        heights = self._cy.heights

        msg =  f"### An (h11,h21)={self.h11, self.h21} "
        msg += f"{'coni-' if self.coni else ''} PFV defined via\n"
        msg += f"verts   = {verts}\n"
        msg += f"heights = {heights}\n"
        if isinstance(verts, str) or isinstance(heights, str):
           pass
        else:
            #msg += f"    cy      = Polytope({verts}).triangulate(heights={heights}).cy()\n"
            msg += f"cy      = Polytope(verts).triangulate(heights=heights).cy()\n"
        if self.coni:
            msg +=  "### for (original-basis) conifold curve\n"
            msg += f"q = {self._cy.coni_curve.tolist()}\n"
            msg +=  "### using basis defined by change-of-basis\n"
            msg += f"cob = {self._cy.cob.tolist()}\n"
        msg +=  "### with fluxes\n"
        msg += f"K = {self.K.tolist()}\n"
        msg += f"M = {self.M.tolist()}\n"
        msg += f"p = {self.pgrading.tolist()}/{self._p_denom}\n"
        Q = -np.dot(self.K,self.M)
        msg += f"Q = {Q} = h11 + h21 + {Q-self.h11-self.h21}"

        return msg

    def __str__(self):
        return self.__repr__()

    # getters
    # =======
    @property
    def h11(self):
        return self._cy.h11

    @property
    def h21(self):
        return self._cy.h21

    @property
    def coni(self):
        """
        Whether this object describes a coni PFV.
        """
        return self._cy.coni

    # basic fluxes
    @property
    def K(self):
        """
        The K-vector
        """
        return self._K.copy()

    @property
    def M(self):
        """
        The M-vector
        """
        return self._M.copy()

    @property
    def kappa(self):
        """
        The intersection numbers
        """
        if self.coni:
            return self._cy.kappa_cob
        else:
            return self._cy.kappa

    @property
    def a(self):
        """
        The a-matrix
        """
        return self._cy.a

    @property
    def b(self):
        """
        The b-vector
        """
        return self._cy.b

    @property
    def gvs(self):
        if self._gvs is not None:
            return self._gvs.copy()

    # setters
    # =======
    @gvs.setter
    def gvs(self, val):
        """
        in coo format
        """
        # set GVs

        # if using p as the grading vector, set it
        if p_grading:
            grading_vec = self.pgrading
        else:
            grading_vec = None

        # check if we need to (re)compute GVs
        bad_GVs = False
        if not hasattr(self.cy,'_gvs'):
            bad_GVs = True
        else:
            if self.cy._gvs.cutoff != max_deg:
                bad_GVs = True
            elif not hasattr(self.cy._gvs,'_pgraded') or\
                    self.cy._gvs._pgraded != p_grading:
                bad_GVs = True
        
        # (re)compute GVs, if needed
        if bad_GVs:
            if (not self.silent) and (verbosity>0):
                msg = "(Computing GVs "
                if p_grading:
                    msg += "with "
                else:
                    msg += "without "
                msg += f"p-grading; max degree={max_deg}... might take time..."
                print(msg,end=' ')

            self.cy._gvs = self.cy.compute_gvs(grading_vec=grading_vec,
                                               max_deg=max_deg,
                                               basis=(self.cob if self.coni else None))
            self.cy._gvs._pgraded = p_grading

            # reset dependent variables
            self._tau0 = None
            self._log10W0 = None
            self._gs = None
            self._series = None
            self._series_gen = None

            if (not self.silent) and (verbosity>0):
                print('done!)\n')

        # return the gvs
        return self.cy._gvs

    # N-matrix and its inverse
    # ------------------------
    @property
    def N(self):
        # the N-matrix
        # defined as kappa @ M
        # for coni, the 0th row and column are trimmed
        if self._N is None:
            self._N = self.kappa @ self._M

            if self.coni:
                self._N = self._N[1:,1:]

        return self._N.copy()

    @property
    def Ninv(self):
        if self._Ninv is None:
            self._Ninv = lattice.inv_scaled(self.N, as_flint=True)

        return self._Ninv

    # p-vector
    # --------
    @property
    def p(self):
        """
        The p-vector

        For non-coni PFVs, this is defined as `N.inv() @ K`
        For     coni PFVs, this is defined as `concatenate([[0], N.inv() @ K[1:]])`
        """
        if self._p is None:
            self._calc_p()
            
        return self._p

    @property
    def pgrading(self):
        """
        The 'pgrading'-vector.

        This is just the primitive vector along the ray defined by p.
        I.e., the unique `pgrading = r*p` for r>0 such that `gcd(r*p) == 1`.
        """
        if self._pgrading is None:
            self._calc_p()
            
        return self._pgrading

    def _calc_p(self):
        """
        Compute the p-vector and pgrading.
        """
        # this computation only makes sense if N is invertible
        if not self.check_Ninvertible():
            self._pgrading = np.empty(len(self.K), dtype=int)
            self._p        = np.empty(len(self.K), dtype=float)
            self._pgrading[:] = np.nan
            self._p[:]        = np.nan
            return
        # N is invertible! the following should work...
        
        # calc pgrading
        # (uses Ninv, which is inv(N), scaled to be integral)
        if self.coni:
            self._pgrading = np.zeros(len(self.K), dtype=int)

            # the coni components of p
            K_flint = flint.fmpz_mat(self.K[1:].reshape(-1,1).tolist())
            tmp = self.Ninv[0]*K_flint

            # remove the gcd
            gcd = functools.reduce(math.gcd, tmp)
            tmp = tmp/gcd
            tmp = tmp.transpose()
            tmp = np.array(tmp.tolist()).astype(int)
            self._pgrading[1:] = tmp
        else:
            K_flint = flint.fmpz_mat(self.K.reshape(-1,1).tolist())
            self._pgrading = self.Ninv[0]*K_flint

            # remove the gcd
            gcd = functools.reduce(math.gcd, self._pgrading)
            self._pgrading = self._pgrading/gcd
            self._pgrading = self._pgrading.transpose()
            self._pgrading = np.array(self._pgrading.tolist()).astype(int)

        # save scaling
        self._p_denom = self.Ninv[1]/gcd # NON INTEGRAL

        # save p
        self._p = self._pgrading/self._p_denom

    # checkers
    # ========
    def check_all(self, stop_at_fail=True):
        # master checking method (calls all functions that begin with 'check_')

        # get check methods
        checks = [func for func in dir(self) if func[:6] == 'check_']
        checks = [check for check in checks if check != 'check_all']

        # always check N-rank first
        if 'check_Ninvertible' in checks:
            checks.remove('check_Ninvertible')
            checks.insert(0, 'check_Ninvertible')

        if 'check_Knonzero' in checks:
            checks.remove('check_Knonzero')
            checks.insert(0, 'check_Knonzero')

        # remove evenness check, optionally
        if ('check_even' in checks) and (not self.require_evenness):
            checks.remove('check_even')
        
        # check them
        passes = True
        for check in checks:
            if not self.__getattribute__(check)():
                # failed a check!
                if not self.silent:
                    # printit!
                    print(f"Check '{check}' failed...")
                passes = False
                if stop_at_fail:
                    break

        if not passes:
            if not self.silent:
                print(f"K = {self.K.tolist() if not isinstance(self.K,ValueError) else 'ERROR'}")
                print(f"M = {self.M.tolist()}")
            return False

        return True

    def check_a(self):
        # check that a@M is even
        tmp = self.a@self.M

        return (tmp%2 == 0).all()
    
    def check_b(self):
        # check that b.M is a multiple of 24
        return np.dot(self.b, self.M)%24 == 0

    def check_tadpole(self):
        # check tadpole bound
        upper = (self.h11+self.h21+2) + 2*self.coni
        return 0 <= -np.dot(self.M,self.K) <= upper

    def check_Knonzero(self):
        # check that K 
        return any([Ki!=0 for Ki in self.K])

    def check_Ninvertible(self, tol=0.5):
        # check that N is full rank
        # use determinant
        if self._Ninvertible == None:
            self._Ninvertible = np.abs(np.linalg.det(self.N))>tol
        return self._Ninvertible

    def check_pcontainment(self):
        # check that p is *strictly* contained in Kcup (the union of 2-face
        # equivalent kahler cones)
        if self.coni:
            return min(self._cy._H_cob@self.pgrading[1:])>0.5
        else:
            return min(self._cy._H@self.p)>0.5
    
    def check_NpK(self, tol=1e-4):
        # check that N@p=K
        if self.coni:
            return (self.pgrading[0] == 0) and\
                   np.linalg.norm(self.N@self.p[1:] - self.K[1:])<tol
        else:
            return np.linalg.norm(self.N@self.p - self.K)<tol

    def check_orthogonality(self):
        # check that K.p=0
        if self.coni:
            return (self.pgrading[0] == 0) and\
                   (np.dot(self.pgrading[1:], self.K[1:]) == 0)
        else:
            return np.dot(self.pgrading, self.K) == 0

    # auxiliary
    # =========
    