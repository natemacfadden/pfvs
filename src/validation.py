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
import numpy as np

# local imports
from . import lattice

class PFV():
    def __init__(self,
        data: "cydata",
        K: "ArrayLike",
        M: "ArrayLike",
        silent: bool = False):
        self._cydata = data
        self._K      = np.array(K)
        self._M      = np.array(M)

        # initialize other variables
        self._p      = None
        self._pgrading = None

        self._N      = None
        self._Ninv   = None

        # whether to print anything
        self.silent  = silent

    # getters
    # -------
    @property
    def coni(self):
        return self._cydata.coni

    # basic fluxes
    @property
    def K(self):
        # the K-vector
        return self._K.copy()

    @property
    def M(self):
        # the M-vector
        return self._M.copy()

    # N-matrix and its inverse
    # ------------------------
    @property
    def N(self):
        # the N-matrix
        # defined as kappa @ M
        # for coni, the 0th row and column are trimmed
        if self._N is None:
            if self.coni:
                self._N = self._cydata._kappa_cob @ self._M
                self._N = self._N[1:,1:]
            else:
                self._N = self._cydata._kappa @ self._M

        return self._N.copy()

    @property
    def Ninv(self):
        if self._Ninv is None:
            self._Ninv = lattice.inv_scaled(self.N)

        return self._Ninv

    # p-vector
    # --------
    @property
    def p(self):
        # the p-vector
        # for non-coni, this is defined as `N.inv() @ K`
        # for coni, this is defined as `concatenate([[0], N.inv() @ K[1:]])`
        if self._p is None:
            self._calc_p()
            
        return self._p

    @property
    def pgrading(self):
        if self._pgrading is None:
            self._calc_p()
            
        return self._pgrading

    def _calc_p(self):
        try:
            # calc pgrading
            # (uses Ninv, which is inv(N), scaled to be integral)
            if self.coni:
                self._pgrading = np.zeros(len(self.K), dtype=int)
                self._pgrading[1:] = self.Ninv[0]@self.K[1:]
            else:
                self._pgrading = self.Ninv[0]@self.K

            # remove the gcd of pgrading, save scaling
            gcd = np.gcd.reduce(self._pgrading)
            self._pgrading = self._pgrading//gcd
            self._p_denom = self.Ninv[1]/gcd # NON INTEGRAL

            # save p
            self._p = self._pgrading/self._p_denom

        except:
            raise ValueError("N likely isn't invertible...")

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
        tmp = self._cydata.a@self.M

        return (tmp%2 == 0).all()
    
    def check_b(self):
        # check that b.M is a multiple of 24
        return np.dot(self._cydata.b, self.M)%24 == 0

    def check_tadpole(self):
        # check tadpole bound
        upper = (self._cydata.h11+self._cydata.h21+2) + 2*self.coni
        return 0 <= -np.dot(self.M,self.K) <= upper

    def check_Knonzero(self):
        # check that K 
        return any([Ki!=0 for Ki in self.K])

    def check_Ninvertible(self, tol=0.5):
        # check that N is full rank
        # use determinant
        return np.abs(np.linalg.det(self.N))>tol

    def check_pcontainment(self):
        # check that p is *strictly* contained in Kcup (the union of 2-face
        # equivalent kahler cones)
        if self.coni:
            return min(self._cydata._H_cob@self.pgrading[1:])>0.5
        else:
            return min(self._cydata._H@self.p)>0.5
    
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
    