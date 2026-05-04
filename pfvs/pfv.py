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
import matplotlib.pyplot as plt
import numpy as np
import warnings

from collections.abc import Generator
from numpy.typing import ArrayLike

# local imports
from . import util, cydata, coniZp
from .cydata import CYData

class PFV():
    """
    Stores and verifies (coni-)PFVs.

    Parameters
    ----------
    data : CYData
        A CYData object defining the CY.
    K : array-like
        K flux vector. Must be integral.
    M : array-like
        M flux vector. Must be integral.
    silent : bool, optional
        Whether to suppress messages. Defaults to False.
    """
    def __init__(self,
        data: CYData,
        K: ArrayLike,
        M: ArrayLike,
        silent: bool = False):
        # read inputs
        self._cydata      = data
        self._K           = np.array(K)
        self._M           = np.array(M)
        self.silent       = silent

        if self._K.shape != (data.h11,):
            raise ValueError(
                f"K must have shape ({data.h11},), got {self._K.shape}.")
        if self._M.shape != (data.h11,):
            raise ValueError(
                f"M must have shape ({data.h11},), got {self._M.shape}.")

        # initialize other variables
        self._p           = None
        self._pgrading    = None

        self._N           = None
        self._Ninvertible = None
        self._Ninv        = None

        # GVs/series info
        self._gvs         = None
        self._gvs_set     = False
        self._series      = []
        self._all_exps    = []
        self._all_gvs     = []
        self._all_charges = []
        self._all_coeffs  = []

        # coni-specific variables
        # -----------------------
        # These properties only exist on coni PFVs. We attach them to the
        # class dynamically (type(self).x = property(...)) rather than
        # defining them unconditionally with @property + a coni guard, so
        # that attribute lookup raises AttributeError for non-coni instances
        # instead of silently returning a wrong value.
        if self.coni:
            # coni curve/basis
            type(self).coni_curve = property(lambda self:
                self._cydata.coni_curve)
            type(self).cob        = property(lambda self:
                self._cydata.cob)

            # Kprime computation/check (must be positive)
            type(self).Kprime = property(lambda self:
                -self.K[0] + (self.M@self.kappa@self.p)[0] )
            type(self).check_Kprime = lambda self: self.Kprime > 0

            # other physics-y variables
            self.ncf = 2

            type(self).gsM = property(lambda self: self.gs*self.M[0])
            type(self).Vtilde = property(lambda self:
                (((self.kappa@self.p)@self.p)@self.p/6)*np.imag(self.tau0)**(3))
            type(self).zcf = property(lambda self:
                np.exp(-2*np.pi*self.Kprime/(self.ncf*self.gsM))/(2.*np.pi) )

            type(self).align = property(lambda self:
                2*89.5643
                * (self.Vtilde**(1/3))
                * (2*(2+self.h11+self.h21))
                * (self.zcf**(4/3))
                / (self.gsM*self.gsM*self.W0()*self.W0()) )

    # alternative constructor
    # -----------------------
    @classmethod
    def from_str(cls, str_: str) -> "PFV":
        """
        Construct a PFV from its string representation.

        Parameters
        ----------
        str_ : str
            String representation of a PFV, in the format produced by
            str(PFV(...)).

        Returns
        -------
        pfv : PFV
            The reconstructed PFV object.
        """
        try:
            import cytools
        except ImportError as e:
            raise ImportError(
                "cytools is required for reading data from a string object..."
            ) from e

        ns = {}
        exec(str_, {'Polytope':cytools.Polytope}, ns)

        # construct the CYData
        # --------------------
        if 'q' in ns:
            # coni...
            data = cydata.CYData.from_cy(
                cy=ns['cy'],
                coni_curve=ns['q'],
                coni_cob=ns['cob']
            )
        else:
            data = cydata.CYData.from_cy(ns['cy'])

        # construct the PFV
        # -----------------
        return cls(
            data=data,
            K=ns['K'],
            M=ns['M'])

    # basic
    # =====
    def __repr__(self):
        verts   = self.vertices
        heights = self.heights

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
            msg += f"q = {self.coni_curve.tolist()}\n"
            msg +=  "### using basis defined by change-of-basis\n"
            msg += f"cob = {self.cob.tolist()}\n"
        msg +=  "### with fluxes\n"
        msg += f"K = {self.K.tolist()}\n"
        msg += f"M = {self.M.tolist()}\n"
        msg += f"#p = {self.pgrading.tolist()}/{self._p_denom}\n"
        Q = -np.dot(self.K,self.M)
        msg += f"#Q = {Q} = h11 + h21 + {Q-self.h11-self.h21}"

        return msg

    def __str__(self):
        return self.__repr__()

    # getters
    # =======
    @property
    def coni(self) -> bool:
        """
        Whether this object describes a coni PFV.
        """
        return self._cydata.coni

    # CY info
    # -------
    @property
    def h11(self) -> int:
        """Hodge number h^{1,1} of the associated CY."""
        return self._cydata.h11

    @property
    def h21(self) -> int:
        """Hodge number h^{2,1} of the associated CY."""
        return self._cydata.h21

    @property
    def vertices(self) -> np.ndarray:
        """Vertices of the associated polytope."""
        return self._cydata.vertices
    verts = vertices

    @property
    def heights(self) -> np.ndarray:
        """Heights of the triangulation."""
        return self._cydata.heights

    @property
    def cy(self) -> "CalabiYau":
        """
        The CY object (requires cytools).
        """
        try:
            import cytools
        except ImportError as e:
            raise ImportError(
                "cytools is required for computation of a formal CY object"
            ) from e

        # check that we know the vertices and heights
        v = self.verts
        h = self.heights
        if isinstance(v,str):
            raise ValueError("Vertices were not known... must set them...")
        if isinstance(h,str):
            raise ValueError("Heights were not known... must set them...")

        # all good :) make the CY
        return cytools.Polytope(v).triangulate(heights=h).cy()

    @property
    def kappa(self) -> np.ndarray:
        """
        The intersection numbers
        """
        if self.coni:
            return self._cydata.kappa_cob
        else:
            return self._cydata.kappa

    # basic fluxes
    # ------------
    @property
    def K(self) -> np.ndarray:
        """
        The K-vector
        """
        return self._K.copy()

    @property
    def M(self) -> np.ndarray:
        """
        The M-vector
        """
        return self._M.copy()

    # search related properties
    # -------------------------
    @property
    def ellipsoid_mat(self) -> np.ndarray:
        """
        The matrix mat defining the M-ellipsoid: c^T @ mat @ c <= Q. Coni only.
        See also `ellipsoid_c`, `ellipsoid_required_dilation`.
        """
        if not self.coni:
            raise NotImplementedError

        mat, Z, Binter = coniZp.coni_M_ellipsoid(self.pgrading, self._cydata)
        return mat

    @property
    def ellipsoid_c(self) -> np.ndarray:
        """
        The lattice vector c such that M = Binter @ c. Coni only.
        See also `ellipsoid_mat`, `ellipsoid_required_dilation`.
        """
        if not self.coni:
            raise NotImplementedError

        mat, Z, Binter = coniZp.coni_M_ellipsoid(self.pgrading, self._cydata)
        c = np.rint(np.linalg.lstsq(Binter, self.M)[0]).astype(int)
        if not np.all(self.M == Binter@c):
            raise RuntimeError(
                "Failed to reconstruct c from M: M != Binter @ c. "
                "This indicates an internal inconsistency."
            )

        return c

    @property
    def ellipsoid_required_dilation(self) -> float:
        """
        The minimum ellipsoid dilation needed to capture this PFV. Coni only.

        Equal to c^T @ mat @ c / Q. See also `ellipsoid_mat`, `ellipsoid_c`.
        """
        if not self.coni:
            raise NotImplementedError

        mat = self.ellipsoid_mat
        c   = self.ellipsoid_c

        return (c.T@mat@c)/(self.h11+self.h21+4)

    # validation of fluxes
    # --------------------
    @property
    def H(self) -> np.ndarray:
        """
        The hyperplanes of the Kahler cone.
        """
        if self.coni:
            return self._cydata.H_cob
        else:
            return self._cydata.H

    @property
    def a(self) -> np.ndarray:
        """
        The a-matrix
        """
        return self._cydata.a

    @property
    def b(self) -> np.ndarray:
        """
        The b-vector
        """
        return self._cydata.b

    @property
    def gvs(self) -> np.ndarray | None:
        """
        The Gopakumar-Vafa invariants in coo format, or None if not yet set.
        """
        if self._gvs_set:
            return self._gvs.copy()

    # N-matrix and its inverse
    # ------------------------
    @property
    def N(self) -> np.ndarray:
        """
        The N-matrix, defined as kappa @ M. For coni, the 0th row and column
        are trimmed.
        """
        if self._N is None:
            self._N = self.kappa @ self._M

            if self.coni:
                self._N = self._N[1:,1:]

        return self._N.copy()

    @property
    def Ninv(self) -> tuple:
        """
        The scaled exact inverse of N, as a (flint matrix, scale) pair.
        """
        if self._Ninv is None:
            self._Ninv = util.inv_scaled(self.N, as_flint=True)

        return self._Ninv

    # p-vector
    # --------
    @property
    def p(self) -> np.ndarray:
        """
        The p-vector

        For non-coni PFVs, this is defined as `N.inv()@K`
        For     coni PFVs, this is defined as `concatenate([[0],N.inv()@K[1:]])`
        """
        if self._p is None:
            self._calc_p()

        return self._p

    @property
    def pgrading(self) -> np.ndarray:
        """
        The 'pgrading'-vector.

        This is just the primitive vector along the ray defined by p.
        I.e., the unique `pgrading = r*p` for r>0 such that `gcd(r*p) == 1`.
        """
        if self._pgrading is None:
            self._calc_p()

        return self._pgrading

    # p-vector computation
    # ====================
    def _calc_p(self) -> None:
        """
        Compute and cache ``self._p`` and ``self._pgrading``.

        Uses the relation N @ p = K (i.e., p = N^{-1} @ K) via the exact
        integer inverse ``self.Ninv``. The result is an integer grading vector
        ``pgrading`` (a scaled version of p with GCD 1) and a float ``p``
        obtained by dividing out the denominator.

        In the coni case the first component of p is constrained to be 0, so
        only K[1:] and the lower-right block of N are used.

        If N is singular (``check_Ninvertible`` returns False), both
        ``_p`` and ``_pgrading`` are filled with NaN.
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
            self._pgrading = np.array(self._pgrading.tolist()).astype(int).flatten()

        # save scaling
        self._p_denom = self.Ninv[1]/gcd # NON INTEGRAL

        # save p
        self._p = self._pgrading/self._p_denom

    # GVs
    # ===
    def compute_gvs(self, max_deg: int) -> None:
        """
        Compute GV invariants up to degree max_deg via cytools and store them.

        Parameters
        ----------
        max_deg : int
            Maximum degree of GV invariants to compute.
        """
        try:
            import cytools
        except ImportError as e:
            raise ImportError(
                "cytools is required for automatic computation of GVs..."
            ) from e

        self._gvs = self.cy.compute_gvs(max_deg=max_deg).coo
        self._gvs_set = True

    @gvs.setter
    def gvs(self, val: np.ndarray | None):
        """
        Accepts GVs in coo format
        """
        # input sanitization
        try:
            # standard input format is coo
            # try to see if we can cast the input to a numeric array...
            if not np.issubdtype(np.array(val).dtype, np.number):
                # likely passed the CYTools GV class
                raise ValueError
            else:
                # all good :)
                val = np.array(val)

        except Exception:
            # not arraylike... maybe CYTools GV class?
            try:
                val = val.coo
            except Exception:
                raise ValueError(
                    "Unknown GV format: expected a numeric array in coo "
                    "format, or a CYTools GV object with a .coo attribute."
                )

        # change the basis
        if self.coni:
            val[:,:-1] = val[:,:-1] @ np.array(self.cob).T

        # sanity check: the p-vector should be in Kcup
        if not np.all(val[:,:-1]@self.p >= 0):
            raise ValueError(
                "GV charges are not compatible with the p-vector: "
                "some charges q have dot(q, p) < 0."
            )

        # set the value :)
        self._gvs  = val.copy()
        self._gvs_set = True

    # checkers
    # ========
    def check_all(self, stop_at_fail: bool = True) -> bool:
        """
        Run all check_* methods. Returns True iff all pass.

        Runs check_Ninvertible first (required for p to exist), then the rest.
        If stop_at_fail is True, short-circuits on first failure.
        """
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

        # check them
        passes = True
        for check in checks:
            if not getattr(self, check)():
                # failed a check!
                if not self.silent:
                    # printit!
                    print(f"Check '{check}' failed...")
                passes = False
                if stop_at_fail:
                    break

        if not passes:
            if not self.silent:
                if isinstance(self.K,ValueError):
                    print("K = ERROR")
                else:
                    print(f"K = {self.K.tolist()}")
                print(f"M = {self.M.tolist()}")
            return False

        return True

    def check_a(self) -> bool:
        """Check that a @ M is even."""
        tmp = self.a@self.M

        return (tmp%2 == 0).all()

    def check_b(self) -> bool:
        """Check that dot(b, M) is a multiple of 24."""
        return np.dot(self.b, self.M)%24 == 0

    def check_tadpole(self) -> bool:
        """Check that 0 <= -dot(K, M) <= h11 + h21 + 2 + 2*coni."""
        upper = (self.h11+self.h21+2) + 2*self.coni
        return 0 <= -np.dot(self.M,self.K) <= upper

    def check_Knonzero(self) -> bool:
        """Check that K is nonzero."""
        return any(Ki != 0 for Ki in self.K)

    def check_Ninvertible(self, tol: float = 0.5) -> bool:
        """Check that N is full rank via |det(N)| > tol.

        Parameters
        ----------
        tol : float, optional
            Threshold for |det(N)|. Defaults to 0.5 (appropriate for integer
            matrices where det is an integer, so 0 vs non-zero is unambiguous).
        """
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}.")
        if self._Ninvertible is None:
            self._Ninvertible = np.abs(np.linalg.det(self.N))>tol
        return self._Ninvertible

    def check_pcontainment(self) -> bool:
        """
        Check that p is strictly contained in Kcup (the union of 2-face
        equivalent Kahler cones).
        """
        if self.coni:
            return min(self.H@self.pgrading[1:])>0.5
        else:
            return min(self.H@self.p)>0.5

    def check_NpK(self, tol: float = 1e-4) -> bool:
        """Check that N @ p = K. Requires `check_Ninvertible` to pass.

        Parameters
        ----------
        tol : float, optional
            Tolerance for the residual norm ||N @ p - K||. Defaults to 1e-4.
        """
        if self.coni:
            return (self.pgrading[0] == 0) and\
                   np.linalg.norm(self.N@self.p[1:] - self.K[1:])<tol
        else:
            return np.linalg.norm(self.N@self.p - self.K)<tol

    def check_orthogonality(self) -> bool:
        """Check that dot(K, p) = 0. Requires `check_Ninvertible` to pass."""
        if self.coni:
            return (self.pgrading[0] == 0) and\
                   (np.dot(self.pgrading[1:], self.K[1:]) == 0)
        else:
            return np.dot(self.pgrading, self.K) == 0

    # more sophisticated analyses
    # ===========================
    # main series method
    # ------------------
    def series_gen(self) -> Generator[tuple[float, float], None, None]:
        """
        Generator yielding the coefficient, exponent of each term in the series
        """
        if not self._gvs_set:
            raise ValueError("GVs must be set to get the series")

        # get the GVs, charges
        gvs = self._gvs[:,-1]
        Q = self._gvs[:,:-1]
        N_charges = len(Q)

        if N_charges==0:
            raise ValueError("Found 0 GV invariants... increase the degree!")

        # helpers
        Qpgrading = Q@self.pgrading
        QM        = Q@self.M

        # compute the sorted, unique exponents
        degs = np.unique(Qpgrading[Qpgrading>0])

        # compute the coefficients
        for deg in degs:
            self._all_exps.append(deg/self._p_denom)
            self._all_gvs.append([])
            self._all_charges.append([])
            self._all_coeffs.append([])

            # calculate the coefficient for this exponent
            coeff = 0
            for i in range(N_charges):
                if Qpgrading[i]==deg:
                    # save the coefficient
                    self._all_coeffs[-1].append(gvs[i]*int(QM[i]))

                    # save the charges and gvs
                    self._all_gvs[-1].append(int(gvs[i]))
                    self._all_charges[-1].append(tuple([int(qj) for qj in Q[i]]))

            # sum the terms to get the actual coefficient
            coeff = sum(self._all_coeffs[-1])

            yield coeff, deg/self._p_denom
        return

    def series(self, N_nonzero: int = float('inf'), verbosity: int = 0) -> list[tuple[float, float]]:
        """
        Compute the superpotential series W = sum_i c_i * exp(2*pi*i*tau*e_i),
        where e_i = dot(p, q) and c_i = sum_{q at e_i} n_q * dot(M, q).

        See also `series_abs_vev`, `series_corrections` for downstream
        diagnostics built on this series.

        Returns a list of [coeff, exponent] pairs for nonzero terms only,
        sorted by exponent. Stops after N_nonzero nonzero terms.
        """
        if N_nonzero <= 0:
            raise ValueError(f"N_nonzero must be > 0, got {N_nonzero}.")

        # initialize series container
        self._series = []
        self._all_exps = []
        self._all_gvs = []
        self._all_charges = []
        self._all_coeffs = []

        len_series = 0

        # need to build the series to size
        for coeff, exp in self.series_gen():
            if verbosity>0:
                print(f"Coefficients={coeff}; exponent={exp}")
            if coeff != 0:
                # nonzero coefficient!
                self._series.append([coeff, exp])
                len_series += 1
                if len_series >= N_nonzero:
                    return self._series

        # built the entire series!
        return self._series

    def valid_coeff_ratio(self) -> bool | None:
        """
        Check that the series has valid leading coefficients, i.e. |c1| > |c0|.
        Returns None if fewer than 2 terms are available.
        """
        terms = self.series(N_nonzero=2)
        if len(terms)<2:
            if (not self.silent):
                print(f'{self} had too few terms...')
            return None
        else:
            return abs(terms[1][0])>abs(terms[0][0])

    # main physics outputs
    # --------------------
    @property
    def tau0(self) -> complex:
        """
        The value of tau minimizing the 2-term racetrack potential.
        Returns nan if the series has invalid leading coefficients.
        """
        # check if PFV has valid leading coefficients
        validQ = self.valid_coeff_ratio()
        if validQ == False:
            return np.nan
        elif validQ is None:
            return np.nan

        # get the two leading terms
        terms = self.series(N_nonzero=2)
        c0, p0 = terms[0]
        c1, p1 = terms[1]

        # (np.log is natural log)
        self._tau0 = np.log( abs((c0*p0)/(c1*p1)) )
        if c0*c1>0:
            # imaginary part of log
            self._tau0 += 1j*np.pi

        self._tau0 *= 1j/(2*np.pi*(p0-p1))

        return self._tau0

    def W0(self,
           as_logs: bool = False,
           check_Ninvertible: bool = True,
           verbosity: int = 0) -> float:
        """
        Compute the flux superpotential W0 from the 2-term racetrack
        approximation.

        Returns the value (or log10 if as_logs=True), or nan if N is singular
        or the series has invalid leading coefficients.
        """
        # check if PFV has valid N-rank
        if check_Ninvertible and (not self.check_Ninvertible()):
            return np.nan

        # check if PFV has valid leading coefficients
        validQ = self.valid_coeff_ratio()
        if validQ == False:
            return np.nan
        elif validQ is None:
            return np.nan

        # helper
        prefactor = 1/((2.0**1.5)*(np.pi**2.5)) # roughly, 0.0202107

        # get the two leading terms
        terms = self.series(N_nonzero=2)
        c0, p0 = terms[0]
        c1, p1 = terms[1]

        # helpers (solely for clarity)
        base = abs(-(c0*p0)/(c1*p1))
        power = p1/(p1-p0)

        if verbosity>0:
            print(f"W0 = ({prefactor})*({c1*(p0-p1)/p0})*({base}**{power})")

        log10W0 = np.log10(base)*power
        log10W0 += np.log10(np.abs(c1*(p0-p1)/p0))
        log10W0 += np.log10(prefactor)

        if as_logs:
            return log10W0
        else:
            return np.power(10,log10W0)

    @property
    def gs(self) -> float:
        """
        The string coupling gs = 1 / Im(tau0).
        Returns nan if the series has invalid leading coefficients.
        """
        # check if PFV has valid leading coefficients
        if not self.valid_coeff_ratio():
            return np.nan

        gs = 1/np.imag(self.tau0)

        if gs < 0:
            warnings.warn("Negative string coupling!")

        return gs

    # diagnostics
    # ===========
    def series_abs_vev(self, as_logs: bool = False) -> list[float]:
        """
        Evaluate |W_i| at the tau0 from the 2-term approximation.

        Built on `series`. See also `series_corrections`.

        Specifically, finds the value of exp(2*pi*i*tau) minimizing W_0 + W_1,
        plugs it into each W_i = c_i * exp(2*pi*i*tau*e_i), and returns |W_i|
        (or log10|W_i| if as_logs=True), one per series term.
        """
        terms = self.series()
        coeffs, exps = zip(*terms)

        # some helper variables
        base = float(-(exps[0]*coeffs[0])/(exps[1]*coeffs[1]))
        base = abs(base) # only looking at absolute values

        if as_logs:
            log_base = math.log10(base)

        # compute the terms
        vevs = []
        for c,e in zip(coeffs,exps):
            power = float(e/(exps[1]-exps[0]))

            if not as_logs:
                term = abs(c)*(base**power)
            else:
                term = math.log10(abs(c))+power*log_base

            vevs.append(term)

        # return the (absolute values of the) VEVs of each term
        return vevs

    def series_corrections(self, as_logs: bool = False) -> list[float]:
        """
        Compute |W_i| / W0 for i >= 2, as a measure of higher-order corrections
        to the 2-term approximation. Returns the ratios (or log10 if
        as_logs=True), starting from the third series term.

        Built on `series_abs_vev`.
        """
        if not self.silent:
            print("THIS USES self.tau0 FROM THE 2-TERM APPROXIMATION")
        log_vevs = self.series_abs_vev(as_logs=True)

        log_W0 = self.W0(as_logs=True)

        corrections = []
        for i in range(2,len(log_vevs)):
            term = log_vevs[i]-log_W0

            if not as_logs:
                corrections.append(10**term)
            else:
                corrections.append(term)

        return corrections

    # diagnostics
    # -----------
    def diagnostics(self, verbosity: int = 0) -> None:
        """
        Print a summary of the PFV: checks, tadpole, W0, tau0, gs, and series
        info. At higher verbosity, dumps the full series and plots corrections.
        """
        # generic info
        print(f"Dumping info for:\n")
        print(self)
        print()

        # passes checks?
        if self.check_all():
            print("Passes checks :^)")
        else:
            for _ in range(20):
                print("FAILS CHECKS!!!")
        print()

        # tadpole saturation
        MK = np.dot(self.M, self.K)
        Q = self.h11 + self.h21 + 2
        print(f"-M.K/Q = {-MK}/{Q} = {-MK/Q}")
        if self.coni:
            print(f"# anti-D3 branes = (|M.K|-Q)/2 = ({-MK}-{Q})/2 = {(-MK-Q)/2}")
        print()

        # main takeaways
        logW0 = self.W0(as_logs=True)
        W0 = 10**logW0

        tau0 = self.tau0
        gs = self.gs
        print( "Main diagnostics:")
        print( "-----------------")
        if not self.valid_coeff_ratio():
            print("INVALID COEFFICIENT RATIO...")
            return

        print(f"W0   = 10**({logW0})")
        print(f"tau0 = {tau0}")
        print(f"gs   = {gs}")
        if W0==float('inf'):
            print("\n")
            for _ in range(10):
                print("There isn't a good coefficient ratio!")
            print()
        else:
            print()

        # coni-diagonstics
        if self.coni:
            print("Coni:")
            print("-----")
            print(f"K'   = {self.Kprime}")
            print(f"gsM  = {self.gsM}")
            print(f"zcf  = {self.zcf}")
            print(f"align= {self.align}")
            print()

        # degrees of leading terms in p-grading
        terms = self.series(N_nonzero=2)
        deg0 = int(round(terms[0][1]*self._p_denom))
        deg1 = int(round(terms[1][1]*self._p_denom))

        print( "Series:")
        print( "-------")
        print(f"The two leading terms have p-graded degrees: ", end="")
        print(f"{deg0} and {deg1}")
        print(f"Resultant W0 exponent is {deg1}/({deg1}-{deg0})=", end="")
        print(f"{deg1/(deg1-deg0)}...")
        print()

        # full series
        if verbosity>0:
            print("Dumping the series...")
            self.dump_series(verbosity=verbosity-1)
            print()

        # absolute value of vev (kinda redundant with plot)
        if verbosity>4:
            print("log10 |W_i|...")
            print(self.series_abs_vev(as_logs=True))
            print()

        # plot the data
        print("Plotting the series... evauluated at tau0 from 2-term")
        self.plot_series()

    def plot_series(self) -> None:
        """Plot log10(|W_i| / W0) vs. term index for i >= 2."""
        corrections = self.series_corrections(as_logs=True)
        plt.plot(range(2,2+len(corrections)),
                 corrections)
        plt.xlabel('ith term')
        plt.ylabel('log$_{10}(|W_i|/W_0)$')

    def dump_series(self, verbosity: int = 0) -> None:
        """
        Print the series terms. At verbosity=0, prints (exponent, coefficient)
        pairs. At verbosity=1, breaks each coefficient into its contributing
        terms. At verbosity>=2, also shows the individual GV invariants and
        charges.
        """
        if verbosity == 0:
            for term in self.series():
                c,e = term
                print(f"Exponent {e:.2f} has coefficient {c}")
        elif verbosity == 1:
            self.series()

            for c,e in zip(self._all_coeffs, self._all_exps):
                print(f"Exponent {e:.2f} has coefficient {sum(c)} "
                      f"arising from terms {c}...")
        else:
            self.series()

            print("(format is (GV_i)*(M.q_i) + ...)")

            for q,gv,c,e in zip(self._all_charges,
                                self._all_gvs,
                                self._all_coeffs,
                                self._all_exps):
                print(f"Exponent {e:.2f} has coefficient {sum(c)} = ", end="")

                first = True
                if verbosity>2:
                    print(f"{self.M.tolist()}*(", end="")
                    for _q,_gv in zip(q, gv):
                        if not first:
                            print(" + ", end="")
                        else:
                            first = False
                        print(f"({_gv} {_q})", end="")
                    print(")",end="")

                else:
                    for _c,_gv in zip(c,gv):
                        if not first:
                            print(" + ", end="")
                        else:
                            first = False
                        print(f"({_gv})*({_c//_gv})", end="")
                print()
