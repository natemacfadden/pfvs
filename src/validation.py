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

# local imports
from . import lattice, cydata

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

        # GVs/series info
        self._gvs         = None
        self._series      = None
        self._all_exps    = []
        self._all_gvs     = []
        self._all_charges = []
        self._all_coeffs  = []

        # coni-specific variables
        if self.coni:
            # Kprime computation/check (must be positive)
            type(self).Kprime = property(lambda self: 
                -self.K[0] + (self.M@self._cy.kappa_cob@self.p)[0] )
            type(self).check_Kprime = lambda self: self.Kprime > 0

            # other physics-y variables
            self.ncf = 2

            type(self).gsM = property(lambda self: self.gs*self.M[0])
            type(self).Vtilde = property(lambda self: 
                (((self.kappa@self.p)@self.p)@self.p/6.)*np.imag(self.tau0)**(3) )
            type(self).zcf = property(lambda self:
                np.exp(-2*np.pi*self.Kprime/(self.ncf*self.gsM))/(2.*np.pi) )

            volProxy = (2*(2+self.h11+self.h21))**(3/2)
            type(self).align = property(lambda self:
                2*89.5643*(self.Vtilde**(1/3))*(volProxy**(2/3))*(self.zcf**(4/3))/(self.gsM*self.gsM*self.W0()*self.W0()) )

        # alternative constructor
    # -----------------------
    @classmethod
    def from_str(cls, str) -> "PFV":
        """
        **Description:**
        Initializes an instance of the PFV validation class

        **Arguments:**
        - `str`: The string format representing the PFV, in the format that.
        """
        try:
            import cytools
        except ImportError as e:
            raise ImportError(
                "cytools is required reading data from a string object..."
            ) from e

        ns = {}
        exec(str, {'Polytope':cytools.Polytope}, ns)

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
            cy=data,
            K=ns['K'],
            M=ns['M'])

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
        msg += f"#p = {self.pgrading.tolist()}/{self._p_denom}\n"
        Q = -np.dot(self.K,self.M)
        msg += f"#Q = {Q} = h11 + h21 + {Q-self.h11-self.h21}"

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
        # input sanitization
        val = np.array(val)

        # change the basis
        if self.coni:
            val[:,:-1] = val[:,:-1] @ np.array(self._cy.cob).T

        # sanity check: the p-vector should be in Kcup
        assert np.all(val[:,:-1]@self.p >= 0)
        
        # set the value :)
        self._gvs  = val.copy()

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

    # more sophisticated analyses
    # ===========================
    # main series method
    # ------------------
    def series_gen(self):
        assert self._gvs is not None

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

    def series(self, N_nonzero=float('inf'), verbosity=0):
        # initialize series container
        if self._series is None:
            self._series = []
            self._all_exps = []
            self._all_gvs = []
            self._all_charges = []
            self._all_coeffs = []

        len_series = len(self._series)

        # series is already long enough!
        if len_series >= N_nonzero:
            return self._series[:N_nonzero]

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

            elif len_series==0:
                # zero coefficient in the front
                self.coeff_gap += 1

        # built the entire series!
        return self._series

    def valid_coeff_ratio(self):
        # checks if the coefficients obey |c1| > |c0|
        terms = self.series(N_nonzero=2)
        if len(terms)<2:
            if (not self.silent):
                print(f'{self} had too few terms...')
            return None
        else:
            return abs(terms[1][0])>abs(terms[0][0])

    # main physics outputs
    # --------------------
    def tau0(self):
        # tau0 is the value of tau that minimizes the 2-term racetrack
        
        # check if PFV has valid leading coefficients
        validQ = self.valid_coeff_ratio()
        if validQ == False:
            return None
        elif validQ is None:
            return None

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
           as_logs=False,
           check_Ninvertible=True,
           verbosity=0):
        # check if PFV has valid N-rank
        if check_Ninvertible and (not self.check_Ninvertible()):
            return np.nan#float('inf')

        # check if PFV has valid leading coefficients
        validQ = self.valid_coeff_ratio()
        if validQ == False:
            return np.nan#float('inf')
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

        self._log10W0 = np.log10(base)*power
        self._log10W0 += np.log10(np.abs(c1*(p0-p1)/p0))
        self._log10W0 += np.log10(prefactor)

        if as_logs:
            return self._log10W0
        else:
            return np.power(10,self._log10W0)

    @property
    def gs(self):
        # check if PFV has valid leading coefficients
        if not self.valid_coeff_ratio():
            return None

        gs = 1/np.imag(self.tau0())
        
        if gs < 0:
            warnings.warn("Negative string coupling!")
        
        return gs

    # diagnostics
    # ===========
    def series_abs_vev(self, as_logs=False):
        # look at the series W = sum_i Wi for Wi = ci exp(2*pi*i*tau*p.qi)
        # find the value of exp(2*pi*i*tau) that minimizes W1+W2
        # plug that in to each Wi, return the (absolute value of the) result
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

    def series_corrections(self, as_logs=False):
        if not self.silent:
            print("THIS USES self.tau0() FROM THE 2-TERM APPROXIMATION")
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
    def diagnostics(self, verbosity=0):
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

        tau0 = self.tau0()
        gs = self.gs
        print( "Main diagnostics:")
        print( "-----------------")
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
        p_graded_degs = np.array(list(self._gvs[:,:-1]))@self.pgrading

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
        corrections = self.series_corrections(as_logs=True)
        plt.plot(range(2,2+len(corrections)),
                 corrections)
        plt.xlabel('ith term')
        plt.ylabel('log$_{10}(|W_i|/W_0)$')

    def dump_series(self,
                    max_deg=15,
                    p_grading=False,
                    n_digits=150,
                    verbosity=0):
        # dump it!
        if verbosity==0:
            for term in self.series(max_deg=max_deg,
                                    p_grading=p_grading,
                                    n_digits=n_digits):
                c,e = term
                print(f"Exponent {e:.2f} has coefficient {c}")
        elif verbosity==1:
            self.series(max_deg=max_deg,
                        p_grading=p_grading,
                        n_digits=n_digits)

            for c,e in zip(self._all_coeffs, self._all_exps):
                print(f"Exponent {e:.2f} has coefficient {sum(c)} "
                      f"arising from terms {c}...")
        else:
            self.series(max_deg=max_deg,
                        p_grading=p_grading,
                        n_digits=n_digits)

            print("(format is (GV_i)·(M.q_i) + ...)")

            for q,gv,c,e in zip(self._all_charges,
                                self._all_gvs,
                                self._all_coeffs,
                                self._all_exps):
                print(f"Exponent {e:.2f} has coefficient {sum(c)} = ", end="")
                
                first = True
                if verbosity>2:
                    print(f"{self.M.tolist()}·(", end="")
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
                        print(f"({_gv})·({_c//_gv})", end="")
                print()

    # auxiliary
    # =========
    def harvest(self, gvs=None, verbosity=0):
        assert self.coni
        import sys; sys.path.append('../../cornell-dev')
        from lib.physics.vacua import PFV_search
        from cytools import Polytope

        cy = Polytope(self._cy.verts).triangulate(heights=self._cy.heights).cy()

        if gvs is None:
            gvs = cy.compute_gvs(max_deg=10).coo


        out = PFV_search.harvestPFVs(cy, [[self.M,self.K,self.p]], gvs=gvs, max_deg=10, coni_LCS=True, basis_matrix=self._cy.cob, return_extra=True, output_format='dict', verbosity=verbosity)
        
        if 'Kprime' in out:
            out['Kprime'] = [-Kp for Kp in out['Kprime']]

        return out
    