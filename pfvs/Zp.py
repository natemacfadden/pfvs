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
# Description:  This module contains methods for constructing non-coni PFVs
#               using the "Zp" style algorithms. These operate by fixing some
#               p-vectors and then searching for lattice points in an ellipsoid,
#               one for each p-vector.
# -----------------------------------------------------------------------------

# external imports
import numpy as np

from numpy.typing import ArrayLike

# local imports
from . import lattice
from .cydata import CYData

# Zp helpers
# ==========
# generic
# -------
def check_singular(Ns: ArrayLike, rtol: float=1e-12):
    # for a length-n stack of mxm matrices Ns (shape nxmxm), return a length-n
    # vector whose ith value is 1 iff Ns[i] is singular
    svals = np.linalg.svdvals(Ns)
    singular = svals[:,-1] <= rtol * svals[:,0]

    return singular

# non-coni
# --------
def allow_gcds(Ks: ArrayLike, Ms: ArrayLike, Qmax: int, h11: int):
    # non-coni PFVs naturally operate on primitive K,M (i.e., gcd(K)=gcd(M)=1)
    # allow re-introduction of nontrivial GCDs (up to the tadpole constraint)

    num_input = len(Ks)
    if num_input == 0:
        return np.zeros((0,h11),dtype=int), np.zeros((0,h11),dtype=int)

    # add the GCDs
    # (don't think there should be duplicates but cheap to do this in a set...)
    KMs_out = set()
    for K,M in zip(Ks,Ms):
        Qtmp = -np.dot(K,M)

        for a in range(1,Qmax//Qtmp+1):
            for b in range(1,Qmax//(a*Qtmp)+1):
                Ktmp = (a*K).tolist()
                Mtmp = (b*M).tolist()
                KMs_out.add(tuple(Ktmp+Mtmp))

    # split back into K and M arrays
    Ks, Ms = [], []
    for KM in KMs_out:
        Ks.append(KM[:h11])
        Ms.append(KM[h11:])

    # return
    if len(Ks):
        return np.vstack(Ks), np.vstack(Ms)
    else:
        msg = f"Found 0 PFVs after introducing GCDs... #input = {num_input}"
        raise ValueError(msg)

# non-coni Zp
# ===========
def ZpM(
    data: CYData,
    ps: ArrayLike,
    Qmax: int = None,
    Qmin: int = 0,
    ellipsoid_dilation: float = 1, # typically want >=1
    fp_recursive: bool = False,
    max_N_pfvs: int = 1_000_000_000,
    verbosity: int = 0
    ) -> tuple[ArrayLike, ArrayLike]:
    """
    A Python "Zp" implementation that takes in p-vectors and outputs PFVs.

    Operates by constructing a lattice of valid M-vectors and writing the
    K-vector in terms of M and p. Then the tadpole constraint 0 <= -K.M <= Q
    defines an ellipsoid of M-vectors

    returns a list of Ks and Ms
    """
    print("NON CONI Zp METHOD ARE SLIGHTLY OUTDATED... THEY EXCLUDE GCD PRUNING, E.G.")
    assert not data.coni

    # misc
    only_positive_news = False

    # read data
    kappa  = data.kappa
    Mbasis = data.M_lattice()

    if Qmax is None:
        Qmax = data.h11+data.h21+2

    # the search
    # ----------
    all_Ks = np.zeros((0,data.h11), dtype=int)
    all_Ms = np.zeros((0,data.h11), dtype=int)

    # iterate over p-vectors
    iterator = ps
    for _i, p in enumerate(iterator):
        # helper variable (K = Z@M)
        Z = kappa@p

        # define the lattices for M
        # -------------------------
        # need K.p = 0
        #
        # note that K = kappa@M@p
        # thus need dot(p, kappa@M@p) = 0
        # equivalently, dot(kappa@p@p, M) = 0
        T = kappa@p@p

        # need T.T @ Mbasis@c = 0
        # thus just need c in the orthogonal lattice to Mbasis.T @ T
        # (the output will be lattice generators of such cs... we'll want
        #  lattice generators of valid Ms so we multiply on left by Mbasis)
        Binter = Mbasis@lattice.orthogonal_lattice(p=T.T@Mbasis)

        # lll-reduce Binter
        # (doesn't seem to have a huge effect...)
        Binter = lattice.lll_reduce(Binter)

        # find lattice points in tadpole
        mat = -(Binter).T@(Z@Binter)

        if np.allclose(mat, np.round(mat)):
            mat = np.rint(mat).astype(int)

        #lattice_points = rejection_ellipsoid(mat,tadpole_mult*Q)
        try:
            lattice_points, _ = lattice.fp_ellipsoid_njit(
                mat=mat,
                Q=ellipsoid_dilation*Qmax,
                max_N_out=max_N_pfvs,
                recursive=fp_recursive,
                verbosity=verbosity-1)
        except Exception as e:
            print("ERROR!!!")
            print(f"LIKELY mat={mat.tolist()} ISN'T POSITIVE DEFINITE...")
            #print(f"vertices = {self.polytope().vertices().tolist()}")
            #print(f"heights  = {self.triangulation().heights().tolist()}")
            print(f"p        = {np.array(p).tolist()}")
            raise e

        # only keep primitive lattice points (can reclaim other PFVs easily)
        primitiveQ = np.gcd.reduce(lattice_points, axis=1) == 1
        lattice_points = lattice_points[primitiveQ]

        # compute Ms
        # ----------
        Ms = Binter@lattice_points.T # as columns

        # filter by N invertibility
        # -------------------------
        if True:#filter_Ninvertible:
            batch_size = 5000
            singular = []
            for i in range(0, len(Ms), batch_size):
                chunk = Ms[i:i+batch_size]

                #Ns = np.tensordot(kappa, Ms, axes=([2], [0]))
                Ns = (kappa.reshape(data.h11*data.h11,data.h11)@Ms).reshape(data.h11,data.h11,-1)
                Ns = Ns.transpose(2,0,1) # (N,h11,h11)

                #sign, logdet = np.linalg.slogdet(Ns)
                #is_zero = (sign == 0) | (logdet <= -1e-4)
                #singular.append(is_zero)
                singular.append(check_singular(Ns))

            singular = np.concatenate(singular)

            if verbosity >= 2:
                if len(singular) and not only_positive_news:
                    print(f"{sum(singular)}/{len(singular)} 'PFVs' had det(N)=0 :(")

            Ms = Ms[:,~singular]

        # compute Ks, reduce by GCDs
        # --------------------------
        Ks = Z@Ms

        #if reduce_gcds:
        #K_gcds = gcd_row(Ks.T)
        K_gcds = np.gcd.reduce(Ks, axis=0)
        Ks = Ks//K_gcds

        if True:#filter_tadpole:
            Qs = -np.sum(Ks*Ms,axis=0)
            in_tadpole = (Qs>Qmin) & (Qs<Qmax)
            if verbosity >= 2:
                if not only_positive_news:
                    print(f"{len(in_tadpole)-sum(in_tadpole)}/{len(in_tadpole)} 'PFVs' violated tadpole :(")
                if sum(in_tadpole):
                    print(f"but {sum(in_tadpole)} in tadpole!!!")
            Ks = Ks[:,in_tadpole]
            Ms = Ms[:,in_tadpole]

        # transpose to row-wise
        Ks, Ms = Ks.T, Ms.T

        # save to data structures
        all_Ks = np.vstack([all_Ks, Ks])
        all_Ms = np.vstack([all_Ms, Ms])

    # return
    return allow_gcds(all_Ks, all_Ms, Qmax, data.h11)

def ZpK(
    data: "cydata",
    ps: "ArrayLike",
    Qmax: int = None,
    Qmin: int = 0,
    ellipsoid_dilation: float = 1, # typically want >=1
    fp_recursive: bool = False,
    max_N_pfvs: int = 1_000_000_000,
    verbosity: int = 0
    ) -> tuple["ArrayLike", "ArrayLike"]:
    """
    A Python Zp implementation that takes in p-vectors and outputs PFVs

    filter_Ninvertible = whether the PFV has a invertible N matrix

    returns a list of Ks and Ms
    """
    print("NON CONI Zp METHOD ARE SLIGHTLY OUTDATED... THEY EXCLUDE GCD PRUNING, E.G.")
    assert not data.coni

    # misc
    only_positive_news = False

    # read data
    kappa  = data.kappa
    Mbasis = data.M_lattice()

    if Qmax is None:
        Qmax = data.h11+data.h21+2

    # the search
    # ----------
    all_Ks = np.zeros((0,data.h11), dtype=int)
    all_Ms = np.zeros((0,data.h11), dtype=int)

    # iterate over p-vectors
    iterator = ps
    for _i, p in enumerate(iterator):
        # helper variables
        A = kappa@p@Mbasis
        try:
            Ainv, scale = lattice.inv_scaled(A)#np.linalg.inv(A)
        except:
            print(f"PANIC!!!! {p.tolist()} CAUSED WEIRD AINV!!!")

        # define the lattices for K
        # -------------------------
        # (need K^T@p = 0)
        B = lattice.orthogonal_lattice(p=p)

        # lll-reduce B
        # (doesn't seem to have a huge effect...)
        B = lattice.lll_reduce(B)

        # find lattice points in tadpole
        mat = -B.T@np.linalg.inv(kappa@p)@B

        if np.allclose(mat, np.round(mat)):
            mat = np.rint(mat).astype(int)

        try:
            lattice_points, _ = lattice.fp_ellipsoid_njit(
                mat,
                ellipsoid_dilation*Qmax,
                max_N_out=max_N_pfvs,
                recursive=fp_recursive,
                verbosity=verbosity-1)
        except Exception as e:
            print("ERROR!!!")
            print(f"LIKELY mat={mat.tolist()} ISN'T POSITIVE DEFINITE...")
            print(f"p        = {np.array(p).tolist()}")
            raise e

        # only keep primitive lattice points (can reclaim other PFVs easily)
        primitiveQ = np.gcd.reduce(lattice_points, axis=1) == 1
        lattice_points = lattice_points[primitiveQ]

        # compute Ms, Ks, and reduced by GCD

        # read the data
        cs = ((Ainv@B)@lattice_points.T).T
        gcds = np.gcd.reduce(cs,axis=1)
        cs_scaled = cs//gcds.reshape(-1,1)

        Ks = B@lattice_points.T # as columns
        Ms = Mbasis@cs_scaled.T

        # filter
        if True:
            Qs = -np.sum(Ks*Ms,axis=0)
            in_tadpole = (Qs>Qmin) & (Qs<Qmax)
            if verbosity >= 2:
                if not only_positive_news:
                    print(f"{len(in_tadpole)-sum(in_tadpole)}/{len(in_tadpole)} 'PFVs' violated tadpole :(")
                if sum(in_tadpole):
                    print(f"but {sum(in_tadpole)} in tadpole!!!")
            Ks = Ks[:,in_tadpole]
            Ms = Ms[:,in_tadpole]

        # filter by N invertibility
        if True:
            batch_size = 5000
            singular = []
            for i in range(0, len(Ms), batch_size):
                chunk = Ms[i:i+batch_size]

                Ns = kappa.reshape(data.h11*data.h11,data.h11)@Ms
                Ns = Ns.reshape(data.h11,data.h11,-1)
                Ns = Ns.transpose(2,0,1) # (N,h11,h11)

                singular.append(check_singular(Ns))

            singular = np.concatenate(singular)

            if verbosity >= 2:
                if not only_positive_news:
                    print(f"{sum(singular)}/{len(singular)} 'PFVs' had det(N)=0 :(")

            Ks = Ks[:,~singular]
            Ms = Ms[:,~singular]

        # transpose to row-wise
        Ks, Ms = Ks.T, Ms.T

        # save to data structures
        all_Ks = np.vstack([all_Ks, Ks])
        all_Ms = np.vstack([all_Ms, Ms])

    # return
    return allow_gcds(all_Ks, all_Ms, Qmax, data.h11)
