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
# Description:  This module contains methods for constructing p-vectors. These
#               are vectors p such that H@p>0 for H defined by the Kahler cone.
#               The exact definition of H varies between coni and non-coni.
# -----------------------------------------------------------------------------

# external imports
import numpy as np
import warnings
from ortools.sat.python import cp_model

from numpy.typing import ArrayLike

# local imports
from latticepts import enum_lattice_points
from .cydata import CYData

def pvecs(
    data: CYData,
    min_N_pts: int,
    verbosity: int = 0) -> np.ndarray:
    """
    Generate primitive p-vectors using a branch-and-bound search (Kannan).

    I.e., finds integral vectors p satisfying H @ p > 0, where H are the
    hyperplanes of the associated Kahler cone (for non-coni PFVs) or the
    hyperplanes of a particular facet of this Kahler cone (for coniPFVs). Only
    primitive vectors (GCD(p) = 1) are returned.

    Wraps `latticepts.enum_lattice_points`, which searches within an L-inf box
    |p_i| <= B and iteratively increases B until at least `min_N_pts` p-vectors
    are found.

    Parameters
    ----------
    data : CYData
        The relevant data from the associated CY, providing the hyperplane
        matrix H (or H_cob for coni).
    min_N_pts : int
        Minimum number of primitive p-vectors to return.
    verbosity : int, optional
        The verbosity level. Higher is more verbose. Defaults to 0.

    Returns
    -------
    pts : ndarray of shape (N, h11)
        Array of primitive p-vectors, where N >= `min_N_pts`. Each row is an
        integer vector satisfying H @ p > 0
    """
    if min_N_pts <= 0:
        raise ValueError(f"min_N_pts must be > 0, got {min_N_pts}.")

    # read hyperplanes (varies for coni and non-coni PFVs)
    if data.coni: H = data.H_cob
    else:         H = data.H

    return enum_lattice_points(
        H, rhs=1, min_N_pts=min_N_pts, primitive=True, verbosity=verbosity)

def pvecs_cpsat(
    data: CYData,
    min_N_pts: int | None = None,
    max_deg: int | None = None,
    min_deg: int = 0,
    max_Linf: int | None = None,
    deg_window: int = 1,
    max_window_i: int = 10_000,
    max_time: float = 120, # 2 min...
    keep_primitive: bool = True,
    verbosity: int = 0) -> np.ndarray:
    """
    (Not recommended in practice - use `pvecs` instead, which is faster)

    Generate primitive p-vectors using CP-SAT (OR-Tools).

    Same goal as `pvecs`: finds integral vectors p satisfying H @ p > 0, where H
    are the hyperplanes of the Kahler cone (non-coni) or a particular facet of
    the Kahler cone (coni).

    Parameters
    ----------
    data : CYData
        The CY geometry, providing the hyperplane matrix H (or H_cob for coni).
    min_N_pts : int, optional
        Minimum number of p-vectors to return. Inclusive. Mutually exclusive
        with `max_deg`
    max_deg : int, optional
        Maximum degree to enumerate p-vectors to. Inclusive. Mutually exclusive
        with `min_N_pts`
    min_deg : int, optional
        Minimum degree for p-vectors. Default is 0
    max_Linf : int, optional
        L-inf bound on p-vector components. Defaults to INT32_MAX - 1 if unset
        (not recommended)
    deg_window : int, optional
        Width of the sliding degree window used in `min_N_pts` mode. Default 1
    max_window_i : int, optional
        Maximum number of degree windows to try in `min_N_pts` mode. Default
        10,000
    max_time : float, optional
        Time limit in seconds per degree window. Default is 120
    keep_primitive : bool, optional
        If True, remove p-vectors with non-trivial GCD. Default is True
    verbosity : int, optional
        The verbosity level. Higher is more verbose. Defaults to 0

    Returns
    -------
    pts : ndarray of shape (N, h11)
        Array of p-vectors satisfying H @ p > 0. Can have N < min_N_pts if
        max_window_i is exhausted.
    """
    # check inputs
    if (max_deg is None) ^ (min_N_pts is None):
        pass # good - exactly one of max_deg or min_N_pts was set
    else:
        raise ValueError("Either `max_deg` or `min_N_pts` must be set...")
    if max_deg is not None and max_deg < min_deg:
        raise ValueError(f"max_deg ({max_deg}) must be >= min_deg ({min_deg}).")
    if max_Linf is not None and max_Linf <= 0:
        raise ValueError(f"max_Linf must be > 0, got {max_Linf}.")
    if deg_window <= 0:
        raise ValueError(f"deg_window must be > 0, got {deg_window}.")

    # read hyperplanes
    if data.coni: H = data.H_cob
    else:         H = data.H

    # requesting N points
    # -------------------
    # use shifting degree windows until we get enough points
    if min_N_pts is not None:
        ps   = np.empty((0,H.shape[1]), dtype=int)
        N_ps = 0

        for window_i in range(max_window_i+1):
            _min = min_deg + (window_i+0)*deg_window + window_i
            _max = min_deg + (window_i+1)*deg_window + window_i

            if verbosity >= 1:
                print(f"Have {N_ps} p-vectors for degrees <{_min}",
                      end=' '*20 + '\r')

            # compute new p-vectors, save them
            new_ps = pvecs_cpsat(
                data,
                min_deg = _min,
                max_deg = _max,
                max_Linf = max_Linf,
                max_time = max_time,
                keep_primitive = keep_primitive)
            if len(new_ps):
                ps     = np.vstack([ps, new_ps])
                N_ps  += len(new_ps)

                # break if done
                if N_ps >= min_N_pts:
                    break

        return ps

    # solving via min/max degree
    # --------------------------
    # compute a grading vector
    # (just needs to be in strict interior of dual cone)
    grading = H.sum(axis=0)
    grading = grading//np.gcd.reduce(grading)

    # define a constraint-programming model to solve
    model  = cp_model.CpModel()
    if max_Linf is None:
        warnings.warn(
            "max_Linf not set; search will be very slow. "
            "Setting max_Linf explicitly is strongly recommended.",
            stacklevel=2
        )
        max_Linf = cp_model.INT32_MAX - 1

    p_vars = [model.NewIntVar(-max_Linf, max_Linf, f'x{i}') for i in\
                                                        range(H.shape[1])]

    for row in H:
        model.Add(sum(int(row[j])*p_vars[j] for j in range(H.shape[1])) >= 1)
    model.Add(sum(int(grading[j])*p_vars[j] for j in range(H.shape[1])) >=\
                                                                    min_deg)
    model.Add(sum(int(grading[j])*p_vars[j] for j in range(H.shape[1])) <=\
                                                                    max_deg)

    # enumerate all solutions up to max_deg
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time

    class SolutionPrinter(cp_model.CpSolverSolutionCallback):
        def __init__(self, p_vars):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.p_vars = p_vars
            self.points = []

        def on_solution_callback(self):
            self.points.append([self.Value(v) for v in self.p_vars])

    printer = SolutionPrinter(p_vars)

    status = solver.SearchForAllSolutions(model, printer)
    if status == cp_model.OPTIMAL:
        pass
    elif status == cp_model.UNKNOWN:
        print("LIKELY HIT TIME LIMIT!!!")
        print(min_deg, max_deg)
    elif status != cp_model.INFEASIBLE:
        print("MISC ERROR")
        print(min_deg, max_deg, status)

    # return
    ps = np.array(printer.points)
    if len(ps) == 0:
        return np.empty((0,H.shape[1]), dtype=int)

    # reduce by GCD
    if keep_primitive:
        gcds = np.gcd.reduce(ps,axis=1)
        ps = ps[gcds==1]
    return ps

def mindeg_pvec_gurobi(
    data: CYData,
    max_deg: int | None = None,
    verbosity: int = 0) -> ArrayLike:
    """
    Generate the minimum-degree (relative to grading vector sum(H,axis=0) / GCD)
    p-vector using Gurobi. See `pvecs` docstring for more detail on what
    p-vectors are.

    If max_deg = None, gives the single smallest-degree p-vector. Otherwise, it
    gives (a non-exhaustive) list of p-vectors up to the stated max_deg.

    Parameters
    ----------
    data : CYData
        The CY geometry, providing the hyperplane matrix H (or H_cob for coni).
    max_deg : int, optional
        Maximum degree to enumerate p-vectors to.
    verbosity : int, optional
        The verbosity level. Higher is more verbose. Defaults to 0.

    Returns
    -------
    pts : ndarray of shape (h11,) or (N, h11)
        Array of p-vectors satisfying H @ p > 0. Shape (h11,) if max_deg=None,
        in which case this is the minimum degree vector. Shape (N,h11) if
        max_deg!=None, in which case these are some low degree p-vectors up to
        the maximum degree. No certificate that this is comprehensive...
    """
    # import gurobi (only needed for this function)
    import gurobipy as gp

    # setup
    # -----
    # read hyperplanes
    if data.coni: H = data.H_cob
    else:         H = data.H

    # compute a grading vector
    # (just needs to be in strict interior of dual cone)
    grading = H.sum(axis=0)
    grading = grading//np.gcd.reduce(grading)

    # define the model
    # ----------------
    model = gp.Model('pFinder')
    model.setParam('OutputFlag', (verbosity > 0))

    p = model.addMVar(
        (H.shape[1],),
        lb=-float('inf'), ub=float('inf'),
        vtype=gp.GRB.INTEGER)
    model.setMObjective(
        Q=None,
        c=grading,
        constant=0,
        xc=p,
        sense=gp.GRB.MINIMIZE)
    model.addMConstr(H, p, '>', np.full(len(H),0.5)) # equiv to H@p>0 for int p
    if max_deg is not None:
        model.addMConstr(grading.reshape(1,-1), p, '<=', [max_deg])

    # optimize
    # --------
    if max_deg is None:
        model.setParam('PoolSolutions', 1)
    else:
        model.setParam('PoolSearchMode', 1)
        model.setParam('PoolSolutions',  1_000_000_000) # effectively unlimited
    model.optimize()

    # retrieve and print the solutions
    nSolutions = model.SolCount
    if verbosity>=1:
        print(f"Found {nSolutions} solutions")

    model.Params.outputFlag = 0 # avoid clutter when extracting data
    sols = []
    for i in range(nSolutions):
        model.setParam('SolutionNumber', i)

        sols.append(np.rint(p.xn).astype(int))
    model.setParam('SolutionNumber', 0)

    if len(sols) == 1:
        return sols[0]
    else:
        return np.array(sols)
