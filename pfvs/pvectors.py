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

    # read hyperplanes (differs for coni and non-coni PFVs)
    if data.coni: H = data.H_cob
    else:         H = data.H

    return enum_lattice_points(
        H, rhs=1, min_N_pts=min_N_pts, primitive=True, verbosity=verbosity)
