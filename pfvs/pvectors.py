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
# Description:  p-vectors: integer p with H p > 0 (H from the Kahler cone, or
#               its coni facet).
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
    Primitive integer p with H p > 0: H is data.H (non-coni) or data.H_cob
    (coni). Returns every such p in the smallest L-inf box |p_i| <= B holding
    at least min_N_pts (`latticepts.enum_lattice_points`).

    Parameters
    ----------
    data : CYData
        The CY.
    min_N_pts : int
        Minimum number of p-vectors to return.
    verbosity : int, optional
        Defaults to 0.

    Returns
    -------
    pts : ndarray of shape (N, h11), or (N, h11-1) for coni
        The p-vectors, N >= min_N_pts.
    """
    if min_N_pts <= 0:
        raise ValueError(f"min_N_pts must be > 0, got {min_N_pts}.")

    # read hyperplanes (differs for coni and non-coni PFVs)
    if data.coni: H = data.H_cob
    else:         H = data.H

    return enum_lattice_points(
        H, rhs=1, min_N_pts=min_N_pts, primitive=True, verbosity=verbosity)
