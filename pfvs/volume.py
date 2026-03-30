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
# Description:  This module contains volume-based methods for estimating the
#               number of lattice points in the Zp ellipsoids.
# -----------------------------------------------------------------------------

# external imports
import numpy as np
import scipy as sp

from numpy.typing import ArrayLike

def vol_ball(r: float, dim: int) -> float:
    """
    **Description:**
    Computes the volume of the radius-r dim-ball,
        V_{dim}(r) = (pi^{dim/2} / Gamma(1 + dim/2)) * r^dim

    See https://en.wikipedia.org/wiki/Volume_of_an_n-ball#Closed_form

    **Arguments:**
    - `r`:   Radius of the ball
    - `dim`: Dimension of the ball.

    **Returns:**
    The volume of the ball.
    """
    prefactor = np.pow(np.pi, dim/2)/sp.special.gamma(1 + dim/2)
    return prefactor * (r**dim)

def vol_cap(r: float, dim: int, offset: float) -> float:
    """
    **Description:**
    Computes the volume of the spherical cap of a radius-r dim-ball, defined by
        |x|^2 <= r
        dot(x, dir) >= offset
    for any unit vector `dir` (volume is independent).

    See https://en.wikipedia.org/wiki/Spherical_cap#Hyperspherical_cap

    **Arguments:**
    - `r`:      Radius of the ball
    - `dim`:    Dimension of the ball.
    - `offset`: The offset of the affine hyperplane defining the cap.

    **Returns:**
    The volume of the spherical cap.
    """
    # quick check if this is 0
    if r < offset:
        return 0

    # compute the volume of the ball
    vol = vol_ball(r,dim)

    # compute the fraction of the ball obeying dot(x, dir) >= offset
    h    = r-offset
    frac = 0.5*sp.special.betainc(
        (dim+1)/2,
        1/2,
        (2*r*h - h**2)/(r**2)
    )

    # return
    return vol*frac

def estimate_num_cs(Q: float, L: ArrayLike, b: ArrayLike, offset: float) -> float:
    """
    Estimate the number of lattice vectors c satisfying
        c.T @ (L @ L.T) @ c <= Q   (ellipsoid)
        b.T @ c >= offset           (linear cut)
    using a volume approximation (one lattice point per unit volume).

    Parameters
    ----------
    Q : float
        Ellipsoid bound.
    L : ArrayLike
        Lower triangular matrix defining the quadratic form.
    b : ArrayLike
        Linear vector defining the spherical cap direction.
    offset : float
        Minimum value of dot(b, c).

    Returns
    -------
    float
        Estimated number of lattice points in the region.
    """
    # preliminary
    r   = np.sqrt(Q)
    dim = L.shape[1]

    # map to a ball via x = L.T @ c
    u      = np.linalg.inv(L) @ b
    offset = offset

    # normalize u
    u_norm = np.linalg.norm(u)
    offset = offset/u_norm

    # find volume of the cap in x-space
    vol = vol_cap(r, dim, offset)

    # map to c-space
    vol = vol/abs(np.linalg.det(L))

    # return (anticipate 1 c-vector for every unit volume in c-space)
    return vol
