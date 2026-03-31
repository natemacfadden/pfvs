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

import numpy as np

from pfvs.c_kernels import coni_kernel
from pfvs.lattice import coni_kernel_njit

# =============================================================================
# Hard-coded Manwe data (from https://arxiv.org/abs/2406.13751)
# Extracted from CYTools.
# =============================================================================

U = np.array([
    [19.131126469708992,  -2.50900019274872,  -12.022292590254283,   9.408750722807701,  10.97687584327565,   8.363333975829066,   24.77637690339361 ],
    [ 0.0,               20.8255832579255,     30.243381795036953,  31.864968637769284,  12.462603346669052,   6.193517125542706,    6.826408301055712],
    [ 0.0,                0.0,                 31.73014873072524,   -6.636894774358689,  15.791162726449791,  12.771246313546088,  -22.268563128240686],
    [ 0.0,                0.0,                  0.0,                10.772688211590271,  -2.932867573104269,  -3.2752582214981767, -13.592907165071438],
    [ 0.0,                0.0,                  0.0,                 0.0,                27.133543485699047,   2.9384531625551986,   1.5013990104854835],
    [ 0.0,                0.0,                  0.0,                 0.0,                 0.0,                15.786970406543043,  -15.837576931579958],
    [ 0.0,                0.0,                  0.0,                 0.0,                 0.0,                 0.0,                 36.49372858726784  ],
], dtype=np.float64)

Q      = 162
LINVEC = np.array([0, 0, 0, 4, -4, -2, -2], dtype=np.int32)
LINMIN = 13.0
H = np.array([
    [2, 0, 0, 0,   62,  58974,  -5086],
    [0, 2, 0, 0,   84, 224666, -19686],
    [0, 0, 2, 0,   12, 234014, -20736],
    [0, 0, 0, 4,   52,  78376,  -6916],
    [0, 0, 0, 0,  120, 161172, -14052],
    [0, 0, 0, 0,    0, 262692, -23292],
    [0, 0, 0, 0,    0,      0,      0],
], dtype=np.int64)

MAX_N_OUT = 100000000

DILATIONS    = [1, 2, 10, 20, 10000, 20000, 30000, 40000, 50000]
EXPECTATIONS = [0, 0,  0,  1,     2,     2,     2,     2,     2]


# =============================================================================
# Tests
# =============================================================================

def test_njit():
    for dilation, expected in zip(DILATIONS, EXPECTATIONS):
        out, Niter = coni_kernel_njit(
            L=U.T,
            Q=Q,
            dilation=dilation,
            Binter0=LINVEC,
            M0min=LINMIN,
            H=H,
            max_N_out=MAX_N_OUT,
        )
        assert out.shape[0] == expected

def test_c():
    for dilation, expected in zip(DILATIONS, EXPECTATIONS):
        out, Qs, status = coni_kernel(
            U, Q=Q, dilation=dilation, linvec=LINVEC,
            linmin=LINMIN, H=H, max_N_out=MAX_N_OUT
        )
        assert out.shape[0] == expected


# =============================================================================
# Benchmarks
# =============================================================================

def test_bench_coni_kernel_njit(benchmark):
    benchmark(
        coni_kernel_njit,
        L=U.T, Q=Q, dilation=50000,
        Binter0=LINVEC, M0min=LINMIN, H=H, max_N_out=MAX_N_OUT,
    )

def test_bench_coni_kernel_c(benchmark):
    benchmark(coni_kernel, U, Q, 50000, LINVEC, LINMIN, H, MAX_N_OUT)
