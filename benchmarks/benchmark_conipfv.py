import numpy as np
import time

from pfvs.c_kernels import coni_kernel
from pfvs.lattice import coni_kernel_njit

# coni_kernel
# ===========
# Manwe from dSv1
# ---------------
dim = 7
U = np.array([
    [19.131126469708992,-2.50900019274872,-12.022292590254283,9.408750722807701,10.97687584327565,8.363333975829066,24.77637690339361],
    [0.0,20.8255832579255,30.243381795036953,31.864968637769284,12.462603346669052,6.193517125542706,6.826408301055712],
    [0.0,0.0,31.73014873072524,-6.636894774358689,15.791162726449791,12.771246313546088,-22.268563128240686],
    [0.0,0.0,0.0,10.772688211590271,-2.932867573104269,-3.2752582214981767,-13.592907165071438],
    [0.0,0.0,0.0,0.0,27.133543485699047,2.9384531625551986,1.5013990104854835],
    [0.0,0.0,0.0,0.0,0.0,15.786970406543043,-15.837576931579958],
    [0.0,0.0,0.0,0.0,0.0,0.0,36.49372858726784]
], dtype=np.float64)

Q = 162
linvec = np.array([0,0,0,4,-4,-2,-2], dtype=np.int32)
linmin = 13.0
H = np.array([
    [2,0,0,0,62,58974,-5086],
    [0,2,0,0,84,224666,-19686],
    [0,0,2,0,12,234014,-20736],
    [0,0,0,4,52,78376,-6916],
    [0,0,0,0,120,161172,-14052],
    [0,0,0,0,0,262692,-23292],
    [0,0,0,0,0,0,0]
], dtype=np.int64)

max_N_out = 100000000

# call the Cython wrapper
# -----------------------
print("CONI-PFV TESTING")
print("----------------")
for dilation in [1,2,10,20] + [10_000*i for i in range(1,20+1)]:
    tic = time.time()
    out, Qs, status = coni_kernel(
        U, Q, dilation, linvec, linmin, H, max_N_out
    )
    toc = time.time()
    
    print(f"dilation = {dilation}; found {out.shape[0]} vectors in {toc-tic}s using C code...")


    tic = time.time()
    out, Niter = coni_kernel_njit(
        L=U.T,
        Q=Q,
        dilation=dilation,
        Binter0=linvec,
        M0min=linmin,
        H=H,
        max_N_out=max_N_out
    )
    toc = time.time()
    
    print(f"dilation = {dilation}; found {out.shape[0]} vectors in {toc-tic}s using njit code...")
