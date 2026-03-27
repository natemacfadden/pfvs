import numpy as np
import time

from pfvs.c_kernels import pvec_kernel
from pfvs.lattice import kannan_box_mat_njit

# pvec_kernel
# ===========
# Manwe from dSv1
# ---------------
dim = 7
numhyps = 73
H = np.array([
    0,-5,0,3,3,0,-1,0,0,0,0,-1,-2,-3,0,0,0,-1,0,-3,-5,0,0,1,0,-1,-2,-3,0,0,-1,-1,0,-3,-4,0,0,0,0,0,-1,-2,0,0,0,0,-1,-2,-3,0,0,0,-1,-1,-5,-8,0,0,1,0,0,0,-1,0,0,0,0,0,0,-1,0,0,0,-1,0,0,-2,0,1,0,-1,1,0,-1,-1,0,0,0,2,0,-1,0,-1,0,0,0,-4,-6,0,0,-4,-1,0,0,-2,-1,0,-2,0,0,0,-1,-1,0,0,0,0,-2,-3,0,0,-1,0,0,-2,-2,0,1,-1,-2,0,-4,-6,0,-1,1,0,0,0,-2,0,-3,0,1,3,0,-1,0,-1,0,0,3,0,-1,0,-1,0,0,0,-3,-4,0,-1,0,0,0,0,-2,0,-1,0,0,2,0,-1,0,0,0,-1,-1,0,-3,0,-1,1,0,0,-2,-4,0,0,0,0,0,0,-1,0,-1,-1,0,0,-2,-2,0,0,0,0,-1,0,-1,0,-1,0,0,0,-3,-5,-1,0,0,0,0,0,-1,0,0,0,0,1,0,-1,0,0,1,0,0,-2,-4,0,1,0,-1,0,-1,-2,0,0,0,0,-1,0,-1,0,0,0,-1,0,-4,-7,0,0,1,0,1,0,-1,0,-1,0,0,0,-2,-3,0,0,0,0,0,-1,-1,0,0,1,0,-1,-1,-2,0,0,1,0,2,0,-2,0,0,-1,0,0,0,0,0,0,0,0,-1,0,-1,-3,0,0,0,2,0,-1,0,-1,0,0,3,0,-2,5,0,0,-2,-2,0,-1,-1,-1,0,1,1,0,0,0,-1,0,0,4,0,-2,0,0,-1,0,0,-1,-1,0,0,1,0,1,0,-2,0,0,0,-1,0,0,-2,0,-1,0,0,0,0,-2,0,-1,-3,0,0,0,-2,0,0,1,0,0,-1,-3,0,0,0,-1,0,-4,-6,0,0,-1,-2,0,0,-4,0,1,0,-1,-1,-2,-3,0,0,-1,-1,3,0,-1,0,-1,-1,0,2,0,0,-1,0,0,0,0,0,-1,3,0,0,-1,-1,0,0,3,0,0,-2,0,0,-1,0,0,0,0,-1,-1,-1,0,0,0,-1,0,0,-2,0,0,-1,-2,0,-7,-11,0,0,1,0,0,-1,-2,5,0,0,-2,0,0,-1,0,0,0,0,0,-1,-1,0,0,-1,0,-1,0,-1,0,0,1,0,-1,0,-1,-1,0,0,0,1,0,0,0,0,0,-1,0,0,-3
    ], dtype=np.int32).reshape(numhyps,dim)
max_N_out = 10000000000
max_N_iter = 1000000000000

# warm up njit
out, Niter = kannan_box_mat_njit(
        B=1,
        linmat=H,
        linmin=1,
        max_N_out=max_N_out,
        max_N_iter=max_N_iter
    )

# do the study
print("PVEC TESTING")
print("------------")
for dilation in [i for i in range(1,10+1)]:
    tic = time.time()
    out, status = pvec_kernel(
        B=dilation,
        linmat=H,
        linmin=1,
        max_N_out=max_N_out,
        max_N_iter=max_N_iter
    )
    toc = time.time()
    
    print(f"dilation = {dilation}; found {out.shape[0]} vectors in {toc-tic}s using C code...")


    tic = time.time()
    out, Niter = kannan_box_mat_njit(
        B=dilation,
        linmat=H,
        linmin=1,
        max_N_out=max_N_out,
        max_N_iter=max_N_iter
    )
    toc = time.time()
    
    print(f"dilation = {dilation}; found {out.shape[0]} vectors in {toc-tic}s using njit code...")