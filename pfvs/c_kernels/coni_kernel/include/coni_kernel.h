#ifndef CONI_KERNEL_H
#define CONI_KERNEL_H

#include <stdint.h>

/*
**Description:**
Adaptation of the (iterative) Fincke-Pohst algorithm for utility in
constructing coni-PFVs. I.e., solves
    0 <= vec^T @ mat     @ vec <= dilation*Q.
    0 <= vec^T @ (U.T@U) @ vec <= dilation*Q
as well as (M[0] cuts)
    linmin <= dot(linvec, vec)
as well as (K'>0 cuts)
    (vec^T @ mat @ vec)//Q <= gcd(Kperp)
                            = gcd([0, 1]@Z@Binter@vec)
                            = gcd(H@vec)
for H the row-HNF of [0, 1]@Z@Binter (this matrix computes Kperp from vec).

Any `vec` satisfying all of the above can generate a coni-PFV, as long as
det(N) != 0.

Most of the work is in writing to `out`, `Qs`, and `N_out`.

**Arguments:**
// output objects
- `out`:       A container for the lattice points vec.
- `Qs`:        A container for the valuations of vec^T @ mat @ vec of the
               outputs.
- `N_out`:     An integer we write to, indicating the number of outputs.
// ellipsoid def
- `dim`      : The dimension of the problem.
- `U`:         The upper triangular matrix such that mat = U.T@L
- `Q`:         The ellipsoid bound.
- `dilation`:  The maximum allowed dilation to allow... As long as
               gcd(Kperp) >= (vec^T @ mat @ vec)//Q, the vector vec can
               still define coni-PFV.
// M0 cuts
- `linvec`:    Binter[0,:]. The vector such that dot(linvec,vec)=M0. BEST TO
               ORDER COLUMNS OF BINTER SUCH THAT linvec HAS A LARGE NUMBER
               OF LEADING 0s.
- `linmin`:    The minimum value of M0 permitted. Inclusive.
// Kprime cuts
- `H`:         Let G be the matrix such that Kperp = G@vec. Then H = HNF(G).
// misc specs
- `max_N_out`: The maximum number of output allowed.
- `eps`:       A small number used for correctly setting bounds despite
               floating point errors.

**Returns:**
A status code.
*/
int _coni_kernel_c(
    int32_t * restrict out,
    double * restrict Qs,
    int * restrict N_out,
    int dim,
    double * restrict U,
    int Q,
    double dilation,
    int * restrict linvec,
    double linmin,
    int * restrict H,
    int max_N_out,
    double eps
);

#endif
