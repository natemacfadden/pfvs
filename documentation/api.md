<a id="pfvs.cydata"></a>

---


# pfvs.cydata

<a id="pfvs.cydata.CYData"></a>

---


## CYData Objects

```python
class CYData()
```

Holds the CY data needed for constructing PFVs.

Parameters
----------
h21 : int
    The Hodge number h^{2,1}.
kappa : ArrayLike
    The triple intersection numbers.
c2 : ArrayLike
    The second Chern class.
H : ArrayLike
    Inwards-facing hyperplanes defining the Kahler cone.
coni_curve : ArrayLike, optional
    The conifold curve. If not provided, non-coni PFVs are assumed.
coni_cob : ArrayLike, optional
    Change-of-basis matrix mapping the coni curve to (1,0,...,0).
vertices : ArrayLike, optional
    Polytope vertices. Defaults to a placeholder string if not provided.
heights : ArrayLike, optional
    Triangulation heights. Defaults to a placeholder string if not
    provided.

<a id="pfvs.cydata.CYData.from_cy"></a>

---


#### from\_cy

```python
@classmethod
def from_cy(cls,
            cy: "cytools.CalabiYau",
            coni_curve: ArrayLike | None = None,
            coni_cob: ArrayLike | None = None) -> "CYData"
```

Construct a CYData object from a cytools.CalabiYau object.

Parameters
----------
cy : cytools.CalabiYau
    The CYTools CalabiYau object of interest.
coni_curve : ArrayLike, optional
    The conifold curve. If not provided, non-coni PFVs are assumed.
coni_cob : ArrayLike, optional
    Change-of-basis matrix mapping the coni curve to (1,0,...,0).

Returns
-------
CYData

<a id="pfvs.cydata.CYData.vertices"></a>

---


#### vertices

```python
@property
def vertices() -> ArrayLike | str
```

Vertices of the associated polytope, or a string identifier.

<a id="pfvs.cydata.CYData.heights"></a>

---


#### heights

```python
@property
def heights() -> ArrayLike | str
```

Heights of the triangulation, or a string identifier.

<a id="pfvs.cydata.CYData.kappa"></a>

---


#### kappa

```python
@property
def kappa() -> np.ndarray
```

Triple intersection numbers, shape (h11, h11, h11).

<a id="pfvs.cydata.CYData.c2"></a>

---


#### c2

```python
@property
def c2() -> np.ndarray
```

Second Chern class coefficients, shape (h11,).

<a id="pfvs.cydata.CYData.H"></a>

---


#### H

```python
@property
def H() -> np.ndarray
```

Kahler cone generators (hyperplanes), shape (n_rays, h11).

<a id="pfvs.cydata.CYData.coni"></a>

---


#### coni

```python
@property
def coni() -> bool
```

True if this CYData is set up for coni PFV search.

<a id="pfvs.cydata.CYData.h11"></a>

---


#### h11

```python
@property
def h11() -> int
```

Hodge number h^{1,1}.

<a id="pfvs.cydata.CYData.h21"></a>

---


#### h21

```python
@property
def h21() -> int
```

Hodge number h^{2,1}.

<a id="pfvs.cydata.CYData.a"></a>

---


#### a

```python
@property
def a()
```

The a-matrix: a_ij = kappa_iij for i >= j, kappa_ijj for i < j (eq 2.52
of arXiv:2406.13751). In the coni basis, with the coni row dropped.

<a id="pfvs.cydata.CYData.b"></a>

---


#### b

```python
@property
def b()
```

The b-vector: c2 (non-coni), or c2 + 2 q_coni in the coni basis (below
eq 3.5 of arXiv:2406.13751).

<a id="pfvs.cydata.CYData.M_lattice"></a>

---


#### M\_lattice

```python
def M_lattice(verify: bool = True) -> np.ndarray
```

Basis (as columns) of the lattice of integral M with (b/24).M and
(a/2)@M integral: the dual of the rows of [b/24; a/2; 1].

Parameters
----------
verify : bool, optional
    Check the result against a and b. Defaults to True.

<a id="pfvs.pfv"></a>

---


# pfvs.pfv

<a id="pfvs.pfv.PFV"></a>

---


## PFV Objects

```python
class PFV()
```

Stores and verifies (coni-)PFVs.

Parameters
----------
data : CYData
    A CYData object defining the CY.
K : array-like
    K flux vector. Must be integral.
M : array-like
    M flux vector. Must be integral.
silent : bool, optional
    Whether to suppress messages. Defaults to False.

<a id="pfvs.pfv.PFV.coni_curve"></a>

---


#### coni\_curve

```python
@_coni_only
def coni_curve()
```

The conifold curve.

<a id="pfvs.pfv.PFV.cob"></a>

---


#### cob

```python
@_coni_only
def cob()
```

Change of basis to the coni basis.

<a id="pfvs.pfv.PFV.Kprime"></a>

---


#### Kprime

```python
@_coni_only
def Kprime()
```

K' = -K_0 + (M kappa p)_0 (must be positive).

<a id="pfvs.pfv.PFV.dilation_bound"></a>

---


#### dilation\_bound

```python
@_coni_only
def dilation_bound()
```

Exact upper bound Q/mu0 on the dilation of coni PFVs with this
direction and tadpole (see ``pfvs.dilation``); None if it doesn't apply.

<a id="pfvs.pfv.PFV.check_Kprime"></a>

---


#### check\_Kprime

```python
def check_Kprime() -> bool
```

(coni PFVs only) K' > 0.

<a id="pfvs.pfv.PFV.from_str"></a>

---


#### from\_str

```python
@classmethod
def from_str(cls, str_: str) -> "PFV"
```

Construct a PFV from its string representation.

Parameters
----------
str_ : str
    String representation of a PFV, in the format produced by
    str(PFV(...)).

Returns
-------
pfv : PFV
    The reconstructed PFV object.

<a id="pfvs.pfv.PFV.coni"></a>

---


#### coni

```python
@property
def coni() -> bool
```

Whether this object describes a coni PFV.

<a id="pfvs.pfv.PFV.h11"></a>

---


#### h11

```python
@property
def h11() -> int
```

Hodge number h^{1,1} of the associated CY.

<a id="pfvs.pfv.PFV.h21"></a>

---


#### h21

```python
@property
def h21() -> int
```

Hodge number h^{2,1} of the associated CY.

<a id="pfvs.pfv.PFV.vertices"></a>

---


#### vertices

```python
@property
def vertices() -> np.ndarray
```

Vertices of the associated polytope.

<a id="pfvs.pfv.PFV.heights"></a>

---


#### heights

```python
@property
def heights() -> np.ndarray
```

Heights of the triangulation.

<a id="pfvs.pfv.PFV.cy"></a>

---


#### cy

```python
@property
def cy() -> "cytools.CalabiYau"
```

The CY object (requires cytools).

<a id="pfvs.pfv.PFV.kappa"></a>

---


#### kappa

```python
@property
def kappa() -> np.ndarray
```

The intersection numbers

<a id="pfvs.pfv.PFV.K"></a>

---


#### K

```python
@property
def K() -> np.ndarray
```

The K-vector

<a id="pfvs.pfv.PFV.M"></a>

---


#### M

```python
@property
def M() -> np.ndarray
```

The M-vector

<a id="pfvs.pfv.PFV.f"></a>

---


#### f

```python
@property
def f() -> np.ndarray | None
```

The F3 flux vector, f = ((b.M)/24, (a@M)/2, 0, M), of length 2*(h11+1).
None if either division is non-integral. Non-coni only.

<a id="pfvs.pfv.PFV.h"></a>

---


#### h

```python
@property
def h() -> np.ndarray
```

The H3 flux vector, h = (0, K, 0, ..., 0), of length 2*(h11+1).
Non-coni only.

<a id="pfvs.pfv.PFV.ellipsoid_mat"></a>

---


#### ellipsoid\_mat

```python
@property
def ellipsoid_mat() -> np.ndarray
```

The matrix mat defining the M-ellipsoid: c^T @ mat @ c <= Q. Coni only.
See also `ellipsoid_c`, `ellipsoid_required_dilation`.

<a id="pfvs.pfv.PFV.ellipsoid_c"></a>

---


#### ellipsoid\_c

```python
@property
def ellipsoid_c() -> np.ndarray
```

The lattice vector c such that M = Binter @ c. Coni only.
See also `ellipsoid_mat`, `ellipsoid_required_dilation`.

<a id="pfvs.pfv.PFV.ellipsoid_required_dilation"></a>

---


#### ellipsoid\_required\_dilation

```python
@property
def ellipsoid_required_dilation() -> float
```

The minimum ellipsoid dilation needed to capture this PFV. Coni only.

Equal to c^T @ mat @ c / Q. See also `ellipsoid_mat`, `ellipsoid_c`.

<a id="pfvs.pfv.PFV.H"></a>

---


#### H

```python
@property
def H() -> np.ndarray
```

The hyperplanes of the Kahler cone.

<a id="pfvs.pfv.PFV.a"></a>

---


#### a

```python
@property
def a() -> np.ndarray
```

The a-matrix

<a id="pfvs.pfv.PFV.b"></a>

---


#### b

```python
@property
def b() -> np.ndarray
```

The b-vector

<a id="pfvs.pfv.PFV.gvs"></a>

---


#### gvs

```python
@property
def gvs() -> np.ndarray | None
```

The Gopakumar-Vafa invariants in coo format, or None if not yet set.

<a id="pfvs.pfv.PFV.N"></a>

---


#### N

```python
@property
def N() -> np.ndarray
```

The N-matrix, defined as kappa @ M. For coni, the 0th row and column
are trimmed.

<a id="pfvs.pfv.PFV.Ninv"></a>

---


#### Ninv

```python
@property
def Ninv() -> tuple
```

The scaled exact inverse of N, as a (flint matrix, scale) pair.

<a id="pfvs.pfv.PFV.p"></a>

---


#### p

```python
@property
def p() -> np.ndarray
```

The p-vector

For non-coni PFVs, this is defined as `N.inv()@K`
For     coni PFVs, this is defined as `concatenate([[0],N.inv()@K[1:]])`

<a id="pfvs.pfv.PFV.pgrading"></a>

---


#### pgrading

```python
@property
def pgrading() -> np.ndarray
```

The 'pgrading'-vector.

This is just the primitive vector along the ray defined by p.
I.e., the unique `pgrading = r*p` for r>0 such that `gcd(r*p) == 1`.

<a id="pfvs.pfv.PFV.compute_gvs"></a>

---


#### compute\_gvs

```python
def compute_gvs(max_deg: int) -> None
```

Compute GV invariants up to degree max_deg via cytools and store them.

Parameters
----------
max_deg : int
    Maximum degree of GV invariants to compute.

<a id="pfvs.pfv.PFV.gvs"></a>

---


#### gvs

```python
@gvs.setter
def gvs(val: np.ndarray | None)
```

Accepts GVs in coo format

<a id="pfvs.pfv.PFV.check_all"></a>

---


#### check\_all

```python
def check_all(stop_at_fail: bool = True) -> bool
```

Run all check_* methods. Returns True iff all pass.

Runs check_Ninvertible first (required for p to exist), then the rest.
If stop_at_fail is True, short-circuits on first failure.

<a id="pfvs.pfv.PFV.check_a"></a>

---


#### check\_a

```python
def check_a() -> bool
```

Check that a @ M is even.

<a id="pfvs.pfv.PFV.check_b"></a>

---


#### check\_b

```python
def check_b() -> bool
```

Check that dot(b, M) is a multiple of 24.

<a id="pfvs.pfv.PFV.check_tadpole"></a>

---


#### check\_tadpole

```python
def check_tadpole() -> bool
```

Check that 0 <= -dot(K, M) <= h11 + h21 + 2 + 2*coni.

<a id="pfvs.pfv.PFV.check_Knonzero"></a>

---


#### check\_Knonzero

```python
def check_Knonzero() -> bool
```

Check that K is nonzero.

<a id="pfvs.pfv.PFV.check_Ninvertible"></a>

---


#### check\_Ninvertible

```python
def check_Ninvertible(tol: float = 0.5) -> bool
```

Check that N is full rank via |det(N)| > tol.

Parameters
----------
tol : float, optional
    Threshold for |det(N)|. Defaults to 0.5 (appropriate for integer
    matrices where det is an integer, so 0 vs non-zero is unambiguous).

<a id="pfvs.pfv.PFV.check_pcontainment"></a>

---


#### check\_pcontainment

```python
def check_pcontainment() -> bool
```

Check that p is strictly contained in Kcup (the union of 2-face
equivalent Kahler cones).

<a id="pfvs.pfv.PFV.check_NpK"></a>

---


#### check\_NpK

```python
def check_NpK(tol: float = 1e-4) -> bool
```

Check that N @ p = K. Requires `check_Ninvertible` to pass.

Parameters
----------
tol : float, optional
    Tolerance for the residual norm ||N @ p - K||. Defaults to 1e-4.

<a id="pfvs.pfv.PFV.check_orthogonality"></a>

---


#### check\_orthogonality

```python
def check_orthogonality() -> bool
```

Check that dot(K, p) = 0. Requires `check_Ninvertible` to pass.

<a id="pfvs.pfv.PFV.series_gen"></a>

---


#### series\_gen

```python
def series_gen() -> Generator[tuple[float, float], None, None]
```

Generator yielding the coefficient, exponent of each term in the series

<a id="pfvs.pfv.PFV.series"></a>

---


#### series

```python
def series(N_nonzero: int = float('inf'), verbosity: int = 0) -> list[tuple[
        float, float]]
```

The superpotential series W = sum_i c_i exp(2 pi i tau e_i), with
e_i = dot(p, q) and c_i = sum_{q at e_i} n_q dot(M, q), as [c_i, e_i]
pairs: nonzero terms only, by exponent, at most N_nonzero of them.

<a id="pfvs.pfv.PFV.valid_coeff_ratio"></a>

---


#### valid\_coeff\_ratio

```python
def valid_coeff_ratio() -> bool | None
```

Check that the series has valid leading coefficients, i.e. |c1| > |c0|.
Returns None if fewer than 2 terms are available.

<a id="pfvs.pfv.PFV.tau0"></a>

---


#### tau0

```python
@property
def tau0() -> complex
```

The value of tau minimizing the 2-term racetrack potential.
Returns nan if the series has invalid leading coefficients.

<a id="pfvs.pfv.PFV.W0"></a>

---


#### W0

```python
def W0(as_logs: bool = False,
       check_Ninvertible: bool = True,
       verbosity: int = 0) -> float
```

Compute the flux superpotential W0 from the 2-term racetrack
approximation.

Returns the value (or log10 if as_logs=True), or nan if N is singular
or the series has invalid leading coefficients.

<a id="pfvs.pfv.PFV.gs"></a>

---


#### gs

```python
@property
def gs() -> float
```

The string coupling gs = 1 / Im(tau0).
Returns nan if the series has invalid leading coefficients.

<a id="pfvs.pfv.PFV.series_abs_vev"></a>

---


#### series\_abs\_vev

```python
def series_abs_vev(as_logs: bool = False) -> list[float]
```

|W_i| (or log10|W_i|) for each series term, at the tau minimizing the
2-term approximation W_0 + W_1.

<a id="pfvs.pfv.PFV.series_corrections"></a>

---


#### series\_corrections

```python
def series_corrections(as_logs: bool = False) -> list[float]
```

|W_i| / W0 (or its log10) for i >= 2: the size of the corrections to
the 2-term approximation.

<a id="pfvs.pfv.PFV.diagnostics"></a>

---


#### diagnostics

```python
def diagnostics(verbosity: int = 0) -> None
```

Print a summary of the PFV: checks, tadpole, W0, tau0, gs, and series
info. At higher verbosity, dumps the full series and plots corrections.

<a id="pfvs.pfv.PFV.plot_series"></a>

---


#### plot\_series

```python
def plot_series() -> None
```

Plot log10(|W_i| / W0) vs. term index for i >= 2.

<a id="pfvs.pfv.PFV.dump_series"></a>

---


#### dump\_series

```python
def dump_series(verbosity: int = 0) -> None
```

Print the series terms. At verbosity=0, prints (exponent, coefficient)
pairs. At verbosity=1, breaks each coefficient into its contributing
terms. At verbosity>=2, also shows the individual GV invariants and
charges.

<a id="pfvs.coniZp"></a>

---


# pfvs.coniZp

<a id="pfvs.coniZp.coni_M_ellipsoid"></a>

---


#### coni\_M\_ellipsoid

```python
def coni_M_ellipsoid(
    p: ArrayLike,
    data: CYData = None,
    kappa: ArrayLike = None,
    Mbasis: ArrayLike = None,
    extra_lll_reduction: bool = True,
    extra_checks: bool = False,
    _maxes: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

The matrices defining coniZpM's M-ellipsoid for one p-vector.

M lies in the lattice M = Binter c (which imposes K.p = 0), and
K[1:] = (Z M)[1:] with Z = kappa.p. Some K[0] > 0 gives -K.M <= Qmax iff
c^T mat c <= Qmax, mat = -Binter^T Z Binter. The ellipsoid may be dilated
if gcd(K[1:]) > c^T mat c / Qmax (see `coni_H_matrix`).

Parameters
----------
p : ndarray of shape (h11,) or (h11-1,)
    The p-vector.
data : CYData, optional
    The CY. Or pass kappa (h11, h11, h11) and Mbasis (h11, h11) instead.
extra_lll_reduction : bool, optional
    LLL-reduce the intermediate basis too (avoids overflows). Defaults to
    True.
extra_checks : bool, optional
    Deprecated, no effect.

Returns
-------
mat : ndarray of shape (h11-1, h11-1)
    The ellipsoid: c^T mat c <= Qmax (times the dilation).
Z : ndarray of shape (h11, h11)
    kappa.p, so K[1:] = (Z M)[1:].
Binter : ndarray of shape (h11, h11-1)
    The M-lattice basis with K.p = 0 imposed.

<a id="pfvs.coniZp.coni_H_matrix"></a>

---


#### coni\_H\_matrix

```python
def coni_H_matrix(ZBinter: ArrayLike, proj: ArrayLike = None)
```

The HNF H of (Z Binter)[1:], used to prune coniZpM's search by gcd(K[1:]).

On a dilated ellipsoid, a point with c^T mat c > Qmax is only valid if
gcd(K[1:]) > c^T mat c / Qmax. Fincke-Pohst sets c from the last entry
down, and since H is upper triangular, gcd(H[-m:,-m:] c[-m:]) bounds
gcd(K[1:]) from above as entries are set, so branches can be cut early.

Parameters
----------
ZBinter : ndarray of shape (h11, h11-1)
    Z @ Binter from `coni_M_ellipsoid`.
proj : ndarray of shape (h11-1, h11), optional
    eye(h11)[1:]; computed if not given.

Returns
-------
H : ndarray of shape (h11-1, h11-1)
    gcd(H c) = gcd(K[1:]).

<a id="pfvs.coniZp.coniZpK"></a>

---


#### coniZpK

```python
def coniZpK(data: CYData,
            ps: ArrayLike,
            D0: int,
            Q: int | None = None,
            M0min: int = 13,
            n_jobs: int = -1,
            max_N_pfvs: int = 1_000_000_000,
            return_formal_pfvs: bool = False,
            verbosity: int = 0) -> tuple[ArrayLike, ArrayLike]
```

The coni PFVs of each direction p_hat with dilation delta > D0
(p = p_hat / delta). With ``coniZpM(..., ellipsoid_dilation=D0)`` this
gives every coni PFV of the direction.

Enumerates the short K_r these PFVs must have (K_r^T S K_r < Q / D0, see
`pfvs.dilation`) with an exact C kernel. p-vectors it cannot handle
exactly are searched by coniZpM up to their dilation bound instead.

Parameters
----------
data : CYData
    The CY (coni).
ps : iterable of shape (N, h11-1)
    The directions, as for ``coniZpM``.
D0 : int
    Positive integer; find the PFVs with dilation > D0.
Q, M0min, n_jobs, max_N_pfvs, return_formal_pfvs, verbosity :
    As for ``coniZpM`` (n_jobs: threads).

Returns
-------
Ks, Ms : ndarrays of shape (N, h11), or a list of PFV objects
    The PFVs, grouped by p-vector (in the order of ps) and sorted by
    (M, K) within one.

Raises
------
IncompleteSearchError
    For a p-vector without a dilation bound, or whose search is not exact.

<a id="pfvs.coniZp.coniZpM"></a>

---


#### coniZpM

```python
def coniZpM(data: CYData,
            ps: ArrayLike,
            Q: int | None = None,
            M0min: int = 13,
            ellipsoid_dilation: float = 1,
            use_gcd_lattice: bool = False,
            use_c_lattice: bool = True,
            low_level_parallelism: bool = False,
            n_jobs: int = -1,
            extra_checks: bool = False,
            extra_lll_reduction: bool = True,
            device: str = "auto",
            max_N_pfvs: int = 1_000_000_000,
            return_formal_pfvs: bool = False,
            verbosity: int = 0,
            exhaustive: bool = False,
            cost_model=None) -> tuple[ArrayLike, ArrayLike]
```

The coni PFVs of each p-vector up to a dilation: the lattice points of
the ellipsoid of `coni_M_ellipsoid`, enumerated by an exact Fincke-Pohst
kernel that prunes on gcd(K[1:]) (`coni_H_matrix`) and on M0.

Parameters
----------
data : CYData
    The CY (coni).
ps : iterable of shape (N, h11-1)
    The p-vectors without their coni entry, p[1:].
Q : integer, optional
    Only PFVs with -K.M = Q exactly. Defaults to h11+h21+4.
M0min : integer, optional
    Only PFVs with M[0] >= M0min. Defaults to 13.
ellipsoid_dilation : float, optional
    The ellipsoid's dilation; runtime grows about linearly with it.
    Defaults to 1.
use_c_lattice : bool, optional
    Build the lattice data in C (exact, falls back to Python on
    overflow). Defaults to True.
use_gcd_lattice : bool, optional
    Use `_Kperp_gcd_lattice` instead of pruning (slower, old). Defaults
    to False.
low_level_parallelism : bool, optional
    Deprecated; only forces n_jobs = 1.
n_jobs : int, optional
    Parallel jobs over p-vectors. Defaults to twice the CPU count.
extra_checks : bool, optional
    Deprecated, no effect.
extra_lll_reduction : bool, optional
    As in `coni_M_ellipsoid`. Defaults to True.
device : str, optional
    "cpu", "gpu" (raises if unavailable) or "auto" (the GPU when
    available and worthwhile; PFVS_DEVICE overrides). Results are
    identical. Defaults to "auto".
max_N_pfvs : int, optional
    Output limit per kernel call. Defaults to 1e9.
return_formal_pfvs : bool, optional
    Return PFV objects instead of (Ks, Ms). Defaults to False.
verbosity : int, optional
    Defaults to 0.
exhaustive : bool, optional
    Find every coni PFV of each direction at any dilation, using each
    p-vector's dilation bound (`pfvs.dilation`): ZpM to the bound, or ZpM
    to D0 plus `coniZpK` above, as `bound_routing` plans. p-vectors
    without a bound are searched to ellipsoid_dilation and reported
    incomplete. Defaults to False.
cost_model : pfvs.dilation.CostModel, optional
    Costs for the exhaustive plan. Defaults to measuring them on a sample.

Returns
-------
Ks, Ms : ndarrays of shape (N, h11)
    The PFVs (or a list of PFV objects if return_formal_pfvs), grouped by
    p-vector and sorted by (M, K) within one.
complete : ndarray of bool, shape (len(ps),)
    Only if exhaustive: whether each p-vector's search is complete.

<a id="pfvs.Zp"></a>

---


# pfvs.Zp

<a id="pfvs.Zp.M_ellipsoid"></a>

---


#### M\_ellipsoid

```python
def M_ellipsoid(
    p: ArrayLike,
    data: CYData = None,
    kappa: ArrayLike = None,
    Mbasis: ArrayLike = None,
    extra_lll_reduction: bool = True,
    extra_checks: bool = False,
    _maxes: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

The M-ellipsoid of non-coni ZpM: M = Binter @ c, K = Z @ M, and the
tadpole -dot(K, M) <= Qmax becomes c^T mat c <= Qmax with
mat = -Binter^T Z Binter. Dilating is allowed where gcd(K) >= c^T mat c / Qmax.

Parameters
----------
p : ndarray of shape (h11,)
    The p-vector.
data : CYData, optional
    The CY. Or pass kappa and Mbasis (data.M_lattice()) instead.
extra_lll_reduction : bool, optional
    Extra LLL reduction of Binter; avoids some overflows. Defaults to True.
extra_checks : bool, optional
    Deprecated, no effect.

Returns
-------
mat : ndarray of shape (h11-1, h11-1)
Z : ndarray of shape (h11, h11)
    K = Z @ M.
Binter : ndarray of shape (h11, h11-1)
    Basis of the M-lattice with dot(K, p) = 0.

<a id="pfvs.Zp.K_ellipsoid"></a>

---


#### K\_ellipsoid

```python
def K_ellipsoid(p: ArrayLike,
                data: CYData = None,
                kappa: ArrayLike = None,
                Mbasis: ArrayLike = None,
                extra_lll_reduction: bool = True,
                extra_checks: bool = False) -> tuple[np.ndarray, np.ndarray]
```

The K-ellipsoid of non-coni ZpK: K = B @ d with B a basis of the lattice
orthogonal to p, M = (kappa p)^{-1} K, and the tadpole becomes
d^T mat d <= Qmax with mat = -B^T (kappa p)^{-1} B.

Parameters
----------
p : ndarray of shape (h11,)
    The p-vector.
data : CYData, optional
    The CY. Or pass kappa and Mbasis (data.M_lattice()) instead.
extra_lll_reduction : bool, optional
    Extra LLL reduction of Binter; avoids some overflows. Defaults to True.
extra_checks : bool, optional
    Deprecated, no effect.

Returns
-------
mat : ndarray of shape (h11-1, h11-1)
B : ndarray of shape (h11, h11-1)
    Basis of the K-lattice orthogonal to p.

<a id="pfvs.Zp.H_matrix"></a>

---


#### H\_matrix

```python
def H_matrix(ZBinter: ArrayLike)
```

Row-HNF of Z@Binter (K = ZBinter @ c), for gcd(K) pruning in ZpM: since
gcd(H @ c) = gcd(K) and the gcd of trailing blocks only decreases as FP
fixes c from the right, a dilated point with gcd(K) < c^T mat c / Qmax is
pruned early. As `coni_H_matrix`, without dropping K[0].

<a id="pfvs.Zp.ZpM"></a>

---


#### ZpM

```python
def ZpM(data: CYData,
        ps: ArrayLike,
        Qmax: int | None = None,
        Qmin: int = 0,
        ellipsoid_dilation: float = 1,
        use_c_kernel: bool = True,
        use_c_lattice: bool = True,
        n_jobs: int = -1,
        extra_checks: bool = False,
        extra_lll_reduction: bool = True,
        max_N_pfvs: int = 1_000_000_000,
        return_formal_pfvs: bool = False,
        verbosity: int = 0) -> tuple[ArrayLike, ArrayLike]
```

Non-coni PFVs from p-vectors: Fincke-Pohst over the M-ellipsoid of each p
(`M_ellipsoid`), with gcd(K) pruning; GCDs are re-introduced afterwards
(`_allow_gcds`).

Parameters
----------
data : CYData
    The CY (non-coni).
ps : ArrayLike of shape (N, h11)
    The p-vectors, inside the Kahler cone (H @ p >= 1).
Qmax, Qmin : integer, optional
    Keep PFVs with Qmin <= -dot(K, M) <= Qmax. Default h11+h21+2 and 0.
ellipsoid_dilation : float, optional
    Ellipsoid dilation; runtime grows about linearly. Defaults to 1.
use_c_kernel : bool, optional
    True: the exact C kernel. False: the Numba reference (float, no GCD
    pruning; same PFVs). Defaults to True.
use_c_lattice : bool, optional
    Build each p's lattice data in C (falls back to Python on overflow;
    same PFVs, possibly in another order). Defaults to True.
n_jobs : int, optional
    Parallel jobs. Defaults to 2x the CPU count.
extra_checks : bool, optional
    Deprecated, no effect.
extra_lll_reduction : bool, optional
    As in `M_ellipsoid`.
max_N_pfvs : int, optional
    Output limit (the C kernel needs one). Defaults to 1e9.
return_formal_pfvs : bool, optional
    Return PFV objects instead of (Ks, Ms). Defaults to False.
verbosity : int, optional
    Defaults to 0.

Returns
-------
Ks, Ms : ndarrays of shape (N, h11)
    The PFVs, one per row (or a list of PFV objects).

Raises
------
ValueError
    If data.coni, ps is empty or outside the cone, Qmax < Qmin,
    ellipsoid_dilation <= 0, or no PFVs survive `_allow_gcds`.

<a id="pfvs.Zp.ZpK"></a>

---


#### ZpK

```python
def ZpK(data: CYData,
        ps: ArrayLike,
        Qmax: int | None = None,
        Qmin: int = 0,
        ellipsoid_dilation: float = 1,
        n_jobs: int = -1,
        extra_checks: bool = False,
        extra_lll_reduction: bool = True,
        max_N_pfvs: int = 1_000_000_000,
        return_formal_pfvs: bool = False,
        verbosity: int = 0) -> tuple[ArrayLike, ArrayLike]
```

Non-coni PFVs from p-vectors: enumerate the K-ellipsoid of each p
(`K_ellipsoid`), without GCD pruning; GCDs are re-introduced afterwards.

Parameters
----------
data : CYData
    The CY (non-coni).
ps : ArrayLike of shape (N, h11)
    The p-vectors, inside the Kahler cone (H @ p >= 1).
Qmax, Qmin : integer, optional
    Keep PFVs with Qmin <= -dot(K, M) <= Qmax. Default h11+h21+2 and 0.
ellipsoid_dilation : float, optional
    Ellipsoid dilation; runtime grows about linearly. Defaults to 1.
n_jobs : int, optional
    Parallel jobs. Defaults to 2x the CPU count.
extra_checks : bool, optional
    Deprecated, no effect.
extra_lll_reduction : bool, optional
    As in `K_ellipsoid`.
max_N_pfvs : int, optional
    Output limit (the C kernel needs one). Defaults to 1e9.
return_formal_pfvs : bool, optional
    Return PFV objects instead of (Ks, Ms). Defaults to False.
verbosity : int, optional
    Defaults to 0.

Returns
-------
Ks, Ms : ndarrays of shape (N, h11)
    The PFVs, one per row (or a list of PFV objects).

Raises
------
ValueError
    If data.coni, ps is empty or outside the cone, Qmax < Qmin,
    ellipsoid_dilation <= 0, or no PFVs survive `_allow_gcds`.

<a id="pfvs.pvectors"></a>

---


# pfvs.pvectors

<a id="pfvs.pvectors.pvecs"></a>

---


#### pvecs

```python
def pvecs(data: CYData, min_N_pts: int, verbosity: int = 0) -> np.ndarray
```

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

<a id="pfvs.dilation"></a>

---


# pfvs.dilation

<a id="pfvs.dilation.coni_mu0"></a>

---


#### coni\_mu0

```python
def coni_mu0(p: ArrayLike,
             kappa: ArrayLike,
             use_c: bool = True) -> tuple[Fraction, list[int]] | None
```

The minimum of K^T S K over nonzero K in Lambda, S = -A_rr^-1.

Parameters
----------
p : array of shape (h11-1,)
    The direction p_hat without its (zero) coni entry, in the coni basis.
kappa : array of shape (h11, h11, h11)
    Triple intersection numbers in the coni basis (``CYData.kappa_cob``).

Returns
-------
(mu0, K) : the exact minimum and a minimizing K (length h11-1), or None
    if the bound's hypotheses fail (A_rr singular, s > 0, or S not
    positive definite on Lambda).
use_c : bool, optional
    Use the C version (falling back to Python where it cannot decide).
    False: the exact python-flint version only.

<a id="pfvs.dilation.coni_dilation_bound"></a>

---


#### coni\_dilation\_bound

```python
def coni_dilation_bound(p: ArrayLike, kappa: ArrayLike,
                        Q: int) -> Fraction | None
```

Upper bound on the dilation of every coni PFV with direction p_hat.

Every coni PFV with -K.M = Q and p = p_hat / delta has delta < Q / mu0
(see the module description). Searching p_hat at ellipsoid dilation
Q / mu0 therefore finds all of them.

Parameters
----------
p : array of shape (h11-1,)
    The direction p_hat without its (zero) coni entry, in the coni basis
    (as ``coniZpM`` takes p-vectors).
kappa : array of shape (h11, h11, h11)
    Triple intersection numbers in the coni basis (``CYData.kappa_cob``).
Q : int
    The tadpole, -K.M.

Returns
-------
Fraction or None
    Q / mu0 (exact; the bound is strict), or None if the hypotheses fail.

<a id="pfvs.dilation.coni_dilation_bound_ceils"></a>

---


#### coni\_dilation\_bound\_ceils

```python
def coni_dilation_bound_ceils(ps: ArrayLike,
                              kappa: ArrayLike,
                              Q: int,
                              n_jobs: int = 1) -> np.ndarray
```

ceil(`coni_dilation_bound`) for each row of `ps` as int64 (0: no bound).

<a id="pfvs.dilation.coni_dilation_bounds"></a>

---


#### coni\_dilation\_bounds

```python
def coni_dilation_bounds(ps: ArrayLike,
                         kappa: ArrayLike,
                         Q: int,
                         n_jobs: int = 1) -> list[Fraction | None]
```

`coni_dilation_bound` for each row of `ps` (shape (n, h11-1)).

<a id="pfvs.dilation.bound_routing"></a>

---


#### bound\_routing

```python
def bound_routing(bounds: ArrayLike,
                  G,
                  K,
                  D0s: ArrayLike,
                  c: float = 0.0,
                  shared: bool = False) -> tuple[float, float, float]
```

Plan an exhaustive search of one geometry's p-vectors: p with
b(p) <= b* run ZpM to their bound (GPU); the rest are split at D0 (ZpM up
to D0 on the GPU, ZpK above it on the CPU). Chooses b* and D0 to minimize
the run time, GPU and CPU running concurrently.

Parameters
----------
bounds : array of shape (n,)
    b(p) for each p-vector (`coni_dilation_bounds`); non-finite ignored.
G : callable
    G(D): GPU time per p-vector of ZpM at dilation D (vectorized,
    increasing).
K : callable
    K(D0): CPU time per p-vector of ZpK above D0.
D0s : array
    Candidate split dilations.
c : float, optional
    CPU wall time per p-vector to compute b(p).
shared : bool, optional
    ZpM and ZpK share the processors (no GPU): minimize T_GPU + T_CPU
    instead of the max.

Returns
-------
(b_star, D0, t) : the threshold (-inf: split every p), the split
    dilation, and the run time per p-vector.

<a id="pfvs.dilation.CostModel"></a>

---


## CostModel Objects

```python
class CostModel()
```

Measured per-p-vector search costs of one geometry, for `bound_routing`.
G is interpolated (and extrapolated) linearly in log-log.

Parameters
----------
D, G : sequences
    Increasing dilations and ZpM's time per p-vector at each.
D0s, K : sequences
    Split dilations (integers) and ZpK's time per p-vector above each.
c : float, optional
    Time per p-vector to compute the bound.
shared : bool, optional
    As in `bound_routing`.

<a id="pfvs.dilation.CostModel.G_at"></a>

---


#### G\_at

```python
def G_at(D)
```

ZpM's time per p-vector at dilation(s) D.

<a id="pfvs.dilation.CostModel.K_at"></a>

---


#### K\_at

```python
def K_at(D0)
```

ZpK's time per p-vector above D0 (one of D0s).

<a id="pfvs.dilation.CostModel.route"></a>

---


#### route

```python
def route(bounds: ArrayLike) -> tuple[float, float, float]
```

`bound_routing` with these costs: (b_star, D0, t).

<a id="pfvs.dilation.CostModel.route_at_price"></a>

---


#### route\_at\_price

```python
def route_at_price(bounds: ArrayLike, lam: float) -> tuple[np.ndarray, int]
```

Routing at a price lam (GPU time per CPU time saved, shared by
geometries on the same hardware): D0 minimizes G(D0) + lam K(D0), and
p goes to its bound iff G(b) - G(D0) <= lam K(D0). Returns
(to_bound mask, D0).

<a id="pfvs.gpu"></a>

---


# pfvs.gpu

<a id="pfvs.gpu.available"></a>

---


#### available

```python
def available() -> bool
```

**Description:**
Whether the GPU backend is built and a GPU (of the kind it was built
for) is present.

**Returns:**
*(bool)* True if `coniZpM(..., device="gpu")` can run.

<a id="pfvs.gpu.unavailable_reason"></a>

---


#### unavailable\_reason

```python
def unavailable_reason() -> str
```

Why `available()` is False ('' if it is True).

<a id="pfvs.gpu.backend"></a>

---


#### backend

```python
def backend() -> str
```

'cuda' (NVIDIA) or 'hip' (AMD): what the GPU backend was built for.

<a id="pfvs.gpu.release"></a>

---


#### release

```python
def release(device: int = 0) -> None
```

Free the device buffers the backend keeps between calls on `device`
(they are reused by later calls; this returns the memory).

<a id="pfvs.gpu.max_h11"></a>

---


#### max\_h11

```python
def max_h11() -> int
```

The largest h11 the GPU build supports (compile-time).

<a id="pfvs.gpu.coni_batch_multi"></a>

---


#### coni\_batch\_multi

```python
def coni_batch_multi(geoms, ps, pgeo, device=0, batch=0, verbose=False)
```

**Description:**
The batched coni pipeline on the GPU for p-vectors of several geometries
at once (what keeps the device busy when each geometry has few
p-vectors).

**Arguments:**
- `geoms` *(list of dict)*: Per geometry: `kappa` (h, h, h), `Mbasis`
(h, h), `Q`, `dilation`, `M0min`.
- `ps` *(list/array of int arrays)*: The p-vectors, each of length h of
its geometry, *with* the leading 0 (as `_coni_batch` takes them).
- `pgeo` *(array of int)*: The geometry index of each p-vector.
- `device` *(int, optional)*: GPU device ordinal.
- `batch` *(int, optional)*: p-vectors per device batch (0: default).
- `verbose` *(bool, optional)*: Per-batch timings on stderr.

**Returns:**
*(tuple)* `(M, Kn, q, pidx, pstat, seconds_gpu)`: for every lattice point
M = Binter c and Kn = Z Binter c (rows of shape (N, max_h11()), first h
entries valid), q = c^T mat c, and the index of its p-vector; pstat[i] is
0 if p-vector i was done on the GPU and 1 if it must go through the CPU
path (none of its points are returned). Points are grouped by p-vector
(ascending) and sorted by (M, Kn) within one.

<a id="pfvs.gpu.coni_batch"></a>

---


#### coni\_batch

```python
def coni_batch(kappa, Mbasis, ps, Q, dilation, M0min, max_N_out, device=0)
```

GPU counterpart of `fp_kernel._coni_batch` for one geometry: (M, Kn, q,
pidx, pstat), pstat 1 marking p-vectors for the CPU path (including any
with more than max_N_out points).

<a id="pfvs.distributed"></a>

---


# pfvs.distributed

<a id="pfvs.distributed.make_jobs"></a>

---


#### make\_jobs

```python
def make_jobs(datas,
              B,
              D,
              Q=None,
              M0min=13,
              ids=None,
              n_p=None,
              exhaustive=False,
              cost_model=None)
```

**Description:**
Jobs for `serve`: one per geometry.

**Arguments:**
- `datas` *(list of CYData)*: The geometries.
- `B` *(int or list)*: p-box half-width: all primitive p in the cone
with |p|_inf <= B.
- `D` *(float or list)*: Ellipsoid dilation.
- `Q` *(int, list or None)*: Tadpole (default h11 + h21 + 4).
- `M0min` *(int, optional)*: As in coniZpM.
- `ids` *(list, optional)*: Job names (default 0, 1, ...); used for the
output file names, so they must be unique.
- `n_p` *(list, optional)*: Estimated p-vector counts, for sharding
large boxes (see `serve`'s target_p).
- `exhaustive` *(bool or list, optional)*: Every coni PFV of each
direction, at any dilation; `D` is then used only for p-vectors
without a dilation bound, reported as incomplete.
- `cost_model` *(CostModel or list, optional)*: Routing costs for
exhaustive jobs (`pfvs.dilation.CostModel`); only their shape
matters. Default: measured for the job's h11.

**Returns:**
*(list of dict)* The jobs.

<a id="pfvs.distributed._Coordinator"></a>

---


## \_Coordinator Objects

```python
class _Coordinator()
```

Lives in the manager's server process; its methods are the RPCs.

Units are keyed (job id, path): (k,) is a job's k-th shard, (k, i) the
i-th piece it was split into, path + (-1,) the ZpK unit of an exhaustive
search unit. A job is done when all its leaves are. Splits are logged
(splits.pkl) so a resumed run rebuilds the same leaves.

<a id="pfvs.distributed._Coordinator.get_work"></a>

---


#### get\_work

```python
def get_work(worker, max_p, kinds=("search", ), gpu=False)
```

Units totalling about max_p p-vectors of the given kinds, in order
of preference; [] if none now, None when all is done. CPU workers get
no exhaustive search units while a GPU worker is active.

<a id="pfvs.distributed._Coordinator.split"></a>

---


#### split

```python
def split(worker, jid, path, n_found, want)
```

A worker found a unit much larger than it should take: split it
(the worker drops it). Returns False if it cannot be split.

<a id="pfvs.distributed.serve"></a>

---


#### serve

```python
def serve(jobs,
          out,
          address="0.0.0.0:5055",
          authkey=None,
          target_p=1 << 20,
          lease_s=3600.0,
          max_tries=3,
          poll_s=10.0,
          verbose=True,
          backup_s=30.0,
          lam0=1.0)
```

**Description:**
Run the coordinator until every unit is done (or has failed `max_tries`
times). Blocks; results go to `out` (see `load_results`).

**Arguments:**
- `jobs` *(list of dict)*: From `make_jobs`.
- `out` *(str)*: Output directory (created; an existing one is resumed).
- `address` *(str)*: host:port to listen on.
- `authkey` *(str, optional)*: Shared secret (default: PFVS_AUTHKEY).
- `target_p` *(int, optional)*: p-vectors per unit for sharding jobs with
an n_p estimate.
- `lease_s` *(float, optional)*: Reissue a unit not returned in this long.
- `max_tries` *(int, optional)*: Attempts per unit before it is recorded
in <out>/failed.txt.
- `backup_s` *(float, optional)*: Once nothing is pending, also give a
unit leased longer than this to an idle worker (first result wins).
- `lam0` *(float, optional)*: Initial routing price of exhaustive jobs.

**Returns:**
*(dict)* The final progress counters.

<a id="pfvs.distributed.load_results"></a>

---


#### load\_results

```python
def load_results(out)
```

**Description:**
The finished jobs of an output directory.

**Returns:**
*(dict)* job id -> dict(K, M, P, n_p, B, D, Q): the PFVs (K, M) and
their p-vectors P = p[1:]. Exhaustive jobs also have `incomplete`: the
p-vectors without a dilation bound (searched at dilation D only).

<a id="pfvs.distributed.work"></a>

---


#### work

```python
def work(address,
         authkey=None,
         device="cpu",
         procs=None,
         batch_p=None,
         verbose=False,
         max_unit_p=None,
         bound_threads=None)
```

**Description:**
Run workers against a coordinator until it has no more work. Blocks.

**Arguments:**
- `address` *(str)*: The coordinator's host:port.
- `authkey` *(str, optional)*: Shared secret (default: PFVS_AUTHKEY).
- `device` *(str, optional)*: "cpu", "gpu:N" (device N) or "gpu" (every
device of this machine, one worker each).
- `procs` *(int, optional)*: CPU worker processes (default: all cores).
- `batch_p` *(int, optional)*: p-vectors per request (GPU default 2^20,
CPU default 2^14).
- `max_unit_p` *(int, optional)*: Larger units are sent back to be split
(GPU default 2^23, CPU default 2^17).
- `bound_threads` *(int, optional)*: GPU worker's host threads for
exhaustive jobs' dilation bounds (default: half the cores).

