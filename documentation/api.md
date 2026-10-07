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
    A fully initialized CYData object populated from the given
    CalabiYau.

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

**Description:**
Returns the a-matrix, used for finding (Coni) PFVs. This matrix is
defined componentwise as
\begin{equation}
\tilde{a}_{ij} = \begin{cases}
kappa_{iij} & i\geq j\
kappa_{ijj} & i < j.
\end{cases}
\end{equation}
See, e.g., eq 2.52 from https://arxiv.org/pdf/2406.13751.

**Arguments:**
Nothing.

**Returns:**
The a-matrix.

<a id="pfvs.cydata.CYData.b"></a>

---


#### b

```python
@property
def b()
```

**Description:**
Returns the b-vector, used for finding (Coni-)PFVs. This vector is
defined differently for non-Coni and Coni PFVs. For non-Coni PFVs, it
is defined as
\begin{equation}
\tilde{b} = c_2.
\end{equation}
For Coni PFVs, it is defined as
\begin{equation}
\tilde{b} = c_2 + n_{cf} q_{coni}
\end{equation}
where $n_{cf} = 2$ and $q_{coni} = `coni_normal`$ (see below eq 3.5 of
https://arxiv.org/pdf/2406.13751).

**Arguments:**
Nothing

**Returns:**
The b vector.

<a id="pfvs.cydata.CYData.M_lattice"></a>

---


#### M\_lattice

```python
def M_lattice(verify: bool = True) -> np.ndarray
```

**Description:**
Computes a basis of the sublattice of all vectors, M, such that
(b/24).M    and    (a/2)@M    and    M
are all integral. This is equivalent to
[b/24; a/2; 1]@M
being integral.

The set of all such M is just the dual to the lattice spanned by the
**rows** of
[b/24; a/2; 1].

Work with column bases throughout.

**Arguments:**
- `verify`: Whether to verify the computation by checking dot products
with a and b.

**Returns:**
A basis for M satisfying the integrality constraints. Basis vectors are
columns.

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

Upper bound on the dilation of any coni PFV with this PFV's
direction and tadpole: delta < Q/mu0 (exact; see ``pfvs.dilation``).
A diagnostic; None if the bound's hypotheses fail.

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

Recall `self.b` is 24x and `self.a` is 2x the prepotential coefficients,
so the two divisions above are exact iff M satisfies the congruences.
Returns None if either division is non-integral. Non-coni only.

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

Compute the superpotential series W = sum_i c_i * exp(2*pi*i*tau*e_i),
where e_i = dot(p, q) and c_i = sum_{q at e_i} n_q * dot(M, q).

See also `series_abs_vev`, `series_corrections` for downstream
diagnostics built on this series.

Returns a list of [coeff, exponent] pairs for nonzero terms only,
sorted by exponent. Stops after N_nonzero nonzero terms.

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

Evaluate |W_i| at the tau0 from the 2-term approximation.

Built on `series`. See also `series_corrections`.

Specifically, finds the value of exp(2*pi*i*tau) minimizing W_0 + W_1,
plugs it into each W_i = c_i * exp(2*pi*i*tau*e_i), and returns |W_i|
(or log10|W_i| if as_logs=True), one per series term.

<a id="pfvs.pfv.PFV.series_corrections"></a>

---


#### series\_corrections

```python
def series_corrections(as_logs: bool = False) -> list[float]
```

Compute |W_i| / W0 for i >= 2, as a measure of higher-order corrections
to the 2-term approximation. Returns the ratios (or log10 if
as_logs=True), starting from the third series term.

Built on `series_abs_vev`.

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

Compute the matrices defining the M-ellipsoid in coni-ZpM.

In brief detail,
    - M lives in a lattice M = Binter@c
    - the component of K perpendicular to the conifold curve can be computed
      as Kperp = (Z@M)[1:] for Z = kappa@p
    - the parallel component of K (i.e., K[0]) is unconstrained, other than
      K[0] > 0 (from the physics)
    - one can show (see `coniZpM`) a K[0]>0 exists s.t. -dot(K,M) <= Qmax
      iff -c^T @ Binter^T @ Z @ Binter @ c <= Qmax. Define
      mat = -Binter^T @ Z @ Binter.
That last constraint c^T @ mat @ c <= Qmax is the ellipsoid constraint. One
can actually dilate the ellipsoid as long as
GCD(Kperp) > (c^T @ mat @ c)/Qmax - see the 'cut on feasibility of finding a
K0 giving K'>0' section of `coniZpM`.

Parameters
----------
p : ndarray of shape (h11,) or (h11-1,)
    The p-vector.
data : CYData, optional
    The relevant data from the associated CY. Mutually exclusive with kappa
    and Mbasis.
kappa : ndarray of shape (h11, h11, h11), optional
    The triple intersection numbers of the CY. Mutually exclusive with data.
    If provided, it is assumed that Mbasis is also provided.
Mbasis : ndarray of shape (h11, h11), optional
    The lattice basis for M-vectors. Mutually exclusive with data.
    If provided, it is assumed that kappa is also provided.
extra_lll_reduction : bool, optional
    Whether to perform an extra (technically unnecessary) LLL reduction on
    the updated M vector lattice basis, Binter. Useful since otherwise
    there are sometimes overflows. Defaults to True.
extra_checks : bool, optional
    Deprecated, no effect (warns if set): mat is always computed in
    exact integer arithmetic, so there is nothing to check.

Returns
-------
mat : ndarray of shape (h11-1, h11-1)
    The matrix defining the ellipsoid. I.e., c^T @ mat @ c <= Qmax. We
    typically dilate this ellipsoid via
    c^T @ mat @ c <= ellipsoid_dilation * Qmax
Z : ndarray of shape (h11, h11)
    The matrix relating M and K. Specifically, K[1:] = (Z@M)[1:]
Binter : ndarray of shape (h11, h11-1)
    Updated M-vector lattice basis, integrating the dot(K,p)=0 constraint.

<a id="pfvs.coniZp.coni_H_matrix"></a>

---


#### coni\_H\_matrix

```python
def coni_H_matrix(ZBinter: ArrayLike, proj: ArrayLike = None)
```

Compute the H-matrix for use in coni-ZpM. This is the HNF of (Z@Binter)[1:].

This is the preferred approach for enforcing the GCD(K[1:]) cut in
`coniZpM`. The alternative lattice-based approach is `_Kperp_gcd_lattice`
(not recommended in practice).

In coni-ZpM, one wants to ensure GCD(K[1:]) is sufficiently large. A point
c in the M-ellipsoid has an associated valuation c^T @ mat @ c. For dilated
ellipsoids, this can have c^T @ mat @ c > Qmax. This would give rise to a K
and M which violates tadpole (i.e., -dot(K,M) > Qmax) unless
    GCD(K[1:]) > (c^T @ mat @ c)/Qmax,
in which case one can divide both p and K by GCD(K[1:]) to bring the
solution back under tadpole. The strict inequality is correct but
unintuitive. See the 'cut on feasibility of finding a K0 giving K'>0'
section of `coniZpM`.

Recall that
    1 The M-vector is built incrementally via the relationship M = Binter c,
      using a modified Fincke-Pohst algorithm.
    2 K[1:] = (Z @ Binter @ c)[1:]
Naively, one would have to fully set c before checking GCD(K[1:]).

A trick, though:
    FP sets c from right to left, beginning with c[-1], then c[-2], etc.

    This uses the fact that FP provides a monotonically increasing lower
    bound on  c^T @ mat @ c as further components of c are set.

    Similarly, since H is upper triangular, H[-m:,-m:] @ c[-m:] is a
    monotonically decreasing upper bound on
        GCD(H@c) = GCD((Z@Binter)[1:,:] @ c) = GCD(K[1:]).
    This is because (H@c)[-m:] = H[-m:,-m:]@c[-m:] and
    GCD((H@c)[-m:]) >= GCD((H@c)[-n:]) for m<n.

    Thus, during FP, one can check if the current upper bound on the GCD
    is sufficiently large compared to the current lower bound on the
    valuation. If not, then one can immediately prune the current branch.

Parameters
----------
ZBinter : ndarray of shape (h11,h11-1)
    The product of matrices Z and Binter from coni_M_ellipsoid. Has
    interpretation that K[1:] = (ZBinter c)[1:].
proj : ndarray of shape (h11-1,h11)
    An optional projection matrix, since we want the HNF of (Z Binter)[1:].
    This is trivial: identity(h11)[1:,:]. If not provided, then it's
    computed using `_get_proj`.

Returns
-------
H : ndarray of shape (h11-1, h11-1)
    The HNF (Z@Binter)[1:]. Has interpretation that GCD(H@c) = GCD(K[1:])
    and that GCD(H[-m:,-m:]@c[-m:]) >= GCD(H[-n:,-n:]@c[-n:]) for m<n.

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
            verbosity: int = 0) -> tuple[ArrayLike, ArrayLike]
```

A 'Zp' implementation that computes coniPFVs from input integer p-vectors.

The logic is
    1 an integer p-vector defines a certain ellipsoid (see `coni_M_ellipsoid`)
    2 a lattice point c in this ellipsoid defines an M-vector via Binter@c.
      this also defines (most of) a K-vector via K[1:] = (Z@Binter@c)[1:]
so one wants to enumerate such c-vectors. This is done via Fincke-Pohst.

As discussed in `coni_M_ellipsoid` and `coni_H_matrix`, this ellipsoid can be
dilated, but then only c vectors that give rise to K[1:] with sufficiently
large GCD are allowed. This is integrated into the Fincke-Pohst solver via
the H-matrix from `coni_H_matrix`. An alternative lattice-based approach is
available via `_Kperp_gcd_lattice` (controlled by `use_gcd_lattice`), but
is not recommended.

Likewise, one can impose constraints on M[0] >= 13 early in FP by ordering
the columns of the M-vector lattice basis such that the first row of this
basis (that corresponding to M[0]) has a maximal number of leading 0s.

Parameters
----------
data : CYData
    The relevant data from the associated CY.
ps : iterable of shape (N, h11-1)
    Each row of the iterable corresponds to the perpendicular component of a
    p-vector. I.e., p[1:]
Q : integer, optional
    Only return PFVs with -dot(K,M) = Q (exact equality, unlike the
    ``Qmin``/``Qmax`` range in non-coni ``ZpM``/``ZpK``). If not
    provided, set to h11+h21+4.
M0min : integer, optional
    Only return PFVs with M[0] >= M0min. Defaults to 13 to match physics.
ellipsoid_dilation : float, optional
    The dilation of the ellipsoid. Typically want >>1 to capture more PFVs.
    Empirically, runtime scales linearly with this value. Defaults to 1.
use_c_lattice : bool, optional
    Whether to build each p-vector's lattice data (the M-lattice basis
    Binter, the ellipsoid and the H-matrix) in C (fast, exact, with
    automatic fallback to the Python path on overflow). Either way the
    PFVs of each p-vector are listed in a canonical order (by M, then K),
    independent of the lattice basis. Defaults to True.
use_gcd_lattice : bool, optional
    Whether to construct explicit lattice bases for guaranteeing sufficient
    GCD of Kperp. Not recommended - it's generally quicker to just prune FP.
    Defaults to False.
low_level_parallelism : bool, optional
    Deprecated, no effect beyond forcing n_jobs = 1 (warns if set): the
    gcd step it parallelized is now vectorized. Parallelize over
    p-vectors with n_jobs instead.
n_jobs : int, optional
    How many jobs to spawn if not doing low-level parallelism. Defaults to
    twice the CPU count.
extra_checks : bool, optional
    Whether to do extra sanity checks in the ellipsoid generation. Never
    seen these fail so defaults to False.
extra_lll_reduction : bool, optional
    Whether to perform an extra (technically unnecessary) LLL reduction on
    the updated M vector lattice basis, Binter. Useful since otherwise
    there are sometimes overflows. Defaults to True.
device : str, optional
    Where the lattice setup and search run: "cpu", "gpu" (the GPU
    backend, NVIDIA or AMD; raises if it is not built or no device is
    present) or
    "auto" (the GPU when available and worthwhile, else the CPU; the
    environment variable PFVS_DEVICE overrides "auto"). Results are
    identical either way. Defaults to "auto".
max_N_pfvs : int, optional
    The maximum number of PFVs that can be output. The C-kernel requires a
    limit. Defaults excessively high to 1,000,000,000.
return_formal_pfvs : bool, optional
    Whether to return "PFV" objects as in pfv.py. Otherwise, an
    array of K-vectors (as rows) and an array of M-vectors (as rows) are
    returned. Defaults to False.
verbosity : int, optional
    The verbosity level. Higher is more verbose. Defaults to 0.

Returns
-------
Ks : ndarray of shape (N, h11)
  K-vectors of the PFVs, one per row. Only returned if
  return_formal_pfvs=False.
Ms : ndarray of shape (N, h11)
    M-vectors of the PFVs, one per row. Only returned if
    return_formal_pfvs=False.
pfvs : list of length N
     PFV objects (see ``pfv.PFV``). Only returned if
     return_formal_pfvs=True.

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

Compute the matrices defining the M-ellipsoid in nonconi-ZpM.

In brief detail,
    - M lives in a lattice M = Binter@c
    - K can be computed as Z @ M
    - tadpole is obeyed (-dot(K,M)<=Qmax) iff
      -c^T @ Binter^T @ Z @ Binter @ c <= Qmax. Define
      mat = -Binter^T @ Z @ Binter.
That last constraint c^T @ mat @ c <= Qmax is the ellipsoid constraint. One
can actually dilate the ellipsoid as long as
GCD(Kperp) >= (c^T @ mat @ c)/Qmax. This is subtly different from coni
contexts since, here, we don't typically enforce K[0] > 0

Parameters
----------
p : ndarray of shape (h11,) or (h11-1,)
    The p-vector.
data : CYData, optional
    The relevant data from the associated CY. Mutually exclusive with kappa
    and Mbasis.
kappa : ndarray of shape (h11, h11, h11), optional
    The triple intersection numbers of the CY. Mutually exclusive with data.
    If provided, it is assumed that Mbasis is also provided.
Mbasis : ndarray of shape (h11, h11), optional
    The lattice basis for M-vectors. Mutually exclusive with data.
    If provided, it is assumed that kappa is also provided.
extra_lll_reduction : bool, optional
    Whether to perform an extra (technically unnecessary) LLL reduction on
    the updated M vector lattice basis, Binter. Useful since otherwise
    there are sometimes overflows. Defaults to True.
extra_checks : bool, optional
    Deprecated, no effect (warns if set): mat is always computed in
    exact integer arithmetic, so there is nothing to check.

Returns
-------
mat : ndarray of shape (h11-1, h11-1)
    The matrix defining the ellipsoid. I.e., c^T @ mat @ c <= Qmax. We
    typically dilate this ellipsoid via
    c^T @ mat @ c <= ellipsoid_dilation * Qmax
Z : ndarray of shape (h11, h11)
    The matrix relating M and K. Specifically, K = Z @ M
Binter : ndarray of shape (h11, h11-1)
    Updated M-vector lattice basis, integrating the dot(K,p)=0 constraint.

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

Compute the matrices defining the K-ellipsoid in nonconi-ZpK.

In brief detail,
    - K lives in the orthog lattice to p. Call a basis for this lattice B
    - M can be computed as (kappa p)^{-1} K
    - tadpole is obeyed (-dot(K,M)<=Qmax) iff
      -d^T @ B^T @ (kappa p)^{-1} @ B @ d <= Qmax. Define
      mat = -B^T @ Ainv @ B.
That last constraint d^T @ mat @ d <= Qmax is the ellipsoid constraint.

Parameters
----------
p : ndarray of shape (h11,) or (h11-1,)
    The p-vector.
data : CYData, optional
    The relevant data from the associated CY. Mutually exclusive with kappa
    and Mbasis.
kappa : ndarray of shape (h11, h11, h11), optional
    The triple intersection numbers of the CY. Mutually exclusive with data.
    If provided, it is assumed that Mbasis is also provided.
Mbasis : ndarray of shape (h11, h11), optional
    The lattice basis for M-vectors. Mutually exclusive with data.
    If provided, it is assumed that kappa is also provided.
extra_lll_reduction : bool, optional
    Whether to perform an extra (technically unnecessary) LLL reduction on
    the updated M vector lattice basis, Binter. Useful since otherwise
    there are sometimes overflows. Defaults to True.
extra_checks : bool, optional
    Deprecated, no effect (warns if set): it is not used here.

Returns
-------
mat : ndarray of shape (h11-1, h11-1)
    The matrix defining the ellipsoid. I.e., d^T @ mat @ d <= Qmax
B : ndarray of shape (h11, h11-1)
    The K-vector lattice basis, integrating just the dot(K,p)=0 constraint.

<a id="pfvs.Zp.H_matrix"></a>

---


#### H\_matrix

```python
def H_matrix(ZBinter: ArrayLike)
```

Compute the H-matrix for use in non-coni ZpM. This is the HNF of Z@Binter.

Analogous to `coni_H_matrix` in coniZp.py, but without the projection that
drops K[0]: in the non-coni context, K = Z @ Binter @ c directly (no free
K[0] component), so the full matrix is used.

In ZpM, one wants to ensure GCD(K) is sufficiently large. A point c in the
M-ellipsoid has an associated valuation c^T @ mat @ c. For dilated ellipsoids,
this can have c^T @ mat @ c > Qmax. This would give rise to a K and M which
violates tadpole (i.e., -dot(K,M) > Qmax) unless
    GCD(K) >= (c^T @ mat @ c)/Qmax,
in which case one can divide K by GCD(K) to bring the solution under tadpole.

Recall that K = Z @ Binter @ c. Since H is the row-HNF of Z @ Binter,
GCD(H @ c) = GCD(K), and GCD(H[-m:,-m:] @ c[-m:]) >= GCD(H[-n:,-n:] @ c[-n:])
for m < n. This gives a monotonically decreasing upper bound on GCD(K) as FP
sets components of c from right to left, enabling early pruning.

Parameters
----------
ZBinter : ndarray of shape (h11, h11-1)
    The product of matrices Z and Binter from M_ellipsoid. Has interpretation
    that K = ZBinter @ c.

Returns
-------
H : ndarray of shape (h11, h11-1)
    The row-HNF of Z@Binter. Has interpretation that GCD(H@c) = GCD(K) and
    that GCD(H[-m:,-m:]@c[-m:]) >= GCD(H[-n:,-n:]@c[-n:]) for m<n.

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

A 'Zp' implementation that computes non-coni PFVs from input integer
p-vectors.

The logic is
    1 an integer p-vector defines a certain ellipsoid (see `M_ellipsoid`)
    2 a lattice point c in this ellipsoid defines an M-vector via Binter@c.
      this also defines a K-vector via K = Z @ Binter @ c
so one wants to enumerate such c-vectors. This is done via Fincke-Pohst.
GCD re-introduction is applied post-hoc via `_allow_gcds`.

Parameters
----------
data : CYData
    The relevant data from the associated CY.
ps : iterable of shape (N, h11-1)
    Each row of the iterable corresponds to the perpendicular component of a
    p-vector. I.e., p[1:]
Qmax : integer, optional
    Only return PFVs with -dot(K,M) <= Qmax. If not provided, set to
    h11+h21+2.
Qmin : integer, optional
    Only return PFVs with -dot(K,M) >= Qmin. If not provided, set to 0.
ellipsoid_dilation : float, optional
    The dilation of the ellipsoid. Typically want >>1 to capture more PFVs.
    Empirically, runtime scales linearly with this value. Defaults to 1.
use_c_lattice : bool, optional
    Whether to build each p-vector's lattice data (Binter, the ellipsoid
    and, with use_c_kernel, the H-matrix) in C (fast, exact, with
    automatic fallback to the Python path on overflow). The C path picks
    a different, equally valid LLL-reduced basis, so it finds the same
    PFVs but may list them in a different order. Set False to reproduce
    the previous order exactly. Defaults to True.
use_c_kernel : bool, optional
    Enumeration backend. True (default) uses `pfv_kernel` (C: exact
    decisions, GCD pruning; much faster for dilated ellipsoids). False
    uses `util.fp_iterative_njit` (Numba, floating-point decisions, no
    GCD pruning), kept for reference. They find the same PFVs.
n_jobs : int, optional
    How many jobs to spawn for per-p-vector parallelism. Defaults to twice
    the CPU count.
extra_checks : bool, optional
    Whether to do extra sanity checks in the ellipsoid generation. Never
    seen these fail so defaults to False.
extra_lll_reduction : bool, optional
    Whether to perform an extra (technically unnecessary) LLL reduction on
    the updated M vector lattice basis, Binter. Useful since otherwise
    there are sometimes overflows. Defaults to True.
max_N_pfvs : int, optional
    The maximum number of PFVs that can be output. The C-kernel requires a
    limit. Defaults excessively high to 1,000,000,000.
return_formal_pfvs : bool, optional
    Whether to return "PFV" objects as in pfv.py. Otherwise, an
    array of K-vectors (as rows) and an array of M-vectors (as rows) are
    returned. Defaults to False.
verbosity : int, optional
    The verbosity level. Higher is more verbose. Defaults to 0.

Returns
-------
Ks : ndarray of shape (N, h11)
  K-vectors of the PFVs, one per row. Only returned if
  return_formal_pfvs=False.
Ms : ndarray of shape (N, h11)
    M-vectors of the PFVs, one per row. Only returned if
    return_formal_pfvs=False.
pfvs : list of length N
     PFV objects (see ``pfv.PFV``). Only returned if
     return_formal_pfvs=True.

Raises
------
ValueError
    If ``data.coni`` is True, ``ps`` is empty, ``Qmax < Qmin``,
    ``ellipsoid_dilation <= 0``, or if ``_allow_gcds`` finds no valid
    GCD expansions.

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

A 'Zp' implementation that computes non-coni PFVs from input integer
p-vectors.

The logic is
    1 an integer p-vector defines a certain ellipsoid (see `K_ellipsoid`)
    2 a lattice point d in this ellipsoid defines a K-vector via B @ d.
      this also defines an M-vector via M = (kappa @ p)^{-1} @ B @ d
so one wants to enumerate such d-vectors. This is done via Fincke-Pohst.
GCD re-introduction is applied post-hoc via `_allow_gcds`. [WIP: GCD
pruning is not yet integrated into the Fincke-Pohst solver here, unlike
`coniZpM`.]

Parameters
----------
data : CYData
    The relevant data from the associated CY.
ps : iterable of shape (N, h11-1)
    Each row of the iterable corresponds to the perpendicular component of a
    p-vector. I.e., p[1:]
Qmax : integer, optional
    Only return PFVs with -dot(K,M) <= Qmax. If not provided, set to
    h11+h21+2.
Qmin : integer, optional
    Only return PFVs with -dot(K,M) >= Qmin. If not provided, set to 0.
ellipsoid_dilation : float, optional
    The dilation of the ellipsoid. Typically want >>1 to capture more PFVs.
    Empirically, runtime scales linearly with this value. Defaults to 1.
n_jobs : int, optional
    How many jobs to spawn for per-p-vector parallelism. Defaults to twice
    the CPU count.
extra_checks : bool, optional
    Deprecated, no effect (warns if set): it is not used by the K-ellipsoid path.
extra_lll_reduction : bool, optional
    Whether to perform an extra (technically unnecessary) LLL reduction on
    the updated M vector lattice basis, Binter. Useful since otherwise
    there are sometimes overflows. Defaults to True.
max_N_pfvs : int, optional
    The maximum number of PFVs that can be output. The C-kernel requires a
    limit. Defaults excessively high to 1,000,000,000.
return_formal_pfvs : bool, optional
    Whether to return "PFV" objects as in pfv.py. Otherwise, an
    array of K-vectors (as rows) and an array of M-vectors (as rows) are
    returned. Defaults to False.
verbosity : int, optional
    The verbosity level. Higher is more verbose. Defaults to 0.

Returns
-------
Ks : ndarray of shape (N, h11)
  K-vectors of the PFVs, one per row. Only returned if
  return_formal_pfvs=False.
Ms : ndarray of shape (N, h11)
    M-vectors of the PFVs, one per row. Only returned if
    return_formal_pfvs=False.
pfvs : list of length N
     PFV objects (see ``pfv.PFV``). Only returned if
     return_formal_pfvs=True.

Raises
------
ValueError
    If ``data.coni`` is True, ``ps`` is empty, ``Qmax < Qmin``,
    ``ellipsoid_dilation <= 0``, or if ``_allow_gcds`` finds no valid
    GCD expansions.

<a id="pfvs.pvectors"></a>

---


# pfvs.pvectors

<a id="pfvs.pvectors.pvecs"></a>

---


#### pvecs

```python
def pvecs(data: CYData, min_N_pts: int, verbosity: int = 0) -> np.ndarray
```

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

<a id="pfvs.dilation"></a>

---


# pfvs.dilation

<a id="pfvs.dilation.coni_mu0"></a>

---


#### coni\_mu0

```python
def coni_mu0(p: ArrayLike,
             kappa: ArrayLike) -> tuple[Fraction, list[int]] | None
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

<a id="pfvs.scoring"></a>

---


# pfvs.scoring

<a id="pfvs.scoring.expected_pfvs_per_p"></a>

---


#### expected\_pfvs\_per\_p

```python
def expected_pfvs_per_p(p: ArrayLike,
                        data: CYData = None,
                        kappa: ArrayLike = None,
                        Mbasis: ArrayLike = None,
                        Q: int | None = None,
                        ellipsoid_dilation: float = 1,
                        M0min: int = 13,
                        return_cumulative: bool = False) -> float | np.ndarray
```

Gaussian-heuristic expected number of coniPFVs coniZpM finds from one p.

See the module description for the formula and its caveats.

Parameters
----------
p : ArrayLike of shape (h11,) or (h11-1,)
    The p-vector (perpendicular components, as passed to coniZpM, or full).
data : CYData, optional
    The relevant data from the associated CY. Mutually exclusive with kappa
    and Mbasis (pass those to avoid recomputing them per p).
kappa : ArrayLike of shape (h11, h11, h11), optional
    data.kappa_cob.
Mbasis : ArrayLike of shape (h11, h11), optional
    data.M_lattice().
Q : integer, optional
    The tadpole. Required with kappa/Mbasis; defaults to coniZpM's
    h11+h21+4 when data is given.
ellipsoid_dilation : float, optional
    As in coniZpM. Defaults to 1.
M0min : integer, optional
    As in coniZpM. Defaults to 13.
return_cumulative : bool, optional
    If True, return E_p for every integer dilation 1..floor(ellipsoid_dilation)
    instead of only the last. Defaults to False.

Returns
-------
float or ndarray
    The expected count (0 when the ellipsoid form is not positive definite,
    as coniZpM then finds nothing for this p).

<a id="pfvs.scoring.estimate_coni_pfvs"></a>

---


#### estimate\_coni\_pfvs

```python
def estimate_coni_pfvs(data: CYData,
                       ps: ArrayLike | None = None,
                       N: int | None = None,
                       ellipsoid_dilation: float = 1,
                       Q: int | None = None,
                       M0min: int = 13,
                       n_samp: int = 512,
                       N0: int = 100_000,
                       seed: int = 0) -> float
```

Estimated number of coniPFVs a coniZpM search finds. For ranking.

Either pass the p-vectors you would give coniZpM (`ps`), or a search size
`N` (the p-vectors `pvecs(data, N)` would return). The estimate averages
`expected_pfvs_per_p` over at most `n_samp` of them and scales by the total.

Parameters
----------
data : CYData
    The relevant data from the associated CY (coni).
ps : ArrayLike of shape (N, h11-1), optional
    The p-vectors of the search. Mutually exclusive with N.
N : integer, optional
    Search size: estimate for the N p-vectors `score_coni_geometries`
    searches. Up to 10^7 they are enumerated and subsampled uniformly;
    beyond, they are sampled without enumerating them all (see
    `_sample_frontier`).
ellipsoid_dilation : float, optional
    As in coniZpM. Defaults to 1.
Q : integer, optional
    As in coniZpM. Defaults to h11+h21+4.
M0min : integer, optional
    As in coniZpM. Defaults to 13.
n_samp : integer, optional
    Number of p-vectors to average over. Per-p estimates are heavy-tailed;
    512 gave stable rankings on the coni_pfvs dataset. Defaults to 512.
N0 : integer, optional
    Base enumeration size when sampling from N > 10^7. Changes the
    absolute level, not the ranking; keep it fixed across compared
    estimates. Defaults to 100,000.
seed : integer, optional
    Seed for the p-vector subsample. Defaults to 0.

Returns
-------
float
    Estimated count. Overestimates by 3-5x (median; see the module
    description) with a wide per-conifold spread; compare estimates with
    each other rather than with absolute counts.

<a id="pfvs.scoring.score_coni_geometries"></a>

---


#### score\_coni\_geometries

```python
def score_coni_geometries(datas: list[CYData],
                          N: int,
                          ellipsoid_dilation: float = 1,
                          method: str = "search",
                          n_prefetch: int = 4,
                          seed: int = 0,
                          verbosity: int = 0,
                          **kwargs) -> np.ndarray
```

Score conifolds by how many coniPFVs a size-N search finds in each.

The search is coniZpM on exactly N p-vectors per conifold: those of
`pvecs(data, N)` (every primitive p in the cone in the smallest L-inf box
|p_i| <= B holding at least N), trimmed to N by keeping a uniform random
subset of the outermost shell |p|_inf = B. On a GPU this takes 1-10 s per
conifold for N = 2M at dilation 150 and h11 = 5-10, most of it generating
the p-vectors.

Parameters
----------
datas : list of CYData
    The conifolds (coni contexts).
N : integer
    Number of p-vectors searched per conifold.
ellipsoid_dilation : float, optional
    As in coniZpM. Defaults to 1.
method : str, optional
    "search" (the default): the exact count; use a GPU. "estimate":
    `estimate_coni_pfvs`, the Gaussian-heuristic prediction of it (CPU
    only; ranks worse, see the module description).
n_prefetch : integer, optional
    For "search": generate the p-vectors of the next conifolds in this
    many worker processes while the current one is searched (generating
    them is single-threaded and, at h11 >= 10, often slower than the GPU
    search). Each holds N x h11 int64. 0 generates them in turn.
    Defaults to 4.
seed : integer, optional
    Seed for the trimming of the outermost shell. Defaults to 0.
verbosity : integer, optional
    >= 1 prints one line per conifold. Defaults to 0.
**kwargs :
    Passed to coniZpM (Q, M0min, device, n_jobs) or `estimate_coni_pfvs`.

Returns
-------
ndarray of shape (len(datas),)
    scores[i]: the number of coniPFVs found in datas[i] (or, for
    "estimate", the predicted number).

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

GPU counterpart of `fp_kernel._coni_batch` for one geometry (the default
lattice options: extra LLL, cut-aware basis): returns (M, Kn, q, pidx,
pstat) with the same meaning -- pstat 1 marks p-vectors for the per-p
CPU path (including any with more than max_N_out lattice points, so that
path reports it as the CPU kernel would).

<a id="pfvs.distributed"></a>

---


# pfvs.distributed

<a id="pfvs.distributed.make_jobs"></a>

---


#### make\_jobs

```python
def make_jobs(datas, B, D, Q=None, M0min=13, ids=None, n_p=None)
```

**Description:**
Jobs for `serve`: one per geometry.

**Arguments:**
- `datas` *(list of CYData)*: The geometries.
- `B` *(int or list)*: p-box half-width (all primitive p in the Kahler
cone with |p|_inf <= B, as coniZpM is usually driven).
- `D` *(float or list)*: Ellipsoid dilation.
- `Q` *(int, list or None)*: Tadpole (default h11 + h21 + 4).
- `M0min` *(int, optional)*: As in coniZpM.
- `ids` *(list, optional)*: Job names (default 0, 1, ...); used for the
output file names, so they must be unique.
- `n_p` *(list, optional)*: Estimated p-vector counts; used to shard large
boxes into units of about `target_p` p-vectors (see `serve`).

**Returns:**
*(list of dict)* The jobs.

<a id="pfvs.distributed._Coordinator"></a>

---


## \_Coordinator Objects

```python
class _Coordinator()
```

Lives in the manager's server process; its methods are the RPCs.

Units are keyed (job id, path): a path is a tuple of ints -- (k,) for the
k-th initial shard of a job, (k, i) for the i-th piece it was split into,
and so on. A job is complete when all its leaf units are done; its PFVs
are the leaves' in path order. Splits are logged (splits.pkl) so a
resumed run rebuilds the same leaves.

<a id="pfvs.distributed._Coordinator.get_work"></a>

---


#### get\_work

```python
def get_work(worker, max_p)
```

Units totalling about max_p p-vectors (estimated), or [] if none
is available now; None when everything is done.

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
          backup_s=30.0)
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
- `backup_s` *(float, optional)*: Once no unit is pending, a unit leased
for longer than this is also given to an idle worker (the first
result wins), so slow workers do not hold up the end.

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
*(dict)* job id -> dict(K, M, P, n_p, B, D, Q): the PFVs (K, M) and each
one's p-vector P = p[1:].

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
         max_unit_p=None)
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
- `max_unit_p` *(int, optional)*: A unit with more p-vectors than this is
sent back to be split (GPU default 2^23, CPU default 2^17), so that no
worker holds a unit for long.

