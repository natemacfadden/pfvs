# pfvs
Tools for computing/verifying perturbatively flat vacua (PFVs). For some references on PFVs, see
- [Vacua with Small Flux Superpotential](https://arxiv.org/abs/1912.10047) by Mehmet Demirtas, Manki Kim, Liam McAllister, Jakob Moritz
- [Small Cosmological Constants in String Theory](https://arxiv.org/abs/1912.10047) by Mehmet Demirtas, Manki Kim, Liam McAllister, Jakob Moritz, Andres Rios-Tascon, and
- [Candidate de Sitter Vacua](https://arxiv.org/abs/2406.13751) by Liam McAllister, Jakob Moritz, Richard Nally, Andreas Schachner,

as well as their references. This repo focuses solely on PFVs as a combinatorial problem abstracted from the physics. In this light, there are also unpublished notes by Richard Nally/Mehmet Demirtas that much of this builds off of. The relevant aspects of such notes will be briefly restated below.

## Problem Statement
The goal will be to construct many (coni)PFVs for an input Calabi-Yau manifold. The relevant (fixed) data from the CY includes
- hodge numbers $h^{1,1}\in\mathbb{Z}\_{\geq 1}$ and $h^{2,1}\in\mathbb{Z}\_{\geq 1}$,
- triple intersection numbers $\kappa\in\mathbb{Z}^{h^{1,1},h^{1,1},h^{1,1}}$ which are totally symmetric (invariant under transposition of any two axes),
- second chern class $c_2\in\mathbb{Z}^{h^{1,1}}$, and
- a pointed Kähler cone $\\{x : Hx\geq 0\\}$ for some $H\in\mathbb{Z}^{N,h^{1,1}}$ with $\text{gcd}(\hat{n})=1$ for any row $\hat{n}$ of $H$.

We will also define two auxiliary variables from these fixed data
- the tadpole $Q = h^{1,1} + h^{2,1} + 2 \in\mathbb{Z}$ and
- the 'a-matrix' $\tilde{a}\in\frac{1}{2}\mathbb{Z}^{h^{1,1},h^{1,1}}$ given by

$$\tilde{a}\_{ij} = \frac{1}{2}\begin{cases} \kappa_{ijj} & i\geq j\\\\ \kappa_{iij} & \text{o.w.} \end{cases}$$

All of the above is fixed and (relatively cheaply) computable using software like [CYTools](https://github.com/LiamMcAllisterGroup/cytools). We now define, in terms of these variables, what a PFV and what a coniPFV is.

Given a CY, a *PFV* is a triple $(K,M,p)\in(\mathbb{Z}^{h^{1,1}}, \mathbb{Z}^{h^{1,1}}, \mathbb{Q}^{h^{1,1}})$ satisfying the following constraints
1. $\tilde{a}M \in\mathbb{Z}^{h^{1,1}}$,
2. $c\_2 \cdot M \in 24\mathbb{Z}$,
3. $Hp>0$,
4. $K\cdot p=0$,
5. $0 \leq -K\cdot M \leq Q$,
6. $\det(\kappa M) \neq 0$, and
7. $K = (\kappa M) p$.

This parameterization is redundant. E.g., with $M$ and $p$ one can compute $K$. This concludes the definition of a non-coni PFV.

More information than just the CY is required to define a coniPFV. Additionally, one needs a hyperplane $\hat{n}\in H$ corresponding to a conifold-curve (also computable using CYTools). It is canonical to choose a basis for this problem such that $\hat{n} = (1,0,\dots,0)$. One must transform the other input data to this basis. In this basis a *coniPFV* is a triple $(K,M,p)\in(\mathbb{Z}^{h^{1,1}}, \mathbb{Z}^{h^{1,1}}, \mathbb{Q}^{h^{1,1}})$ satisfying the following constraints
1. $\tilde{a}M \in\mathbb{Z}^{h^{1,1}}$,
2. $(c\_2 + (2,0,\dots,0)) \cdot M \in 24\mathbb{Z}$,
3. $p_0 = 0$ and $(H\setminus\hat{n})p>0$,
4. $K_{1:}\cdot p_{1:}=0$,
5. $0 \leq -K\cdot M \leq Q$,
6. $\det((\kappa M)_{1:,1:}) \neq 0$, and
7. $K\_{1:} = (\kappa M)\_{1:,1:} p\_{1:}$.

In some ways, the problem of enumerating coniPFVs is easier since, e.g., $p$ lives in a lower-dimensional cone... $p_0$ is fixed. In other ways, coniPFVs are more complicated than PFVs: a specification of $M$ and $p$ does not uniquely define $K$... $K_0$ is left semi-free (up to constraint #5).

In either case, for PFVs or coniPFVs, specification of $K$ and $M$ suffices to define the object. This will be the standard output.

## Algorithm
We provide only cursory descriptions of the algorithms here. - full detail will be provided in an upcoming (as of March 2026) paper. There are subtle differences between the non-coni and coniPFV algorithms - the following discussion will implicitly be non-coni PFV focused.

There are two general classes of algorithms
1. 'box-style algorithms': (non-exhaustively) enumerate $K$ and $M$ satisfying constraints #1, #2, and #5. This can be done by trying all $|K_i|\leq bound_K$ and $|M_i|\leq bound_M$, hence the name 'box' (there are better ways of enumerating such $K$, $M$ though). One can then rejection sample on constraint #6. Likewise, one can compute $p$ using #7 and then allows checking of constraints #3 and #4.
2. 'Zp-style algorithms': (non-exhaustively) enumerate $\hat{p} \in \mathbb{Z}^{h^{1,1}}$ obeying #3. Define $p = \hat{p}/p_{denom}$ for some $p\_{denom} \in \mathbb{Z}\_{>0}$. Use #7 to rewrite constraint #5 as an ellipsoidal constraint on $M$, $0\leq -M^T (\kappa \hat{p}) M \leq p_{denom} Q$. This defines the 'ZpM algorithm'. For non-coni PFVs only, one can invert constraint #7 to rewrite constraint #5 as an allipsoid on $K$, $0\leq -K^T (\kappa \hat{p})^{-1} K \leq Q/p_{denom}$. This defines the 'ZpK algorithm'. One can integrate constraints #1, #2, and #4 as modifications to the ellipsoid via certain lattice bases.

Zp algorithms require special care with $p_{denom}$. Focus on ZpM and call $0\leq -M^T (\kappa \hat{p}) M \leq Q$ the 'base' ellipsoid. To generate a non-coni PFV with $p_{denom}=d$, one needs to dilate the base ellipsoid $d$-times. An $M$ in this $d$-dilated ellipsoid only gives rise to $p_{denom}=d$ if $g | (\kappa M) \hat{p}$. This is a strong cut on an increasingly wide search space, making large $p_{denom}$ expensive/difficult to find with ZpM (in contrast to box which has no difficulty finding such $p_{denom}$). The purpose of ZpK was to invert this: the base ellipsoid in ZpK is sensitive to any $p_{denom} \geq 1$.

This gets to the point of efficiency (no careful analysis is done here). First, box-style algorithms are efficient at low $h^{1,1}$ but scale poorly with $h^{1,1}$. This is potentially due to the increasing narrowness of the Kähler cone $\\{x : Hx\geq 0\\}$ as dimension increases. In contrast, Zp-style algorithms typically scale better with $h^{1,1}$ than box. ZpM is particularly efficient, arguably running up to $h^{1,1}=60$. Unfortunately, the base ZpK ellipsoid is typically too large for ZpK to be usable.

## Demos
The demos are currently in demo_notebooks/. This will be modified soon.

## Organization

```
pfvs/
├── pfvs/
│   ├── conipfv_kernel/          # C kernel* for coniPFV enumeration
│   │   └── testing_benchmarks/  # C-code testing/benchmarks
│   ├── pfv_kernel/              # C kernel* for non-coni PFV enumeration
│   ├── cydata.py                # simple class holding CY-related data
│   └── Zp.py                    # main methods for generating PFVs
├── tests/
│   └─ FILL IN
├── environment.yml
├── pyproject.toml
└── setup.py
```

*: This C code originally was the bottleneck/core of the problem, hence the name 'kernel'. In the current state, unless one is studying large $p_{denom}$, these kernels represent a relatively small fracton of the total computation

**Note:** `util.py` also contains `conipfv_kernel_njit`, a Numba implementation of the same algorithm. It is not recommended for production use; prefer `conipfv_kernel`. The Numba version uses native integer arithmetic and can silently produce wrong results when H-matrix entries overflow 64-bit integers — the C version uses GMP for arbitrary-precision arithmetic and handles these cases correctly.

## Installation

### Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate pfvs
pip install -e .
```

### Or install dependencies separately:
```bash
conda install -c conda-forge gmp numpy cython
pip install -e .
```
