# pfvs
*[Nate MacFadden](https://github.com/natemacfadden), Liam McAllister Group, Cornell*

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

We provide only cursory descriptions of the algorithms here. Full detail will be provided in an upcoming (as of March 2026) paper. There are subtle differences between the non-coni and coniPFV algorithms - the following discussion will implicitly be non-coni PFV focused.

There are two general classes of algorithms
1. 'box-style algorithms': (non-exhaustively) enumerate $K$ and $M$ satisfying constraints #1, #2, and #5. This can be done by trying all $|K_i|\leq bound_K$ and $|M_i|\leq bound_M$, hence the name 'box' (there are better ways of enumerating such $K$, $M$ though). One can then rejection sample on constraint #6. Likewise, one can compute $p$ using #7 and then allows checking of constraints #3 and #4.
2. 'Zp-style algorithms': (non-exhaustively) enumerate $\hat{p} \in \mathbb{Z}^{h^{1,1}}$ obeying #3. Define $p = \hat{p}/p_{denom}$ for some $p\_{denom} \in \mathbb{Z}\_{>0}$. Use #7 to rewrite constraint #5 as an ellipsoidal constraint on $M$, $0\leq -M^T (\kappa \hat{p}) M \leq p_{denom} Q$. This defines the 'ZpM algorithm'. For non-coni PFVs only, one can invert constraint #7 to rewrite constraint #5 as an ellipsoid on $K$, $0\leq -K^T (\kappa \hat{p})^{-1} K \leq Q/p_{denom}$. This defines the 'ZpK algorithm'. One can integrate constraints #1, #2, and #4 as modifications to the ellipsoid via certain lattice bases.

Zp algorithms require special care with $p_{denom}$. Focus on ZpM and call $0\leq -M^T (\kappa \hat{p}) M \leq Q$ the 'base' ellipsoid. To generate a non-coni PFV with $p_{denom}=d$, one needs to dilate the base ellipsoid $d$-times. An $M$ in this $d$-dilated ellipsoid only gives rise to $p_{denom}=d$ if $g | (\kappa M) \hat{p}$. This is a strong cut on an increasingly wide search space, making large $p_{denom}$ expensive/difficult to find with ZpM (in contrast to box which has no difficulty finding such $p_{denom}$). The purpose of ZpK was to invert this: the base ellipsoid in ZpK is sensitive to any $p_{denom} \geq 1$.

This gets to the point of efficiency (no careful analysis is done here). First, box-style algorithms are efficient at low $h^{1,1}$ but scale poorly with $h^{1,1}$. This is potentially due to the increasing narrowness of the Kähler cone $\\{x : Hx\geq 0\\}$ as dimension increases. In contrast, Zp-style algorithms typically scale better with $h^{1,1}$ than box. ZpM is particularly efficient, arguably running up to $h^{1,1}=60$. Unfortunately, the base ZpK ellipsoid is typically too large for ZpK to be usable.

## Installation

### Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate pfvs
pip install -e .
```

### Or install dependencies separately:
`gmp` is the only non-Python dependency; `pip install -e .` resolves the rest
(numpy, cython, python-flint, numba, scipy, latticepts, joblib, matplotlib).
```bash
conda install -c conda-forge gmp
pip install -e .
```

## Demos

Example notebooks are in `demo_notebooks/`; `manwe_demo.ipynb` is the self-contained starting point -- it finds the perturbatively-flat vacua of the "Manwe" geometry from scratch.

## Performance

The core of the coniPFV search is enumerating integer vectors in a (dilated) ellipsoid subject to several cuts (see the [Algorithm](#algorithm) section for what "dilation" means -- it is the flux denominator $p_{denom}$). The current kernel, `conipfv_kernel` (C), does this with a Fincke-Pohst search that prunes on every cut as it goes. The previous implementation used in [arXiv:2406.13751](https://arxiv.org/abs/2406.13751) ("dSv1") instead materialized a bounding box, filtered it down to the ellipsoid, then rejection-sampled the cuts.

Both take the same ellipsoid (Zp-style) approach and return identical results; they differ only in how they enumerate it. The new kernel keeps only its output and an $O(h^{1,1})$ recursion stack, while dSv1 also materializes the whole bounding box, whose size grows as $(Q\cdot p_{denom})^{h^{1,1}/2}$. That box is what drives dSv1's time and memory up sharply as the dilation grows.

Measured on the $h^{1,1}=7$ "Manwe" example (identical inputs, identical output, same cuts), on one CPU (Intel Core Ultra 7 270K, 24 cores, 30 GB), three runs each, reported as mean $\pm$ std:

| dilation $p_{denom}$ | C (ms) | dSv1 (ms) | speedup | dSv1 memory | box (est.) |
|---:|---:|---:|---:|---:|---:|
| 1   | 0.005 $\pm$ 0.000 | 0.29 $\pm$ 0.03 | 53x      | 0.3 MB  | 1 MB    |
| 2   | 0.006 $\pm$ 0.000 | 0.91 $\pm$ 0.02 | 150x     | 3.6 MB  | 3 MB    |
| 5   | 0.006 $\pm$ 0.000 | 11.3 $\pm$ 5.6  | 1,921x   | 38 MB   | 30 MB   |
| 10  | 0.007 $\pm$ 0.000 | 84 $\pm$ 4      | 12,028x  | 354 MB  | 330 MB  |
| 15  | 0.011 $\pm$ 0.000 | 293 $\pm$ 3     | 26,699x  | 1.24 GB | 1.28 GB |
| 20  | 0.014 $\pm$ 0.000 | 711 $\pm$ 4     | 49,282x  | 3.02 GB | 3.14 GB |
| 25  | 0.018 $\pm$ 0.001 | 1,545 $\pm$ 46  | 86,820x  | 6.4 GB  | 6.68 GB |
| 30  | 0.022 $\pm$ 0.002 | 6,131 $\pm$ 159 | 281,495x | 13.4 GB | 14.1 GB |
| 40  | 0.031 $\pm$ 0.004 | out of memory   | --       | --      | 24.9 GB |
| 60  | 0.049 $\pm$ 0.001 | out of memory   | --       | --      | 120 GB  |
| 100 | 0.094 $\pm$ 0.001 | out of memory   | --       | --      | 659 GB  |

- The new kernel's working set is just its output and the recursion stack, below RSS resolution here (at most two output vectors), so its speedup keeps growing with dilation while dSv1 follows its box. The "box (est.)" column is $3\times$ the candidate array $(Q\cdot p_{denom})^{h^{1,1}/2}$: the vectorized ellipsoid test holds three same-size arrays at once (the integer coordinate grid, the float `candidates @ Zp` product, and their elementwise product) before the sum reduces them. Measured memory matches this estimate from $p_{denom}\gtrsim 5$; below that the box is smaller than page/pool granularity and RSS reads under it.
- By $p_{denom}=30$ the box is 13.4 GB (measured). Past that the harness skips dSv1 rather than risk exhausting the 30 GB machine: the budget guard triggers once the estimated box crosses its threshold, which is why those rows show only "box (est.)". The C kernel keeps running at flat cost through $p_{denom}=100$ (0.09 ms), where dSv1's box would need hundreds of GB.

Notes:

- Results are three runs on one machine, reported as mean $\pm$ std. dSv1 timings under about 10 ms have larger relative spread from system noise on short measurements (for example $p_{denom}=5$); the larger-dilation figures are tight.
- Memory is peak resident set above the run's baseline (Linux VmHWM, reset per measurement so a fork-inherited high-water mark cannot leak in). The C kernel also reserves an untouched `max_N_out` output buffer that never becomes resident and is not counted.
- The "box (est.)" column is `predict_box_gb`: $3\times$ the candidate-box footprint $(Q\cdot p_{denom})^{h^{1,1}/2}$. The harness's budget guard uses it to skip runs that would exhaust memory, so the high-dilation rows read "out of memory" rather than a measured number.
- The dSv1 baseline is a faithful reimplementation: verbatim `points_in_ellipsoid` plus a reimplemented metric-LLL, giving identical output and box memory, with `maximum_box_size` set to infinity (else it truncates) and `fluxbound=2`.
- Both methods are timed on enumeration only: the C kernel receives its factorization ($U$) precomputed, so the baseline's metric-LLL is likewise hoisted out of the timed region (reported separately as `prep_lll_s`).

Reproduce (self-contained, needs only this repo):
```bash
python benchmarks/benchmark_dSv1_vs_new.py
```

## Organization

```
pfvs/
├── pfvs/
│   ├── conipfv_kernel/    # C kernel* + Cython binding for coni-PFV enumeration
│   ├── pfv_kernel/        # C kernel* + Cython binding for non-coni PFV enumeration
│   ├── coniZp.py          # coniZpM: coni-PFV generation pipeline
│   ├── Zp.py              # ZpM / ZpK: PFV generation pipeline
│   ├── cydata.py          # CYData: CY-data holder
│   ├── pfv.py             # PFV class + diagnostics
│   ├── pvectors.py        # p-vector generation
│   └── util.py            # shared helpers (+ njit kernels)
├── tests/                 # test_manwe.py, test_conipfv_kernel.py, test_util.py
├── benchmarks/            # benchmark_dSv1_vs_new.py (headline), benchmark_conipfv.py
├── demo_notebooks/        # manwe_demo.ipynb (self-contained)
├── environment.yml
├── pyproject.toml
└── setup.py
```

*: This C code was originally the bottleneck/core of the problem, hence the name 'kernel'. In the current state, unless one is studying large $p_{denom}$, these kernels are a relatively small fraction of the total computation.
