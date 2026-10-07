# pfvs
*[Nate MacFadden](https://github.com/natemacfadden), Liam McAllister Group, Cornell*

Tools for computing/verifying perturbatively flat vacua (PFVs). For some references on PFVs, see
- [Vacua with Small Flux Superpotential](https://arxiv.org/abs/1912.10047) by Mehmet Demirtas, Manki Kim, Liam McAllister, Jakob Moritz
- [Small Cosmological Constants in String Theory](https://arxiv.org/abs/2107.09064) by Mehmet Demirtas, Manki Kim, Liam McAllister, Jakob Moritz, Andres Rios-Tascon, and
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

$$\tilde{a}\_{ij} = \frac{1}{2}\begin{cases} \kappa_{iij} & i\geq j\\\\ \kappa_{ijj} & \text{o.w.} \end{cases}$$

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

Requires Python 3.10+ on Linux or macOS.

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

### Machine-specific build
For a slightly faster (~5%), machine-specific kernel, set `PFVS_NATIVE=1` (adds `-march=native`):
```bash
PFVS_NATIVE=1 pip install -e .
```

### GPU backend (optional; NVIDIA or AMD)
`coniZpM` can run its lattice setup and search on a GPU, with results identical to the CPU path (same arrays, same order). The backend is a small shared library built from `pfvs/fp_kernel/cuda/` by `nvcc` (NVIDIA, CUDA 11.5+) or `hipcc` (AMD, ROCm 6+). By default (`PFVS_GPU=auto`) the build uses whichever toolchain it finds and never fails the install over it:
```bash
pip install -e .                    # + GPU backend if nvcc/hipcc is found
PFVS_GPU=cuda pip install -e .      # require the NVIDIA backend
PFVS_GPU=hip  pip install -e .      # require the AMD backend
```
The GPUs of the build machine are targeted (`native`); set `PFVS_CUDA_ARCH` / `PFVS_HIP_ARCH` (e.g. `sm_80,sm_90` or `gfx90a,gfx1100`) to build for others. A pip-installed ROCm SDK is found automatically, or point `PFVS_ROCM_PATH` at a ROCm tree; `NVCC_CCBIN` selects nvcc's host compiler. See `setup.py` for the rest.

Then `coniZpM(..., device="auto")` (the default) uses the GPU for batches of 256+ p-vectors, `device="gpu"` requires it and `device="cpu"` avoids it; `PFVS_DEVICE` overrides `"auto"`. `pfvs.gpu.available()` reports whether a device is usable, and `pfvs.gpu.coni_batch_multi` runs many geometries in one device call.

## Running on many machines

`pfvs.distributed` spreads coni-PFV searches over any number of machines and devices -- NVIDIA and AMD GPUs and CPUs, Linux or macOS. A coordinator splits each geometry's p-box into units; workers lease units, search them, and send back PFVs. Units are checkpointed as they arrive (a restarted coordinator resumes), units of vanished workers are reissued, oversized units are split on the fly, and slow workers get backup copies near the end.
```python
from pfvs import distributed
jobs = distributed.make_jobs(datas, B=..., D=..., n_p=...)   # one job per CYData
pickle.dump(jobs, open("jobs.pkl", "wb"))
```
```bash
export PFVS_AUTHKEY=...                                          # same secret everywhere
python -m pfvs.distributed serve jobs.pkl out/ --address 0.0.0.0:5055    # coordinator
python -m pfvs.distributed work HOST:5055 --device gpu           # a worker per GPU of this machine
python -m pfvs.distributed work HOST:5055 --device cpu --procs 16
```
`distributed.load_results("out/")` then gives each job's PFVs (K, M) and their p-vectors. The connection is authenticated, but it exchanges pickles: use it on a trusted network only.

## Ranking conifolds

`pfvs.prediction` ranks conifolds by how many coni PFVs a search of the same size finds in each, e.g. to decide which to search in depth:
```python
from pfvs.prediction import rank_coni_geometries, count_coni_pfvs
order, counts = rank_coni_geometries(datas, N=2_000_000, ellipsoid_dilation=150)
n_pfvs, n_p = count_coni_pfvs(data, ps=ps, ellipsoid_dilation=150)   # or a given set of p-vectors
```
Both run the search itself (`coniZpM`) on the first N p-vectors of `pvecs` (trimmed to exactly N), so the counts are exact for those p-vectors. On a GPU a geometry takes seconds, mostly generating the p-vectors, which `rank_coni_geometries` does for the next geometries in worker processes. On 63 dataset conifolds (h11 = 5-11, N = 2M, dilation 150) the counts rank the dataset's recorded counts with Spearman 0.98. Without a GPU, `method="estimate"` uses a parameter-free lattice-point estimate instead (Spearman 0.89 on the same conifolds; it overcounts 3-5x, see `pfvs/prediction.py`).

## Examples

Scripts in `examples/` (run in CI):

- [`examples/manwe.py`](examples/manwe.py) is the starting point: it checks the "Manwe" coni PFV of [arXiv:2406.13751](https://arxiv.org/abs/2406.13751), finds it again from scratch, and ranks the PFVs found by $W_0$ (`--plot` saves $W_0$ vs. alignment). It needs no CYTools.
- [`examples/h11_11.py`](examples/h11_11.py) builds an $(h^{1,1}, h^{2,1}) = (11, 347)$ geometry with CYTools and scans its conifold.

The API reference is [`documentation/api.md`](documentation/api.md), generated from the docstrings (CI checks it is current).

## Performance

The core of the coniPFV search is enumerating integer vectors in a (dilated) ellipsoid subject to several cuts (see the [Algorithm](#algorithm) section for what "dilation" means -- it is the flux denominator $p_{denom}$). The current kernel, `conipfv_kernel` (C), does this with a Fincke-Pohst search that prunes on every cut as it goes. The previous implementation used in [arXiv:2406.13751](https://arxiv.org/abs/2406.13751) ("dSv1") instead materialized a bounding box, filtered it down to the ellipsoid, then rejection-sampled the cuts.

Both take the same ellipsoid (Zp-style) approach and return identical results; they differ only in how they enumerate it. The new kernel keeps only its output and an $O(h^{1,1})$ recursion stack, while dSv1 also materializes the whole bounding box, whose size grows as $(Q\cdot p_{denom})^{h^{1,1}/2}$. That box is what drives dSv1's time and memory up sharply as the dilation grows.

**Current pipeline (`fp_kernel`).** Relative to the previous version of this repo, the search is now:

- **Exact.** A single C kernel serves both pipelines (see [`pfvs/fp_kernel/README.md`](pfvs/fp_kernel/README.md)), and every accept/reject decision is exact integer arithmetic. The per-p-vector lattice setup (M-lattice basis, ellipsoid, $H$-matrix) is also C, and also exact.
- **Batched.** `coniZpM` runs the lattice setup and the kernel for a whole chunk of p-vectors in one C call, then post-processes all lattice points at once.
- **Built on a cut-aware basis.** The lattice basis is chosen so that the $M_0\ge M_{0,\min}$ cut is decided at the first coordinate searched, and it is LLL-reduced with respect to the ellipsoid itself. This makes searches ~4–5× smaller at high $h^{1,1}$ and $p_{denom}$.

End-to-end `coniZpM` speedup over the previous version. Each cell covers 2–3 dataset geometries, with ≤200 p-vectors per geometry. Old and new ran on identical inputs, pinned to the same core, and found identical PFV sets, apart from PFVs with $\gcd(K_{1:})\ge2$ that only the new code finds (measured 2026-09-28):

| $h^{1,1}$ \ $p_{denom}$ | 20 | 50 | 150 | 400 |
|---:|---:|---:|---:|---:|
| 4  | 28.0× | 26.1× | 20.9× | 14.8× |
| 5  | 17.5× | 17.3× | 13.3× | 11.5× |
| 6  | 11.0× | 10.3× | 8.0×  | 5.2×  |
| 7  | 8.9×  | 9.2×  | 8.4×  | 6.6×  |
| 8  | 7.8×  | 8.3×  | 6.5×  | 5.1×  |
| 9  | 10.6× | 13.2× | 11.3× | 9.0×  |
| 10 | 8.1×  | 13.6× | 15.6× | 12.6× |
| 11 | 7.6×  | 10.7× | 10.4× | 8.3×  |

- **Low $h^{1,1}$.** The old code's time was mostly per-p overhead in Python/flint, which is now gone.
- **High $h^{1,1}$.** The gain comes from the smaller searches. For example, $h^{1,1}=11$ at $p_{denom}=400$ went from 7.0 to 0.84 ms per p-vector.
- **Whole dataset.** The full [coni-PFV dataset](https://huggingface.co/datasets/natemacfadden/calabi-yau-coni-pfvs) covers 18,253 geometries and 8.6M p-vectors at $|p|_\infty\le5$ and $p_{denom}=150$. All 33,376 of its PFVs are found, none missing. Because $K_{1:}$ is no longer assumed primitive (automatic Kperp handling), 820 further PFVs are found as well; each has $\gcd(K_{1:})\ge2$ and passes `PFV.check_all`. The dataset predates that change. The run takes 3,093 CPU-seconds, about 3 minutes on 23 cores.

**On a GPU.** The same pipeline on an RTX 5090 does the whole dataset run above in 9.8 s of device time (8.6M p-vectors, 1.14 µs each; 15 s for the whole call), with exactly the CPU's lattice points. Against all 22 cores of the CPU box, the GPU is ~3-6x faster at $p_{denom}\le10$ (where the per-p lattice setup dominates) and ~10-20x at $p_{denom}=200$-$400$ for $h^{1,1}\ge 9$ (where the search dominates). An RX 6700 XT runs at about 1/6, and a Radeon 8060S (integrated) about 1/9, of the RTX 5090.

Behaviour changes relative to the previous version (PFV order, errors instead of skipped p-vectors, defaults) are listed in [CHANGELOG.md](CHANGELOG.md).

The table below compares the *original* C kernel with dSv1, measured before these changes. It is kept for the box-vs-Fincke–Pohst comparison, which the current pipeline only improves on. Measured on the $h^{1,1}=7$ "Manwe" example (identical inputs, identical output, same cuts), on one CPU (Intel Core Ultra 7 270K, 24 cores, 30 GB), three runs each, reported as mean $\pm$ std:

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

- The new kernel's working set is just its output and the recursion stack (below RSS resolution here), so its speedup keeps growing with dilation while dSv1 follows its box.
- "box (est.)" is $3\times$ the candidate array $(Q\cdot p_{denom})^{h^{1,1}/2}$ (the vectorized ellipsoid test holds three such arrays at once); measured memory matches it from $p_{denom}\gtrsim 5$. Past $p_{denom}=30$ (13.4 GB measured) the harness skips dSv1 rather than exhaust the 30 GB machine, hence "out of memory".
- Memory is peak resident set above the run's baseline (Linux VmHWM, reset per measurement). dSv1 timings under ~10 ms are noisier (e.g. $p_{denom}=5$).
- The dSv1 baseline is a faithful reimplementation: verbatim `points_in_ellipsoid` plus a reimplemented metric-LLL, giving identical output and box memory, with `maximum_box_size` set to infinity (else it truncates) and `fluxbound=2`.
- Both methods are timed on enumeration only: the C kernel receives its factorization ($U$) precomputed, so the baseline's metric-LLL is likewise hoisted out of the timed region (reported separately as `prep_lll_s`).

Reproduce (self-contained, needs only this repo):
```bash
python benchmarks/benchmark_dSv1_vs_coniZpM.py
```

## Organization

```
pfvs/
├── pfvs/
│   ├── fp_kernel/         # C kernel* (fp_kernel.h) + C lattice setup (pfv_lattice.h) + Cython binding
│   │   └── cuda/          # GPU backend (CUDA/HIP): pipeline + C API (pfvs_gpu.cu)
│   ├── conipfv_kernel/    # re-exports fp_kernel.conipfv_kernel (coni-PFV enumeration)
│   ├── pfv_kernel/        # re-exports fp_kernel.pfv_kernel (non-coni PFV enumeration)
│   ├── coniZp.py          # coniZpM: coni-PFV generation pipeline
│   ├── Zp.py              # ZpM / ZpK: PFV generation pipeline
│   ├── cydata.py          # CYData: CY-data holder
│   ├── pfv.py             # PFV class + diagnostics
│   ├── gpu.py             # GPU backend loader (ctypes)
│   ├── distributed.py     # multi-machine coordinator/workers
│   ├── pvectors.py        # p-vector generation
│   └── util.py            # shared helpers (+ njit kernels)
├── tests/                 # pytest suite, exact oracle (oracle.py), fixtures (data/)
│   └── c/                 # standalone C driver for the kernel (profiling/debugging)
├── benchmarks/            # benchmark_dSv1_vs_coniZpM.py (headline), benchmark_conipfv.py
├── examples/              # manwe.py (self-contained), h11_11.py (CYTools)
├── documentation/         # api.md (generated by pydoc-markdown)
├── environment.yml
├── pyproject.toml
└── setup.py
```

*: This C code was originally the bottleneck/core of the problem, hence the name 'kernel'. See [`pfvs/fp_kernel/README.md`](pfvs/fp_kernel/README.md) for its exact contract and how it prunes. At high $h^{1,1}$ and $p_{denom}$ it remains the dominant cost.

## Correctness

The kernel's output is specified exactly (see [`pfvs/fp_kernel/README.md`](pfvs/fp_kernel/README.md)). All accept/reject decisions are exact integer arithmetic, and floating point is only used for pruning, widened by a rigorous error bound. The test suite checks this:

- **Kernel vs. an exact oracle.** [`tests/oracle.py`](tests/oracle.py) is a rational-arithmetic Fincke–Pohst with no floating point. The tests compare the kernel's output against it point for point, in order, on 219 real instances from the dataset below (some with $H$ entries beyond $2^{100}$) and on adversarial synthetic ones: non-echelon or rank-deficient $H$, boundary points, and ill-conditioned ellipsoids with huge entries.
- **End to end vs. the published dataset.** For 71 stored geometries ($h^{1,1}=3,\dots,11$), `coniZpM` runs over every p-vector with $|p|_\infty\le B$ at dilation $D$, with no p-vector skipped. Among its output, the PFVs with primitive $K_{1:}$ must equal *exactly* those in [calabi-yau-coni-pfvs](https://huggingface.co/datasets/natemacfadden/calabi-yau-coni-pfvs); the dataset predates automatic Kperp handling. Every other PFV found must have $\gcd(K_{1:})\ge2$ and pass `PFV.check_all`. The fixtures are built by [`tests/data/build_fixtures.py`](tests/data/build_fixtures.py) from a pinned dataset revision.
- **Regression tests for integer overflow.** Large p-vectors used to overflow int64 silently while the M-lattice was being built, and were then skipped. The tests use real large-p cases.
- **Independent routes agree.** Non-coni `ZpM` via the numba enumerator, the C kernel, and the C kernel with the C lattice setup find identical PFVs. `coniZpM` finds identical PFVs with and without the C lattice setup, and its exact post-processing fallback matches the int64 path.
- **The papers' examples.** Every named example of [arXiv:2406.13751](https://arxiv.org/abs/2406.13751) (coni) and [arXiv:2107.09064](https://arxiv.org/abs/2107.09064) (non-coni) is found from its p-vector. Every coni example, and the vacuum of Sec. 6.3, is also found by a scan of every p-vector in a box of the Kähler cone, with no knowledge of the paper's p. The other non-coni vacua need a cone other than the Kähler cone or too large a box. The geometries are rebuilt with CYTools by [`tests/data/build_paper_examples.py`](tests/data/build_paper_examples.py) and checked against the papers through their fluxes; `PFVS_SLOW_TESTS=1` adds the two slow scans.
- **Exact helpers.** The singularity test keeps ill-conditioned nonsingular $N$. The int128 factorization matches GMP on 200k random matrices (a C test run in CI).

Run `pytest tests/` (add `-n auto` with `pytest-xdist` to parallelize).

## Citing

See [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository"), and cite the papers above that introduced the method.

## License

[GPLv3 or later](LICENSE). Copyright (c) 2026 Nate MacFadden.
