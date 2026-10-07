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

Details will be in an upcoming paper. Two classes of algorithm (described for non-coni PFVs):
1. **Box:** enumerate $K$, $M$ obeying #1, #2, #5 in a box, then check #6, compute $p$ from #7 and check #3, #4. Efficient at low $h^{1,1}$; scales poorly.
2. **Zp:** enumerate integer $\hat p$ obeying #3 and set $p = \hat p/p_{denom}$. Through #7, constraint #5 becomes an ellipsoid on $M$, $0\leq -M^T (\kappa \hat{p}) M \leq p_{denom} Q$ (**ZpM**), or on $K$, $0\leq -K^T (\kappa \hat{p})^{-1} K \leq Q/p_{denom}$ (**ZpK**). #1, #2, #4 become lattice bases.

ZpM is the workhorse (usable up to $h^{1,1}\approx 60$). A PFV with $p_{denom}=d$ needs the base ellipsoid dilated $d$ times (the `ellipsoid_dilation` argument), so large $p_{denom}$ is expensive for ZpM. ZpK's ellipsoid covers every $p_{denom}$ but is usually too large on its own; **Zp** (`coniZp`) combines the two.

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
`coniZpM` can run on a GPU, with results identical to the CPU path. The build uses `nvcc` (CUDA 11.5+) or `hipcc` (ROCm 6+) if found:
```bash
pip install -e .                    # + GPU backend if nvcc/hipcc is found
PFVS_GPU=cuda pip install -e .      # require the NVIDIA backend
PFVS_GPU=hip  pip install -e .      # require the AMD backend
```
It targets the build machine's GPUs; `PFVS_CUDA_ARCH` / `PFVS_HIP_ARCH` build for others (see `setup.py` for the remaining variables). `coniZpM(..., device="auto")` uses the GPU for batches of 256+ p-vectors; `"gpu"` requires it, `"cpu"` avoids it. `pfvs.gpu.available()` reports whether a device is usable.

## Every PFV of a direction: `coniZp`

Every coni PFV with direction $\hat p$ has dilation $\delta < Q/\mu_0(\hat p)$ (`pfvs.dilation`), so searching up to that bound finds all of them:
```python
Ks, Ms, complete = coniZp(data, ps)   # complete[i]: p-vector i fully searched
```
The bound is ~10³ at $h^{1,1}\ge 8$, so each p-vector is searched either by ZpM up to its bound, or by ZpM up to a split dilation $D_0$ plus ZpK above it (`coniZpK`, CPU), whichever measured costs favour (`pfvs.dilation.bound_routing`).

## Running on many machines

`pfvs.distributed` spreads coni-PFV searches over machines, GPUs and CPUs. A coordinator splits each geometry's p-box into units, checkpoints results, and reissues units of vanished workers.
```python
from pfvs import distributed
jobs = distributed.make_jobs(datas, B=..., D=..., n_p=...)   # exhaustive=True: coniZp instead of coniZpM
pickle.dump(jobs, open("jobs.pkl", "wb"))
```
```bash
export PFVS_AUTHKEY=...                                          # same secret everywhere
python -m pfvs.distributed serve jobs.pkl out/ --address 0.0.0.0:5055    # coordinator
python -m pfvs.distributed work HOST:5055 --device gpu           # a worker per GPU
python -m pfvs.distributed work HOST:5055 --device cpu --procs 16
```
`distributed.load_results("out/")` returns each job's PFVs. It exchanges pickles: use it on a trusted network only.

## Scoring geometries

To pick which geometries to search in depth:
```python
from pfvs.scoring import score_geometries
scores = score_geometries(datas, N=2_000_000, ellipsoid_dilation=150)
```
`scores[i]` is the number of PFVs `coniZpM` (coni) or `ZpM` (non-coni) finds on the first N p-vectors of `datas[i]`. For coni, use a GPU; without one, `method="estimate"` predicts the count instead (ranks worse). Non-coni finds ~10 PFVs per p-vector, so N ~ 10^4-10^5 suffices.

## Examples

Scripts in `examples/` (run in CI):
- [`examples/manwe.py`](examples/manwe.py): checks the "Manwe" coni PFV of [arXiv:2406.13751](https://arxiv.org/abs/2406.13751), finds it again from scratch, and ranks the PFVs found by $W_0$. Needs no CYTools.
- [`examples/h11_11.py`](examples/h11_11.py): builds an $(h^{1,1}, h^{2,1}) = (11, 347)$ geometry with CYTools and scans its conifold.

## Performance

The search enumerates lattice points in a dilated ellipsoid with a Fincke-Pohst search in C (`pfvs/fp_kernel/`) that prunes on every cut and makes every accept/reject decision in exact integer arithmetic. The implementation used in [arXiv:2406.13751](https://arxiv.org/abs/2406.13751) ("dSv1") materialized a bounding box instead, whose size grows as $(Q\cdot p_{denom})^{h^{1,1}/2}$.

- **Kernel vs dSv1** (Manwe, $h^{1,1}=7$, one core): 53× faster at $p_{denom}=1$, 280,000× at $p_{denom}=30$; dSv1 runs out of 30 GB beyond that. Reproduce with `python benchmarks/benchmark_dSv1_vs_coniZpM.py`.
- **End-to-end `coniZpM` vs the previous version of this repo:** 5-28× faster ($h^{1,1}=4$-$11$, $p_{denom}=20$-$400$).
- **Whole [coni-PFV dataset](https://huggingface.co/datasets/natemacfadden/calabi-yau-coni-pfvs)** (18,253 geometries, 8.6M p-vectors, $p_{denom}=150$): about 3 minutes on 23 CPU cores, 15 s on an RTX 5090. It finds all 33,376 dataset PFVs, plus 820 with $\gcd(K_{1:})\ge2$ that the dataset's search excluded.

Behaviour changes relative to dSv1: PFVs are listed in a canonical order (by M, then K) within each p-vector; a p-vector that cannot be searched completely raises `pfvs.IncompleteSearchError` instead of being skipped; `ZpM` uses the C kernel by default; det N ≠ 0 is decided exactly.

## Organization

```
pfvs/
├── pfvs/
│   ├── fp_kernel/         # C kernel* (fp_kernel.h) + C lattice setup (pfv_lattice.h) + Cython binding
│   │   └── cuda/          # GPU backend (CUDA/HIP): pipeline + C API (pfvs_gpu.cu)
│   ├── conipfv_kernel/    # re-exports fp_kernel.conipfv_kernel (coni-PFV enumeration)
│   ├── pfv_kernel/        # re-exports fp_kernel.pfv_kernel (non-coni PFV enumeration)
│   ├── coni.py            # coniZp / coniZpM / coniZpK: coni-PFV search
│   ├── nonconi.py         # ZpM / ZpK: non-coni PFV search
│   ├── cydata.py          # CYData: CY-data holder
│   ├── pfv.py             # PFV class + diagnostics
│   ├── gpu.py             # GPU backend loader (ctypes)
│   ├── distributed.py     # multi-machine coordinator/workers
│   ├── dilation.py        # dilation bound + ZpM/ZpK routing
│   ├── scoring.py         # scoring geometries by PFV count
│   ├── pvectors.py        # p-vector generation
│   └── util.py            # shared helpers (+ njit kernels)
├── tests/                 # pytest suite, exact oracle (oracle.py), fixtures (data/)
│   └── c/                 # standalone C driver for the kernel (profiling/debugging)
├── benchmarks/            # benchmark_dSv1_vs_coniZpM.py (headline), benchmark_conipfv.py
├── examples/              # manwe.py (self-contained), h11_11.py (CYTools)
├── environment.yml
├── pyproject.toml
└── setup.py
```

*: This C code was originally the bottleneck/core of the problem, hence the name 'kernel'. See [`pfvs/fp_kernel/README.md`](pfvs/fp_kernel/README.md) for its exact contract and how it prunes. At high $h^{1,1}$ and $p_{denom}$ it remains the dominant cost.

## Correctness

All accept/reject decisions are exact integer arithmetic; floating point is only used for pruning, widened by a rigorous error bound. The tests check:
- the kernel against an exact rational-arithmetic oracle ([`tests/oracle.py`](tests/oracle.py)), point for point, on 219 dataset instances and adversarial synthetic ones;
- `coniZpM` against the [published dataset](https://huggingface.co/datasets/natemacfadden/calabi-yau-coni-pfvs) on 71 geometries ($h^{1,1}=3$-$11$): exactly its PFVs, plus only valid PFVs with $\gcd(K_{1:})\ge2$;
- that independent routes (numba vs C kernel, Python vs C lattice setup, CPU vs GPU) agree;
- that every named example of [arXiv:2406.13751](https://arxiv.org/abs/2406.13751) and [arXiv:2107.09064](https://arxiv.org/abs/2107.09064) is found (`PFVS_SLOW_TESTS=1` adds full-box scans).

Run `pytest tests/` (add `-n auto` with `pytest-xdist` to parallelize).

## Citing

See [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository"), and cite the papers above that introduced the method.

## License

[GPLv3 or later](LICENSE). Copyright (c) 2026 Nate MacFadden.
