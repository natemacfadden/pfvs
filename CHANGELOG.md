# Changelog

## Unreleased (0.1.0)

First packaged release. Relative to the code used for arXiv:2406.13751 ("dSv1"):

### Search
- One exact Fincke-Pohst kernel (C) for coni and non-coni PFVs: every
  accept/reject decision is integer arithmetic, and floating point only prunes,
  widened by rigorous error bounds. Overflow raises; no p-vector is skipped.
- The per-p lattice setup (M-lattice basis, ellipsoid, H-matrix) is C as well:
  row-insertion HNF in int128 with a GMP fallback, incremental-Gram LLL, and a
  cut-aware basis (4-5x smaller searches).
- Automatic handling of non-primitive K-perp; f/h fluxes for non-coni PFVs.

### Behaviour changes
- PFV order: within each p-vector, PFVs are listed in a canonical order (by M,
  then K), independent of the lattice basis, so the CPU and GPU paths return
  identical arrays. The PFVs themselves are unchanged.
- A p-vector that cannot be searched exactly and completely raises
  `pfvs.IncompleteSearchError` (more than `max_N_pfvs` outputs, coordinates
  beyond int32, a non-positive-definite ellipsoid); it used to be skipped or
  truncated silently.
- `ZpM` defaults to the C kernel (`use_c_kernel=True`); `ZpK` uses the exact C
  kernel too, with its rational ellipsoid scaled to an integer one.
- `extra_checks` and `low_level_parallelism` have no effect and emit a
  `FutureWarning` when set.
- PFVs are filtered on det N != 0 exactly: a float SVD only prefilters, and
  every "singular" verdict is confirmed by an exact rank (a float test with
  rtol=1e-12 could drop a valid, badly conditioned N).

### GPUs and clusters
- Optional GPU backend for the batched coni pipeline, one source for NVIDIA
  (CUDA) and AMD (HIP), built with `PFVS_GPU=auto|cuda|hip`. Results are
  identical to the CPU path's; batches size themselves to free device memory and
  recover from out-of-memory. `coniZpM(..., device="auto"|"cpu"|"gpu")`.
- `pfvs.distributed`: a coordinator and CPU/GPU workers over TCP, with
  checkpoints, resume, dynamic splitting and backup copies for stragglers.

### Fixes
- `PFV`'s coni-only attributes no longer leak to non-coni PFVs.
- `PFV.from_str` parses its input as literals instead of executing it.
- The search functions no longer print a line per job at `verbosity=0`.

### Testing and packaging
- Tests against an exact rational oracle, against the published dataset, and
  for the named examples of arXiv:2406.13751 and arXiv:2107.09064.
- The sdist now contains every source needed to build (it lacked the Cython
  source) and the test suite with its data; CI builds it, installs it and runs
  its tests. Wheels contain only the runtime files.
- Examples are scripts in `examples/` (run in CI), replacing the notebooks.
- ruff lint and an API-reference drift check in CI.
- Requires Python >= 3.10 (3.9 never worked: annotations use PEP 604).
