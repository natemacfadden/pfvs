# Changelog

## Unreleased (0.1.0)

First packaged release. Relative to the code used for arXiv:2406.13751 ("dSv1"):

### Search
- One exact Fincke-Pohst kernel (C) for coni and non-coni PFVs: decisions in
  integer arithmetic, floating point only prunes. Overflow raises; no p-vector
  is skipped.
- Per-p lattice setup in C, with a cut-aware basis (4-5x smaller searches).
- Automatic handling of non-primitive K-perp; f/h fluxes for non-coni PFVs.
- `pfvs.dilation.coni_dilation_bound` / `PFV.dilation_bound`: an exact upper
  bound on the dilation of the coni PFVs of a direction (C, python-flint
  fallback).
- `coniZpK`: the coni PFVs of each direction above a dilation D0.
- `coniZpM(..., exhaustive=True)`: every coni PFV of each direction, routing
  each p-vector to ZpM or ZpM + ZpK by measured cost
  (`pfvs.dilation.bound_routing`).

### Behaviour changes
- Within each p-vector, PFVs are in a canonical order (by M, then K), so CPU
  and GPU return identical arrays.
- A p-vector that cannot be searched completely raises
  `pfvs.IncompleteSearchError` instead of being skipped or truncated.
- `ZpM` defaults to the C kernel; `ZpK` uses it too.
- `extra_checks` and `low_level_parallelism` have no effect and warn.
- det N != 0 is decided exactly (a float test could drop valid PFVs).

### GPUs and clusters
- Optional GPU backend for `coniZpM` (CUDA or HIP, `PFVS_GPU`), identical to
  the CPU path; `device="auto"|"cpu"|"gpu"`.
- `pfvs.distributed`: a coordinator and CPU/GPU workers over TCP, with
  checkpoints and resume; also runs exhaustive jobs.

### Fixes
- `PFV`'s coni-only attributes no longer leak to non-coni PFVs.
- `PFV.from_str` parses literals instead of executing its input.
- No per-job printing at `verbosity=0`.

### Testing and packaging
- Tests against an exact oracle, the published dataset, and the papers'
  named examples.
- A complete sdist, tested in CI; examples as scripts; ruff and API-doc
  checks in CI.
- Requires Python >= 3.10.
