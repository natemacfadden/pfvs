# fp_kernel

A single-header ([stb-style](https://github.com/nothings/stb)) C kernel, `fp_kernel.h`, plus a Cython binding. It runs the Fincke–Pohst lattice-point enumeration behind both PFV pipelines:

- `pfvs.conipfv_kernel.conipfv_kernel` (used by `coniZpM`)
- `pfvs.pfv_kernel.pfv_kernel` (used by `ZpM`)

Both entry points keep their original signatures. They used to be two near-identical kernels; they are now one.

## Contract

The kernel returns **exactly** the integer vectors $c$ with

| cut | condition | when |
|---|---|---|
| ellipsoid | $q(c) := c^T M c \le \lfloor \text{dilation}\cdot Q \rfloor$ | always |
| M0 | $\ell\cdot c \ge \ell_{\min}$ | coni only |
| GCD | $g = 0$ or $Q g \ge q(c)$ (coni: $Q g > q(c)$), with $g=\gcd(Hc)$ | if $H$ has rows |

The output is listed in depth-first search order: the last coordinate is outermost, and each coordinate is ascending. Alongside it the kernel returns the exact values $q(c)$.

Every accept/reject decision at a leaf is made in exact integer arithmetic: `__int128`, with GMP when that overflows. Floating point is only used to *prune* the search, and every floating-point test is widened by a rigorous error bound (see below). So pruning never discards a valid vector. If a coordinate range does not fit in int32, the kernel returns status `-8` rather than a partial answer.

`tests/oracle.py` is an independent exact reference: a rational-arithmetic Fincke–Pohst. `tests/test_kernels.py` checks the kernel against it point for point, in order, on real instances from the [coni-PFV dataset](https://huggingface.co/datasets/natemacfadden/calabi-yau-coni-pfvs) and on adversarial synthetic ones: non-echelon $H$, rank-deficient $H$, $H$ entries beyond $2^{100}$, boundary points, and ill-conditioned ellipsoids with entries up to $2^{50}$.

## How it prunes

The search fixes $c$ from its last component to its first. At level $i$ ($c_{i+1:}$ fixed, $c_i$ being chosen):

- **Ellipsoid.** Standard Fincke–Pohst on $M = U^TU$. The remaining budget bounds $c_i$ to an interval.
- **GCD.** A row of $H$ whose first nonzero entry is in column $j$ is fully determined once $c_{j:}$ is set. So the gcd $G$ of the determined rows never increases down the tree, and it bounds the final $g$ from above. The partial norm is a nondecreasing lower bound on $q(c)$. This allows three kinds of pruning:
  - the $c_i$ interval narrows to where $Q\,G \ge q_{\text{lb}}$;
  - candidates are rejected with an early-exit binary gcd, which stops as soon as $g$ provably falls short;
  - when $G$ is small ($\le 2^{20}$), the only possible survivors are generated directly. For each divisor $d \ge$ need of $G$, the $c_i$ with $d \mid \text{pre} + h c_i$ form one residue class, so the kernel enumerates those classes instead of every $c_i$.
- **M0.** The largest $\ell\cdot c$ attainable in the rest of the ellipsoid is $\ell\cdot c_{\text{fixed}} - V_i\cdot c_{i+1:} + \sqrt{N_i\,\text{rem}}$, with $N_i, V_i$ precomputed exactly. Branches that cannot reach $\ell_{\min}$ are cut. Where M0 becomes fully determined, the cut is an exact integer bound on $c_i$.

None of these change the output or its order. They only avoid visiting candidates that would be rejected anyway.

## Floating-point safety

$U$ is computed from the exact integer $M$ by fraction-free (Bareiss) $LDL^T$ elimination in GMP, then rounded entrywise. Each float coefficient is therefore within a few ulps of its *true* value, however ill-conditioned $M$ is. A float Cholesky only guarantees a small backward error, and its forward error is amplified by the condition number, which is exactly what goes wrong for partial sums.

Along each search path the kernel carries the bound $K\sum_i(a_i^2 + \text{rem}_i)$ with $a_i = \sum_{j\ge i}|U_{ij}c_j|$ and $K \approx 2(\dim+8)\,2^{-53}$. This bounds the deviation of every float partial norm from the exact one. Every float test is widened by it: interval bounds, ellipsoid rejection, $q_{\text{lb}}$ in the GCD cut, and the M0 bound. For well-scaled inputs the bound is negligible. It matters for ellipsoids with huge entries but small $q$ (heavy cancellation), where a fixed epsilon silently loses points.

## Lattice setup (`pfv_lattice.h`) and batching

For each p-vector, `coniZpM`/`ZpM` need a lattice basis `Binter` for the M-vectors, the ellipsoid $M = -B^T Z B$ with $Z=\kappa\cdot p$, and $H = \mathrm{HNF}(Z B)$. `pfv_lattice.h` builds them in C:

1. An orthogonal lattice, by Bezout elimination.
2. LLL: floating-point Gram–Schmidt chooses the operations, but the basis updates themselves are exact, overflow-checked integer operations. So the lattice is always preserved exactly.
3. A 128-bit HNF.

On any overflow the p-vector falls back to the exact Python/flint path. If only the HNF overflows, only the HNF is computed in Python.

**Cut-aware basis.** The kernel's cuts depend on the basis it searches in, so the basis is chosen for them.

- **Coni.** Take $T = [K\,|\,w]$ with:
  - $K$ an LLL basis of $\ker(\ell)$, where $\ell$ is the M0 row. $K$ is reduced *with respect to the ellipsoid form* $M$, not the Euclidean norm.
  - $w$ a vector with $\ell\cdot w = \gcd(\ell)$, size-reduced against $K$.

  Then $\ell = (0,\dots,0,g)$, so $M_0$ is fixed at the first level searched, where $M_0 \ge M_{0,\min}$ becomes an interval bound.
- **Non-coni.** $T$ is the LLL reduction of the whole basis with respect to $M$.

It is the same lattice either way, so the same PFVs are found, possibly in a different order. On 340 heavy $h^{1,1}=10$, $p_{denom}=150$ problems this made searches 4.7× smaller: the median per problem was 3.6×, and no problem got slower.

**Batching.** `_coni_batch` runs lattice setup plus kernel for a whole chunk of p-vectors in one C loop, without the GIL. For each lattice point $c$ it returns only what the post-processing needs: $M = B c$, $K_{nat} = Z B c$, $q$ and the p-index. All of these are exact and checked. $H$ goes to the kernel as int64 (`fpk_problem.H64`), which avoids GMP. The kernel's exact factorization uses an int128 fraction-free (Bareiss) fast path, with GMP as the fallback.

**Portability.** Checked 128-bit multiplies are written out explicitly rather than with `__builtin_mul_overflow`, which older Clang lowers to a call libgcc lacks. The code builds warning-free with `-Wpedantic` on GCC and on Clang ≥ 13. It needs a 64-bit GCC/Clang target, which covers Linux and macOS.

## Building / testing the C directly

`tests/c/fp_kernel_cli.c` is a standalone driver (no Python), for profiling and debugging:

```
make -C tests/c            # fp_kernel_cli, fp_kernel_cli_stats (-DFPK_STATS), test_factor
tests/c/fp_kernel_cli 3 < problems.txt
```

The input format is documented at the top of the file. The `-DFPK_STATS` build also reports rejected candidates by reason and level.
