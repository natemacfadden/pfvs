# fp_kernel

A single-header ([stb-style](https://github.com/nothings/stb)) C kernel, `fp_kernel.h`, plus a Cython binding. It runs the Fincke–Pohst lattice-point enumeration behind both PFV pipelines:

- `pfvs.conipfv_kernel.conipfv_kernel` (used by `coniZpM`)
- `pfvs.pfv_kernel.pfv_kernel` (used by `ZpM`)

## Contract

The kernel returns **exactly** the integer vectors $c$ with

| cut | condition | when |
|---|---|---|
| ellipsoid | $q(c) := c^T M c \le \lfloor \text{dilation}\cdot Q \rfloor$ | always |
| M0 | $\ell\cdot c \ge \ell_{\min}$ | coni only |
| GCD | $g = 0$ or $Q g \ge q(c)$ (coni: $Q g > q(c)$), with $g=\gcd(Hc)$ | if $H$ has rows |

The output is listed in depth-first search order: the last coordinate is outermost, and each coordinate is ascending. Alongside it the kernel returns the exact values $q(c)$.

Leaf decisions are exact (`__int128`, GMP on overflow). Floating point only prunes, and every float test is widened by a rigorous error bound, so pruning never discards a valid vector. A coordinate range beyond int32 returns status `-8`. `tests/test_kernels.py` checks the kernel point for point against an exact rational reference, `tests/oracle.py`.

## How it prunes

The search fixes $c$ from its last component to its first. At level $i$:
- **Ellipsoid:** standard Fincke–Pohst on $M = U^TU$.
- **GCD:** a row of $H$ starting in column $j$ is determined once $c_{j:}$ is set, so the gcd $G$ of determined rows bounds $g$ from above. The kernel narrows the $c_i$ interval to $Q\,G \ge q_{\text{lb}}$, rejects with an early-exit gcd, and for $G \le 2^{20}$ enumerates only the residue classes of $c_i$ that can survive.
- **M0:** branches whose largest attainable $\ell\cdot c$ is below $\ell_{\min}$ are cut.

None of these change the output or its order.

## Floating-point safety

$U$ comes from an exact fraction-free (Bareiss) $LDL^T$ of the integer $M$, rounded entrywise, so each coefficient is within a few ulps of its true value however ill-conditioned $M$ is. Each search path carries a bound on the deviation of its float partial norms from the exact ones, $K\sum_i(a_i^2 + \text{rem}_i)$ with $a_i = \sum_{j\ge i}|U_{ij}c_j|$ and $K \approx 2(\dim+8)\,2^{-53}$, and every float test is widened by it.

## Lattice setup (`pfv_lattice.h`) and batching

Per p-vector, `pfv_lattice.h` builds the M-lattice basis `Binter` (Bezout elimination, then LLL with exact integer updates), the ellipsoid $M = -B^T Z B$ with $Z=\kappa\cdot p$, and $H = \mathrm{HNF}(Z B)$ in 128 bits. On overflow it falls back to Python/flint.

The basis is chosen for the cuts: for coni, LLL of $\ker(\ell)$ with respect to $M$ plus one vector with $\ell\cdot w = \gcd(\ell)$, so the M0 cut becomes an interval bound at the first level; for non-coni, LLL of the whole basis with respect to $M$. Same lattice, same PFVs, possibly in a different order.

`_coni_batch` runs setup plus kernel for a chunk of p-vectors in one C call without the GIL, returning $M$, $K$, $q$ and the p-index per point.

The code needs a 64-bit GCC or Clang (≥ 13) target and builds warning-free with `-Wpedantic`.

## Building / testing the C directly

`tests/c/fp_kernel_cli.c` is a standalone driver (no Python), for profiling and debugging:

```
make -C tests/c            # fp_kernel_cli, fp_kernel_cli_stats (-DFPK_STATS), test_factor
tests/c/fp_kernel_cli 3 < problems.txt
```

The input format is at the top of the file; `-DFPK_STATS` also counts rejections by reason and level.
