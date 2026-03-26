# pfvs
Tools for computing/verifying perturbatively flat vacua (PFVs). For some references on PFVs, see
- [Vacua with Small Flux Superpotential](https://arxiv.org/abs/1912.10047) by Mehmet Demirtas, Manki Kim, Liam McAllister, Jakob Moritz
- [Small Cosmological Constants in String Theory](https://arxiv.org/abs/1912.10047) by Mehmet Demirtas, Manki Kim, Liam McAllister, Jakob Moritz, Andres Rios-Tascon, and
- [Candidate de Sitter Vacua](https://arxiv.org/abs/2406.13751) by Liam McAllister, Jakob Moritz, Richard Nally, Andreas Schachner,

as well as their references. This repo focuses solely on PFVs as a combinatorial problem abstracted from the physics. In this light, there are also unpublished notes by Mehmet Demirtas/Richard Nally that much of this builds off of. The relevant aspects of such notes will be briefly restated below.

## Problem Statement
The goal will be to construct many (coni-)PFVs for an input Calabi-Yau manifold. The relevant (fixed) data from the CY includes
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

This parameterization is redundant. E.g., with $M$ and $p$ one can compute $K$.

More information than just the CY is required to define a coniPFV. Additionally, one needs a hyperplane $\hat{n}\in H$ corresponding to a conifold-curve (also computable using CYTools). It is canonical to choose a basis for this problem such that $\hat{n} = (1,0,\dots,0)$. This requires representing the other input data in this basis. In this basis a *coniPFV* is a triple $(K,M,p)\in(\mathbb{Z}^{h^{1,1}}, \mathbb{Z}^{h^{1,1}}, \mathbb{Q}^{h^{1,1}})$ satisfying the following constraints
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
We will not contain a detailed algorithm in this README. The algorithm is visible in the code and will be described in detail in an upcoming (as of March 2026) paper. We provide a very brief description here, however. First, as a general strategy, some constraints such as #1 and #2 can be viewed as saying one variable (in this case $M$) lives in a certain lattice. By writing a basis for this lattice, one can thus trivialize these constraints. Whenever such a strategy is possible, it is generally valuable.

There are two general classes of algorithms
1. 'box-style algorithms': enumerate $K$ and $M$ satisfying constraints #1, #2, #5, and #6. This involves rejection sampling for constraint #6. Then, compute $p$ using #7 and then check constraints #3 and #4.
2. 'Zp-style algorithms': enumerate $\hat{p} \in \mathbb{Z}^{h^{1,1}}$ obeying #3. Define $p = \hat{p}/p_{denom}$ for some $p\_{denom} \in \mathbb{Z}\_{>0}$. Use #7 to rewrite constraint #5 as an ellipsoidal constraint either on $M$ (defining the ZpM algorithm) or on $K$ (defining the ZpK) algorithm. The size of this ellipsoid is controlled by $p_{denom}$. This ellipsoid can be made to include all constraints other than #6, which would still be checked via rejection sampling.

The Zp-style is generally more efficient, although one has to appropriately handle $p_{denom}$ for this method to be useful. There are two primary approaches
1. set a max $p_{denom}$ value and run ZpM with the ellipsoid dilated by this amount or
2. (for non-coni PFVs) run ZpM and ZpK for an undilated ellipsoid - this can be shown to be sensitive to any $p_{denom}$.

The choice between the two approaches depends on the maximum $p_{denom}$ that one is interested in: ZpK is fairly expensive but ZpM empirically scales approximately linearly with the max allowed $p_{denom}$. If one is curious about non-coni PFVs with truly large $p_{denom}$, then the first approach is better. Otherwise, the second approach is likely more efficient.

## Generation
See the demo notebooks, specifically the Manwe demo, for a dmeo of coni PFV searches...

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
