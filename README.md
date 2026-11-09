# diffmrc

An end-to-end differentiable multireference electronic structure stack, spanning
configuration interaction (CASCI), complete active space self-consistent field
(CASSCF), state-averaged CASSCF, and second-order N-electron valence perturbation
theory (NEVPT2). Energies, nuclear gradients, and second derivatives are obtained
from a single code path by automatic differentiation; no hand-derived derivative
equation appears at any layer.

Built on [PySCF](https://pyscf.org/) and [pyscfad](https://github.com/fishjojo/pyscfad),
with [JAX](https://github.com/jax-ml/jax) providing the differentiable array operations.
All arithmetic is double precision.

> **Project status: early development.** This is a research codebase, released
> to accompany a methods paper and to make the results reproducible. It is not
> a finished library. Interfaces may change without notice, the test coverage is
> partial, and the numerical behaviour of the orbital-optimization and
> perturbation layers depends on convergence settings that are described in the
> Open problems section rather than enforced by the code. Several of the open
> problems listed there are properties of the method family and are not
> expected to be resolved by further development of this package. Use it for
> research, check the convergence diagnostics it reports, and do not rely on it
> as a black box.

---

## What is here

```
src/diffmrc/    the package
  core/             numerical kernel: differentiable CI, CASSCF, gradients, NEVPT2
  analysis/         wavefunction analysis: entropies, diagnostics, composition

tests/          unit and integration tests
experiments/    scripts reproducing the results in the paper
examples/       small runnable examples for the public interfaces
manual/         Tutorial and Manual (LaTeX sources and built PDFs)
```

## What is new in 0.3.x

- CI spin-sector projection: the NEVPT2 core classes project the determinant
  CI vector onto the alpha-beta layout of one M_S sector. An odd
  active-electron count has two exactly degenerate lowest sectors and the
  blocked solver may return either, so the fixed projection could be empty and
  the renormalization 0/0 (measured on OH/6-31G CAS(5,4): NaN for the Sr and
  Si classes from the partner sector). `impl.ci_ab_from_flat` now selects the
  occupied sector and transposes the coefficient matrix when the partner
  carries the vector; both sectors reproduce the eight class energies to
  6e-16 and the base value is unchanged.
- CASPT2 TUVA/ITUV classes (0.3.2): `Vm1_a` and `V1_i` are three-active-index
  functions with an orthonormal class basis; N2/6-31G CAS(6,6) default ratio
  reaches 0.983 against ORCA CASPT2-D for the release reference
  (OMP_NUM_THREADS=1; the Wave 12 frozen reference gives 0.989). The remaining
  class-space residual is the frozen-core setting: ORCA freezes the two N 1s
  orbitals, and `frozen_core=2` with the exact class-block solve reproduces
  the ORCA class table (see Open problems).
- `core.caspt2_three_center` (0.3.1): three-center contracted CASPT2 using the
  same `cross - direct` VM convention as the materialized path. The release
  test compares the two paths on deterministic exact factor tensors and
  reproduces the materialized result to machine precision; a H2O/CO
  cross-check script is not included in the release tree.
- Validation drivers (0.3.1): N2/6-31G CAS(6,6) CASPT2, ECP core-Hamiltonian
  gradient test, and an H2O/cc-pVDZ NEVPT2 gradient basis-set check.
- `driver.nevpt2_zvector`: relaxed NEVPT2 nuclear gradients. The near-zero
  `Si` channels are evaluated in exact algebraic form, displaced orbital
  references are aligned blockwise, and the response term is supplied by the
  CASSCF Z-vector. Validated against finite differences on H2O/6-31G; the
  LiH/6-31G anchor is covered by the algebraic and chain-level tests rather
  than by a finite-difference driver in the release tree.
- `core.caspt2`: a coupled full-Fock contraction that retains the inter-class
  coupling of the zeroth-order Hamiltonian, next to the differentiable
  diagonal-Fock path. The relative weight of the `VM` component is corrected
  to the convention used by the reference implementation.
- `core.somf_ecp`: a valence-level spin-orbit mean-field operator for
  effective-core-potential calculations, with gradient prototypes.
- `driver.ri_scf` and `driver.ri_scf_driver`: a RIJONX SCF loop
  (RI-J Coulomb plus exact exchange) on top of pyscfad RHF/ROHF, including the
  SCF-level overlap-derivative term.
- `driver.scf_ad`: implicit-differentiation ROHF references for f-block
  molecules. pyscfad's default unrolled SCF AD freezes the DIIS
  coefficients and is not the derivative of the converged solution
  (YbF: 3.42 times the analytic gradient magnitude); this module
  differentiates the fixed point (3.0e-7 against the analytic ROHF
  gradient, 1.7e-6 against central differences) and selects
  multi-solution branches from a fixed energy-ordered ladder. The route
  is opt-in; existing drivers are unchanged.
- `core.x2c_casscf_df`: scalar (spin-free) DF-X2C CASSCF energies, analytic
  nuclear gradients, and state-averaged response.

## Acceleration layer

`diffmrc.core.ri` adds a differentiable resolution-of-identity (RI/DF)
layer. It supplies three-center factors, the auxiliary metric, the
symmetric inverse square root with an analytic derivative, and an
explicit JVP for the four-center tensor. The nuclear-derivative working
set scales as `nao^2 * naux` per atom instead of `nctr * nao^4`; the
measured peak memory of the RI integral-derivative chain is 2.11 GB at
`nao = 96` and 16.05 GB at `nao = 175`, against 55.0 GB and 2033 GB for
the dense single-array path. The RI approximation is an approximation:
its physical error is documented in Manual Chapter 12, and the SCF layer
of the present chain still uses dense integral derivatives.

The performance choices in the differentiable chain are documented in
Manual Chapter 13: the four-step MO transform, the S_z-blocked CI
solver and its `auto` selection rule, the array-argument jit form, and
the CI matvec contraction. The Tutorial chapters 10 and 11 walk through
the RI and performance examples in `examples/`.

## Layers

| Layer | Contents | Status |
|---|---|---|
| `diffmrc.core` | Differentiable CI, CASSCF, gradients, NEVPT2 | Frozen, validated against reference values |
| `diffmrc.analysis` | Entropies, plateau detection, diagnostics, composition | Pure functions over numpy arrays |

`diffmrc.core` is frozen: its numerics are validated and must not move. The
migration into `src/diffmrc/core` changed import paths and removed the
demonstration blocks only; no algorithm, constant, or convergence threshold was
touched. `tests/core/test_frozen_reference.py` pins an energy from the
manuscript so that a future edit to the kernel fails loudly.

`diffmrc.analysis` is independent of the kernel by contract -- it may not
import `core` or any upper layer. Run `lint-imports` to verify the boundary
holds. See `docs/analysis_layer.md`.

## Requirements

```
python >= 3.10
numpy, scipy              # required
pyscfad >= 0.3.3          # provides jax-based gto, scf
pyscf >= 2.5              # for reference comparisons
jax >= 0.4                # jax_enable_x64 must be True
matplotlib                # for the figure sources
```

The analysis layer needs only numpy; the numerical kernel needs the rest.
Install into a dedicated environment, since `pyscfad` pins a specific JAX
version.

```bash
conda create -n diffmrc python=3.11
conda activate diffmrc
pip install -e ".[core,dev]"          # kernel plus test and lint tooling
pip install matplotlib                # only for the figures
```

For the analysis layer alone, `pip install -e ".[dev]"` is enough.

```bash
pytest tests/ -v        # unit and integration tests
lint-imports            # verifies the layer boundary
```

## Quick start

Compute a CASSCF nuclear gradient for water.

```python
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
          'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS'):
    os.environ[v] = '8'

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
from jax import numpy as jnp
from pyscfad import gto as pgto, scf as pscf
from diffmrc.core.rotation import block_rotation
from diffmrc.core.orblib import orbital_pairs
from diffmrc.core.optimizer import optimize_theta
from diffmrc.core.grad import (make_energy_fn, make_grad_fn,
                               solve_theta_star, nuclear_gradient)

mol = pgto.Mole(); mol.atom = 'O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880'
mol.basis = '6-31g'; mol.build()
nmo, ncore, nact, nele = mol.nao_nr(), 4, 2, 2
pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)

E_cas = make_energy_fn(ncore, nact, nele)
mf = pscf.RHF(mol); mf.kernel()
C_rhf = jnp.asarray(np.asarray(mf.mo_coeff))
h1e = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
eri = mol.intor('int2e', aosym='s1')

E_theta = lambda t: E_cas(C_rhf @ block_rotation(t, nmo, pairs), h1e, eri,
                          mol.energy_nuc())
th_star, gnorm = solve_theta_star(mol, C_rhf, E_theta, len(pairs),
                                  optimize_theta, steps=2000, g_conv=1e-12)
g = nuclear_gradient(mol, make_grad_fn(nmo, pairs, th_star, E_cas, pscf))
print(g)          # (natm, 3) in Eh/bohr
```

## The differentiable layers

**Configuration interaction.** The CI matrix is diagonalized in full, so its
eigenvalues and eigenvectors are differentiable array functions of the matrix
elements; the Hellmann–Feynman relation supplies the derivative and the
eigenvector is differentiated through the same operation. Because the
Hamiltonian is block-diagonal in the spin projection $S_z$, the diagonalization
is performed blockwise (`cs_casci_blocked`); a single `eigh` over the full
determinant space encounters degeneracies across blocks and returns NaN in
reverse mode. Block separation is necessary but not sufficient: the reverse
rule forms 1/(w_j - w_i) for every pair in a block whether or not that pair
carries cotangent, so one bit-exact degenerate pair anywhere in any block
turns the untouched entries into 0 × inf = NaN. Measured on CO/6-31G
CAS(6,6), five of the seven blocks carry bit-exact pairs (24 in the
400-determinant winning block, whose own ground state is simple) and the
raw-eigh theta gradient is all-NaN while the energy is finite. The block
eigenproblem therefore goes through `core.eigh.eigh_safe`, whose
Lorentzian-softened rule freezes the rotation inside a degenerate subspace;
that rule is first-order safe by direct measurement. Second derivatives of
any decomposition-based path remain NaN on such systems. The remedy is
`core.lanczos.lowest_eigpair_lanczos`, selected with `eig_solver='lanczos'`:
a reorthogonalized Lanczos forward, polished by Newton steps and then
stopped, whose derivatives come from one bordered implicit-function
(Newton) step on the eigenequation — nothing decomposes on the
differentiated path. Measured on the same CO winning block (400
determinants, 24 bit-exact pairs): the AD second derivative agrees with
gradient finite differences and with value-based second differences to
1.4 × 10⁻⁶, independent of the Krylov depth from 24 to 160, where every
decomposition kernel returns NaN. The design measurements — in
particular why the forward runs under `stop_gradient` (the Lanczos
basis's sensitivity to the matrix grows like the product of 1/beta) —
are recorded in the module docstring.

**Orbital relaxation.** Orbitals are parametrized by an exponential map of
antisymmetric rotation parameters, $C = C_0 R(\theta)$, so the optimization is a
continuous function of a fixed-size parameter vector and the envelope theorem
applies to the first derivative. The parameter set must include core–virtual
rotations; a set restricted to core–active and active–virtual rotations leaves
the optimization at a non-stationary point.

**State averaging.** A differentiable weighted average over a multi-state
diagonalization.

**Perturbation correction.** NEVPT2 assembled by differentiable contraction of
the fourth-order reduced density matrix, with all core, semi-core, and cross
terms.

Its derivatives are validated in the same reference-free way as the CASSCF
layer (2026-09 measurements, `driver/nevpt2.py`): the strongly contracted
energy agrees with an independent implementation to 3 × 10⁻¹⁰; the nuclear
gradient is translationally and rotationally invariant at the 10⁻¹⁵ level;
with the correlation term scaled to zero it reproduces the validated CASSCF
gradient to 8 × 10⁻¹⁷; and its Hessian-vector products match a
finite-difference ladder to 6 × 10⁻⁷ at the finest step, with the Hessian
symmetric to 10⁻¹¹. Second derivatives of this chain go through the blocked
CI eigensolver (`cs_casci_blocked`) — the plain eigensolver's second
derivative is NaN at a cross-`S_z` degeneracy, which its own docstring
records — and on systems whose block spectra carry bit-exact degenerate
pairs (linear molecules) through the same solver with
`eig_solver='lanczos'`; the end-to-end check on CO/6-31G measured the
theta gradient finite at 1.705953 × 10⁻¹ and the HVP finite, where every
decomposition-based path returns NaN.

At actinide scale (ThO and UF with a Stuttgart ECP and 6-31G ligands, nao
= 96) the correlation energy cross-checks against an independent
implementation to 2.5 × 10⁻⁶ (ThO) and 2.9 × 10⁻⁸ (UF), and the theta
gradients are finite. Those runs carry a ~10⁻⁶ reproducibility band, and
their theta-space derivatives a ~8% degenerate-branch uncertainty: the
linear molecules' generalized-Fock blocks contain machine-precision
degeneracies whose inverse gaps amplify every numerical path choice, so
the numerical route (contraction order, eigensolver rule) must be tested
on such a system rather than assumed equivalent. The nuclear-derivative
layer of the actinide protocol is blocked upstream: pyscfad's
integral-derivative path allocates a dense (nctr, nao⁴) array — 51 GiB at
nao = 96 — before any chemistry happens.

## Second derivatives

The orbital-response term enters the second derivative even though it cancels
from the first. Two independent implementations are provided and agree to
1.6 × 10⁻¹¹ where both are carried to convergence.

*Unrolled iteration.* Evaluate $N$ Newton steps inside the computational graph
and back-propagate through them. The self-consistent-field step that supplies
the reference orbitals must be evaluated outside the loop (it cannot be traced);
the orbital Hessian requires regularization $\lambda I$; and the iteration
should start from the converged parameters at the reference geometry.

*Implicit differentiation.* Apply the implicit-function theorem to the
stationarity condition $g(\theta^*, R) = 0$:

```
dθ*/dR  = −H_θθ⁻¹ · ∂²E/∂θ∂R
d²E/dR² = ∂²E/∂R²|_θ + (∂²E/∂R∂θ) · (dθ*/dR)
```

The orbital Hessian is not positive definite and can be near-singular — a
redundant active-orbital rotation contributes a near-null direction whose
gradient component vanishes identically. **Use a pseudo-inverse, not a direct
solve.**

The implicit route is the faster of the two and scales better in the number of
rotation parameters; the unrolled route serves as its independent check.

## Verification

Two checks used throughout require no external reference and are exact rather
than threshold-based:

*Invariance.* Translational invariance requires $\sum_a \partial E/\partial R_a = 0$;
rotational invariance requires $\sum_a R_a \times \partial E/\partial R_a = 0$.
Both hold to $10^{-14}$ or better for the systems reported here.

*Variational ordering.* Because the active space contains the starting
determinant, the orbital-optimized energy cannot lie above the energy of the
orbitals it started from. A violation means the two calculations described
different electronic states.

Agreement with a finite difference of the same code path establishes internal
consistency only: automatic differentiation propagates a numerically correct
derivative of an incorrectly implemented energy. An independent reference is
required, and where that reference is itself an iterative calculation its
convergence must be checked at every point — an unconverged reference enters a
central difference as an intermediate value, not as a stationary point.

An independent reference for the CASSCF-layer nuclear Hessian now exists in the
project's record: on CO/6-31G, a construction that uses plain PySCF 2.14 and
none of this package's derivative machinery — analytic CASSCF gradients
central-differenced in Cartesian coordinates, with the orbitals re-converged at
every displaced geometry — reproduces the base energy to 1.1e-8 Eh, the relaxed
bond curvature to 3.1e-5 through the implicit route and to 1.8e-3 through the
value route at a 3e-3 bohr step, and the relaxed wavenumber to 0.09%. That
construction also carries its own branch certificate (the displaced energies'
central difference reproduces the analytic base gradient to 1.0e-5 relative,
while warm-started solutions off the continuous branch fail that test).

## Open problems

The entries below are open problems, not permanent exclusions. Each states the
current status and, where known, the direction of work.

*Active-space size.* Full diagonalization in the CI step limits the active-space
size. `cs_casci_blocked` now takes `eig_solver='davidson'`, which applies the
matrix-free operator of `core.ci_matvec` through the Davidson kernel and uses
the implicit-function reverse of `core.iterative_ci`; the determinant space is
then limited by working memory rather than by the $O(\mathrm{dim}^2)$ block
matrix. On the local Windows replica the memory crossover lies between
CAS(6,6) (dense 0.50 GiB, matrix-free 1.66 GiB) and CAS(8,8) (dense exceeds
8 GiB, matrix-free 2.18 GiB); CAS(10,10) runs on the matrix-free route with
184,756 determinants at 2.0 GiB peak and 552 s per energy. The first-order
derivative is validated against finite differences through a rotated-MO chain
(relative 1.4e-9 at CAS(6,6), 3.2e-9 at CAS(8,8)). `core.ci_second_deriv`
carries the matrix-free eigenpair second derivative as a prototype:
`mf_eigenpair_hessian` reproduces the dense AD Hessian to 1.07e-10 at CAS(6,6)
and the finite-difference of the first derivative to 3.3e-8 / 2.4e-7 at
CAS(6,6) / CAS(8,8). It is not wired into `hessian.py` yet: the four blocks
there sit on the `custom_vjp` path, where forward-over-reverse is silently
wrong and reverse-over-reverse raises inside the MINRES `while_loop`, so they
need explicit response rules, and the adjoint response solve is not
implemented. A definition that carries both the response and the integral
curvature term is required -- the response term alone has the wrong sign (the
CAS(6,6) (0,0) element is -3.12 without it against +63.29 with it). (Branch
`wave14/eigensolver`, commit `99d0cfb`.)

*Degenerate pairs at a stationary point.* The cluster tolerance of the
eigensolver must sit inside the measured gap between the floating-point noise
that splits symmetry-degenerate pairs at a rotated stationary point (1e-8 to
1.2e-5 in the systems measured here) and the physical gaps of the same spectrum
(1.9e-2 and above). A tolerance below the noise scale exposes both the first
and the second derivative to a bare inverse gap; one above the physical gaps
clusters distinct states. The window measured here spans 1e-5 to 1e-2, and the
rule has no admissible value when a physical gap falls inside the noise band.
`core.eigh.cluster_tolerance` and `lowest_pair_auto` now derive the tolerance
from the spectrum: a ratio test separates the noise floor from the smallest
physical gap, the tolerance is their geometric mean, and a spectrum without the
required separation returns the decomposition-free Lanczos kernel explicitly.
On the recorded CO spectra the rule returns 1.26e-3 for the 1-RDM block and
7.55e-4 for the Fock virtual block, and 4.69e-4 when the measured noise scale
is supplied. The rule is a decision helper, not a differentiable function, and
an a priori guarantee that the two scales separate remains open. (Branch
`wave14/eigensolver`, commit `99d0cfb`.)

*Stationary-point selection.* The CASSCF energy is non-convex in the orbital
parameters; different starting points can reach different stationary points,
and the selection is not stable even at fixed settings. Current practice is to
report a stationarity certificate and the stationary-point spread alongside any
multireference number. `core.stationary.select_stationary_point` makes the
selection deterministic: it keeps the candidates that reached the gradient
target, prefers certified minima, orders by energy, gradient norm, label and
the canonical form of the parameters, and reports `minimum_degenerate`,
`saddle_only` or `curvature_unverified` rather than calling a saddle a minimum.
Generating the starting points and running the refinements remain the caller's
job; a search strategy that guarantees the global minimum remains open. (Branch
`wave14/eigensolver`, commit `99d0cfb`.)

*Perturbation-layer convergence.* NEVPT2 depends on the orbitals more
sensitively than the CASSCF energy does, so energy-based convergence criteria
are insufficient when the target quantity is the correction. Measured on
H2O/6-31G CAS(2,2): between optimizations converged to 1e-4 and 1e-6 the
CASSCF energy moves by 1.3e-9 Eh while the correction moves by 1.1e-5 Eh
(ratio about 8300); the correction stabilizes to 1.9e-10 once the orbital
gradient reaches 5e-7, the practical floor of the L-BFGS-B optimizer. Same
system, independent starting points: the zero start lands on a theta-space
saddle (Hessian minimum eigenvalue -1.6e-2) and a 21.2 mEh higher stationary
point, whose correction differs by 18.7 mEh. The current practice is therefore
a branch certificate: three or more independent starts that agree in energy to
1e-8 with a positive-definite theta Hessian, followed on the selected branch
by an orbital-gradient threshold together with a bound on the change of the
correction between two independently converged orbital sets. The first-order
bound `|H^-1 s| |g|` is recorded as a diagnostic only: on the saddle it has the
wrong sign. A criterion that removes the branch dependence remains open.
(Branch `wave14/caspt2-nevpt2`, commit `ea26401`.)

*Second-derivative cost.* The vectorized implicit route requires the orbital
Hessian in memory, and its cost is set by the tangent broadcast rather than by
the matrix itself; the unrolled route scales with the number of iterations.
Blocked and matrix-free implicit routes are now available. The blocked route
takes one tangent per column and reproduces the vectorized Hessian to
$5\times10^{-15}$ on H2O/6-31G and $3\times10^{-15}$ on CO/6-31G while using
32% and 35% less peak memory; the matrix-free route solves the response
equation with LSMR, never forms $H_{\theta\theta}$, $M$ or $H_{RR}$, and
reproduces the same Hessian to $6\times10^{-12}$ and $8\times10^{-12}$ while
using 34% and 40% less memory. The unrolled route with three Newton steps
matches the implicit route to $6\times10^{-14}$ on LiH/STO-3G
($n_\mathrm{par}=11$, 3.05 GiB) but exceeds 20 GiB on H2O/6-31G CAS(2,2)
($n_\mathrm{par}=50$) and did not produce a Hessian.

The matrix-free response solve no longer carries an iteration cap. It stops on
the relative residual $\lVert r\rVert/\lVert b\rVert$ (or on the
normal-equation criterion when the right-hand side has a null-space
component), with a wall-clock guard and a stagnation guard; storage is six
vectors of length $n_\mathrm{par}$ regardless of the iteration count, so
memory is bounded by construction. `core.lsmr.near_null_basis` builds a basis
of the smallest-$|\lambda|$ orbital directions matrix-free -- diagonal-seeded,
grown by coordinate expansion, Rayleigh-Ritz at each round -- and certifies it
against $\lVert H v-\theta v\rVert/\lambda_\mathrm{max}$ before it is used;
`core.lsmr.deflated_solve` inverts that block with the same relative cutoff
the dense pseudo-inverse applies and hands the complement to LSMR, with
residual-correction rounds and a fallback to the plain solve when the
deflated route has not converged. On the local Windows replica (pyscfad is
unavailable there, so $M$ and $H_{RR}$ are central differences shared by both
routes while $H_{\theta\theta}$ is exact AD) the deflated solve reproduces the
dense route to $8\times10^{-17}$ on H2O/6-31G in 901 LSMR iterations plus a
256-product basis, against $1.2\times10^{-10}$ in 2192 iterations without
deflation, and to $2\times10^{-15}$ on CO/6-31G in 1720 iterations plus 774
products, against $1.5\times10^{-11}$ in 3439 iterations. The accuracy gain,
not only the iteration count, is the point: with five constructed near-null
directions at $10^{-9}$ to $10^{-7}$ of $\lambda_\mathrm{max}$ the plain solve
satisfies its residual test and is still wrong by $6\times10^{-4}$ relative,
while the deflated solve reaches $1.5\times10^{-12}$. Analytic second
derivatives for the single-reference RIJONX surface remain open. (Branches
`wave14/ri-hessian`, commit `5024db7`, and the second-derivative cost work,
2026-11-09.)

*Relaxed NEVPT2 gradient scope.* The relaxed-path nuclear gradient is validated
on two closed-shell, frozen-core systems (H2O/6-31G and LiH/6-31G) and on two
scalar-relativistic ECP systems (HI/def2-SVP with ncore=12 and HI/LANL2DZ with
ncore=3; the ECP core-Hamiltonian derivative test now runs in the suite). For
the open-shell doublet OH/6-31G CAS(3,2) the CASSCF stationary point coincides
with the ROHF orbitals at every geometry, so the response term vanishes
identically and only the frozen gradient is testable. The non-trivial
open-shell case, OH/6-31G CAS(5,4) with ncore=2 at theta* = -75.387977003714
Eh, carries an 18.5% discrepancy in the frozen e2 nuclear derivative along the
O-H stretch (release AD +1.9137e-3 Eh/bohr against +2.3471e-3; the reported
19.4% used a 3e-4 finite difference of +2.3746e-3, which the M_S sector fix
shows to be 1.2% high -- the clean 3e-4 value is +2.3472183e-3, and the 1e-3
and 5e-3 values extrapolate to the same limit). The contraction is not
responsible: contracting the reverse-mode partials
d e2/d(C, h1, eri) with finite-difference tangents of the same inputs
reproduces the finite difference to 7.3e-7 on 2.35e-3 (3.1e-4 relative), and
each of the eight classes agrees with its own finite difference to 1e-9..7e-7.
The derivative is a cancellation of three channels, C -2.0011e-2,
h -4.5755e-2 and eri +6.8113e-2, summed to +2.3471e-3. Splitting the orbital
tangent by the overlap metric into a rotation and a symmetric part gives the
discriminating number: the CASSCF energy's rotational contraction is 2.5e-14
-- it is stationary under every orbital rotation, so its 6.6e-6 closure
validates only the symmetric part -- while e2's is +4.176e-3, 1.8 times the
total. The cosine between the two orbital gradients is 0.093. A 10.4% error in
the rotational response accounts for the whole discrepancy, and the release
builder differentiates pyscfad's ROHF through the default unrolled iteration,
the failure mode documented in driver/scf_ad.py.
The contract (enable ``config.scf_implicit_diff`` for open-shell references) is
now recorded in ``make_nevpt2_joint_fn`` and
``nevpt2_relaxed_nuclear_gradient``; the implicit-differentiation rerun that
would close the number runs on the server and remains open. f-block and
large-basis relaxed gradients are memory-bound,
measured on LrF at 109 spherical basis functions with a 78-core Stuttgart RLC
ECP (CAS(8,6), 13 frozen core orbitals, one thread): the pinned NEVPT2 energy
forward peaks at 12.1 GB and the displaced-geometry SCF plus NEVPT2 chain at
16.5 GB, while the reverse pass of `s = d e2/dtheta` is oom-killed by the
kernel at the 60 GB cgroup limit, with a measured peak of 60 728 MB after
1 h 55 min (the same pass is oom-killed at 20 GB, and a 55 GB proactive
threshold stops it at 56.5 GB). The dense `nctr * nao^4` integral-derivative
allocation is a separate upstream limit; the `core.adint` custom-VJP backend
reduces the bare integral-derivative peak from 9.6-19 GB to 4.2-7.8 GB at
nao = 104-124 and is the natural next step for this chain. The exact
near-zero treatment applies to the `Si` class only; the other four contraction
classes retain the release value mask, and no validation system with a masked
channel outside `Si` has been found. (Branch `wave14/caspt2-nevpt2`, commit
`ea26401`.)

*CASPT2 residual.* The coupled full-Fock path is a NumPy value-layer
implementation. The differentiable path is the diagonal-Fock approximation
(CASPT2-D level). The default path's `(u,u)` active-row truncation is exact
only for a two-orbital active space: the plain-product class projection
reproduces it to 4e-16 on the CAS(2,2) anchors, while on N2 removing it without
projection doubles the `Vm2_ab` and `V2_ij` classes. The
`class_diagonal='block', v0_ia='class'` option replaces the truncation and the
per-block `V0_ia` solve with exact class projections; on N2/6-31G CAS(6,6) it
moves `Vm2_ab` to 1.0000 of ORCA CASPT2-D and `V0_ia` to 0.9970, and the total
to 1.0232 of the diagonal reference and 1.0023 of the full-FIC column. The
release default-path total is 0.983 against ORCA CASPT2-D (OMP_NUM_THREADS=1);
the Wave 12 frozen reference gives 0.989, and the spread across reference sets
(0.980–0.989) comes from the active degenerate-pair gauge, while the five
classes without active-pair rows are stable to 1e-7 across reference sets. The
default total's
apparent agreement is a cancellation between `V1_ija` (+55%), `V2_ij` (-77%),
`V0_ia` (+10%) and `Vm2_ab` (+12%). The diagonal-to-full coupling ratio of the
two paths, 0.9795831, matches the ORCA CASPT2-D/full-FIC ratio 0.9795829 to
2e-7 relative: the coupling is right.

The class-space residual is the frozen-core setting, not a contraction rule.
ORCA prints one function count per class; the counts of the archived N2/6-31G
CAS(6,6), H2O/6-31G CAS(2,2) and CO/6-31G CAS(2,2) runs are reproduced by the
plain-product cardinalities on the correlated internal orbitals,
`m = ncore - nfrozen` (`nfrozen` from the log's `Last frozen orbital` line;
N2, H2O and CO have nfrozen 2, 1 and 2):

| class | rule | N2 | H2O | CO |
|---|---|---:|---:|---:|
| ITAB (`Vm1_iab`) | `m*nact*nvir**2` | 768 | 294 | 800 |
| TUAB (`Vm2_ab`) | `nact**2*nvir**2` | 2304 | 196 | 400 |
| IJTA (`V1_ija`) | `m**2*nact*nvir` | 192 | 126 | 320 |
| IJTU (`V2_ij`) | `m*(m+1)/2*nact**2` | 108 | 24 | 40 |
| ITUA (`V0_ia`) | `2*m*nact**2*nvir` | 1152 | 168 | 320 |
| TUVA (`Vm1_a`) | `nact**3*nvir` | 1728 | 56 | 80 |
| ITUV (`V1_i`) | `m*nact**3` | 432 | 24 | 32 |

All 21 entries match. The N2 `IJTA` count of 192 is therefore the frozen-core
plain-product count (`m = 2`), not evidence of a reduced class space: the two
N 1s orbitals are frozen in ORCA, while the release reference and its class
table correlate all four core orbitals.

`caspt2_contracted_e2(..., frozen_core=2)` restricts the class index pools to
the correlated core orbitals (the reference, E0 and the denominators are
unchanged). With `class_diagonal='block', v0_ia='class', frozen_core=2` the
exact class-block solve reproduces the ORCA CASPT2-D table: `V1_ija`
-0.001224022 against -0.001224040 and `V2_ij` -0.007510716 against
-0.007510645 (both 1.0000 in ratio), and `V0_ijab`/`Vm1_iab`/`Vm2_ab` at
1.0001/1.0000/1.0000. The full-core values (`V1_ija` 1.5529, `V2_ij` 1.0542,
total 1.0232) compare a full-core CASPT2 with a frozen-core ORCA CASPT2. The
exact class-block reading is gauge invariant in both pools (function-order
permutation spread 4e-19); the diagonal approximation is gauge dependent in
the full-core pool (8 random gauges inside the degenerate overlap blocks span
0.879-0.895 of ORCA, the raw LAPACK basis 1.293, the production block-A gauge
1.553) and nearly gauge-free in the frozen-core pool (random gauges
0.978-0.994, production gauge 1.0000).

The two remaining readings are exposed by opt-in switches and close as well.
`v0_ia='class_ordered'` builds ORCA's ITUA/ITAU class from both operator types
over ordered active pairs (1152 functions at N2/6-31G, ORCA's cardinality) and
solves it exactly: -0.020417988 against -0.020418190 (1.0000), against 0.9804
for the unordered `'class'` construction. `three_active='block'` uses the
exact class-block reading of Vm1_a and V1_i: 0.9995 (-0.003607008) and 0.9998
(-0.001610267), against 1.0108 and 0.9409 with the orthonormal-diagonal
default. With `class_diagonal='block', v0_ia='class_ordered',
three_active='block', frozen_core=2` the complete eight-class table is within
6e-4 of ORCA CASPT2-D in every class and the total is -0.071518650 against
-0.071520733 (0.999971). The release defaults keep `frozen_core=0`,
`v0_ia='block'` and `three_active='diagonal'` because the reference and its
pins were generated with those settings; the ORCA-compatible setting is
opt-in. Extending the coupled path to the differentiable layer remains open.
(Branch `wave14/caspt2-nevpt2`, commits `ea26401` and `339e480`; frozen-core
investigation 2026-11-09.)

*ECP SOMF.* The valence-level ECP SOMF operator is a prototype, and no unique
ECP SOMF definition exists: the split between the ECP spin-orbit operator and
the explicit valence mean field is a fitting freedom that the ECP data alone do
not fix. `core.somf_ecp` now exposes the decomposition family (ECPso-only, full
valence, one-center, on-site-subtracted, point-charge and a double-counting
upper view) in one convention, and `experiments/somf_ecp/compare_variants.py`
implements a free-atom subtraction that removes the mean field the ECP already
contains. On the YbF model window the free-atom-subtracted and full-valence
values differ by about 10%, and the Ge+ 2P calibration leaves an absolute
offset of about 12%; both percentages come from the legacy spin-summed
transition-density assembly and are order-of-magnitude brackets only -- the
row is pending a determinant-space recalculation, and the density-combination
offset alone is 2.4-9.1% on GeF (see the script warnings). A unique definition and a
double-counting correction remain open; it is not a production absolute-SOC
method, so report the variant bracket and use the free-atom subtraction only
with its atomic-reference caveat. (Branch `wave14/relativistic`, commits
`dc2b836` and `193e546`.)

*SOMF derivatives and RI.* `core.somf_jax` provides a differentiable
explicit-kernel path for the two-electron SOMF operator, including the SI
assembly and a degeneracy-safe eigenvalue derivative; it is validated against
finite differences on OH and OF. `core.somf_ri` provides a value-layer
RI-SOMF(1X) prototype in which RI applies to the Coulomb term only and the
exchange is one-center exact; on OH/OF/6-31G the RI error is
0.0007/0.0010 cm^-1 against the four-center route, and RI-SOMF(1X) is within
0.5 cm^-1 of the four-center SOMF. The RI factors have no derivative layer, so
a differentiable RI-SOMF gradient remains open. (Branches `wave13/somf-diff` at
`989e173` and `wave13/ri-somf` at `1a66c0f`.)

*Two-component X2C.* The X2C-CASSCF chain is scalar (spin-free).
`core.spinor_ci` adds a minimal complex determinant CI with frozen-core folding,
a JAX value path and a Kramers-safe eigenvalue derivative; it reproduces pyscf
FCI in the non-relativistic spin-orbital limit and the Kramers theorem on a
synthetic time-reversal-invariant Hamiltonian. The physical X2C path remains
open: the pyscf `int2e_spinor` array does not obey the chemist pair symmetry
(measured violation 0.62 on H2O/6-31G), and the CI Hamiltonian built from it
with the determinant-energy ordering does not commute with many-body time
reversal on OH/6-31G. Kramers-restricted response, complex orbital rotations
and ECP support remain unimplemented. (Branch `wave14/relativistic`, commits
`dc2b836` and `193e546`.)

*RI SCF.* The RIJONX driver keeps exact exchange dense: the RI-J factors
remove the Coulomb four-center work, not the exchange four-center work.
Explicit analytic gradients now cover RHF, ROHF and UHF references (total
density for RI-J and the core Hamiltonian, spin densities for exact exchange,
spin-resolved energy-weighted density for the orbital response), checked
against implicit differentiation and central differences on OH, OF and an ECP
IO doublet. RI-JK and RIJCOSX remain out of scope, and the exact-exchange
gradient is validation scale because it still forms the dense derivative
tensors. (Branch `wave14/ri-hessian`, commit `5024db7`.)

## Reproducing the paper

Scripts in `experiments/` correspond to the results section by section:

| Script | Result |
|---|---|
| `gradient_gold_validation.py` | nuclear gradient validation; invariance conditions |
| `gradient_regression.py` | gradient regression against an analytic reference |
| `hessian_gold_refine.py` | golden-standard Hessian by re-optimizing at each displaced geometry |
| `mixed_derivative_ad_vs_fd.py` | why finite differences fail for the mixed derivative |
| `orbital_response_unrolled.py` | response term by unrolled iteration |
| `orbital_response_generalize.py` | unrolled route on CO and N₂ |
| `orbital_response_implicit.py` | response term by implicit differentiation |
| `hessian_to_frequencies.py` | harmonic frequencies from the mass-weighted Hessian |
| `frequencies_basis_active_space.py` | frequency dependence on basis set and active space |
| `collect_frequency_results.py` | frequency summary |
| `lanthanide_multistart.py`, `europium_scf_convergence.py`, `europium_dissociation_rerun.py` | lanthanide dissociation energies |
| `somf_ecp/validate_ecp_somf.py` | valence-level SOMF operator for ECP calculations (GeF, YbF) |
| `somf_ecp/validate_somf_grad.py` | SOMF gradient prototypes |
| `df_x2c_casscf_validation.py` | scalar DF-X2C CASSCF energy and gradient validation |
| `measure_ri_scf_driver.py` | RIJONX SCF driver comparison |

Each script sets the thread environment variables internally and writes
JSON to a `results/` directory. The scripts import `diffmrc` directly; make
it importable first (`pip install -e .`, or `PYTHONPATH=src`).

## Debugging

The package logs through Python's standard ``logging`` under the
``diffmrc.*`` namespaces; nothing is emitted unless the level is raised.

```python
import logging
logging.basicConfig(level=logging.DEBUG)   # INFO for optimizer progress
```

The CLI exposes the same switch: ``diffmrc mol.xyz -v`` shows optimizer
progress (INFO); ``-vv`` adds internals (DEBUG). ``CASPT2_DEBUG=1`` in
the environment still forces the CASPT2 debug records out.

## License

[MIT](LICENSE) — see the license file.

## Citing

If you use this code, please cite the repository:

```bibtex
@software{diffmrc,
  title  = {diffmrc: a differentiable multireference electronic structure stack},
  author = {{The diffmrc developers}},
  year   = {2026},
  url    = {https://github.com/jqChen1566/diffmrc},
  note   = {Version 0.3.0, early development}
}
```

A manuscript describing the method is in preparation.
