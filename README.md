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
> Known limitations section rather than enforced by the code. Several of the
> limitations listed there are properties of the method family and are not
> expected to be removed by further development of this package. Use it for
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
manual/         Tutorial and Manual (LaTeX sources and built PDFs)
```

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

## Known limitations

*Full diagonalization* in the CI step limits the active-space size. Larger
spaces require a differentiable iterative eigensolver.

*Degenerate pairs at a stationary point.* The cluster tolerance of the
eigensolver must sit inside the measured gap between the floating-point noise
that splits symmetry-degenerate pairs at a rotated stationary point (1e-8 to
1.2e-5 in the systems measured here) and the physical gaps of the same spectrum
(1.9e-2 and above). A tolerance below the noise scale exposes both the first
and the second derivative to a bare inverse gap; one above the physical gaps
clusters distinct states. The window measured here spans 1e-5 to 1e-2, and the
rule has no admissible value when a physical gap falls inside the noise band —
the fallback for that case is the decomposition-free Lanczos kernel.

*Multiple stationary points.* The CASSCF energy is non-convex in the orbital
parameters; different starting points can reach different stationary points,
and the selection is not stable even at fixed settings. Report a stationarity
certificate and the stationary-point spread alongside any multireference
number.

*Convergence of the perturbation layer.* NEVPT2 depends on the orbitals more
sensitively than the CASSCF energy does, so energy-based convergence criteria
are insufficient when the target quantity is the correction. Use the stability
of the correction between independent runs together with an orbital-gradient
threshold.

*Cost of second derivatives.* The implicit route requires the orbital Hessian in
memory and does not exploit sparsity; the unrolled route scales with the number
of iterations. Neither is competitive with analytic second derivatives for
single-reference methods.

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
  note   = {Version 0.1.0, early development}
}
```

A manuscript describing the method is in preparation.
