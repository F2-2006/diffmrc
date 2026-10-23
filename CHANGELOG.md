# Changelog

## 0.3.0 (2026-10-23)

### Added

- `driver.nevpt2_zvector`: relaxed NEVPT2 nuclear gradients with exact
  near-zero `Si` channels, blockwise reference alignment, and the CASSCF
  Z-vector response. Validated against finite differences on H2O/6-31G and
  LiH/6-31G.
- `core.caspt2.caspt2_contracted_e2_coupled`: coupled full-Fock contraction
  retaining the inter-class coupling of the zeroth-order Hamiltonian.
- `core.somf_ecp`: valence-level spin-orbit mean-field operator for
  effective-core-potential calculations, with gradient prototypes.
- `driver.ri_scf`, `driver.ri_scf_driver`: RIJONX SCF (RI-J Coulomb plus exact
  exchange) on pyscfad RHF/ROHF, including the SCF overlap-derivative term.
- `core.x2c_casscf_df`: scalar (spin-free) DF-X2C CASSCF energies, analytic
  nuclear gradients, and state-averaged response.

### Fixed

- CASPT2: the relative weight of the `VM` component now follows the
  `cross - direct` convention. The previous `direct - sqrt(3) * cross`
  expression is not a scalar multiple of the correct operator.
- `driver.ri_scf`: the fixed-coefficient gradient was not equal to the SCF
  gradient at a converged density; the missing overlap-derivative term and the
  ECP scalar contribution to the core Hamiltonian are restored.

### Notes

- New modules are covered by unit tests. Scope limits are listed in the
  Known limitations section of the README.

## 0.2.0 (2026-10-22)

- SOMF value layer for molecular spin-orbit coupling.
- Trust-region orbital optimization with a saddle-escape test.
- AVAS active-space construction, including the f-block fixed-7f workflow.
- Scalar DF-X2C and SAP-X2C value and derivative layers.
- Differentiable RI-J, RIJONX, and RIJCOSX prototypes.
- Manual and tutorial examples; package data for the SAP auxiliary basis.

## 0.1.0 (2026-10-11)

- Initial public release: differentiable CASCI, CASSCF, state averaging,
  strongly contracted NEVPT2, matrix-free iterative CI, and DMRG candidate
  space; analytic first and second derivatives for CASSCF and NEVPT2.
