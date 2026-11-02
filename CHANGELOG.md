# Changelog

## Unreleased

### Added

- `core.kramers`: a Kramers-adapted spinor basis and a T-closed active-space
  slice, with `x2c_spinor_ci` as an opt-in physical X2C determinant-space
  entry. The threaded complex GHF basis is not guaranteed to be
  Kramers-paired; on OH/6-31G the unadapted core breaks time-reversal
  invariance to 5.06 and opens a pseudo Kramers gap of 0.0495 Eh, while the
  adapted basis gives `T H* T† - H` of 6.1e-13 (H2O) and 3.7e-13 (OH) with
  a Kramers gap of 9.9e-14. `_build_matrix` is the physical Coulomb operator;
  the frozen-core reduction was not: its exchange contraction averaged the
  two index orders, which is only valid for real orbitals. Using the single
  physical order `E[p,i,i,q]` makes the reduced matrix agree with the
  full-space projection to 2.0e-13 and restores core-rotation invariance
  from 0.17 Eh to 8.5e-14; the frozen-core energies become -85.222187844
  (H2O/6-31G) and -77.679324114 Eh (OH/6-31G). The non-relativistic FCI
  comparison (4.3e-11/4.9e-11) validates determinant algebra and the ERI
  convention, not relativistic FCI.
- `driver.scf_response`: canonical closed-shell RHF reference response for
  the excited-state property chain. PySCF's CPHF returns the occupied
  response in the `-S1/2` gauge, which keeps the density derivative exact
  but not the individual orbitals, and says nothing about the virtual
  columns; adding the occupied-occupied and virtual-virtual canonical
  rotations, the symmetric overlap term and the pyscf/pyscfad column-sign
  alignment closes the response. H2O/6-31G SA-CASSCF total nuclear
  derivatives of mu_z and f then agree with central differences to 8.0e-8
  and 2.4e-10 at theta=0 (4.1e-7 and 8.5e-8 at theta*). Open-shell
  references are refused: PySCF 2.13 has no ROHF CPHF, and a UHF
  substitute breaks symmetry.

## 0.3.4 (2026-11-02)

### Added

- `driver.scf_ad`: implicit-differentiation ROHF reference for f-block
  molecules. The unrolled pyscfad SCF gradient is not the derivative of
  the converged solution when DIIS participates (its error vectors are
  detached in the AD graph, so the linearized iteration map does not
  contract); on YbF/stuttgart_rsc it returns +2.418e-1 Eh/Bohr, 3.42 times
  the magnitude of the analytic +7.070e-2. `rohf_energy` differentiates
  the fixed point
  instead (+7.0703535e-2, 3.0e-7 from the analytic ROHF gradient,
  1.7e-6 from central differences), `scf_reference` selects the branch
  by a fixed energy-ordered candidate ladder shared with
  `driver.scf_protocol`, and the coordinate-only pytree view keeps the
  YbF gradient at 3.7 GB in the single-purpose probe (4.1 GB with the
  candidate ladder held in the same process) instead of above 20 GB.
  CeO's two solutions,
  0.112 Eh apart with identical occupations, become bit-repeatable
  across runs. The flag is set per call; existing callers are unchanged.
- `analysis.ci.spin_free_*`: Pittner's spin-free orbital entropy, pair
  entropy and mutual information (arXiv:2502.04800): the three-weight and
  nine-pattern forms, built from the same occupation triple and requiring
  no additional density matrix. For a spin-adapted state the
  manifold-averaged reading satisfies
  `S_manifold = S_spin_free + (N - 2 p2) ln 2`; the identity is checked
  numerically and on N/6-31G, CH2/6-31G and CeO. The two quantities
  measure different objects: the spin-free form removes the spin
  degeneracy of singly occupied orbitals, the manifold average keeps it.

### Changed

- `core.caspt2`: the per-function diagonal path now warns when
  `active_row='replica'` is used with more than two active orbitals,
  because the dropped `(u, u)` rows are replicas only for `nact = 2`.
  The N2/6-31G CAS(6,6) audit behind the warning: the default total
  0.9893 (Wave 12 frozen reference) is a cancellation of `V2_ij`
  +5.71 mEh against -4.95 mEh over the other seven classes; the
  class-complete block solve with `v0_ia='class'` gives `Vm2_ab` 1.0000,
  `V2_ij` 1.0542 and `V0_ia` 0.9970 and a total of 1.0232, while it
  leaves the H2O and CO CAS(2,2) anchors bit-identical. `V1_ija` stays
  at 1.553 in the gauge-invariant class-block solve, with its normalized
  minimum denominator 2.7e-6 from ORCA, and is registered as a
  class-space definition boundary. `active_row='keep'` alone double
  counts under the per-function diagonal (N2 total 1.3280).

### Fixed

- `core.gauge.canonical_mo_phases` and its three call sites in
  `driver.nevpt2`: a deterministic column phase for the SCF reference
  orbitals. The threaded eigen-decomposition returns the same orbitals
  with different column signs (MKL/OMP dependent through the JAX/XLA
  path), and `theta` is measured from those columns; a fixed `theta`
  therefore described different physical rotations under different
  thread counts. After the fix, H2O/6-31G CAS(2,2) at one fixed `theta`
  is bit-identical over `OMP_NUM_THREADS = 1, 2, 4, 8, 32` (ten runs),
  the optimized-`theta` energy and `e2` are unchanged, and HI/LANL2DZ
  agrees across the same thread sweep. The phase call is `O(nmo^2)`
  (28 us at 13 columns, 418 us at 256) and is not measurable against a
  NEVPT2 evaluation. Explicit column-sign flips on the reference change
  the old fixed-`theta` result by 1.8 Ha on H2O and 0.34 Ha on HI; they
  change nothing after the fix.

## 0.3.3 (2026-11-02)

### Added

- `core.adint`: a memory-bounded integral-derivative backend for the nuclear
  gradient chains. `use_vjp_integrals(True)` switches the four-centre
  integrals to the `pyscfad` `custom_vjp` kernel, whose backward contracts the
  `nao**4` cotangent in C instead of transposing a `custom_jvp` and
  materialising `nctr * nao**4` doubles. `int2e_s1` requests the `s4` packing
  the kernel requires and restores `s1` with two differentiable gathers;
  `build_mole` leaves the basis exponents and contraction coefficients
  untraced, which the backend requires. Measured peak RSS of the bare
  integral derivative: 18 999 MB -> 7 797 MB at nao = 124, natm = 8, and
  9 565 MB -> 4 151 MB at nao = 104, natm = 4; the gradient agrees with the
  default backend to 1e-12 relative. The switch is off by default, so the
  existing chains are unchanged. The two-centre integrals stay on
  `custom_jvp`: the `custom_vjp` two-centre kernel segfaults on `ECPscalar`
  (measured on HI/lanl2dz).
  Actinide check: on ThO (`Th=stuttgart_rsc` ECP, `O=6-31G`, nao = 96,
  CAS(8,8)) the stage ladder SCF -> MO transform -> CI goes from
  oom-killed at the CI rung to 19 267 MB (SCF 8 606 -> 2 537 MB,
  +MO transform 14 398 -> 6 125 MB). That rung's analytic gradient then
  completes at 19 278.5 MB peak against a central difference of the same
  energy: 1.22e-5 Eh/bohr absolute, 2.04e-7 relative. The canonicalize
  and NEVPT2 rungs remain above 20 GB on both backends; their cost is the
  perturbation layer, not the integral derivative (ThO/CAS(6,6) forward
  alone: 8 709 MB, 1 152 s).
- `driver.nevpt2_zvector`: the exact-Si correction now evaluates the Si
  class once and applies `si_exact - si_masked` to the release total,
  instead of rebuilding the six core classes twice and cancelling five of
  them. The classes, `core.nevpt2.core.e2_core` and `_e2_core_exact_si`
  are unchanged; the latter two are kept so the equivalence test can
  rebuild the old value. H2O/6-31G: 0.936 -> 0.872 s per NEVPT2
  evaluation (-6.8%), total agrees to 1.79e-16 relative; HI8/LANL2DZ:
  0.942 -> 0.881 s (-6.5%), bit-identical. ThO/CAS(6,6) forward:
  1641 -> 1152 s (-29.8%), peak unchanged at 8.7 GB. The residual cost of
  the exact-Si path is `_core_blocks`, which rebuilds `dm1/dm2/dm3` and
  `f3ca/f3ac` that `nevpt2_e2_full2` already formed (95% of the
  overhead); removing it needs a change in `core/nevpt2/impl.py`.
- `driver.ri_scf_driver`: explicit analytic RIJONX nuclear gradients for ROHF
  and UHF references, and the `RIJONX_UHF` mean-field class. The skeleton
  gradient uses the total density for RI-J and the core Hamiltonian and the
  spin densities for exact exchange; the response term is the spin-resolved
  energy-weighted density built from the RIJONX Fock. Validated against
  implicit differentiation and central differences on OH and OF, and against
  implicit differentiation on an ECP IO doublet.
- `driver.ri_scf.exact_k_gradient_spin`: spin-summed exact-exchange skeleton
  gradient with the open-shell `-1/2 sum_sigma Tr(D_sigma K[D_sigma])`
  convention. `ri_j_gradient` and `core_hamiltonian_gradient` accept a density
  directly, which a spin-separated reference needs.
- `core.hessian.implicit_hessian_columnwise`: one tangent per column; identical
  to the vectorized route (5e-15 on H2O/6-31G, 3e-15 on CO/6-31G) at lower peak
  memory.
- `core.hessian.implicit_hessian_hvp`: matrix-free relaxed Hessian with an LSMR
  response solve; never forms `H_tt`, `M` or `H_RR`. Matches the dense route to
  6e-12 (H2O) and 8e-12 (CO); `H_tt` products are jitted, and the default
  iteration cap is `8 n_par`.
- `core.hessian.make_theta_energy_fn`: fixed-geometry theta-only energy, which
  keeps the SCF out of the `H_tt` graph.
- `experiments/hessian_cost_benchmark.py`: wall time, peak RSS, accuracy and
  LSMR diagnostics for the three Hessian routes.
- `analysis.ci.manifold_occupations` and
  `analysis.ci.manifold_entropy_from_ci`: four-state entropy of a weighted
  mixture of CI vectors, assembled on the occupation triple (the weights are
  linear in `(n_a, n_b, p2)`, so no density matrix is built). The
  differentiable siblings are `driver.entropy.manifold_entropy_from_ci_masks`
  and `driver.entropy.manifold_single_orbital_entropy_from_ci`, bound to the
  NumPy route at 1e-12.
- `driver.kernel.casci_vectors_at`: CI vectors at one orbital rotation, on the
  integral block and rotation shared with `energy_function`. Used by the
  active-space verification and available for analysis of a converged point.
- `policy.rules.candidate_from_occupations` and a fallback in
  `driver.kernel.select_active_space`: a composition that no element rule
  covers is no longer refused outright. Every orbital with a fractional
  reference occupation enters the space; a closed-shell reference
  contributes the frontier pair; degenerate partners enter with a selected
  orbital. On the six-system benchmark this makes H2O (orbitals 4 and 5),
  triplet CH2 (the two singly occupied orbitals) and O3 (the frontier pair,
  CASSCF 0.114 Eh below RHF) runnable. N2 selects a degenerate frontier
  pair; its analytic Hessian is not finite at the symmetric point, so the
  curvature comes from a symmetric difference of the gradient and the run
  completes with `curvature_method = 'gradient-difference'`.
- `driver.kernel.compute_ground_state`: when the space came from the
  occupation fallback and its orbital optimization stalls (gradient above
  1e-3), the frontier pair is retried and kept if its point is better.
  Measured on HI/def2-SVP, the three-orbital completed space did not
  converge in hundreds of steps while the frontier pair reached 4.3e-8 in
  128 s. The first attempt on a fallback space is also given a step
  budget (60 L-BFGS-B steps, 20 refinement steps) so a stall is
  established in minutes rather than after the default cap.
- `driver.kernel.scf_reference` and the end-to-end path accept an `ecp`
  parameter (a name, or a mapping keyed by element) and pass it to
  `Mole.ecp`. Without it a heavy element is built as the all-electron atom
  against a valence basis and the reference cannot be constructed
  (`Nocc > Nmo`). Verified on HI/def2-SVP with `{'I': 'def2-svp'}`
  at I = origin, H at 1.6 Bohr: 26 electrons, 31 orbitals; the reference
  RHF is -297.2315508861 Eh and the end-to-end CASSCF with active
  orbitals [12, 13] is -297.2377012677 Eh.

### Changed

- `core.x2c_casscf_df.df_x2c_casscf` gains a Newton polish after the
  L-BFGS-B stage: from the minimum-gradient iterate it builds the orbital
  Hessian by central differences of the analytic gradient, applies a
  pseudoinverse with a relative cutoff of 1e-10, caps a step at 0.3 rad,
  and accepts a step only if the gradient strictly decreases and the
  energy does not rise. On H2O/6-31G the final orbital gradient drops from
  2.14e-6 to 4.31e-12 in two steps, the envelope finite-difference
  relative error from 1.38e-4 to 1.57e-5, and the rotation invariant from
  1.67e-6 to 5.58e-10, with the energy 1.10e-11 Ha from pyscf. No test
  threshold was relaxed. The polish is skipped above 256 parameters,
  where the central-difference Hessian would dominate; the stage runs
  explicitly with `polish=False` to keep the previous behaviour.
- `driver.orbitals` gains a two-stage orbital optimization for spaces whose
  core-virtual rotation block dominates. When the core-virtual pair count
  exceeds twice the core-active plus active-virtual count, the first stage
  runs on the subset without core-virtual rotations and the second stage
  refines the embedded point over the full set (`embed_theta` maps by pair
  identity, `converge_orbital_point_two_stage`). Measured on HI/def2-SVP
  CAS(2,2) (262 full / 58 subset parameters): final gradient 1.41e-9 in
  378.5 s, against 420.5 s and 7.48e-10 for the single-stage full-space
  route, with the two converged points 1.1e-13 Ha apart. H2O/6-31G and
  N2/6-31G keep the single-stage route and are bit-identical. The gate is
  required because the subset optimum is not always a minimum: on H2O it
  is a saddle with two negative curvature directions, while the full-space
  refinement reaches the correct minimum. The gate is an engineering rule
  supported by three systems, not a theorem.
- The first-stage orbital-gradient target defaults to 1e-4, down from 1e-6,
  in `policy.params` and `driver.orbitals`. The measured floor of the
  L-BFGS-B stage is 1.5e-5, so a 1e-6 target keeps the first stage
  iterating in a flat valley: measured on HI/def2-SVP CAS(2,2), the run
  passed 55 minutes without converging, while stopping the first stage at
  1e-4 and letting the Newton refinement finish reached 4.3e-8 in 128 s.
- `make_joint_energy_fn` accepts an `scf` constructor.
- `rijonx_scf` accepts `uhf=True`; the open-shell class remains ROHF by
  default.
- The robust level now performs the entanglement verification of the active
  space instead of reporting it as unimplemented. The driver merges the lowest
  `2S+1` roots before reading the entropy, so the `M_S = +S` component of a
  high-spin determinant no longer reports zero for every open shell. A
  single-reference verdict fails the check and refuses the run; size
  mismatches are recorded but do not veto. On N/6-31G CAS(3,3) the merged
  entropy is ln 2 = 0.693147 for all three 2p orbitals, against 2e-25 to 0.66
  for the individual roots; H2O/6-31G CAS(2,2) and H2/STO-3G CAS(2,2) report
  4.79e-3 and 6.79e-2 and are refused as single-reference.

### Fixed

- `driver.kernel` assembled the one-electron Hamiltonian as
  `int1e_kin + int1e_nuc`, which drops the ECP scalar term on a
  pseudopotential system. Measured on ThO, the matrix traces differ by
  150.7 Ha and the energy by 71 Ha. `kernel._core_hamiltonian` now adds
  `ECPscalar` when the molecule reports an ECP, and the same construction in
  the on-top terms of `driver.nevpt2` is corrected. A molecule without an ECP
  is unchanged: the H2O/6-31G energy is bit-identical before and after.
  Verified on I/def2-SVP (ECP): the helper equals kinetic + nuclear + ECP
  and differs from the raw sum, whose trace moves by 90.5 Ha.
- A non-finite orbital Hessian no longer kills the run inside `eigvalsh`.
  Measured on N/6-31G CAS(3,3) quartet (26 parameters) and N2/6-31G with
  its degenerate frontier pair (101 parameters): every analytic Hessian
  entry is NaN at the symmetric stationary point while the energy and the
  analytic gradient are finite. The energy is smooth, so
  `core.stationary.curvature_from_gradient` recovers the curvature from a
  symmetric difference of the analytic gradient; on N2 the spectra at steps
  1e-5, 1e-4 and 1e-3 agree (min 0.0648, max 66.96, no negative
  eigenvalues). The route costs two gradient evaluations per parameter and
  is capped at 300 parameters, above which the driver refuses with the
  diagnosis. `algorithm_data['curvature_method']` records which route ran.
  End to end, N2/6-31G CAS(4,4) (E = -108.97003525320882 Eh) and the N
  quartet CAS(3,3) (E = -54.382051137489576 Eh) now complete with
  `curvature_method = 'gradient-difference'` and zero negative
  eigenvalues.
- The Newton refinement no longer stops at a point whose analytic orbital
  Hessian is not finite. `_trust_region_polish` probes one
  Hessian-vector product with a vector of ones; a non-finite entry hands
  the point to `_finite_difference_newton`, which rebuilds the Hessian
  from symmetric differences of the analytic gradient (step 1e-5, dense
  solve 82 kB at 101 parameters, 300-parameter cap) and keeps the
  existing damping ladder and energy-decrease acceptance. Measured on
  N2/6-31G CAS(4,4) with the automatic space {5,6,7,8}: the L-BFGS-B
  stage stops at 7.325e-5, two refinement steps reach 8.81e-14, energy
  -108.97003525321175 Eh, 302 s, and the robust level passes all four
  checks; the same space was refused on the 1e-8 gradient before. The
  probe costs one Hessian-vector product, so the finite-Hessian path is
  unchanged: HI/def2-SVP [12,13] (-297.23770126765163 Eh, 1.409e-9),
  H2O/6-31G [4,5] (-75.9849511873311 Eh, 1.891e-12, three negative
  eigenvalues, refused) and CH2/6-31G triplet [3,4]
  (-38.906546701973426 Eh, 3.387e-14) reproduce the previous run. No
  threshold was relaxed.

### Known limitations

- NEVPT2 fixed-`theta` values were thread-dependent before 0.3.4. The cause
  is not the CI eigen-decomposition but the arbitrary sign of the SCF
  eigenvectors: `theta` is measured from those columns, so a
  thread-dependent phase gauge makes one numeric `theta` describe
  different physical rotations. Measured on H2O/6-31G CAS(2,2) at one
  fixed `theta`, `e2` spanned 2.96e-3 Ha over `OMP_NUM_THREADS = 1, 2, 4,
  8, 32`, while the SCF energies agreed to 2e-14 and the coefficient
  matrices differed by column signs only (off-diagonal overlap 1.5e-12).
  The NEVPT2 driver now canonicalizes the MO phases (`core.gauge`);
  optimized values are unchanged and fixed-`theta` values are
  thread-independent. The same convention is applied at the kernel,
  Hessian, X2C and frozen-gradient rotation bases; the five runtime
  anchors are bit-identical before and after. A different freedom
  remains: an exactly degenerate subspace can return a different
  orthonormal basis (measured at one fixed `theta`, N2/6-31G differs by
  3.5e-2 Ha and HI/def2-SVP by 1.5e-1 Ha between one and four threads,
  with the SCF energies agreeing to 3e-14). A column-sign rule does not
  fix that, so `OMP_NUM_THREADS=1` remains the safe setting for numbers
  compared across runs.
- The matrix-free route needs the LSMR iteration cap to reach convergence on
  the tested stationary points; the unrolled route with three Newton steps
  exceeds 20 GiB on H2O/6-31G CAS(2,2).
- The exact-exchange RIJONX gradient remains validation scale (dense
  `int2e_ip1`/`ip2` tensors). RI-JK and RIJCOSX are not implemented.

## 0.3.2 (2026-10-25)

### Changed

- CASPT2: `Vm1_a` (TUVA) and `V1_i` (ITUV) are now three-active-index
  classes, `E(a,t)E(u,v)` and `E(t,i)E(u,v)`, instead of single
  excitations. The contracted energy uses an orthonormal class basis
  (H1) with a canonical basis inside degenerate blocks; the JAX path uses
  `eigh_safe`. N2/6-31G CAS(6,6) default-path ratio to ORCA CASPT2-D moves
  from 0.916 to 0.989; H2O/CO CAS(2,2) anchors are unchanged.
- The three-center CASPT2 path uses the same class set and H1 treatment.

### Known limitations

- N2 default-path residual: `V1_ija` 1.55 and `V2_ij` 0.24 relative to
  ORCA CASPT2-D, from the class-space definition and the `(u,u)` active-row
  approximation. The fully coupled path totals 1.023 on N2; its O(nfn^2)
  Python assembly is not used by the public entry point.

## 0.3.1 (2026-10-25)

### Added

- `core.caspt2_three_center`: three-center contracted CASPT2 with the same
  `cross - direct` VM convention as the materialized path; agrees with the
  materialized RI path to machine precision on H2O/CO.
- `scripts/caspt2_n2_validation.py` and `scripts/caspt2_sparse_assembly.py`:
  N2/6-31G CAS(6,6) validation driver and sparse assembly.
- `experiments/basis_set_gradient_validation.py` and stage runners: basis-set
  independence check for the relaxed NEVPT2 gradient (H2O/cc-pVDZ).
- `tests/driver/test_ecp_gradient.py`: pins the ECP core-Hamiltonian nuclear
  gradient.

### Changed

- `make_dm123_sf` accepts `max_rank`; `canonicalize_jax` requests only the
  one-particle density, avoiding the `ncas**6` dm3 contraction at large
  active spaces. Default behaviour is unchanged.

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
  strongly contracted NEVPT2, matrix-free iterative CI, and a DMRG
  candidate-space hook (`core.mps.pin_degenerate_columns`); analytic first and
  second derivatives for CASSCF and NEVPT2. The DMRG solver itself is not part
  of this release.
