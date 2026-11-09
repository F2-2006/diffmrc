"""Multistate CASPT2 with a selectable zeroth-order Hamiltonian.

PURPOSE
-------
Implements the MS-CASPT2 and XMS-CASPT2 effective Hamiltonians on a
determinant model space whose zeroth-order operator can be chosen as the
state-averaged or the state-specific generalized Fock matrix.

The two zeroth-order choices follow Park et al., Chem. Rev. 2020, 120,
5878, eqs 11/16 (MS-CASPT2, diagonal projection of the Fock operator in
the reference space) and eq 17 (XMS-CASPT2, full ``P f P + Q f Q``).  The
effective Hamiltonian is the symmetrized form of Shiozaki, Woywod,
Werner, Phys. Chem. Chem. Phys. 2013, 15, 262, eq 2 (equivalently Park
and Shiozaki, J. Chem. Theory Comput. 2017, 13, 3643, eq 8) with the
level shift set to zero:

    H^eff_MN = <M|H|N> + 1/2 [ <Psi_M^(1)|H|N> + <M|H|Psi_N^(1)> ].

The first-order amplitudes are those of the contracted solver
``diffmrc.core.caspt2.caspt2_contracted_e2_coupled`` with the Fock matrix
supplied explicitly; the solver itself is not modified.  The solver and
the Slater-Condon kernel are imported lazily so that importing this
subpackage stays numpy-only.

PUBLIC INTERFACE
----------------
Densities and Fock matrices in the model space:
    ``state_averaged_density``, ``state_focks``, ``averaged_fock``,
    ``reference_fock_matrix``.
Amplitude solve and effective Hamiltonian:
    ``solve_states``, ``first_order_vectors``, ``coupling_matrix``,
    ``build_heff``.
Scheme entry points:
    ``solve_ms``, ``solve_xms``.
Utility: ``random_unitary``.

``civs`` are CI vectors over the determinant list ``dets``, one row per
reference state.  ``fock_mode='sa'`` selects the state-averaged Fock
operator (MS-MR and XMS), ``fock_mode='state'`` the state-specific
operator (SS-SR).  Energies are in Eh and exclude nuclear repulsion.

CONVENTIONS
-----------
* ``H_eff`` is built from the reference-space Hamiltonian
  ``H_ref = C H_model C^T`` plus the symmetrized first-order coupling.
* ``reference_fock_matrix`` returns ``F_MN = <M|f|N>`` in the reference
  basis; ``solve_xms`` diagonalizes it and rotates the references onto
  its eigenbasis (Chem. Rev. 2020 eq 18).
* With ``fock_mode='state'`` and ``symmetrize_fock=True`` the
  state-specific elements enter as ``(f^M + f^N)/2``, the SS-SR
  construction of Park, J. Chem. Theory Comput. 2019, 15, 3960,
  Supporting Information eq S4.

VALIDATED SCOPE
---------------
H2O CAS(2,2)/6-31G, 13 basis functions, 4 determinants, three equally
weighted singlet SA-CASSCF references at r_OH = 0.96, 1.30, 1.80 A:

* single-state limit: every scheme collapses onto the contracted solver
  and reproduces it bitwise (difference 0.0 Eh at all three geometries,
  against a 1e-8 Eh target);
* XMS rotation invariance: in the development run the eigenvalue
  spectrum changed by 3.9e-13 to 4.1e-11 eV over four random reference
  rotations (seeds 7/11/23/37), while the MS and SS-SR spectra moved by
  0.3 to 1.3 eV.  The residual is a few units in the last place of the
  total energy (order 1e-14 Eh) and moves within that floor between
  platforms; it is not a physical symmetry breaking;
* Hermiticity: ``max|H_eff - H_eff^T|`` between 5.1e-28 and 6.1e-15 Eh;
* the diagonal identity ``diag(H_eff) = E_CASCI + E2`` holds to at most
  1.4e-14 Eh.

The CH2/STO-3G CAS(2,2) two-state case is the unit-test fixture.

BOUNDARIES
----------
* The MS-MR implementation is a state-averaged-Fock zeroth order with a
  per-state contraction; the reference-space coupling term of Shiozaki
  et al. 2013 eq 1 is not included, and the first-order space is not
  generated from all reference functions jointly.  It therefore
  implements the zeroth-order operator of Chem. Rev. 2020 eq 16, not the
  full MS-MR contraction of eq 15.
* SS-SR is not invariant under a rotation of the reference states; this
  is a property of the state-specific zeroth order, not an implementation
  defect.  The SS-SR-XMS diagnostic variant is likewise not invariant in
  this implementation because the state-specific Fock operator is
  rebuilt from the rotated densities.
* The amplitude solver assembles the function-pair matrices in
  ``O(n_fn^2)``; N2 CAS(6,6)/6-31G (n_fn = 8052) exceeded the 20 min job
  limit and was not run.  The validated systems have n_fn = 1363 (H2O)
  and smaller.
* No IPEA shift, no real or imaginary level shift (E_shift = 0), no
  gradients, no spin-orbit coupling.  The model space is ``S_z = 0``
  only.
* Equal state-averaging weights only; a weight scan was not performed.
* The literature formulas were used through the secondary presentations
  CR20, S13, PS17 and the P19 Supporting Information; the original XMS
  papers (S11, G11) were not available in full text.
"""
from __future__ import annotations

import numpy as np

from .model import ci_1rdm, embed_density, state_fock, transition_1rdm


def _contracted_solver():
    """The release contracted CASPT2 solver (lazy, keeps imports light)."""
    from ..caspt2 import caspt2_contracted_e2_coupled
    return caspt2_contracted_e2_coupled


def state_averaged_density(civs, dets, norb):
    """Equal-weight state-averaged 1-RDM."""
    civs = np.asarray(civs, dtype=float)
    D = np.zeros((norb, norb))
    for ci in civs:
        D += ci_1rdm(ci, dets, norb)
    return D / civs.shape[0]


def state_focks(h1e_mo, eri_mo, civs, dets, norb, nmo=None):
    """State-specific generalized Fock matrices."""
    if nmo is None:
        nmo = np.asarray(h1e_mo).shape[0]
    return [state_fock(h1e_mo, eri_mo,
                       embed_density(ci_1rdm(ci, dets, norb), nmo))
            for ci in np.asarray(civs, dtype=float)]


def averaged_fock(h1e_mo, eri_mo, civs, dets, norb, nmo=None):
    """Generalized Fock matrix from the equal-weight state-averaged density."""
    if nmo is None:
        nmo = np.asarray(h1e_mo).shape[0]
    D = state_averaged_density(civs, dets, norb)
    return state_fock(h1e_mo, eri_mo, embed_density(D, nmo))


def reference_fock_matrix(focks, civs, dets, norb, symmetrize=False):
    """Fock matrix in the reference-state basis, ``F_MN``.

    ``focks`` is either one matrix shared by all states (state-averaged
    Fock) or one matrix per state.  With ``symmetrize=True`` the
    state-specific elements are averaged as ``(f^M + f^N)/2``, the
    SS-SR construction of Park, J. Chem. Theory Comput. 2019, 15, 3960,
    Supporting Information eq S4.
    """
    civs = np.asarray(civs, dtype=float)
    n = civs.shape[0]
    shared = np.ndim(focks) == 2
    F = np.zeros((n, n))
    for M in range(n):
        for N in range(n):
            G = transition_1rdm(civs[M], civs[N], dets, norb)
            if shared:
                fM = fN = np.asarray(focks)[:norb, :norb]
            elif symmetrize:
                fM = np.asarray(focks[M])[:norb, :norb]
                fN = np.asarray(focks[N])[:norb, :norb]
            else:
                fM = np.asarray(focks[M])[:norb, :norb]
                fN = fM
            F[M, N] = 0.5 * np.sum((fM + fN) * G)
    return 0.5 * (F + F.T)


def solve_states(h1e_mo, eri_mo, civs, dets, *, ncore, nact, nele,
                 focks=None, rtol=1e-8):
    """One contracted CASPT2 solve per reference state.

    Returns a list of slimmed solver records (amplitudes and contracted
    functions only) plus the per-state second-order energies.
    """
    solver = _contracted_solver()
    civs = np.asarray(civs, dtype=float)
    if focks is None:
        focks = [None] * civs.shape[0]
    elif np.ndim(focks) == 2:
        focks = [focks] * civs.shape[0]
    records = []
    e2 = np.zeros(civs.shape[0])
    for M, ci in enumerate(civs):
        e2[M], _, system = solver(
            h1e_mo, eri_mo, ci, dets, ncore=ncore, nact=nact, nele=nele,
            fock=focks[M], rtol=rtol, return_system=True)
        records.append({'x': np.asarray(system['x'], dtype=float),
                        'functions': system['functions'],
                        'E0': float(system['E0'])})
    return records, e2


def first_order_vectors(records):
    """Collect ``|Psi_M^(1)> = sum_i x_i |Omega_i^M>`` as determinant maps."""
    psis = []
    for rec in records:
        psi = {}
        for i, (_, vec) in enumerate(rec['functions']):
            xi = rec['x'][i]
            if xi == 0.0:
                continue
            for dd, c in vec.items():
                psi[dd] = psi.get(dd, 0.0) + xi * c
        psis.append(psi)
    return psis


def coupling_matrix(psis, civs, h1e_mo, eri_mo, dets):
    """``cross_MN = <Psi_M^(1)|H|N>`` for all state pairs.

    The matrix elements come from ``diffmrc.core.caspt2._sc_kernel``,
    the same kernel the contracted solver uses for ``W``.  Its convention
    (zero for identical determinants) is kept so that
    ``H_eff[M, M] = <M|H|M> + E2_M`` holds exactly.
    """
    from ..caspt2 import _sc_kernel

    civs = np.asarray(civs, dtype=float)
    n = civs.shape[0]
    h_elem, _ = _sc_kernel(h1e_mo, eri_mo)
    refs = [[(int(d), float(c)) for d, c in zip(dets, ci) if abs(c) > 1e-14]
            for ci in civs]
    support = set()
    for psi in psis:
        support.update(psi.keys())

    g = []
    for N in range(n):
        ref = refs[N]
        gN = {}
        for dd in support:
            acc = 0.0
            for J, c in ref:
                if (J ^ dd).bit_count() <= 4:
                    acc += c * h_elem(J, dd)
            gN[dd] = acc
        g.append(gN)

    cross = np.zeros((n, n))
    for M, psi in enumerate(psis):
        for N in range(n):
            gN = g[N]
            cross[M, N] = sum(c * gN[dd] for dd, c in psi.items())
    return cross


def build_heff(h1e_mo, eri_mo, civs, dets, *, ncore, nact, nele,
               H_model=None, det_index=None, fock_mode='sa', focks=None,
               symmetrize_fock=False, rtol=1e-8):
    """Build and diagonalize the symmetrized effective Hamiltonian.

    ``fock_mode='sa'`` uses one state-averaged Fock operator for every
    state (the MS-MR and XMS choice); ``fock_mode='state'`` uses the
    state-specific operator of each reference (the SS-SR choice).
    """
    civs = np.asarray(civs, dtype=float)
    n = civs.shape[0]
    norb = ncore + nact
    if H_model is None:
        from .model import model_hamiltonian
        H_model = model_hamiltonian(h1e_mo, eri_mo, dets, ncore, nact, nele)
    if det_index is None:
        det_index = {d: i for i, d in enumerate(dets)}

    if fock_mode == 'sa':
        f = averaged_fock(h1e_mo, eri_mo, civs, dets, norb)
        solve_focks = f
        ref_focks = f
    elif fock_mode == 'state':
        solve_focks = state_focks(h1e_mo, eri_mo, civs, dets, norb)
        ref_focks = solve_focks
    else:
        raise ValueError("fock_mode must be 'sa' or 'state'")

    if focks is not None:
        solve_focks = focks
        ref_focks = focks

    records, e2 = solve_states(h1e_mo, eri_mo, civs, dets, ncore=ncore,
                               nact=nact, nele=nele, focks=solve_focks,
                               rtol=rtol)
    psis = first_order_vectors(records)
    cross = coupling_matrix(psis, civs, h1e_mo, eri_mo, dets)
    Href = civs @ H_model @ civs.T
    H_eff = Href + 0.5 * (cross + cross.T)
    energies = np.linalg.eigvalsh(H_eff)
    F = reference_fock_matrix(ref_focks, civs, dets, norb,
                              symmetrize=(fock_mode == 'state'
                                          and symmetrize_fock))
    return {
        'H_eff': H_eff,
        'H_ref': Href,
        'cross': cross,
        'energies': energies,
        'e2': e2,
        'E0': np.array([r['E0'] for r in records]),
        'reference_fock': F,
        'civs': civs,
        'records': records,
        'psis': psis,
    }


def solve_ms(h1e_mo, eri_mo, civs, dets, **kwargs):
    """MS-CASPT2: diagonal Fock projection, no rotation of the references."""
    out = build_heff(h1e_mo, eri_mo, civs, dets, **kwargs)
    out['scheme'] = 'MS-CASPT2'
    return out


def solve_xms(h1e_mo, eri_mo, civs, dets, *, ncore, nact, nele,
              fock_mode='sa', rtol=1e-8, **kwargs):
    """XMS-CASPT2: rotate the references onto the Fock eigenbasis first.

    The rotation is the ``P f P`` diagonalization of Chem. Rev. 2020 eq
    18.  With ``fock_mode='state'`` the state-specific Fock elements are
    symmetrized first (Park 2019 Supporting Information eq S4).
    """
    civs = np.asarray(civs, dtype=float)
    n = civs.shape[0]
    norb = ncore + nact
    if fock_mode == 'sa':
        ref_focks = averaged_fock(h1e_mo, eri_mo, civs, dets, norb)
    elif fock_mode == 'state':
        ref_focks = state_focks(h1e_mo, eri_mo, civs, dets, norb)
    else:
        raise ValueError("fock_mode must be 'sa' or 'state'")

    F = reference_fock_matrix(ref_focks, civs, dets, norb,
                              symmetrize=(fock_mode == 'state'))
    fock_eigvals, U = np.linalg.eigh(F)
    civs_rot = U.T @ civs
    out = build_heff(h1e_mo, eri_mo, civs_rot, dets, ncore=ncore,
                     nact=nact, nele=nele, fock_mode=fock_mode,
                     symmetrize_fock=(fock_mode == 'state'), rtol=rtol,
                     **kwargs)
    out['scheme'] = 'XMS-CASPT2'
    out['rotation'] = U
    out['fock_model'] = F
    out['fock_eigvals'] = fock_eigvals
    out['civs_input'] = civs
    return out


def random_unitary(n, seed=0):
    """Haar-ish orthogonal matrix for the invariance check."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.diag(R))
