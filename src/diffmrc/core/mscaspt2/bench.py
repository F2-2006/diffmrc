"""Benchmark systems for the multistate CASPT2 zeroth-order Hamiltonians.

PURPOSE
-------
Builds a state-averaged CASSCF reference and re-expresses its CI vectors
in the module's own ``S_z = 0`` determinant basis, so that the reference
states, the model Hamiltonian and the CASPT2 amplitudes share one
convention.

PUBLIC INTERFACE
----------------
``transform_integrals`` -- AO to MO integral transform (chemist notation).
``build_system`` -- SA-CASSCF orbitals plus prototype-convention CI
vectors and the model Hamiltonian for one geometry.

``build_system`` imports pyscf inside the function, so importing this
module does not require pyscf.  Coordinates are in angstrom, energies in
Eh.  The returned ``sa_energies`` are total energies (nuclear repulsion
included); ``H_model`` is electronic only, and the caller adds ``e_nuc``
to compare.

CONVENTIONS
-----------
* The SA-CASSCF run uses pyscf's ``fix_spin_`` penalty with ``ss=0`` so
  that the averaged manifold stays in the singlet sector.
* The CI vectors are recomputed as eigenvectors of the model
  Hamiltonian; each SA-CASSCF state is matched to the closest eigenvalue
  (tolerance 1e-6 Eh) and rejected if the match is missing or duplicated.
* The sign of every CI vector is fixed by making the first coefficient
  that is tied with the largest magnitude (to 1e-10 relative) positive.
  The tie rule keeps the sign deterministic when symmetry makes two
  coefficients equal in magnitude, where a plain argmax would depend on
  round-off.

VALIDATED SCOPE
---------------
H2O CAS(2,2)/6-31G at r_OH = 0.96, 1.30, 1.80 A (13 basis functions, 4
determinants) and CH2/STO-3G CAS(2,2) with two states: the model
Hamiltonian eigenvalues reproduce the SA-CASSCF state energies and the
matched CI vectors are eigenstates to 1e-8 Eh.  The H2O systems are the
ones behind the validation numbers listed in
``diffmrc.core.mscaspt2.ms``.

BOUNDARIES
----------
* pyscf is an optional dependency; ``build_system`` raises the ordinary
  ``ImportError`` when it is absent.
* Equal state-averaging weights; no point-group symmetry (C1), no
  frozen-core selection beyond ``(nelectron - nele) // 2``.
* The determinant space is ``S_z = 0``, so only even active electron
  counts are supported (``sz0_dets`` rejects odd counts).
* This is benchmark scaffolding, not part of the production solver path.
"""
from __future__ import annotations

import numpy as np

from .model import model_hamiltonian, sz0_dets


def transform_integrals(C, h1e, eri):
    """AO to MO transform (chemist notation)."""
    h1 = np.einsum('ai,ab,bj->ij', C, h1e, C)
    z = np.einsum('pi,pqrs->iqrs', C, eri)
    z = np.einsum('qj,iqrs->ijrs', C, z)
    z = np.einsum('rk,ijrs->ijks', C, z)
    e4 = np.einsum('sl,ijks->ijkl', C, z)
    return h1, e4


def build_system(atom, basis, nact, nele, nstates, *, verbose=0):
    """SA-CASSCF orbitals plus model-convention CI vectors."""
    from pyscf import fci, gto, mcscf, scf

    mol = gto.M(atom=atom, basis=basis, verbose=verbose)
    mf = scf.RHF(mol).run(verbose=verbose)
    mc = mcscf.CASSCF(mf, nact, nele)
    mc.fcisolver.nroots = nstates
    # fix_spin_ returns a solver object; it does not modify in place.
    mc.fcisolver = fci.addons.fix_spin_(mc.fcisolver, shift=0.5, ss=0)
    mc.state_average_([1.0 / nstates] * nstates)
    mc.kernel()

    ncore = (mol.nelectron - nele) // 2
    C = np.asarray(mc.mo_coeff)
    h1e = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    eri = mol.intor('int2e', aosym='s1')
    h1_mo, eri_mo = transform_integrals(C, h1e, eri)

    dets = sz0_dets(ncore, nact, nele)
    H = model_hamiltonian(h1_mo, eri_mo, dets, ncore, nact, nele)
    w, V = np.linalg.eigh(H)

    # ``mc.e_states`` are total energies; the determinant Hamiltonian is
    # electronic only, so the nuclear repulsion is added for matching.
    w_total = w + float(mol.energy_nuc())
    sa_energies = np.asarray(mc.e_states, dtype=float)
    picked = []
    for e in sa_energies:
        j = int(np.argmin(np.abs(w_total - e)))
        if abs(w_total[j] - e) > 1e-6:
            raise RuntimeError(
                f"SA-CASSCF state at {e:.9f} has no model CI match "
                f"(closest {w_total[j]:.9f})")
        picked.append(j)
    if len(set(picked)) != len(picked):
        raise RuntimeError(f"duplicate CI match: {picked}")

    civs = np.array([V[:, j] for j in picked])
    # Deterministic sign: among the coefficients tied with the largest
    # magnitude (to round-off), the one with the lowest index is made
    # positive.  Symmetry makes some coefficients exactly equal in
    # magnitude (the two open-shell components of a singlet, for
    # example), so a plain argmax would let the last bits of the
    # diagonalization decide the overall sign of the vector.
    for i, ci in enumerate(civs):
        mag = np.abs(ci)
        tied = mag >= mag.max() - 1e-10 * mag.max()
        first = int(np.argmax(tied))
        if ci[first] < 0:
            civs[i] = -ci

    return {
        'mol': mol,
        'mc': mc,
        'ncore': ncore,
        'nact': nact,
        'nele': nele,
        'nmo': h1_mo.shape[0],
        'h1_mo': h1_mo,
        'eri_mo': eri_mo,
        'dets': dets,
        'H_model': H,
        'det_index': {d: i for i, d in enumerate(dets)},
        'civs': civs,
        'casci_energies': np.array([w[j] for j in picked]),
        'sa_energies': sa_energies,
        'e_nuc': float(mol.energy_nuc()),
        'ndet': len(dets),
    }
