"""Interface between pyscf references and the MRCI kernel.

PURPOSE
-------
Runs RHF/ROHF and CASCI with pyscf and packages the reference orbitals,
active-space integrals, frozen-core energy and CASCI energy into the
``Reference`` record consumed by ``diffmrc.core.mrci.workflow.run_mrci``.
All CI algebra is done in this package.

DEPENDENCY
----------
This module requires pyscf and is not part of the JAX-differentiable
path: it is a reference builder, not a differentiable energy.  It is not
imported by ``diffmrc.core.mrci.__init__``, so importing the MRCI package
itself stays numpy-only.

CONVENTIONS
-----------
* ``h1`` and ``eri`` are in the MO basis of the CASCI orbitals, sliced to
  the non-frozen window; ``eri`` is in chemist notation (pq|rs).
* ``e_core`` contains the frozen-core one- and two-electron contributions
  and the nuclear repulsion, so ``e_core + E(CI)`` is directly comparable
  with pyscf's ``e_tot``.

BOUNDARIES
----------
* Closed-shell RHF for ``spin=0``, ROHF otherwise; no UHF, no CASSCF
  orbital optimization.
* Requires an active-space electron count consistent with the molecular
  electron count and a closed-shell frozen core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import ao2mo, gto, mcscf, scf


@dataclass
class Reference:
    mol: object
    mo_coeff: np.ndarray
    h1: np.ndarray          # effective one-electron integrals in the CI space
    eri: np.ndarray         # (pq|rs) in the CI space
    e_core: float           # frozen-core energy, nuclear repulsion included
    e_casci: float
    ncore: int
    nact: int
    nact_elec: int
    norb_ci: int
    mf_energy: float


def build_reference(atom, basis, nact, nact_elec, spin=0, unit="Angstrom", verbose=0) -> Reference:
    """Run RHF/ROHF + CASCI and return integrals for the CI space."""
    mol = gto.M(atom=atom, basis=basis, unit=unit, spin=spin, verbose=verbose)
    if spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    mf.kernel()

    ncore = (mol.nelectron - nact_elec) // 2
    if ncore < 0 or nact_elec > mol.nelectron:
        raise ValueError("inconsistent active-space electron count")

    casci = mcscf.CASCI(mf, nact, nact_elec)
    casci.verbose = verbose
    e_casci = casci.kernel()[0]
    mo = casci.mo_coeff

    norb = mo.shape[1]
    h1_ao = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    h1_mo = mo.T @ h1_ao @ mo
    eri_mo = ao2mo.full(mol, mo, compact=False).reshape(norb, norb, norb, norb)

    core = list(range(ncore))
    e_core = 0.0
    for c in core:
        e_core += 2.0 * h1_mo[c, c]
    for c in core:
        for d in core:
            e_core += 2.0 * eri_mo[c, c, d, d] - eri_mo[c, d, d, c]
    # Total-energy convention: include the nuclear repulsion so that
    # e_core + E(CI) is directly comparable with pyscf's e_tot.
    e_core += float(mol.energy_nuc())

    # Fold the frozen core into an effective one-electron operator.
    h1_eff = h1_mo.copy()
    for p in range(norb):
        for q in range(norb):
            for c in core:
                h1_eff[p, q] += 2.0 * eri_mo[p, q, c, c] - eri_mo[p, c, c, q]

    sl = slice(ncore, norb)
    h1_ci = h1_eff[sl, sl]
    eri_ci = eri_mo[sl, sl, sl, sl]

    return Reference(
        mol=mol,
        mo_coeff=mo,
        h1=h1_ci,
        eri=eri_ci,
        e_core=e_core,
        e_casci=float(e_casci),
        ncore=ncore,
        nact=nact,
        nact_elec=nact_elec,
        norb_ci=norb - ncore,
        mf_energy=float(mf.e_tot),
    )
