"""Bridge from selected-CI wavefunctions to pyscf NEVPT2.

PURPOSE
-------
Embeds a determinant list and its coefficients into the pyscf alpha-major
CI matrix, optionally truncates the wavefunction by coefficient weight
(Guo et al., J. Chem. Phys. 162, 144110 (2025), Eqs. 8-9), and runs
``pyscf.mrpt.nevpt2.NEVPT`` on the resulting reference.

DEPENDENCY
----------
This module requires pyscf and is not part of the JAX-differentiable
path: it wraps pyscf's NEVPT2 implementation.  It is not imported by
``diffmrc.core.selected_ci.__init__``, so importing the selected-CI
package stays numpy-only.

PUBLIC INTERFACE
----------------
``embed_ci_vector``, ``truncate_wavefunction``, ``selected_ci_nevpt2``.

CONVENTIONS
-----------
* Unselected determinants have zero coefficients in the CI vector passed
  to NEVPT2.
* ``truncate_wavefunction`` returns the expectation value of the
  truncated and renormalized wavefunction, not a re-diagonalized
  eigenvalue.
* The pyscf ``mc`` object must carry canonicalized orbitals
  (``canonicalized=True``) for a meaningful NEVPT2 result.

VALIDATED SCOPE
---------------
H2O/cc-pVDZ CAS(8,6): the full-space bridge reproduces a direct
``NEVPT(mc).kernel()`` call to 1e-12 Eh and the embedded vector
reproduces ``mc.ci`` elementwise (tests/core/test_selected_ci.py).

BOUNDARIES
----------
* Standard pyscf NEVPT2 only; FR-NEVPT2 is not implemented.
* Accuracy under strong truncation is not guaranteed and is reported as
  a measured deviation in the prototype report.
"""

import numpy as np
from pyscf.fci import cistring
from pyscf.mrpt.nevpt2 import NEVPT


def embed_ci_vector(dets, coeffs, norb, neleca, nelecb):
    # Map bitmask determinants onto the PySCF alpha-major CI vector layout.
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci = np.zeros((na, nb))
    for det, coefficient in zip(dets, coeffs):
        addra = cistring.str2addr(norb, neleca, det[0])
        addrb = cistring.str2addr(norb, nelecb, det[1])
        ci[addra, addrb] += coefficient
    return ci


def truncate_wavefunction(hamiltonian, dets, coeffs, tvar):
    # Eq. (8) of Guo et al.: keep configurations with T_I^2 > TVAR.
    # Eq. (9): reference energy of the truncated, renormalized wavefunction.
    kept = [(det, coefficient) for det, coefficient in zip(dets, coeffs)
            if coefficient * coefficient > tvar]
    kept_dets = [item[0] for item in kept]
    kept_coeffs = np.array([item[1] for item in kept])
    norm = np.linalg.norm(kept_coeffs)
    kept_coeffs = kept_coeffs / norm
    matrix = hamiltonian.dense_matrix(kept_dets)
    energy = float(kept_coeffs @ matrix @ kept_coeffs)
    return kept_dets, kept_coeffs, energy


def selected_ci_nevpt2(mc, dets, coeffs, e_variational, canonicalized=True):
    # The selected determinants define the reference space; unselected
    # coefficients are zero in the CI vector passed to NEVPT2.
    norb = mc.ncas
    neleca, nelecb = mc.nelecas
    ci = embed_ci_vector(dets, coeffs, norb, neleca, nelecb)
    mc.ci = ci
    solver = NEVPT(mc)
    solver.canonicalized = canonicalized
    e_corr = solver.kernel()
    return {
        'e_corr': float(e_corr),
        'e_variational': float(e_variational),
        'e_total': float(e_variational + e_corr),
        'ci_shape': ci.shape,
    }
