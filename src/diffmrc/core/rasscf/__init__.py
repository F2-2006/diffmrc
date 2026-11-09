"""RASSCF: restricted-active-space CI with orbital optimization.

Determinant-basis RAS-CI, its Slater-Condon matrix elements, the spin-summed
density matrices and the analytic orbital gradient, on the project's
numpy/scipy stack.

Scope: RASSCF only, with the occupation constraints of Malmqvist,
J. Chem. Phys. 129 (2008) 164106.  RASPT2 dynamic correlation, FOCIS
construction, state averaging, spin adaptation and spin-orbit coupling are
not implemented.
"""

from .determinants import (
    apply_sequence,
    build_hamiltonian,
    enumerate_active_determinants,
    enumerate_determinants,
    hamiltonian_element,
    occupation,
    one_rdm,
    popcount,
    ras_allowed,
    ras_counts,
    solve_ci,
    two_rdm,
)
from .mcscf import RASSCF

__all__ = [
    "RASSCF",
    "apply_sequence",
    "build_hamiltonian",
    "enumerate_active_determinants",
    "enumerate_determinants",
    "hamiltonian_element",
    "occupation",
    "one_rdm",
    "popcount",
    "ras_allowed",
    "ras_counts",
    "solve_ci",
    "two_rdm",
]
