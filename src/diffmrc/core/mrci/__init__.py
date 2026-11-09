"""Multireference configuration interaction in a determinant basis.

PURPOSE
-------
Determinant-basis MRCISD with optional internal contraction of the
doubly-external configurations, following Werner & Knowles,
J. Chem. Phys. 1988, 89, 5803, plus the Langhoff-Davidson a posteriori
correction (Int. J. Quantum Chem. 1974, 8, 61).

The package is a self-contained numerical kernel: it takes active-space
one- and two-electron integrals and returns CI energies, vectors and
spin diagnostics.  It is not a differentiable JAX path.

PUBLIC INTERFACE
----------------
Space and Hamiltonian:
    ``build_space``, ``build_hamiltonian``, ``dets_with_ms``,
    ``s2_expectation``, ``active_pairs``, ``external_pairs``,
    ``reference_vector``.
Internally contracted MRCI:
    ``pair_function``, ``build_contracted_matrix``, ``orthonormalize``,
    ``solve_mrci``, ``MRCIResult``.
Correction and workflow:
    ``davidson_correction``, ``run_mrci``.

The pyscf reference builder lives in
``diffmrc.core.mrci.pyscf_iface`` (``build_reference``, ``Reference``).
It requires pyscf and is deliberately not imported here, so importing
this package stays numpy-only.

CONVENTIONS
-----------
* Determinants are Python int bitmasks over 2*norb spin orbitals, alpha
  block first.
* Integrals are in the MO basis of the CI window with the frozen core
  folded in; two-electron integrals use chemist notation (pq|rs).
* Energies are in Eh and include the frozen-core energy and nuclear
  repulsion when the reference comes from ``pyscf_iface``.

VALIDATED SCOPE
---------------
H2O/6-31G CAS(2,2) and CH2/STO-3G CAS(4,4) reproduce pyscf FCI to
1e-12 Eh; CH2/6-31G CAS(4,4) reproduces the frozen prototype values
(uncontracted MRCISD 9.147e-4 Eh above FCI, contraction error
1.028e-4 Eh); the H2...H2 and He...He size-consistency deviations match
the prototype within 5e-5 Eh (tests/core/test_mrci.py).

BOUNDARIES
----------
* Single reference root (lowest CASCI root), M_S = 0 only.
* Dense Hamiltonian: memory 8 N^2 bytes, so the practical ceiling is of
  order 1e4 determinants.
* No point-group symmetry, no spin-adapted CSF basis, no analytic
  gradients, no state-averaged reference.
* The direct-CI residual algorithm of WK88 is not implemented, so the
  reduced-scaling property of the original method does not carry over.
"""

from .davidson import davidson_correction
from .determinants import (
    annihilate,
    apply_single,
    build_hamiltonian,
    create,
    dets_with_ms,
    s2_expectation,
)
from .icmrci import (
    MRCIResult,
    build_contracted_matrix,
    orthonormalize,
    pair_function,
    solve_mrci,
)
from .space import (
    active_pairs,
    build_space,
    external_pairs,
    reference_vector,
)
from .workflow import run_mrci

__all__ = [
    "MRCIResult",
    "active_pairs",
    "annihilate",
    "apply_single",
    "build_contracted_matrix",
    "build_hamiltonian",
    "build_space",
    "create",
    "davidson_correction",
    "dets_with_ms",
    "external_pairs",
    "orthonormalize",
    "pair_function",
    "reference_vector",
    "run_mrci",
    "s2_expectation",
    "solve_mrci",
]
