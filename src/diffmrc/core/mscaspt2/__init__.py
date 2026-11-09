"""Multistate CASPT2 on a determinant model space (MS / XMS).

PURPOSE
-------
Multistate second-order perturbation theory for a set of state-averaged
CASSCF reference states: the effective Hamiltonian is built from the
model-space Hamiltonian plus the symmetrized first-order coupling, with
the zeroth-order operator selectable between the state-averaged Fock
operator (MS-CASPT2, XMS-CASPT2) and the state-specific one (SS-SR).

PUBLIC INTERFACE
----------------
Model-space algebra (``diffmrc.core.mscaspt2.model``):
    ``sz0_dets``, ``dets_to_so``, ``model_hamiltonian``,
    ``transition_1rdm``, ``ci_1rdm``, ``state_fock``, ``embed_density``.
Solvers (``diffmrc.core.mscaspt2.ms``):
    ``state_averaged_density``, ``state_focks``, ``averaged_fock``,
    ``reference_fock_matrix``, ``solve_states``, ``first_order_vectors``,
    ``coupling_matrix``, ``build_heff``, ``solve_ms``, ``solve_xms``,
    ``random_unitary``.

The benchmark helpers ``build_system`` and ``transform_integrals`` live in
``diffmrc.core.mscaspt2.bench`` and are not re-exported here; that module
imports pyscf lazily.

VALIDATED SCOPE AND BOUNDARIES
------------------------------
See the module docstrings of ``model`` and ``ms``.  In short: H2O
CAS(2,2)/6-31G three-scheme runs, XMS rotation invariance (3.9e-13 to
4.1e-11 eV in the development run, a round-off floor of the total
energy), Hermiticity at or below 6.1e-15 Eh, and a bitwise
single-state limit against the contracted CASPT2 solver.  The MS-MR path
uses a state-averaged-Fock zeroth order with per-state contraction (no
reference-space coupling term), SS-SR is not rotation invariant, and the
contracted assembly is limited to small active spaces.
"""
from .model import (ci_1rdm, dets_to_so, embed_density, model_hamiltonian,
                    state_fock, sz0_dets, transition_1rdm)
from .ms import (averaged_fock, build_heff, coupling_matrix,
                 first_order_vectors, random_unitary, reference_fock_matrix,
                 solve_ms, solve_states, solve_xms, state_averaged_density,
                 state_focks)

__all__ = [
    'sz0_dets', 'dets_to_so', 'model_hamiltonian',
    'transition_1rdm', 'ci_1rdm', 'state_fock', 'embed_density',
    'state_averaged_density', 'state_focks', 'averaged_fock',
    'reference_fock_matrix', 'solve_states', 'first_order_vectors',
    'coupling_matrix', 'build_heff', 'solve_ms', 'solve_xms',
    'random_unitary',
]
