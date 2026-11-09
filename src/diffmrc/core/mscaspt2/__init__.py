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
    ``reference_fock_matrix``, ``union_function_vectors``,
    ``coupled_reference_system``, ``solve_coupled_amplitudes``,
    ``solve_states``, ``first_order_vectors``, ``coupling_matrix``,
    ``build_heff``, ``solve_ms``, ``solve_xms``, ``random_unitary``.

The benchmark helpers ``build_system`` and ``transform_integrals`` live in
``diffmrc.core.mscaspt2.bench`` and are not re-exported here; that module
imports pyscf lazily.

VALIDATED SCOPE AND BOUNDARIES
------------------------------
See the module docstrings of ``model`` and ``ms``.  In short: H2O
CAS(2,2)/6-31G runs with the S13 eq. 1 reference-space coupling (union
MS-MR contraction), rotation invariance of the coupled MS-MR and XMS
spectra (at most 3e-12 eV over four random reference rotations), the
legacy diagonal-projection MS-MR contrast (0.28 to 1.31 eV), Hermiticity
at or below 6.1e-15 Eh, and a bitwise single-state limit against the
contracted CASPT2 solver.  SS-SR is not rotation invariant.  The coupled
assembly is sparse in the determinant representation but still
diagonalizes the union overlap densely, so it is limited to small active
spaces.
"""
from .model import (ci_1rdm, dets_to_so, embed_density, model_hamiltonian,
                    state_fock, sz0_dets, transition_1rdm)
from .ms import (averaged_fock, build_heff, coupled_reference_system,
                 coupling_matrix, first_order_vectors, random_unitary,
                 reference_fock_matrix, solve_coupled_amplitudes, solve_ms,
                 solve_states, solve_xms, state_averaged_density,
                 state_focks, union_function_vectors)

__all__ = [
    'sz0_dets', 'dets_to_so', 'model_hamiltonian',
    'transition_1rdm', 'ci_1rdm', 'state_fock', 'embed_density',
    'state_averaged_density', 'state_focks', 'averaged_fock',
    'reference_fock_matrix', 'union_function_vectors',
    'coupled_reference_system', 'solve_coupled_amplitudes',
    'solve_states', 'first_order_vectors', 'coupling_matrix',
    'build_heff', 'solve_ms', 'solve_xms', 'random_unitary',
]
