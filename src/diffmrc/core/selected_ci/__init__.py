"""Selected configuration interaction: HCI and ASCI in a determinant basis.

PURPOSE
-------
Provides two determinant-space selectors on a common active-space
Hamiltonian:

* ``HCI`` - heat-bath CI (Holmes, Tubman & Umrigar, J. Chem. Theory
  Comput. 12, 3674 (2016));
* ``ASCI`` - adaptive sampling CI (Levine et al., J. Chem. Theory
  Comput. 16, 2340 (2020)).

Both build a selected determinant space, diagonalize it densely, and
optionally add an Epstein-Nesbet second-order correction.  The package
is a numerical kernel, not a differentiable JAX path.

PUBLIC INTERFACE
----------------
``ActiveSpaceHamiltonian``, ``enumerate_determinants``, ``HCI``, ``ASCI``.

The pyscf NEVPT2 bridge lives in
``diffmrc.core.selected_ci.nevpt2_interface`` (``embed_ci_vector``,
``truncate_wavefunction``, ``selected_ci_nevpt2``).  It requires pyscf
and is deliberately not imported here, so importing this package stays
numpy-only.

CONVENTIONS
-----------
* A determinant is ``(amask, bmask)`` over spatial orbitals; alpha spin
  orbitals come first, then beta.
* The Hamiltonian takes active-space integrals in chemist notation
  ``(pq|rs)``; it is the same object in both selectors and in the
  ``pyscf_iface`` counterparts of the other subpackages.
* Energies are in Eh.

VALIDATED SCOPE
---------------
H2O/cc-pVDZ CAS(8,6) and N2/6-31G CAS(6,6): full-space HCI and ASCI
return the CASCI energy to 1e-12 Eh, the HCI eps1 scan reproduces the
frozen prototype selection counts and energy deviations, the pyscf
NEVPT2 bridge matches the native call to 1e-12 Eh, and the documented 1%
stopping-rule defect on N2 is reproduced exactly
(tests/core/test_selected_ci.py).

BOUNDARIES
----------
* Ground state only; no symmetry adaptation, no excited states, no
  degeneracy handling.
* Dense diagonalization limits the selected space to a few thousand
  determinants.
* Selection counts depend on the orbital basis; use ``new_fraction=0``
  for limit checks.
"""
from .asci import ASCI
from .determinants import ActiveSpaceHamiltonian, enumerate_determinants
from .hci import HCI

__all__ = ['ActiveSpaceHamiltonian', 'enumerate_determinants', 'HCI', 'ASCI']
