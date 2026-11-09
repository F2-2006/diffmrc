"""Heat-bath configuration interaction (HCI).

PURPOSE
-------
Implements the heat-bath CI selector of Holmes, Tubman and Umrigar,
J. Chem. Theory Comput. 12, 3674 (2016): determinants are added when
``|H_ki c_i|`` exceeds ``eps1`` (the heat-bath threshold is
``eps1 / |c_i|``), the selected space is diagonalized, and the loop stops
when fewer than ``new_fraction`` times the current space size are added.
The second-order Epstein-Nesbet correction is summed over external
determinants with threshold ``eps2`` (Holmes et al. Eq. 9).

PUBLIC INTERFACE
----------------
``HCI(hamiltonian, neleca, nelecb, eps1, eps2=None, max_iter=200,
new_fraction=0.01)`` with ``run()``, ``pt2(eps2=None)`` and
``summary()``.  After ``run()`` the attributes ``dets``, ``coeffs`` and
``energy`` describe the final selected space; ``dets`` and ``coeffs`` are
in matching order, which is not the enumeration order.

VALIDATED SCOPE
---------------
H2O/cc-pVDZ CAS(8,6): the eps1 scan reproduces the frozen prototype
points (eps1 = 1e-2/1e-3/3e-4/1e-4/3e-5 selects 20/46/59/63/65
determinants); with ``eps1=0`` and ``new_fraction=0`` the full space
returns the CASCI energy; the NEVPT2 bridge agrees with native pyscf
NEVPT2 to 1e-12 Eh.  N2/6-31G CAS(6,6) reproduces the documented 1%
stopping-rule defect (tests/core/test_selected_ci.py).

BOUNDARIES
----------
* The initial determinant is the lowest-diagonal determinant, not
  necessarily the RHF determinant.
* The default 1% stopping rule has integer-resolution failure modes on
  spaces of order 100 determinants; use ``new_fraction=0`` for limit
  checks, at the cost of running ``max_iter`` iterations.
* With ``new_fraction=0`` there is no "no new determinants" exit.
* Ground state only; no symmetry or degeneracy handling.
* The selection depends on the orbital basis through the integral
  magnitudes; selected counts are not basis-independent.
"""

import numpy as np

from .determinants import enumerate_determinants


class HCI:
    # Holmes, Tubman, Umrigar, J. Chem. Theory Comput. 12, 3674 (2016).
    # eps1 controls the variational selection, eps2 the perturbative sum.
    def __init__(self, hamiltonian, neleca, nelecb, eps1, eps2=None,
                 max_iter=200, new_fraction=0.01):
        self.hamiltonian = hamiltonian
        self.neleca = int(neleca)
        self.nelecb = int(nelecb)
        self.eps1 = float(eps1)
        self.eps2 = float(eps1 if eps2 is None else eps2)
        self.max_iter = int(max_iter)
        self.new_fraction = float(new_fraction)
        self.dets = None
        self.coeffs = None
        self.energy = None
        self.history = []

    def _initial_determinant(self):
        dets = enumerate_determinants(
            self.hamiltonian.norb, self.neleca, self.nelecb)
        return min(dets, key=self.hamiltonian.diagonal)

    def run(self):
        dets = [self._initial_determinant()]
        for iteration in range(self.max_iter):
            matrix = self.hamiltonian.dense_matrix(dets)
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            energy = float(eigenvalues[0])
            coeffs = eigenvectors[:, 0]
            selected = set(dets)
            new_dets = set()
            for index, det in enumerate(dets):
                coefficient = coeffs[index]
                if abs(coefficient) < 1e-14:
                    continue
                threshold = self.eps1 / abs(coefficient)
                for new_det, _ in self.hamiltonian.heat_bath_connected(det, threshold):
                    if new_det not in selected:
                        new_dets.add(new_det)
            self.history.append({
                'iteration': iteration,
                'num_dets': len(dets),
                'energy': energy,
                'num_new': len(new_dets),
            })
            if len(new_dets) < self.new_fraction * len(dets):
                break
            dets.extend(sorted(new_dets))
        self.dets = dets
        self.coeffs = coeffs
        self.energy = energy
        return self

    def pt2(self, eps2=None):
        # Eq. (9) of Holmes et al.: sum over external determinants of
        # (sum_i H_ki c_i)^2 / (E0 - H_kk), with |H_ki c_i| >= eps2.
        if self.dets is None:
            self.run()
        threshold = self.eps2 if eps2 is None else float(eps2)
        selected = set(self.dets)
        numerators = {}
        for index, det in enumerate(self.dets):
            coefficient = self.coeffs[index]
            if abs(coefficient) < 1e-14:
                continue
            local_threshold = threshold / abs(coefficient)
            for new_det, value in self.hamiltonian.heat_bath_connected(
                    det, local_threshold):
                if new_det in selected:
                    continue
                numerators[new_det] = numerators.get(new_det, 0.0) + value * coefficient
        correction = 0.0
        skipped = 0
        for det, numerator in numerators.items():
            denominator = self.energy - self.hamiltonian.diagonal(det)
            if abs(denominator) < 1e-12:
                skipped += 1
                continue
            correction += numerator * numerator / denominator
        return correction, skipped

    def summary(self):
        correction, skipped = self.pt2()
        return {
            'eps1': self.eps1,
            'eps2': self.eps2,
            'num_dets': len(self.dets),
            'energy_variational': self.energy,
            'energy_pt2_correction': correction,
            'energy_total': self.energy + correction,
            'pt2_skipped': skipped,
            'iterations': len(self.history),
        }
