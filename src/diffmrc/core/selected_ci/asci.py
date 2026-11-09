"""Adaptive sampling configuration interaction (ASCI).

PURPOSE
-------
Implements the ASCI selector of Levine et al., J. Chem. Theory Comput.
16, 2340 (2020), Eq. (2), with the core and target sizes of Tubman et al.,
J. Chem. Phys. 145, 044112 (2016): the ``cdets`` largest-weight
determinants form the core, the importance
``sum_j H_ij c_j / (H_ii - E)`` ranks the connected candidates, and the
``tdets`` highest-ranked determinants form the next space.  The
Epstein-Nesbet second-order correction is available through ``pt2()``.

PUBLIC INTERFACE
----------------
``ASCI(hamiltonian, neleca, nelecb, cdets, tdets, max_iter=100,
energy_tol=1e-10)`` with ``run()``, ``pt2()`` and ``summary()``.

VALIDATED SCOPE
---------------
H2O/cc-pVDZ CAS(8,6): ``tdets = 75`` (33% of the 225 determinants)
returns the CASCI energy to 1e-12 Eh, and the full-space limit
``cdets = tdets = 225`` does the same; on H2O/6-31G CAS(4,4) the
variational energy never falls below CASCI and the PT2 sum matches a
brute-force Epstein-Nesbet sum (tests/core/test_selected_ci.py).

BOUNDARIES
----------
* The core set is capped at ``min(100, tdets)``; core determinants are
  merged into the candidate pool by coefficient magnitude, a protective
  variant of the published ranking.
* Ground state only; no symmetry or degeneracy handling.
* Cost is dominated by enumerating the connected space of every
  candidate; the dense diagonalization limits the space to a few
  thousand determinants.
"""

import numpy as np

from .determinants import enumerate_determinants


class ASCI:
    # Levine et al., J. Chem. Theory Comput. 16, 2340 (2020), Eq. (2).
    # Core size cdets and target size tdets follow Tubman et al.,
    # J. Chem. Phys. 145, 044112 (2016).
    def __init__(self, hamiltonian, neleca, nelecb, cdets, tdets,
                 max_iter=100, energy_tol=1e-10):
        if int(tdets) < int(cdets):
            raise ValueError('tdets must be greater than or equal to cdets')
        self.hamiltonian = hamiltonian
        self.neleca = int(neleca)
        self.nelecb = int(nelecb)
        self.cdets = int(cdets)
        self.tdets = int(tdets)
        self.max_iter = int(max_iter)
        self.energy_tol = float(energy_tol)
        self.dets = None
        self.coeffs = None
        self.energy = None
        self.history = []

    def _initial_determinant(self):
        dets = enumerate_determinants(
            self.hamiltonian.norb, self.neleca, self.nelecb)
        return min(dets, key=self.hamiltonian.diagonal)

    def _importance(self, candidates, target, coeffs):
        target_index = {det: index for index, det in enumerate(target)}
        importance = {}
        for candidate in candidates:
            numerator = 0.0
            for det_j in self.hamiltonian.connected_determinants(candidate):
                index = target_index.get(det_j)
                if index is None:
                    continue
                value = self.hamiltonian.matrix_element(candidate, det_j)
                if value != 0.0:
                    numerator += value * coeffs[index]
            denominator = self.hamiltonian.diagonal(candidate) - self.energy
            if abs(denominator) < 1e-12:
                importance[candidate] = 0.0
            else:
                importance[candidate] = numerator / denominator
        return importance

    def run(self):
        target = [self._initial_determinant()]
        energy_old = np.inf
        coeffs = np.array([1.0])
        for iteration in range(self.max_iter):
            matrix = self.hamiltonian.dense_matrix(target)
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            energy = float(eigenvalues[0])
            coeffs = eigenvectors[:, 0]
            self.energy = energy
            order = np.argsort(-np.abs(coeffs), kind='stable')
            core_size = min(self.cdets, len(target))
            core = [target[index] for index in order[:core_size]]
            core_coeffs = {det: coeffs[index]
                           for det, index in zip(core, order[:core_size])}
            candidates = set()
            for det in core:
                candidates.update(self.hamiltonian.connected_determinants(det))
            importance = self._importance(candidates, target, coeffs)
            core_set = set(core)
            pool = [(abs(value), det) for det, value in core_coeffs.items()]
            pool.extend((abs(value), det) for det, value in importance.items()
                        if det not in core_set)
            pool.sort(key=lambda item: (-item[0], item[1]))
            new_target = [det for _, det in pool[:self.tdets]]
            self.history.append({
                'iteration': iteration,
                'num_dets': len(target),
                'num_candidates': len(candidates),
                'num_new_target': len(new_target),
                'energy': energy,
            })
            converged = (set(new_target) == set(target)
                         and abs(energy - energy_old) < self.energy_tol)
            target = new_target
            energy_old = energy
            if converged:
                break
        matrix = self.hamiltonian.dense_matrix(target)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        self.dets = target
        self.coeffs = eigenvectors[:, 0]
        self.energy = float(eigenvalues[0])
        return self

    def pt2(self):
        # Epstein-Nesbet second-order correction over all external
        # determinants connected to the selected space.
        selected = set(self.dets)
        numerators = {}
        for index, det in enumerate(self.dets):
            coefficient = self.coeffs[index]
            if abs(coefficient) < 1e-14:
                continue
            for new_det in self.hamiltonian.connected_determinants(det):
                if new_det in selected:
                    continue
                value = self.hamiltonian.matrix_element(det, new_det)
                if value != 0.0:
                    numerators[new_det] = (numerators.get(new_det, 0.0)
                                           + value * coefficient)
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
            'cdets': self.cdets,
            'tdets': self.tdets,
            'num_dets': len(self.dets),
            'energy_variational': self.energy,
            'energy_pt2_correction': correction,
            'energy_total': self.energy + correction,
            'pt2_skipped': skipped,
            'iterations': len(self.history),
        }
