"""Determinant space and Hamiltonian for selected CI.

PURPOSE
-------
Provides the determinant representation and the active-space Hamiltonian
used by the HCI and ASCI selectors: enumeration of the alpha/beta string
pairs, Slater-Condon matrix elements, the diagonal, the single and double
connection lists, and an optional dense-matrix cache.

CONVENTIONS
-----------
* A determinant is a pair ``(amask, bmask)`` of spatial-orbital bitmasks.
  Spin orbitals are ordered alpha 0..norb-1 then beta norb..2*norb-1.
* Two-electron integrals use chemist notation, ``h2e[p, q, r, s] =
  (pq|rs)``.
* ``enable_cache`` materializes the full matrix over a determinant list;
  the cache returns exactly the same values as the on-the-fly path and is
  a pure performance switch.

VALIDATED SCOPE
---------------
The dense matrix agrees elementwise with pyscf
``fci.direct_spin1.pspace`` for H2O/6-31G CAS(4,4) (36 determinants) and
N2/6-31G CAS(6,6) (400 determinants) to 1e-10, and the lowest eigenvalue
equals the pyscf CASCI energy (tests/core/test_selected_ci.py).

BOUNDARIES
----------
* The dense matrix is N x N float64; N = 1e4 is about 0.8 GB, so the
  practical ceiling is a few thousand determinants.
* No symmetry adaptation and no spin adaptation; only the alpha and beta
  electron counts are fixed.
* ``build_double_lists`` scales as O((2*norb)^4) entries and is intended
  for small active spaces.
"""

import itertools

import numpy as np


def popcount(value):
    return int(value).bit_count()


def parity_below(occupied, orbital):
    # Sign of a_q acting on a determinant whose occupied spin orbitals are `occupied`.
    return -1.0 if sum(1 for p in occupied if p < orbital) % 2 else 1.0


def combination_masks(norb, nelec):
    return [sum(1 << i for i in comb)
            for comb in itertools.combinations(range(norb), nelec)]


def enumerate_determinants(norb, neleca, nelecb):
    amasks = combination_masks(norb, neleca)
    bmasks = combination_masks(norb, nelecb)
    return [(a, b) for a in amasks for b in bmasks]


def occupied_spin_orbitals(det, norb):
    amask, bmask = det
    occupied = [i for i in range(norb) if (amask >> i) & 1]
    occupied.extend(norb + i for i in range(norb) if (bmask >> i) & 1)
    return occupied


def det_with_excitation(det, norb, removed, added):
    amask, bmask = det
    for orbital in removed:
        if orbital < norb:
            amask &= ~(1 << orbital)
        else:
            bmask &= ~(1 << (orbital - norb))
    for orbital in added:
        if orbital < norb:
            amask |= 1 << orbital
        else:
            bmask |= 1 << (orbital - norb)
    return (amask, bmask)


class ActiveSpaceHamiltonian:
    # Spin orbital ordering: alpha orbitals 0..norb-1, beta orbitals norb..2*norb-1.
    # Two-electron integrals use chemists' notation h2e[p,q,r,s] = (pq|rs).
    def __init__(self, h1e, h2e, norb):
        self.norb = int(norb)
        self.h1e = np.asarray(h1e, dtype=float)
        self.h2e = np.asarray(h2e, dtype=float).reshape(
            self.norb, self.norb, self.norb, self.norb)
        self._double_lists = None
        self._cache_matrix = None
        self._cache_index = None

    def enable_cache(self, dets):
        dets = list(dets)
        matrix = np.zeros((len(dets), len(dets)))
        for i, det_i in enumerate(dets):
            for j in range(i + 1):
                value = self.matrix_element(det_i, dets[j])
                matrix[i, j] = value
                matrix[j, i] = value
        self._cache_index = {det: index for index, det in enumerate(dets)}
        self._cache_matrix = matrix
        return self

    def _unpack(self, orbital):
        if orbital < self.norb:
            return orbital, 0
        return orbital - self.norb, 1

    def h1_spin(self, p, q):
        pi, sp = self._unpack(p)
        qi, sq = self._unpack(q)
        if sp != sq:
            return 0.0
        return self.h1e[pi, qi]

    def h2_spin(self, p, q, r, s):
        pi, sp = self._unpack(p)
        qi, sq = self._unpack(q)
        ri, sr = self._unpack(r)
        si, ss = self._unpack(s)
        if sp != sq or sr != ss:
            return 0.0
        return self.h2e[pi, qi, ri, si]

    def diagonal(self, det):
        if self._cache_matrix is not None:
            index = self._cache_index.get(det)
            if index is not None:
                return self._cache_matrix[index, index]
        occupied = occupied_spin_orbitals(det, self.norb)
        energy = 0.0
        for p in occupied:
            energy += self.h1_spin(p, p)
        for p in occupied:
            for q in occupied:
                energy += 0.5 * (self.h2_spin(p, p, q, q)
                                 - self.h2_spin(p, q, q, p))
        return energy

    def matrix_element(self, det_i, det_j):
        if self._cache_matrix is not None:
            index_i = self._cache_index.get(det_i)
            index_j = self._cache_index.get(det_j)
            if index_i is not None and index_j is not None:
                return self._cache_matrix[index_i, index_j]
        occ_i = set(occupied_spin_orbitals(det_i, self.norb))
        occ_j = set(occupied_spin_orbitals(det_j, self.norb))
        removed = sorted(occ_i - occ_j)
        added = sorted(occ_j - occ_i)
        if not removed and not added:
            return self.diagonal(det_i)
        if len(removed) == 1 and len(added) == 1:
            q = removed[0]
            p = added[0]
            value = self.h1_spin(p, q)
            for spectator in occ_i & occ_j:
                value += (self.h2_spin(p, q, spectator, spectator)
                          - self.h2_spin(p, spectator, spectator, q))
            sign = parity_below(occ_i, q)
            sign *= parity_below(occ_i - {q}, p)
            return sign * value
        if len(removed) == 2 and len(added) == 2:
            q1, q2 = removed
            p1, p2 = added
            sign = parity_below(occ_i, q1)
            occ_1 = occ_i - {q1}
            sign *= parity_below(occ_1, q2)
            occ_2 = occ_1 - {q2}
            sign *= parity_below(occ_2, p2)
            occ_3 = occ_2 | {p2}
            sign *= parity_below(occ_3, p1)
            value = (self.h2_spin(p1, q1, p2, q2)
                     - self.h2_spin(p1, q2, p2, q1))
            return sign * value
        return 0.0

    def double_magnitude(self, p, q, r, s):
        # |<D(rs)|H|D(pq)>| for a double excitation p,q -> r,s.
        return abs(self.h2_spin(r, p, s, q) - self.h2_spin(r, q, s, p))

    def build_double_lists(self):
        nspin = 2 * self.norb
        lists = {}
        for p in range(nspin):
            for q in range(p + 1, nspin):
                entries = []
                for r in range(nspin):
                    if r == p or r == q:
                        continue
                    for s in range(r + 1, nspin):
                        if s == p or s == q:
                            continue
                        magnitude = self.double_magnitude(p, q, r, s)
                        if magnitude > 0.0:
                            entries.append((r, s, magnitude))
                entries.sort(key=lambda item: item[2], reverse=True)
                lists[(p, q)] = entries
        self._double_lists = lists
        return lists

    def heat_bath_connected(self, det, eps):
        if self._double_lists is None:
            self.build_double_lists()
        occupied = set(occupied_spin_orbitals(det, self.norb))
        occupied_list = sorted(occupied)
        for index, p in enumerate(occupied_list):
            for q in occupied_list[index + 1:]:
                for r, s, magnitude in self._double_lists[(p, q)]:
                    if magnitude < eps:
                        break
                    if r in occupied or s in occupied:
                        continue
                    new_det = det_with_excitation(det, self.norb, (p, q), (r, s))
                    yield new_det, self.matrix_element(det, new_det)
        for p in occupied_list:
            for r in range(2 * self.norb):
                if r in occupied:
                    continue
                if (p < self.norb) != (r < self.norb):
                    continue
                new_det = det_with_excitation(det, self.norb, (p,), (r,))
                value = self.matrix_element(det, new_det)
                if abs(value) > 0.0 and abs(value) >= eps:
                    yield new_det, value

    def connected_determinants(self, det):
        # All single and double excitations, without an energy threshold.
        occupied = set(occupied_spin_orbitals(det, self.norb))
        occupied_list = sorted(occupied)
        virtual = [p for p in range(2 * self.norb) if p not in occupied]
        nalpha = det[0].bit_count()
        nbeta = det[1].bit_count()
        connected = set()
        for p in occupied_list:
            for r in virtual:
                new_det = det_with_excitation(det, self.norb, (p,), (r,))
                if (new_det[0].bit_count() == nalpha
                        and new_det[1].bit_count() == nbeta):
                    connected.add(new_det)
        for index, p in enumerate(occupied_list):
            for q in occupied_list[index + 1:]:
                for index_r, r in enumerate(virtual):
                    for s in virtual[index_r + 1:]:
                        new_det = det_with_excitation(
                            det, self.norb, (p, q), (r, s))
                        if (new_det[0].bit_count() == nalpha
                                and new_det[1].bit_count() == nbeta):
                            connected.add(new_det)
        return connected

    def dense_matrix(self, dets):
        if self._cache_matrix is not None:
            index = [self._cache_index[det] for det in dets]
            return self._cache_matrix[np.ix_(index, index)]
        size = len(dets)
        matrix = np.zeros((size, size))
        for i, det_i in enumerate(dets):
            matrix[i, i] = self.diagonal(det_i)
            for j in range(i):
                value = self.matrix_element(det_i, dets[j])
                matrix[i, j] = value
                matrix[j, i] = value
        return matrix
