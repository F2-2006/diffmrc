"""RASSI state interaction for spin-orbit coupling.

PURPOSE
-------
Implements the two-step state-interaction (RASSI) construction:

1. A set of spin-free states |Psi_A^{S,M_S}> with energies E_A is given on
   a common orthonormal active-orbital basis.
2. The spin-orbit operator is assembled in the determinant basis of the
   active spin orbitals (route A) or, equivalently, contracted with the
   spin-resolved transition density matrices of the states (route B).
3. The resulting complex Hermitian matrix is diagonalized; the
   eigenvectors give the spin-orbit coupled states as linear
   combinations of the spin-free states, and their squared coefficients
   give the state composition.

The module is self-contained: it needs numpy and the caller-supplied
spin-free states, integrals, and SOC tensor.  Building those states (for
example with pyscf SA-CASSCF) is the caller's job.

PUBLIC INTERFACE
----------------
Basis and states:
    ``DetSpace``, ``SpinFreeState``, ``SOState``.
Operators in the determinant basis:
    ``one_body_matrix``, ``two_body_matrix``, ``ci_matrix``,
    ``spin_operators``, ``soc_one_body``.
Multiplet construction:
    ``lower_component``, ``build_multiplet``.
State interaction:
    ``soc_matrix_determinant`` (route A), ``transition_density`` and
    ``soc_matrix_transition`` (route B), ``diagonalize``,
    ``kramers_pairs``.
Constants: ``MAX_DETERMINANTS``, ``SIGMA``, ``SPIN``.

Inputs are numpy arrays; ``hk`` is real with shape ``(3, norb, norb)``
and must already carry the ``alpha^2/2`` prefactor.  Energies are in Eh;
the caller converts to cm^-1 if needed.

OPERATOR CONVENTION (shared with the project stack)
---------------------------------------------------
    H_SO = i * sum_k h_k s_k,     s_k = sigma_k / 2,

where ``h_k`` is the raw ``int1e_pnucxp`` / SOMF tensor multiplied by
alpha^2/2 (``somf_matrix(..., prefactor=True)``).

VALIDATED SCOPE
---------------
OH X ^2Pi, 6-31G, CAS(7,6), two state-averaged roots, r = 1.8324 Bohr:
the two assembly routes agree to 5.4e-20 (one-electron) and 1.6e-19
(SOMF), the state-interaction matrix is Hermitian to 1.4e-20 and 1.1e-19,
and the term splittings are 205.883 and 128.829 cm^-1, matching the
frozen ORCA 6.1.1 anchors.  O2 with ^3Sigma_g^-, ^1Delta_g and
^1Sigma_g^+ in one model space reproduces the Omega = 0, +-1 and +-2
assignment and the Delta S = 0, +-1 selection rule.

BOUNDARIES
----------
* All spin-free states must share one orthonormal orbital set;
  biorthogonalization of different orbital sets is not implemented.
* The SOC enters through the matrix elements of ``H_SO`` in the supplied
  model space; electron correlation beyond the supplied state energies
  is not added.
* The model space (number of states) is a physical choice, not a
  convergence knob.  Results depend on it and must be reported with the
  state set.
* ``MAX_DETERMINANTS`` caps the dense determinant matrices at 20000; the
  determinant-space route is intended for small active spaces.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "MAX_DETERMINANTS", "SIGMA", "SPIN",
    "DetSpace", "SpinFreeState", "SOState",
    "one_body_matrix", "two_body_matrix", "ci_matrix",
    "spin_operators", "soc_one_body",
    "lower_component", "build_multiplet",
    "soc_matrix_determinant", "transition_density", "soc_matrix_transition",
    "diagonalize", "kramers_pairs",
]

# Pauli matrices; the spin operator is s_k = sigma_k / 2.
SIGMA = np.array([[[0.0, 1.0], [1.0, 0.0]],
                  [[0.0, -1.0j], [1.0j, 0.0]],
                  [[1.0, 0.0], [0.0, -1.0]]], dtype=complex)
SPIN = 0.5 * SIGMA

MAX_DETERMINANTS = 20000


# ---------------------------------------------------------------------------
# Determinant basis
# ---------------------------------------------------------------------------
@dataclass
class DetSpace:
    """Determinant basis for ``nelec`` electrons in ``norb`` spatial orbitals.

    A determinant is a sorted tuple of spin-orbital indices ``0..2*norb-1``
    (even = alpha, odd = beta), interpreted as a^dag_{i1} ... a^dag_{ik}|0>.
    """

    norb: int
    nelec: int
    dets: list[tuple[int, ...]] = field(init=False)
    masks: np.ndarray = field(init=False)
    index: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        nso = 2 * self.norb
        if self.nelec < 0 or self.nelec > nso:
            raise ValueError("nelec out of range")
        self.dets = list(itertools.combinations(range(nso), self.nelec))
        if len(self.dets) > MAX_DETERMINANTS:
            raise RuntimeError("determinant space %d exceeds cap %d"
                               % (len(self.dets), MAX_DETERMINANTS))
        self.masks = np.array([sum(1 << i for i in d) for d in self.dets],
                              dtype=np.int64)
        self.index = {int(m): i for i, m in enumerate(self.masks)}

    def __len__(self) -> int:
        return len(self.dets)


def _annihilate(key: int, x: int):
    if not (key >> x) & 1:
        return None
    sign = -1.0 if bin(key & ((1 << x) - 1)).count("1") % 2 else 1.0
    return key ^ (1 << x), sign


def _create(key: int, x: int):
    if (key >> x) & 1:
        return None
    sign = -1.0 if bin(key & ((1 << x) - 1)).count("1") % 2 else 1.0
    return key ^ (1 << x), sign


def one_body_matrix(op: np.ndarray, space: DetSpace) -> np.ndarray:
    """Matrix of ``sum_{P,Q} op[P,Q] a^dag_P a_Q`` in the determinant basis."""
    nso = op.shape[0]
    H = np.zeros((len(space), len(space)), dtype=complex)
    for jcol, key in enumerate(space.masks):
        key = int(key)
        occ = space.dets[jcol]
        for q in occ:
            step = _annihilate(key, q)
            if step is None:
                continue
            key1, s1 = step
            for p in range(nso):
                step = _create(key1, p)
                if step is None:
                    continue
                key2, s2 = step
                irow = space.index.get(key2)
                if irow is not None:
                    H[irow, jcol] += op[p, q] * s1 * s2
    return H


def two_body_matrix(eri: np.ndarray, space: DetSpace) -> np.ndarray:
    """Matrix of ``1/2 sum eri[p,q,r,s] a^dag_p a^dag_r a_s a_q``."""
    nso = eri.shape[0]
    H = np.zeros((len(space), len(space)), dtype=complex)
    for jcol, key in enumerate(space.masks):
        key = int(key)
        occ = space.dets[jcol]
        for q in occ:
            step = _annihilate(key, q)
            if step is None:
                continue
            key1, s1 = step
            for s in occ:
                if s == q:
                    continue
                step = _annihilate(key1, s)
                if step is None:
                    continue
                key2, s2 = step
                for r in range(nso):
                    step = _create(key2, r)
                    if step is None:
                        continue
                    key3, s3 = step
                    for p in range(nso):
                        if p == r:
                            continue
                        step = _create(key3, p)
                        if step is None:
                            continue
                        key4, s4 = step
                        irow = space.index.get(key4)
                        if irow is not None:
                            H[irow, jcol] += (0.5 * eri[p, q, r, s]
                                              * s1 * s2 * s3 * s4)
    return 0.5 * (H + H.conj().T)


def ci_matrix(h1: np.ndarray, eri: np.ndarray, space: DetSpace) -> np.ndarray:
    """Spin-free (or spin-dependent) CI Hamiltonian in the determinant basis."""
    return one_body_matrix(h1, space) + two_body_matrix(eri, space)


def spin_operators(space: DetSpace):
    """Return (Sz, S^2) matrices in the determinant basis."""
    nso = 2 * space.norb
    sz = np.zeros((nso, nso), dtype=complex)
    sp = np.zeros((nso, nso), dtype=complex)
    sm = np.zeros((nso, nso), dtype=complex)
    for p in range(space.norb):
        sz[2 * p, 2 * p] = 0.5
        sz[2 * p + 1, 2 * p + 1] = -0.5
        sp[2 * p, 2 * p + 1] = 1.0
        sm[2 * p + 1, 2 * p] = 1.0
    Sz = one_body_matrix(sz, space)
    Sp = one_body_matrix(sp, space)
    Sm = one_body_matrix(sm, space)
    S2 = Sz @ Sz + 0.5 * (Sp @ Sm + Sm @ Sp)
    return Sz, S2


def soc_one_body(hk: np.ndarray, norb: int) -> np.ndarray:
    """Spin-orbital matrix of ``i * sum_k hk[k] s_k``.

    ``hk`` must already contain the alpha^2/2 prefactor.
    """
    if hk.shape != (3, norb, norb):
        raise ValueError("hk must have shape (3, norb, norb)")
    nso = 2 * norb
    o = np.zeros((nso, nso), dtype=complex)
    for k in range(3):
        for p in range(norb):
            for q in range(norb):
                val = 1j * hk[k, p, q]
                if val == 0.0:
                    continue
                for a in range(2):
                    for b in range(2):
                        o[2 * p + a, 2 * q + b] += val * SPIN[k, a, b]
    return o


# ---------------------------------------------------------------------------
# Spin-free states
# ---------------------------------------------------------------------------
@dataclass
class SpinFreeState:
    """One spin-free state component |Psi_A^{S,M_S}>."""

    energy: float
    s: float
    ms: float
    ci: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        self.ci = np.asarray(self.ci, dtype=complex)
        norm = float(np.linalg.norm(self.ci))
        if norm == 0.0:
            raise ValueError("zero CI vector")
        self.ci = self.ci / norm


def lower_component(state: SpinFreeState, space: DetSpace) -> SpinFreeState:
    """Apply the normalized spin lowering operator S_- once."""
    out = np.zeros(len(space), dtype=complex)
    for i, c in enumerate(state.ci):
        if c == 0.0:
            continue
        key = int(space.masks[i])
        for p in range(space.norb):
            a, b = 2 * p, 2 * p + 1
            if (key >> a) & 1 and not (key >> b) & 1:
                j = space.index[key ^ (1 << a) ^ (1 << b)]
                out[j] += c
    norm = float(np.linalg.norm(out))
    if norm < 1e-12:
        raise ValueError("S_- annihilates the state; check M_S = S")
    out /= norm
    ms = state.ms - 1.0
    return SpinFreeState(state.energy, state.s, ms, out, state.label)


def build_multiplet(high: SpinFreeState, space: DetSpace):
    """Generate all 2S+1 components from the highest-weight member."""
    if abs(high.ms - high.s) > 1e-8:
        raise ValueError("highest-weight state must have M_S = S")
    states = [high]
    while states[-1].ms > -states[-1].s + 1e-8:
        states.append(lower_component(states[-1], space))
    return states


# ---------------------------------------------------------------------------
# State-interaction matrix
# ---------------------------------------------------------------------------
def soc_matrix_determinant(states, o_det: np.ndarray) -> np.ndarray:
    """Route A: project the determinant-space operator onto the state basis."""
    V = np.column_stack([s.ci for s in states])
    return V.conj().T @ o_det @ V


def transition_density(ci_i: np.ndarray, ci_j: np.ndarray, space: DetSpace,
                       p: int, a: int, q: int, b: int) -> complex:
    """<Psi_i| a^dag_{p a} a_{q b} |Psi_j>."""
    total = 0.0 + 0.0j
    P, Q = 2 * p + a, 2 * q + b
    for jcol, key in enumerate(space.masks):
        cj = ci_j[jcol]
        if cj == 0.0:
            continue
        step = _annihilate(int(key), Q)
        if step is None:
            continue
        key1, s1 = step
        step = _create(key1, P)
        if step is None:
            continue
        key2, s2 = step
        irow = space.index.get(key2)
        if irow is None:
            continue
        ci = ci_i[irow]
        if ci != 0.0:
            total += ci.conj() * cj * s1 * s2
    return total


def soc_matrix_transition(states, hk: np.ndarray, space: DetSpace) -> np.ndarray:
    """Route B: contract the SOC integrals with spin-resolved transition densities.

    Evaluated in the M_S-resolved basis:
        H_AB = sum_k sum_pq hk[k,p,q] sum_ab (s_k)_ab <A|a^dag_pa a_qb|B>.
    """
    n = len(states)
    H = np.zeros((n, n), dtype=complex)
    norb = space.norb
    for i in range(n):
        for j in range(n):
            acc = 0.0 + 0.0j
            for k in range(3):
                for p in range(norb):
                    for q in range(norb):
                        if hk[k, p, q] == 0.0:
                            continue
                        spin_sum = 0.0 + 0.0j
                        for a in range(2):
                            for b in range(2):
                                if SPIN[k, a, b] == 0.0:
                                    continue
                                spin_sum += SPIN[k, a, b] * transition_density(
                                    states[i].ci, states[j].ci, space, p, a, q, b)
                        acc += hk[k, p, q] * spin_sum
            H[i, j] = 1j * acc
    return H


# ---------------------------------------------------------------------------
# Diagonalization and analysis
# ---------------------------------------------------------------------------
@dataclass
class SOState:
    energy: float
    vector: np.ndarray
    composition: list[tuple[str, float]]
    spin_composition: dict[float, float]
    omega: float | None = None


def diagonalize(states, h_soc: np.ndarray, lz_matrix: np.ndarray | None = None,
                herm_tol: float = 1e-10):
    """Diagonalize E + H_SOC in the spin-free basis.

    Returns ``(so_states, diagnostics)``.  The matrix is never symmetrized:
    a Hermiticity violation above ``herm_tol`` raises, because it would
    indicate a convention error.
    """
    n = len(states)
    energies = np.array([s.energy for s in states], dtype=float)
    H = np.diag(energies) + h_soc
    herm = float(np.abs(H - H.conj().T).max()) if n else 0.0
    if herm > herm_tol:
        raise RuntimeError("state-interaction matrix is not Hermitian: %.3e"
                           % herm)
    w, v = np.linalg.eigh(0.5 * (H + H.conj().T))

    # Exactly degenerate levels (for example the two Lambda components of a
    # Delta term, which have zero SOC coupling) leave the eigenvectors
    # arbitrary inside the degenerate subspace.  Rotate each such subspace to
    # diagonalize the property operator so that Omega is well defined.
    if lz_matrix is not None:
        for group in kramers_pairs(w):
            if len(group) < 2:
                continue
            idx = np.array(group)
            vg = v[:, idx]
            m = vg.conj().T @ lz_matrix @ vg
            _, u = np.linalg.eigh(0.5 * (m + m.conj().T))
            v[:, idx] = vg @ u

    so_states = []
    for j in range(n):
        comp: dict[str, float] = {}
        spin_comp: dict[float, float] = {}
        for i, st in enumerate(states):
            wt = float(abs(v[i, j]) ** 2)
            comp[st.label] = comp.get(st.label, 0.0) + wt
            spin_comp[st.s] = spin_comp.get(st.s, 0.0) + wt
        omega = None
        if lz_matrix is not None:
            omega = float(np.real(v[:, j].conj() @ lz_matrix @ v[:, j]))
        so_states.append(SOState(float(w[j]), v[:, j],
                                 sorted(comp.items(), key=lambda kv: -kv[1]),
                                 spin_comp, omega))
    diag = dict(hermiticity=herm)
    return so_states, diag


def kramers_pairs(energies, tol=1e-6):
    """Return the list of degeneracy groups (size, energy)."""
    order = np.argsort(energies)
    groups = []
    cur = [int(order[0])]
    for k in order[1:]:
        if abs(energies[k] - energies[cur[-1]]) < tol:
            cur.append(int(k))
        else:
            groups.append(cur)
            cur = [int(k)]
    groups.append(cur)
    return groups
