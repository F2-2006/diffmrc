"""Spin-orbit state interaction and QDPT on SA-CASSCF references.

PURPOSE
-------
Provides the spin-orbit (SO) state-interaction layer used by the f-block
SOC benchmarks: a spin-free reference, the SO operator in the active
orbital basis, and the quasi-degenerate perturbation theory (QDPT)
effective Hamiltonian whose eigenvalues are the SO-corrected levels.

CONTENTS
--------
1. Determinant-space one-body operator (diagnostic oracle).  Builds the
   spin-orbital ``H_SO`` in pyscf's FCI determinant ordering; used to
   check spectra, traces and symmetry, not to produce reported numbers
   (see the sign-convention boundary below).
2. SOC operators in the MO basis.  Reuses the two implementations
   already validated in this stack: ``diffmrc.core.somf`` (one-electron
   Breit-Pauli and the SOMF mean field) and ``diffmrc.core.somf_ecp``
   (the ``ECPso`` operator).
3. Spin-free reference.  ROHF followed by state-averaged CASSCF over the
   spatial components of the target term, with state energies and
   alpha/beta transition densities.
4. Doublet state interaction and QDPT diagonalization.

PUBLIC INTERFACE
----------------
Determinant space (diagnostic):
    ``spinorbital_soc``, ``fci_strings``, ``one_body_det_matrix``,
    ``ci_matrix_element``, ``soc_state_matrix``, ``scalar_ci_matrix``,
    ``soc_levels``.
Operators:
    ``ALPHA2_HALF``, ``PREFACTORS``, ``mo_soc``, ``check_antisymmetry``.
Reference:
    ``ScalarSpace``.
State interaction:
    ``CM``, ``spherical_components``, ``trans_rdm1s``, ``si_matrix_s12``,
    ``qdpt_levels``, ``term_splitting_cm``.

``ScalarSpace`` takes a system dict with keys ``atom``, ``basis``,
``charge``, ``spin``, ``ncore``, ``ncas``, ``nelecas`` and ``nroots``;
coordinates are in Bohr because the constructor passes ``unit='Bohr'``.
Energies are in Eh except ``CM``-converted level arrays, which are in
cm^-1.

PREFACTORS (fixed by the project, cross-checked against ORCA)

    operator                          H_SO
    int1e_pnucxp / SOMF               i * (alpha^2/2) * sum_k h_k S_k
    ECPso                             -i * sum_k V_k S_k

Both raw operators are real and antisymmetric per Cartesian component;
the ``i`` and ``alpha^2/2`` factors are restored once, in ``mo_soc``, so
the state-interaction layer receives a ready complex one-body operator.

VALIDATED SCOPE
---------------
Doublet (2S = 1) same-multiplet coupling.  With the frozen benchmark
settings (OH 6-31G CAS(7,6), ncore = 1, r = 1.8324 Bohr; OF 6-31G
CAS(13,8), ncore = 2, r = 2.4 Bohr), two state-averaged roots:

    OH  one-electron  205.8828 cm^-1   (frozen 205.883)
    OH  SOMF          128.8287 cm^-1   (frozen 128.829)
    OF  one-electron  272.0614 cm^-1   (frozen 272.061)
    OF  SOMF          176.1253 cm^-1   (frozen 176.125)

The frozen values are ORCA 6.1.1 anchors; reproduction is exact to the
printed digits, which pins the prefactors, the 1/2 and the 1/sqrt(2) in
the matrix elements.  On both benchmarks the state-interaction matrix is
Hermitian and the Kramers pairs are degenerate at the 1e-12 level or
below.

BOUNDARIES
----------
* ``si_matrix_s12`` handles doublets only.  The general spin-coupling
  coefficients for S >= 1 and for inter-multiplet coupling require the
  explicit RASSI expressions of Malmqvist, Roos and Schimmelpfennig,
  Chem. Phys. Lett. 354 (2002) 223; the general path raises
  ``NotImplementedError`` instead of guessing them.
* The effective Hamiltonian is
  ``H_eff[(I,m),(J,n)] = E_I delta_IJ delta_mn + H_SO[(I,m),(J,n)]``.
  Dynamic correlation is not computed: ``qdpt_levels(diagonal=...)``
  accepts external MS-CASPT2 or NEVPT2 diagonal energies for the
  SO-CASPT2 variant.
* The number of states is a physical choice, not a convergence knob.
  Diagonalizing the same operator over the full OH determinant space
  gives 168.7 cm^-1 instead of 205.9 cm^-1 because the ^2Pi components
  couple at second order to ^2Sigma.  Report the SA-CASSCF root count and
  the state-interaction window with every number.
* The determinant-space oracle follows pyscf's FCI string ordering, but
  direct contraction with pyscf CI vectors differs by about 7e-4
  relative because of a determinant-basis sign convention.  It is a
  diagnostic for spectra, traces and norms only.
* X2C two-component integrals and the Park 2019 PT2 amplitudes are not
  implemented; only the multistate effective-Hamiltonian form is used.
"""
from __future__ import annotations

import numpy as np
from pyscf import gto, mcscf, scf
from pyscf.fci import cistring, direct_spin1

from . import somf, somf_ecp

__all__ = [
    "ALPHA2_HALF", "PREFACTORS", "CM",
    "spinorbital_soc", "fci_strings", "one_body_det_matrix",
    "ci_matrix_element", "soc_state_matrix", "scalar_ci_matrix", "soc_levels",
    "mo_soc", "check_antisymmetry",
    "ScalarSpace",
    "spherical_components", "trans_rdm1s", "si_matrix_s12",
    "qdpt_levels", "term_splitting_cm",
]

CM = 219474.6313702
ALPHA2_HALF = 0.5 * 7.2973525693e-3 ** 2

#: complex prefactor multiplying ``sum_k h_k S_k`` for each operator family
PREFACTORS = {
    "one_electron": 1j * ALPHA2_HALF,
    "somf": 1j * ALPHA2_HALF,
    "ecp": -1j,
}


# ---------------------------------------------------------------------------
# Determinant-space one-body operator (diagnostic oracle)
# ---------------------------------------------------------------------------
def spinorbital_soc(h, prefactor):
    """Embed ``(3, n, n)`` spatial components into ``(2n, 2n)`` spin orbitals.

    Spin basis order is ``2p = alpha``, ``2p+1 = beta``; the spin matrices
    are ``S = sigma/2``.
    """
    n = h.shape[1]
    hso = np.zeros((2 * n, 2 * n), dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    for k, s in enumerate((sx, sy, sz)):
        hk = h[k]
        for a in range(2):
            for b in range(2):
                if s[a, b] == 0:
                    continue
                hso[a::2, b::2] += prefactor * hk * s[a, b]
    return hso


def fci_strings(ncas, na, nb):
    return (cistring.gen_strings4orblist(range(ncas), na),
            cistring.gen_strings4orblist(range(ncas), nb))


def one_body_det_matrix(hso, ncas, na, nb):
    """Dense one-body matrix in pyscf's determinant ordering."""
    sa, sb = fci_strings(ncas, na, nb)
    nd = len(sa) * len(sb)
    nso = 2 * ncas
    # occupation list of every determinant
    occ = []
    for ia in range(len(sa)):
        va = int(sa[ia])
        oa = [2 * p for p in range(ncas) if (va >> p) & 1]
        for ib in range(len(sb)):
            vb = int(sb[ib])
            ob = [2 * p + 1 for p in range(ncas) if (vb >> p) & 1]
            occ.append(sorted(oa + ob))
    keys = {}
    for idx, o in enumerate(occ):
        keys[tuple(o)] = idx

    op = np.zeros((nd, nd), dtype=complex)
    for j, oj in enumerate(occ):
        for q in oj:
            for p in range(nso):
                v = hso[p, q]
                if v == 0.0:
                    continue
                if p == q:
                    op[j, j] += v
                    continue
                if p in oj:
                    continue
                s = (-1) ** oj.index(q)
                rest = [x for x in oj if x != q]
                s *= (-1) ** sum(1 for x in rest if x < p)
                i = keys.get(tuple(sorted(rest + [p])))
                if i is not None:
                    op[i, j] += s * v
    return op, sa, sb


def ci_matrix_element(op, ci_i, ci_j):
    """<ci_i| op |ci_j> for CI vectors shaped as pyscf FCI arrays."""
    return complex(np.asarray(ci_i).ravel() @ op @ np.asarray(ci_j).ravel())


def soc_state_matrix(space, h, prefactor, states=None):
    """Exact ``<I|H_SO|J>`` between the SA-CASSCF CI vectors.

    Returns a complex matrix of shape ``(nstates, nstates)``.
    """
    if states is None:
        states = range(space.nroots)
    states = list(states)
    op, _, _ = one_body_det_matrix(
        spinorbital_soc(h, prefactor), space.ncas, space.na, space.nb)
    out = np.zeros((len(states), len(states)), dtype=complex)
    for a, i in enumerate(states):
        for b, j in enumerate(states):
            out[a, b] = ci_matrix_element(op, space.ci[i], space.ci[j])
    return out


def scalar_ci_matrix(space):
    """Scalar determinant-space Hamiltonian in the active space."""
    h1, _ = space.h1eff()
    h2 = space.h2eff()
    h2a = direct_spin1.absorb_h1e(h1, h2, space.ncas,
                                  (space.na, space.nb), 0.5)
    sa, sb = fci_strings(space.ncas, space.na, space.nb)
    nd = len(sa) * len(sb)
    H = np.zeros((nd, nd))
    for ia in range(len(sa)):
        for ib in range(len(sb)):
            j = ia * len(sb) + ib
            civec = np.zeros((len(sa), len(sb)))
            civec[ia, ib] = 1.0
            H[:, j] = direct_spin1.contract_2e(
                h2a, civec, space.ncas, (space.na, space.nb)).ravel()
    return H


def soc_levels(space, h, prefactor):
    """Exact SOC levels in the fixed-M_S determinant space (cm^-1)."""
    H0 = scalar_ci_matrix(space)
    op, _, _ = one_body_det_matrix(
        spinorbital_soc(h, prefactor), space.ncas, space.na, space.nb)
    H = H0 + op
    w = np.linalg.eigvalsh(H)
    return (w - w[0]) * CM


# ---------------------------------------------------------------------------
# Spin-orbit operators in the MO basis
# ---------------------------------------------------------------------------
def mo_soc(mol, mo_coeff, density_ao=None, mode="somf"):
    """Return ``(h, prefactor)`` for the requested SOC operator.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    mo_coeff : ndarray, shape (nao, nmo)
        Orbital basis; normally the CASSCF MO coefficients truncated to
        the core + active block.
    density_ao : ndarray, shape (nao, nao)
        Spin-summed reference density, required for ``mode='somf'``.
    mode : {'one_electron', 'somf', 'ecp'}

    Returns
    -------
    h : ndarray, shape (3, nmo, nmo)
        Raw real antisymmetric operator components.
    prefactor : complex
        Multiply ``sum_k h_k S_k`` by this to obtain H_SO in Eh.
    """
    if mode in ("one_electron", "somf"):
        h = somf.somf_matrix(mol, mo_coeff, P=density_ao, mode=mode,
                             contract="auto")
    elif mode == "ecp":
        if not mol.has_ecp_soc():
            raise ValueError("mode='ecp' requires a spin-orbit ECP")
        h_ao = somf_ecp.ecp_soc(mol)
        h = np.einsum("ap,kab,bq->kpq", mo_coeff, h_ao, mo_coeff,
                      optimize=True)
    else:
        raise ValueError(f"unknown SOC mode {mode!r}")
    return np.asarray(h, dtype=float), PREFACTORS[mode]


def check_antisymmetry(h, tol=1e-10):
    """Raise if a component is not real antisymmetric."""
    imag = float(np.abs(np.imag(h)).max()) if np.iscomplexobj(h) else 0.0
    asym = float(np.abs(h + np.transpose(h, (0, 2, 1))).max())
    if imag > tol or asym > tol:
        raise ValueError(f"SOC operator symmetry violated: imag={imag:.2e}, "
                         f"max|h+h^T|={asym:.2e}")
    return dict(imag=imag, antisymmetry=asym)


# ---------------------------------------------------------------------------
# Spin-free reference: state-averaged CASSCF states and their densities
# ---------------------------------------------------------------------------
def _unpack(nelecas, spin):
    if isinstance(nelecas, (tuple, list)):
        return int(nelecas[0]), int(nelecas[1])
    return (int(nelecas) + int(spin)) // 2, (int(nelecas) - int(spin)) // 2


class ScalarSpace:
    """SA-CASSCF reference states for one spin multiplicity."""

    def __init__(self, system, verbose=False):
        self.system = system
        self.mol = gto.M(atom=system["atom"], basis=system["basis"],
                         charge=system["charge"], spin=system["spin"],
                         unit="Bohr", verbose=0, symmetry=False)
        self.mf = scf.ROHF(self.mol)
        self.mf.conv_tol = 1e-10
        self.mf.max_cycle = 300
        self.mf.verbose = 0
        self.mf.kernel()
        if not self.mf.converged:
            raise RuntimeError("ROHF did not converge")

        self.ncore = int(system["ncore"])
        self.ncas = int(system["ncas"])
        self.nelecas = int(system["nelecas"])
        self.nroots = int(system["nroots"])

        mc = mcscf.CASSCF(self.mf, self.ncas, self.nelecas)
        mc.ncore = self.ncore
        mc.fcisolver.spin = int(system["spin"])
        mc.verbose = 0
        mc.conv_tol = 1e-9
        mc.conv_tol_grad = 1e-5
        mc.max_cycle = 200
        if self.nroots > 1:
            mc.state_average_([1.0 / self.nroots] * self.nroots)
            mc.fcisolver.nroots = self.nroots
        mc.kernel()
        self.mc = mc
        if not mc.converged:
            raise RuntimeError("SA-CASSCF did not converge")

        self.mo_coeff = np.asarray(mc.mo_coeff)
        self.core = slice(0, self.ncore)
        self.active = slice(self.ncore, self.ncore + self.ncas)
        self.c_act = self.mo_coeff[:, self.active]
        self.c_core_act = self.mo_coeff[:, :self.ncore + self.ncas]
        self.ci = [np.asarray(c, dtype=float) for c in mc.ci]
        self.na, self.nb = _unpack(mc.nelecas, system["spin"])

    # -- energies ---------------------------------------------------------
    def state_energies(self):
        """Spin-free state energies in Eh (common zero)."""
        es = getattr(self.mc.fcisolver, "e_states", None)
        if es is not None and len(np.atleast_1d(es)) >= self.nroots:
            return np.asarray(es[:self.nroots], dtype=float)
        h1, ecore = self.mc.get_h1eff()
        e, _ = self.mc.fcisolver.kernel(h1, self.mc.get_h2eff(), self.ncas,
                                        self.mc.nelecas, nroots=self.nroots)
        return np.asarray(e[:self.nroots], dtype=float) + ecore

    # -- densities --------------------------------------------------------
    def state_rdm1(self, i):
        """Active-space 1-RDM of spin-free state ``i`` (spin-summed)."""
        return direct_spin1.make_rdm1(self.ci[i], self.ncas, (self.na, self.nb))

    def state_rdm1_spin(self, i):
        """Active-space (alpha, beta) 1-RDMs of state ``i``."""
        return direct_spin1.make_rdm1s(self.ci[i], self.ncas,
                                       (self.na, self.nb))

    def trans_rdm1(self, i, j):
        """Spin-summed transition 1-RDM between states ``i`` and ``j``."""
        return direct_spin1.trans_rdm1(self.ci[i], self.ci[j], self.ncas,
                                       (self.na, self.nb))

    def state_average_rdm1(self):
        dm = np.zeros((self.ncas, self.ncas))
        for i in range(self.nroots):
            dm += self.state_rdm1(i)
        return dm / self.nroots

    def density_ao(self, dm_act=None):
        """Total spin-summed AO density: frozen core + active part."""
        if dm_act is None:
            dm_act = self.state_average_rdm1()
        P = 2.0 * np.einsum("pi,qi->pq", self.mo_coeff[:, self.core],
                            self.mo_coeff[:, self.core])
        P += np.einsum("pi,ij,qj->pq", self.c_act, dm_act, self.c_act)
        return P

    # -- effective integrals ---------------------------------------------
    def h1eff(self):
        h1, ecore = self.mc.get_h1eff()
        return np.asarray(h1), float(ecore)

    def h2eff(self):
        h2 = self.mc.get_h2eff()
        if np.ndim(h2) == 2:
            from pyscf import ao2mo
            h2 = ao2mo.restore(1, np.asarray(h2), self.ncas)
        return np.asarray(h2)


# ---------------------------------------------------------------------------
# State interaction and QDPT
# ---------------------------------------------------------------------------
def spherical_components(h):
    """Return ``(h_plus, h_minus, h_zero)`` from Cartesian components."""
    hp = -(h[0] + 1j * h[1]) / np.sqrt(2.0)
    hm = (h[0] - 1j * h[1]) / np.sqrt(2.0)
    return hp, hm, h[2]


def trans_rdm1s(space, i, j):
    """Alpha and beta transition 1-RDMs between spin-free states i, j."""
    return direct_spin1.trans_rdm1s(space.ci[i], space.ci[j],
                                    space.ncas, (space.na, space.nb))


def si_matrix_s12(space, h, prefactor, states=None):
    """State-interaction matrix for a doublet manifold.

    Parameters
    ----------
    space : qdpt.ScalarSpace
        Reference with ``nroots`` doublet states.
    h : ndarray, shape (3, ncas, ncas)
        Active-space SOC operator components (raw, real, antisymmetric).
    prefactor : complex
        ``i*alpha^2/2`` for pnucxp/SOMF, ``-i`` for ECPso.
    states : sequence of int, optional
        Model-space state indices; default all ``space.nroots``.

    Returns
    -------
    H : ndarray, shape (2n, 2n), complex
        Basis order ``|0+>, |0->, |1+>, |1->, ...``.
    """
    if int(space.system["spin"]) != 1:
        raise NotImplementedError(
            "si_matrix_s12 handles doublets (2S = 1) only; the general "
            "spin-coupling coefficients require Malmqvist et al., Chem. "
            "Phys. Lett. 354 (2002) 223, which is not available in this "
            "module.  Do not guess them.")
    if states is None:
        states = list(range(space.nroots))
    states = list(states)
    hp, hm, h0 = spherical_components(h)
    ns = len(states)
    H = np.zeros((2 * ns, 2 * ns), dtype=complex)
    for a, i in enumerate(states):
        for b, j in enumerate(states):
            da, db = trans_rdm1s(space, i, j)
            z = 0.5 * prefactor * np.sum(h0 * (da - db))
            H[2 * a, 2 * b] = z
            H[2 * a + 1, 2 * b + 1] = -z
            H[2 * a, 2 * b + 1] = prefactor / np.sqrt(2.0) * np.sum(hm * da)
            H[2 * a + 1, 2 * b] = prefactor / np.sqrt(2.0) * np.sum(hp * db)
    return H


def qdpt_levels(space, h, prefactor, states=None, diagonal=None):
    """Diagonalize the QDPT effective Hamiltonian.

    Parameters
    ----------
    diagonal : ndarray, optional
        Spin-free diagonal energies in Eh, one per model-space state.
        Defaults to the SA-CASSCF state energies.  Pass MS-CASPT2 or
        NEVPT2 corrected energies to obtain the SO-CASPT2 variant.

    Returns
    -------
    dict with ``levels_cm`` (shifted to the lowest level), ``levels_eh``,
    ``eigenvectors``, ``H_eff``, ``hermiticity`` and ``kramers_gap``.
    """
    if states is None:
        states = list(range(space.nroots))
    states = list(states)
    ns = len(states)
    Hso = si_matrix_s12(space, h, prefactor, states)
    if diagonal is None:
        e = np.asarray(space.state_energies(), dtype=float)[states]
    else:
        e = np.asarray(diagonal, dtype=float)
        if e.size != ns:
            raise ValueError("diagonal must have one entry per model state")
    H = np.zeros((2 * ns, 2 * ns), dtype=complex)
    for a in range(ns):
        H[2 * a, 2 * a] = e[a]
        H[2 * a + 1, 2 * a + 1] = e[a]
    H = H + Hso
    herm = float(np.abs(H - H.conj().T).max())
    # ``eigh`` assumes a Hermitian matrix and reads one triangle; feeding the
    # raw assembly would let a small non-Hermitian residual leak into the
    # spectrum (observed at 5e-10 in a long full-suite run, which split the
    # Kramers pairs by 1e-4 cm^-1).  Symmetrize before diagonalizing and keep
    # the pre-symmetrization residual as the diagnostic reported above.
    H = 0.5 * (H + H.conj().T)
    w, u = np.linalg.eigh(H)
    w = w - w[0]
    # Kramers: every level must be at least doubly degenerate.
    gaps = np.abs(w[1::2] - w[0::2]) if 2 * ns >= 2 else np.zeros(0)
    return dict(levels_eh=w, levels_cm=w * CM, eigenvectors=u,
                H_eff=H, hermiticity=herm,
                kramers_gap_cm=float(np.max(gaps)) if gaps.size else 0.0)


def term_splitting_cm(result, nlevels=2):
    """Span of the lowest ``nlevels`` Kramers doublets in cm^-1."""
    w = result["levels_cm"]
    if len(w) < 2 * nlevels:
        raise ValueError("not enough levels")
    return float(w[2 * nlevels - 1] - w[0])
