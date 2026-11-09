"""Zero-field splitting (D, E) and g-tensor from spin-orbit state interaction.

PURPOSE
-------
This module builds the spin-orbit state-interaction (SI) matrix on a
spin-free CASCI spectrum and extracts

* the effective spin Hamiltonian of the ground spin multiplet,
      H_eff = D [S_z^2 - S(S+1)/3] + E (S_x^2 - S_y^2) + const,
  by second-order quasi-degenerate perturbation theory (QDPT2) in the
  |S, M_S> basis, and
* the g-tensor of the ground Kramers doublet from the Zeeman operator
  M = L + g_e S projected onto the SOC-mixed eigenstates.

Everything is built from the project's own stack: pyscf integrals, the
release SOMF operator ``diffmrc.core.somf.somf_matrix`` and the spin
recognition helper ``diffmrc.analysis.states.spin_square``.  No external
quantum-chemistry program is called.

PUBLIC INTERFACE
----------------
Determinant space and scalar spectrum:
    ``det_space``, ``block_hamiltonian``, ``solve_scalar``.
Integrals and one-electron operators:
    ``active_integrals``, ``soc_mo_operator``, ``l_mo_operator``,
    ``spin_orbital_components``, ``spin_orbital_blocks``,
    ``build_connections``, ``onee_matrix``, ``state_matrix``,
    ``rdm1_state``.
Effective Hamiltonian and analysis:
    ``spin_matrices``, ``qdpt2_effective``, ``fit_spin_hamiltonian``,
    ``si_diagonalize``, ``g_tensor_doublet``.

Inputs are pyscf ``Mole``/MO arrays plus active-space integrals; outputs
are numpy arrays.  Energies are in Eh unless a ``_cm1``-style suffix or
the caller's conversion says cm^-1.

CONVENTIONS
-----------
* Spin orbitals of the active space are indexed ``a = 2*j + sigma`` with
  ``sigma = 0`` (alpha) and ``1`` (beta); determinant bitmaps use bit
  ``a``.  This is the same relative ordering as the project kernel, and
  S^2 is invariant under the constant bit shift (the frozen core is
  closed shell).
* SOC operator: ``H_SO = I_SOC * (alpha^2/2) * sum_k h_k (x) s_k`` with
  ``h_k`` the raw real antisymmetric ``int1e_pnucxp`` (plus the SOMF
  two-electron mean field) and ``s_k = sigma_k/2``.  ``I_SOC = +1j``;
  the phase is fixed by the experimental OH ^2Pi ordering (with ``-1j``
  the ^2Pi_1/2 lies below ^2Pi_3/2, opposite to experiment).  Splitting
  magnitudes do not depend on this phase.
* Orbital angular momentum: ``L = I_SOC * int1e_cg_irxp`` with the same
  ``I_SOC``, so that L and the orbital part of H_SO carry the same phase.
  Only the product of L and H_SO enters the g-shift, so the g-tensor is
  invariant under a global flip of ``I_SOC``.
* Energies in Eh unless a ``_cm1`` suffix says cm^-1.

VALIDATED SCOPE
---------------
OH/6-31G CAS(4,4), ROHF reference, all four M_S blocks: the lowest block
eigenvalue reproduces the pyscf FCI energy to 1e-9 Eh and <S^2> to 1e-6;
the spin-summed active-space 1-RDM reproduces the pyscf natural
occupations to 1e-7; <S_z> equals M_S to 1e-10 while <S_x> and <S_y>
vanish to 1e-10.  The QDPT2 + D/E machinery was exercised on MnO
CAS(7,7) in the development study (D = 0.19 cm^-1 on the warm branch,
with a window spread of about 20%).

EXPERIMENTAL
------------
``g_tensor_doublet`` is experimental and unvalidated.  In the
development study its principal values moved by more than 30% when the
QDPT2 window changed from 1.5 to 2.0 Eh, and a system with a quenched
orbital angular momentum came out strongly anisotropic, which is
unphysical.  Do not use it to produce reported numbers until the
projection and the operator construction are re-derived.

BOUNDARIES
----------
* QDPT2 is second order in H_SO and first order in the model-space
  degeneracy; the model space is the degenerate ground manifold and the
  perturbers are the scalar states inside ``ecut``.
* The frozen core is closed shell; SOC-induced core excitations are not
  in the determinant space.
* The SOMF operator uses one reference density, so the SI matrix is not
  state dependent beyond that approximation.
* The g-tensor is first order in the magnetic field (all orders in SOC
  through the SI eigenvectors); the second-order Zeeman response is not
  included.
"""

from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix

__all__ = [
    "CM1", "ALPHA2_HALF", "GE", "I_SOC",
    "det_space", "block_hamiltonian", "solve_scalar",
    "active_integrals", "soc_mo_operator", "l_mo_operator",
    "build_connections", "onee_matrix", "state_matrix",
    "rdm1_state", "qdpt2_effective", "fit_spin_hamiltonian",
    "si_diagonalize", "g_tensor_doublet", "spin_matrices",
]

CM1 = 219474.6313702            # Eh -> cm^-1
ALPHA2_HALF = 0.5 * 7.2973525693e-3 ** 2
GE = 2.00231930436256
I_SOC = 1j                      # phase of the raw r x p integrals;
                                # fixed by the experimental OH 2Pi ordering

PAULI = np.array([[[0, 1], [1, 0]],
                  [[0, -1j], [1j, 0]],
                  [[1, 0], [0, -1]]], dtype=complex)
SPIN_HALF = 0.5 * PAULI         # s_k = sigma_k / 2


# ----------------------------------------------------------------- determinants
def _below(bitmap, x):
    return (int(bitmap) & ((1 << x) - 1)).bit_count()


def excitation_sign(bitmap, i, a):
    """Parity of ``a^dag_a a_i`` on ``|bitmap>`` (i occupied, a empty)."""
    lo, hi = (i, a) if i < a else (a, i)
    parity = _below(bitmap, lo) + _below(bitmap, hi) - (1 if a > i else 0)
    return -1.0 if parity & 1 else 1.0


def det_space(nact, nele_act):
    """All determinants of the active space, grouped by ``(na, nb)``.

    Returns
    -------
    blocks : dict (na, nb) -> int64 array of bitmaps
    all_masks : int64 array, concatenation of the blocks in sorted order
    index : dict bitmap -> global row index
    """
    abits = [2 * j for j in range(nact)]
    bbits = [2 * j + 1 for j in range(nact)]
    blocks = {}
    for na in range(nele_act + 1):
        nb = nele_act - na
        if na > nact or nb > nact:
            continue
        masks = []
        for ca in combinations(abits, na):
            for cb in combinations(bbits, nb):
                m = 0
                for b in ca:
                    m |= 1 << b
                for b in cb:
                    m |= 1 << b
                masks.append(m)
        blocks[(na, nb)] = np.array(masks, dtype=np.int64)
    order = sorted(blocks)
    all_masks = np.concatenate([blocks[k] for k in order])
    index = {int(m): i for i, m in enumerate(all_masks)}
    return blocks, all_masks, index


def block_hamiltonian(h1, eri, masks, nact):
    """Spin-free Hamiltonian (numpy port of ``diffmrc.core.casci._block_H``).

    Parameters
    ----------
    h1 : (nact, nact) active-space one-electron integrals with the frozen
        core folded in (``h_core + sum_i 2J_i - K_i``).
    eri : (nact, nact, nact, nact) active-space two-electron integrals,
        chemist notation.
    masks : determinant bitmaps of one M_S block.
    nact : number of active spatial orbitals.
    """
    nso = 2 * nact
    ndet = len(masks)
    pi = np.arange(nso) // 2
    si = np.arange(nso) % 2
    h_so = np.where(si[:, None] == si[None, :], h1[np.ix_(pi, pi)], 0.0)
    # The kernel orders the two-electron tensor as h2e[i,k,j,l] = (ij|kl),
    # i.e. the spin pairing is (i,k) and (j,l); reproduce that permutation
    # before expanding to spin orbitals.
    eri_t = np.ascontiguousarray(eri.transpose(0, 2, 1, 3))
    m4 = ((si[:, None, None, None] == si[None, None, :, None])
          & (si[None, :, None, None] == si[None, None, None, :]))
    eri_so = np.where(m4, eri_t[np.ix_(pi, pi, pi, pi)], 0.0)

    occ = ((masks[:, None] >> np.arange(nso)[None, :]) & 1).astype(bool)
    diag1 = occ @ np.diag(h_so)
    e2 = np.einsum("ijij->ij", eri_so) - np.einsum("ijji->ij", eri_so)
    diag2 = ((occ @ np.triu(e2, k=1)) * occ).sum(axis=1)

    xa = occ[:, None, :] & ~occ[None, :, :]
    xb = ~occ[:, None, :] & occ[None, :, :]
    na, nb = xa.sum(axis=2), xb.sum(axis=2)
    pos = np.cumsum(occ, axis=1) - 1

    mask1 = (na == 1) & (nb == 1)
    i1 = np.argmax(xa, axis=2)
    k1 = np.argmax(xb, axis=2)
    pos_i = np.take_along_axis(pos[:, None, :], i1[:, :, None], axis=2)[..., 0]
    pos_k = np.take_along_axis(pos[None, :, :], k1[:, :, None], axis=2)[..., 0]
    sign1 = np.where((pos_i + pos_k) % 2 == 0, 1.0, -1.0)
    xijk = np.einsum("ijkj->ijk", eri_so) - np.einsum("ijjk->ijk", eri_so)
    t = np.einsum("aj,ijk->aik", occ.astype(float), xijk)
    t_pair = t[np.arange(ndet)[:, None], i1, k1]
    occ_i = np.take_along_axis(occ[:, :, None], i1[:, :, None],
                               axis=1)[..., 0].astype(float)
    x_ii_k = xijk[i1, i1, k1]
    val1 = sign1 * (h_so[i1, k1] + t_pair - occ_i * x_ii_k)
    ham = np.where(mask1, val1, 0.0)

    mask2 = (na == 2) & (nb == 2)
    ax = np.arange(nso)[None, None, :]

    def second_idx(xm, first):
        return np.argmax(xm & (ax != first[..., None]), axis=2)

    i2 = second_idx(xa, i1)
    k2 = second_idx(xb, k1)
    pos_i2 = np.take_along_axis(pos[:, None, :], i2[:, :, None], axis=2)[..., 0]
    pos_k2 = np.take_along_axis(pos[None, :, :], k2[:, :, None], axis=2)[..., 0]
    sign2 = np.where((pos_i + pos_i2 + pos_k + pos_k2) % 2 == 0, 1.0, -1.0)
    val2 = sign2 * (eri_so[i1, i2, k1, k2] - eri_so[i1, i2, k2, k1])
    ham = ham + np.where(mask2, val2, 0.0)
    return ham + np.diag(diag1 + diag2)


def solve_scalar(h1, eri, nact, nele_act, nkeep=24, ecut=None):
    """Lowest scalar states of every M_S block.

    Returns a list of dicts with keys ``E``, ``ms``, ``block``, ``ci``
    (vector over the block masks), ``masks`` (block bitmaps) and
    ``s2`` (filled by the caller, which owns the spin_square helper).
    """
    from diffmrc.analysis.states import spin_square

    blocks, all_masks, _ = det_space(nact, nele_act)
    e_min = None
    raw = []
    for key in sorted(blocks):
        masks = blocks[key]
        ham = block_hamiltonian(h1, eri, masks, nact)
        w, v = np.linalg.eigh(0.5 * (ham + ham.T))
        e_min = w[0] if e_min is None else min(e_min, w[0])
        keep = min(nkeep, len(w))
        raw.append((key, masks, w[:keep], v[:, :keep]))

    states = []
    for key, masks, w, v in raw:
        for r in range(len(w)):
            if ecut is not None and w[r] - e_min > ecut:
                continue
            ci = np.ascontiguousarray(v[:, r])
            states.append(dict(E=float(w[r]), ms=0.5 * (key[0] - key[1]),
                               block=key, ci=ci, masks=masks,
                               s2=float(spin_square(ci, masks))))
    states.sort(key=lambda s: s["E"])
    return states


# ------------------------------------------------------------------- integrals
def active_integrals(mol, C, ncore, nact):
    """Active-space h1 (core folded in) and ERI in the MO basis."""
    h1e_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri_ao = mol.intor("int2e", aosym="s1")
    nkeep = ncore + nact
    Ck = C[:, :nkeep]
    h1e_mo = np.einsum("ai,ab,bj->ij", Ck, h1e_ao, Ck, optimize=True)
    eri_mo = np.einsum("pi,qj,rk,sl,pqrs->ijkl", Ck, Ck, Ck, Ck, eri_ao,
                       optimize=True)
    h1_act = np.array(h1e_mo[ncore:, ncore:], dtype=float)
    for k in range(ncore):
        # 2 (pq|ii) - (pi|iq)
        h1_act += (2.0 * eri_mo[ncore:, ncore:, k, k]
                   - eri_mo[ncore:, k, k, ncore:])
    eri_act = np.ascontiguousarray(eri_mo[ncore:, ncore:, ncore:, ncore:])
    return h1_act, eri_act, h1e_mo


def soc_mo_operator(mol, C, P_ao, mode="somf", contract="auto"):
    """SOC operator in the MO basis, units Eh, shape ``(3, nmo, nmo)``.

    ``mode='somf'`` uses ``diffmrc.core.somf.somf_matrix`` (one-electron
    Breit-Pauli plus the two-electron mean field); ``mode='one_electron'``
    uses ``int1e_pnucxp`` only.  The returned array already carries
    ``alpha^2/2``; the caller applies ``I_SOC``.
    """
    from diffmrc.core.somf import somf_matrix

    C = np.asarray(C, dtype=float)
    if mode == "somf":
        h = somf_matrix(mol, C=C, P=P_ao, mode="somf", contract=contract,
                        prefactor=True, check=True)
    elif mode == "one_electron":
        h = somf_matrix(mol, C=C, P=None, mode="one_electron",
                        prefactor=True)
    else:
        raise ValueError(f"unknown SOC mode: {mode!r}")
    return np.asarray(h, dtype=float)


def l_mo_operator(mol, C):
    """Orbital angular momentum in the MO basis, shape ``(3, nmo, nmo)``.

    Raw ``int1e_cg_irxp`` (real antisymmetric); the caller applies
    ``I_SOC`` to obtain the physical L.
    """
    C = np.asarray(C, dtype=float)
    l_ao = mol.intor("int1e_cg_irxp", 3)
    return np.einsum("ai,cab,bj->cij", C, l_ao, C, optimize=True)


# ------------------------------------------------- one-electron operator layer
def spin_orbital_components(h3, kind):
    """Three spin-orbital matrices, one per Cartesian component.

    kind='L' : ``I_SOC * h3[k,p,q] * delta_{sigma sigma'}``
    kind='S' : ``delta_{pq} * (s_k)_{sigma sigma'}`` (h3 only sets the size)
    kind='soc': ``I_SOC * h3[k,p,q] * (s_k)_{sigma sigma'}`` per component
    """
    n = h3.shape[1]
    out = []
    for k in range(3):
        mat = np.zeros((2 * n, 2 * n), dtype=complex)
        if kind == "S":
            for p in range(n):
                mat[2 * p:2 * p + 2, 2 * p:2 * p + 2] += SPIN_HALF[k]
        else:
            blk = I_SOC * np.asarray(h3[k], dtype=float)
            if kind == "L":
                for a in range(2):
                    mat[a::2, a::2] += blk
            elif kind == "soc":
                for a in range(2):
                    for b in range(2):
                        mat[a::2, b::2] += blk * SPIN_HALF[k][a, b]
            else:
                raise ValueError(f"unknown kind: {kind!r}")
        out.append(mat)
    return out


def spin_orbital_blocks(h3, kind):
    """Sum of the three Cartesian components (the scalar SOC operator)."""
    return sum(spin_orbital_components(h3, kind))


def build_connections(all_masks, index, nact):
    """Single-excitation connections of the determinant space.

    Returns ``(row, col, created, annihilated, sign, occ)`` where a
    one-electron operator ``h`` has matrix element
    ``H[row, col] = sign * h[created, annihilated]``.
    """
    nso = 2 * nact
    rows, cols, created, annihilated, signs = [], [], [], [], []
    for j, m in enumerate(all_masks):
        m = int(m)
        occ_list = [p for p in range(nso) if (m >> p) & 1]
        emp_list = [p for p in range(nso) if not (m >> p) & 1]
        for q in occ_list:
            for p in emp_list:
                m2 = m ^ (1 << q) ^ (1 << p)
                i = index.get(m2)
                if i is None:
                    continue
                rows.append(i)
                cols.append(j)
                created.append(p)
                annihilated.append(q)
                signs.append(excitation_sign(m, q, p))
    occ = ((all_masks[:, None] >> np.arange(nso)[None, :]) & 1).astype(bool)
    return (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64),
            np.asarray(created, dtype=np.int64),
            np.asarray(annihilated, dtype=np.int64),
            np.asarray(signs, dtype=float), occ)


def onee_matrix(conn, h_sp):
    """Sparse determinant matrix of a one-electron operator ``h_sp``."""
    row, col, cre, ann, sign, occ = conn
    ndet = len(occ)
    diag = occ @ np.diag(h_sp)
    vals = sign * h_sp[cre, ann]
    mat = coo_matrix((vals, (row, col)), shape=(ndet, ndet)).tocsr()
    mat = mat + coo_matrix((diag.astype(complex), (np.arange(ndet),
                                                   np.arange(ndet))),
                           shape=mat.shape).tocsr()
    return mat


def state_matrix(states, mat, index, block_map=None):
    """Matrix of a one-electron operator between scalar states.

    ``mat`` is the sparse determinant matrix in the global ordering of
    :func:`det_space`; ``index`` maps a bitmap to that ordering.
    """
    if block_map is None:
        block_map = {}
        for s in states:
            key = id(s["masks"])
            if key not in block_map:
                block_map[key] = np.array(
                    [index[int(m)] for m in s["masks"]], dtype=np.int64)
    n = len(states)
    ndet = mat.shape[0]
    vec = np.zeros((ndet, n), dtype=complex)
    for k, s in enumerate(states):
        vec[block_map[id(s["masks"])], k] = s["ci"]
    return vec.conj().T @ (mat @ vec)


def rdm1_state(state, conn, index):
    """Spin-summed active-space 1-RDM of one scalar state."""
    row, col, cre, ann, sign, occ = conn
    nso = occ.shape[1]
    ci = np.asarray(state["ci"], dtype=float)
    local = {index[int(m)]: k for k, m in enumerate(state["masks"])}
    lr = np.fromiter((local.get(int(r), -1) for r in row), dtype=np.int64,
                     count=len(row))
    lc = np.fromiter((local.get(int(c), -1) for c in col), dtype=np.int64,
                     count=len(col))
    ok = (lr >= 0) & (lc >= 0)
    mask = np.zeros((nso, nso))
    np.add.at(mask, (cre[ok], ann[ok]),
              sign[ok] * ci[lr[ok]] * ci[lc[ok]])
    occ_block = occ[np.array([index[int(m)] for m in state["masks"]])]
    mask[np.arange(nso), np.arange(nso)] += (occ_block * (ci ** 2)[:, None]).sum(0)
    nact = nso // 2
    return mask[0::2, 0::2] + mask[1::2, 1::2]


# -------------------------------------------------------------- QDPT and fit
def spin_matrices(S, ms):
    """S_z, S_x, S_y in the |S, M_S> basis ordered as ``ms``."""
    n = len(ms)
    idx = {round(m, 6): i for i, m in enumerate(ms)}
    sz = np.diag(np.asarray(ms, dtype=float))
    sp = np.zeros((n, n))
    for i, m in enumerate(ms):
        j = idx.get(round(m + 1.0, 6))
        if j is not None:
            sp[j, i] = np.sqrt(S * (S + 1) - m * (m + 1))
    sm = sp.T
    sx = 0.5 * (sp + sm)
    sy = -0.5j * (sp - sm)
    return sz, sx, sy


def qdpt2_effective(states, hso_state, model_idx, perturber_idx):
    """Second-order effective Hamiltonian in the model space."""
    model_idx = np.asarray(model_idx, dtype=int)
    perturber_idx = np.asarray(perturber_idx, dtype=int)
    e = np.array([s["E"] for s in states])
    e0 = float(np.mean(e[model_idx]))
    heff = hso_state[np.ix_(model_idx, model_idx)].astype(complex).copy()
    if len(perturber_idx):
        denom = e0 - e[perturber_idx]
        if np.any(np.abs(denom) < 1e-10):
            raise ValueError("a perturber is degenerate with the model space")
        block = hso_state[np.ix_(model_idx, perturber_idx)]
        heff += block @ (block.conj().T / denom[:, None])
    return heff, e0


def fit_spin_hamiltonian(heff, S, ms, include_fourth=False):
    """Least-squares D/E (optionally B4^0) fit of a model-space matrix.

    Returns a dict with D, E, residual, and the fitted matrices.
    """
    heff = np.asarray(heff, dtype=complex)
    sz, sx, sy = spin_matrices(S, ms)
    ops = [np.eye(len(ms)),
           sz @ sz - (S * (S + 1) / 3.0) * np.eye(len(ms)),
           sx @ sx - sy @ sy]
    names = ["const", "D", "E"]
    if include_fourth:
        # Stevens O4^0 = 35 Sz^4 - [30 S(S+1)-25] Sz^2 + const
        s2 = sz @ sz
        o40 = (35.0 * s2 @ s2 - (30.0 * S * (S + 1) - 25.0) * s2)
        o40 = o40 - np.trace(o40) / len(ms) * np.eye(len(ms))
        ops.append(o40)
        names.append("B4")
    real = np.real(heff)
    imag = np.imag(heff)
    a = np.column_stack([o.reshape(-1).real for o in ops])
    b = real.reshape(-1)
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    resid = b - a @ coef
    scale = np.linalg.norm(b - b.mean())
    out = dict(zip(names, [float(c) for c in coef]))
    out["residual"] = float(np.linalg.norm(resid) / scale) if scale > 0 else 0.0
    out["imag_norm"] = float(np.linalg.norm(imag))
    out["herm_norm"] = float(np.linalg.norm(heff - heff.conj().T))
    out["fit_matrix"] = sum(c * o for c, o in zip(coef, ops))
    return out


def si_diagonalize(states, hso_state):
    """Diagonalize the full SI matrix ``diag(E) + H_SO``."""
    e = np.array([s["E"] for s in states])
    si = np.diag(e).astype(complex) + hso_state
    herm = float(np.abs(si - si.conj().T).max())
    vals, vecs = np.linalg.eigh(0.5 * (si + si.conj().T))
    return vals, vecs, herm


def g_tensor_doublet(vecs, mats, ndoublet=2):
    """g-tensor of the lowest ``ndoublet`` SO levels.

    EXPERIMENTAL AND UNVALIDATED.  The projection and the operator
    construction behind this function have not been pinned down; see the
    module docstring.  It is kept for the development record only.

    ``mats[k]`` is the matrix of ``M_k = L_k + g_e S_k`` in the scalar
    state basis.  Returns the 3x3 matrix ``g[field, spin]`` and its
    principal values.
    """
    psi = vecs[:, :ndoublet]
    g = np.zeros((3, 3))
    for k in range(3):
        meff = psi.conj().T @ mats[k] @ psi
        for j in range(3):
            g[k, j] = np.trace(PAULI[j] @ meff).real
    sv = np.linalg.svd(g, compute_uv=False)
    return g, sv
