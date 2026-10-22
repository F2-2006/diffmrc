"""SAP-X2C: two-electron picture-change corrections from atomic potentials.

WHY THIS MODULE EXISTS

One-electron X2C (1eX2C) transforms only the nuclear potential.  The
two-electron picture-change (2ePC) terms -- the small-component blocks of
the Coulomb operator that appear when the four-component Hamiltonian is
block-diagonalized -- are omitted, which is the dominant residual error of
1eX2C for heavy elements.  The SAP-X2C ansatz (Surjuse and Valeev,
J. Chem. Theory Comput. 2026, 22, 3443, DOI 10.1021/acs.jctc.6c00032)
replaces the mean-field 2ePC operator by a local potential built as a
superposition of atomic potentials (Lehtola, J. Chem. Theory Comput. 2019,
15, 1593): every atom contributes a spherical screening potential, so the
operator is short-ranged, size-intensive and no more expensive than 1eX2C.

EQUATIONS AND CONVENTIONS

For atom A at R_A, with r_A = |r - R_A| and the Gaussian screening density

    theta_A(r) = sum_k c_k exp(-alpha_k r_A^2),                    (1)

the atomic potential is written in the SAP-X2C paper (Eqs. 13-14) as

    V_A^SAP(r) = -Z_A/r_A + integral dr' theta_A(r')/|r-r'|.       (2)

The coefficients shipped here are the Basis Set Exchange `sap_grasp_large`
fit (Lehtola, Visscher and Engel, J. Chem. Phys. 2020, 152, 144105).  Their
convention is the one used by Eq. 16 of the paper: they multiply
unit-integral s Gaussians g_k = (alpha_k/pi)^(3/2) exp(-alpha_k r^2) and
their sum over k equals -Z_A for a neutral atom.  The screening piece of
the potential is therefore

    V_A^e(r) = -sum_k c_k erf(sqrt(alpha_k) r_A)/r_A,              (3)

which makes V_A^SAP = -Z_A/r_A + V_A^e decay superpolynomially, as required
by Eq. 20 of the paper.

The matrix elements are (Eqs. 15-19)

    V^e_{mu nu} = -sum_A (theta_A | phi_mu phi_nu),
    W^e_{mu nu} = -sum_A (theta_A | [(sigma.p) phi_mu][(sigma.p) phi_nu]),

where (theta_A|...) is a three-centre two-electron integral over the
contracted s Gaussians.  The SAP-dressed four-component Hamiltonian is
(Eq. 21)

    H_SAP = [[V^SAP, T], [T, W^SAP/(4c^2) - T]],

and its X2C block diagonalization gives (Eq. 22)

    H_SAP-X2C = R'^dag [ V^SAP + T X' + X'^dag T
                         + X'^dag (W^SAP/(4c^2) - T) X' ] R' - V^e.

The final -V^e removes the screening contribution already accounted for by
the two-electron mean field of the subsequent SCF calculation.  The
transformation itself (X', R') is the standard one-electron X2C transform
evaluated with V -> V^SAP and W -> W^SAP; this module therefore calls
``pyscf.x2c.x2c._x2c1e_get_hcore`` instead of reimplementing it.

IMPLEMENTATION NOTES

* The three-centre integrals are computed with PySCF's ``aux_e2`` against an
  auxiliary Mole whose basis is the SAP s-Gaussian set, one shell per
  exponent.  PySCF L2-normalizes each primitive; the factor
  (alpha/(2 pi))^(3/4) converts it to the unit-integral Gaussian g_k.
* W^e uses libcint's spinor integral ``int3c2e_spsp1_spinor``, which
  evaluates <(sigma.p) phi_mu| V_g |(sigma.p) phi_nu> in the same
  j-adapted spinor convention as ``int1e_spnucsp_spinor``.  The identity
  was checked in the tight-Gaussian limit: at alpha = 1e12 the 3c matrix
  reproduces the point-charge ``int1e_spnucsp_spinor`` to 9e-13 relative,
  and the scalar counterpart reproduces ``int1e_nuc`` to 2e-11.
* The scalar (spin-free) SAP-X2C route is NOT implemented.  PySCF's
  spinor basis is j-adapted, not a direct product of spatial AOs with
  alpha/beta spins, so the spin trace of W^e is not the scalar
  p.V.p matrix (measured: the spin trace of the nuclear W differs from
  ``int1e_pnucp`` by 4.4% on the O atom).  A scalar W^e needs either the
  sph2spinor transformation or a dressed-Boys 3-centre kernel; raising is
  preferable to returning a silently wrong matrix.
* Only the value layer is claimed here.  The nuclear-coordinate derivative
  design is documented in the Wave 7 report; the derivative integrals
  (``int3c2e_ip1`` / ``int3c2e_ip2`` for V^e and
  ``int3c2e_ipspsp1_spinor`` / ``int3c2e_spsp1ip2_spinor`` for W^e) are
  available in libcint but are not wired into a production gradient.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.special import erf

__all__ = [
    "DEFAULT_SAP_BASIS",
    "load_sap_table",
    "sap_auxmol",
    "sap_aux_weights",
    "sap_potential",
    "sap_screening_potential",
    "sap_potential_matrix",
    "sap_kinetic_matrix",
    "sap_matrices",
    "x2c_sap_hcore",
    "x2c_hcore_1e",
]

DEFAULT_SAP_BASIS = "sap_grasp_large"
_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "sap_grasp_large.json")


# ---------------------------------------------------------------------------
# SAP parameter table
# ---------------------------------------------------------------------------
def load_sap_table(source=None):
    """Return {Z: {"alpha": ndarray, "coef": ndarray}} for the SAP fit.

    Parameters
    ----------
    source : None, path-like or mapping
        None reads the bundled ``sap_grasp_large`` table.  A path may point
        to the bundled compact schema or to a BSE molssi-schema JSON.  A
        mapping is interpreted as {Z: {"alpha": ..., "coef": ...}} or
        {Z: (alpha, coef)}.

    Notes
    -----
    The stored coefficients follow the BSE convention described in the
    module docstring: they sum to -Z for a neutral atom and multiply
    unit-integral s Gaussians.  Sign flips here silently destroy the
    long-range cancellation in V^SAP, so the table is validated on load.
    """
    if source is None:
        raw = _read_json(_DATA_PATH)
    elif isinstance(source, (str, bytes, os.PathLike)):
        raw = _read_json(source)
    elif isinstance(source, dict):
        raw = source
    else:
        raise TypeError("source must be None, a path, or a mapping")

    if "elements" in raw:
        elements = raw["elements"]
        first = next(iter(elements.values()))
        if isinstance(first, dict) and "electron_shells" in first:
            table = _table_from_bse_schema(elements)
        else:
            table = _table_from_compact_schema(elements)
    else:
        table = _table_from_mapping(raw)

    for z, entry in table.items():
        alpha = np.asarray(entry["alpha"], dtype=float)
        coef = np.asarray(entry["coef"], dtype=float)
        if alpha.shape != coef.shape:
            raise ValueError("SAP table Z=%d: alpha and coef lengths differ" % z)
        if not np.all(alpha > 0.0):
            raise ValueError("SAP table Z=%d: non-positive exponent" % z)
        table[z] = {"alpha": alpha, "coef": coef}
    return table


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _table_from_compact_schema(elements):
    out = {}
    for key, entry in elements.items():
        out[int(key)] = {"alpha": [float(x) for x in entry["alpha"]],
                         "coef": [float(x) for x in entry["coef"]]}
    return out


def _table_from_bse_schema(elements):
    out = {}
    for key, entry in elements.items():
        alphas, coefs = [], []
        for shell in entry["electron_shells"]:
            if shell["angular_momentum"] != [0]:
                raise ValueError("SAP basis must be s-type only (Z=%s)" % key)
            alphas.extend(float(x) for x in shell["exponents"])
            for row in shell["coefficients"]:
                coefs.extend(float(x) for x in row)
        if len(alphas) != len(coefs):
            raise ValueError("SAP basis Z=%s: exponent/coefficient mismatch" % key)
        out[int(key)] = {"alpha": alphas, "coef": coefs}
    return out


def _table_from_mapping(raw):
    out = {}
    for key, entry in raw.items():
        if isinstance(entry, dict):
            out[int(key)] = {"alpha": entry["alpha"], "coef": entry["coef"]}
        else:
            out[int(key)] = {"alpha": entry[0], "coef": entry[1]}
    return out


def _entry_for_atom(table, z):
    entry = table.get(int(z))
    if entry is None:
        raise KeyError(
            "no SAP atomic potential for Z=%d; extend the table or supply "
            "load_sap_table(path)" % int(z))
    return entry


def _resolve_table(table):
    """Normalize an optional table argument; None reads the bundled data."""
    return load_sap_table(table)


# ---------------------------------------------------------------------------
# Real-space potential
# ---------------------------------------------------------------------------
def sap_screening_potential(mol, grid, table=None):
    """Screening part V^e(r) = sum_A V_A^e(r) on Cartesian grid points.

    ``grid`` is an (N, 3) array in Bohr.  Points exactly on a nucleus are
    returned as -inf for the bare term and are excluded from the screening
    sum; callers that need a finite value should shift or mask them.
    """
    table = _resolve_table(table)
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError("grid must have shape (N, 3)")
    out = np.zeros(grid.shape[0])
    for ia in range(mol.natm):
        z = mol.atom_charge(ia)
        entry = _entry_for_atom(table, z)
        r = np.linalg.norm(grid - np.asarray(mol.atom_coord(ia)), axis=1)
        good = r > 1e-12
        rr = r[good]
        acc = np.zeros_like(rr)
        for alpha, coef in zip(entry["alpha"], entry["coef"]):
            acc += coef * erf(np.sqrt(alpha) * rr) / rr
        out[good] += -acc
    return out


def sap_potential(mol, grid, table=None):
    """Total SAP potential V_SAP(r) = sum_A [-Z_A/r_A + V_A^e(r)].

    This is the potential whose matrix elements (plus the kinetic operator)
    enter the SAP-dressed four-component Hamiltonian, Eq. 21 of the paper.
    """
    table = _resolve_table(table)
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError("grid must have shape (N, 3)")
    out = np.zeros(grid.shape[0])
    for ia in range(mol.natm):
        z = mol.atom_charge(ia)
        r = np.linalg.norm(grid - np.asarray(mol.atom_coord(ia)), axis=1)
        good = r > 1e-12
        out[good] += -z / r[good]
    out += sap_screening_potential(mol, grid, table=table)
    return out


# ---------------------------------------------------------------------------
# Matrix elements
# ---------------------------------------------------------------------------
def sap_auxmol(mol, table=None):
    """Auxiliary Mole holding the SAP s-Gaussian fit for every atom.

    One shell per exponent, single primitive with coefficient 1.0.  PySCF
    normalizes the primitive to unit L2 norm; ``sap_aux_weights`` supplies
    the conversion to the unit-integral Gaussians of Eq. 1.
    """
    from pyscf import gto

    table = _resolve_table(table)
    basis = {}
    for ia in range(mol.natm):
        z = int(mol.atom_charge(ia))
        if z == 0:
            raise ValueError("SAP auxiliary basis is undefined for ghost atoms")
        symbol = mol.atom_symbol(ia)
        if symbol in basis:
            continue
        entry = _entry_for_atom(table, z)
        basis[symbol] = [[0, [float(alpha), 1.0]] for alpha in entry["alpha"]]
    aux = gto.Mole()
    aux.atom = mol.atom
    aux.unit = mol.unit
    aux.verbose = 0
    aux.cart = False
    aux.basis = basis
    # The auxiliary Mole is a basis container, not a physical electron
    # system: make its neutral electron count consistent for charged parent
    # molecules (e.g. XeH+) by choosing the matching spin parity.
    aux.charge = 0
    aux.spin = int(sum(mol.atom_charge(ia) for ia in range(mol.natm))) % 2
    aux.build()
    return aux


def sap_aux_weights(mol, auxmol, table=None):
    """Contraction weights for the SAP auxiliary functions.

    For an auxiliary function p on atom A with exponent alpha_k, the weight
    is  w_p = -c_k (alpha_k/(2 pi))^(3/4), such that

        V^e_{mu nu} = sum_p w_p (mu nu | p)

    reproduces Eq. 16 with the BSE sign convention.  The exponent is read
    back from the built Mole so the mapping does not rely on shell order.
    """
    table = _resolve_table(table)
    if auxmol.natm != mol.natm:
        raise ValueError("auxmol and mol must have the same atom list")
    weights = np.zeros(auxmol.nao_nr())
    slices = auxmol.aoslice_by_atom()
    for ia in range(auxmol.natm):
        sh0, sh1, p0, p1 = slices[ia]
        z = int(auxmol.atom_charge(ia))
        entry = _entry_for_atom(table, z)
        alphas = np.asarray(entry["alpha"])
        coefs = np.asarray(entry["coef"])
        if sh1 - sh0 != alphas.size or p1 - p0 != alphas.size:
            raise RuntimeError("auxmol shell/AO count does not match the SAP table")
        for shell in range(sh0, sh1):
            exps = np.atleast_1d(auxmol.bas_exp(shell))
            if exps.size != 1:
                raise RuntimeError("SAP auxiliary shells must be single primitives")
            idx = int(np.argmin(np.abs(alphas - exps[0])))
            if abs(alphas[idx] - exps[0]) > 1e-10 * max(1.0, exps[0]):
                raise RuntimeError("auxmol exponent not found in the SAP table")
            weights[p0 + shell - sh0] = (
                -coefs[idx] * (exps[0] / (2.0 * np.pi)) ** 0.75)
    return weights


def sap_potential_matrix(mol, table=None, spinor=True):
    """Screening matrix V^e in the orbital basis of ``mol``.

    ``spinor=True`` returns the (2 nao, 2 nao) spinor matrix, with the same
    spin ordering as PySCF's spinor integrals; ``spinor=False`` returns the
    (nao, nao) scalar matrix.  Eq. 16 with the BSE sign convention.
    """
    from pyscf.df.incore import aux_e2

    auxmol = sap_auxmol(mol, table=table)
    weights = sap_aux_weights(mol, auxmol, table=table)
    intor = "int3c2e_spinor" if spinor else "int3c2e"
    eri3 = aux_e2(mol, auxmol, intor=intor)
    return np.einsum("ijp,p->ij", eri3, weights)


def sap_kinetic_matrix(mol, table=None, spinor=True):
    """Screening matrix W^e = -sum_A (theta_A | (s.p)phi_mu (s.p)phi_nu).

    Eq. 19, returned in the j-adapted spinor basis of PySCF (the same basis
    used by ``int1e_spnucsp_spinor``).  The scalar (spin-free) w^e is not
    available from this tensor; see the module docstring.
    """
    if not spinor:
        raise NotImplementedError(
            "scalar SAP W^e is not implemented: PySCF's spinor basis is "
            "j-adapted, so the spin trace of W^e is not p.V.p "
            "(measured 4.4% discrepancy on the O atom).  Use spinor=True.")
    from pyscf.df.incore import aux_e2

    auxmol = sap_auxmol(mol, table=table)
    weights = sap_aux_weights(mol, auxmol, table=table)
    eri3 = aux_e2(mol, auxmol, intor="int3c2e_spsp1_spinor")
    return np.einsum("ijp,p->ij", eri3, weights)


def sap_matrices(mol, table=None, spinor=True):
    """Return (V^e, W^e) in the given representation.

    Only ``spinor=True`` is implemented for W^e; ``spinor=False`` raises in
    ``sap_kinetic_matrix``.  The scalar V^e alone is available through
    ``sap_potential_matrix(..., spinor=False)``.
    """
    ve = sap_potential_matrix(mol, table=table, spinor=spinor)
    we = sap_kinetic_matrix(mol, table=table, spinor=spinor)
    return ve, we


# ---------------------------------------------------------------------------
# X2C transformation
# ---------------------------------------------------------------------------
def _integrals(xmol, spinor):
    from pyscf import lib

    c = lib.param.LIGHT_SPEED
    if spinor:
        s = xmol.intor_symmetric("int1e_ovlp_spinor")
        t = xmol.intor_symmetric("int1e_spsp_spinor") * 0.5
        v = xmol.intor_symmetric("int1e_nuc_spinor")
        w = xmol.intor_symmetric("int1e_spnucsp_spinor")
    else:
        s = xmol.intor_symmetric("int1e_ovlp")
        t = xmol.intor_symmetric("int1e_kin")
        v = xmol.intor_symmetric("int1e_nuc")
        w = xmol.intor_symmetric("int1e_pnucp")
    return s, t, v, w, c


def _contract(contr, spinor, ntarget):
    """Contraction matrix in the spinor (interleaved) or scalar ordering.

    ``get_xmol`` returns either the scalar contraction (PySCF 2.14) or the
    already spinor-expanded one, depending on the helper; the target
    dimension decides which case applies.
    """
    if contr is None:
        return None
    contr = np.asarray(contr)
    if not spinor or contr.shape[0] == ntarget:
        return contr
    npp, ncc = contr.shape
    out = np.zeros((2 * npp, 2 * ncc), dtype=contr.dtype)
    out[0::2, 0::2] = contr
    out[1::2, 1::2] = contr
    return out


def _x2c_helpers(mol, spinor):
    from pyscf.x2c import sfx2c1e, x2c

    if spinor:
        helper = x2c.SpinorX2CHelper(mol)
    else:
        helper = sfx2c1e.SpinFreeX2CHelper(mol)
    xmol, contr = helper.get_xmol(mol)
    return helper, xmol, contr


def x2c_hcore_1e(mol, spinor=True, c=None):
    """One-electron X2C core Hamiltonian (no SAP), for reference.

    Returns the contracted matrix in the original AO/spinor basis of ``mol``.
    This is the same object as ``pyscf.x2c``'s helper, kept here so the SAP
    and non-SAP routes share one code path.
    """
    from pyscf.x2c import x2c

    _, xmol, contr = _x2c_helpers(mol, spinor)
    s, t, v, w, c0 = _integrals(xmol, spinor)
    c = c0 if c is None else c
    h1 = x2c._x2c1e_get_hcore(t, v, w, s, c)
    cc = _contract(contr, spinor, h1.shape[0])
    return h1 if cc is None else np.dot(np.dot(cc.conj().T, h1), cc)


def x2c_sap_hcore(mol, spinor=True, table=None, c=None, return_terms=False):
    """SAP-X2C core Hamiltonian, Eq. 22 of Surjuse and Valeev (2026).

    Parameters
    ----------
    mol : pyscf.gto.Mole
        All-electron molecule (ECP is not supported by the two-component
        X2C helper in PySCF 2.14 and raises there before this function).
    spinor : bool
        True for the two-component spinor X2C.  False is reserved for the
        scalar sf-X2C route and raises NotImplementedError because the scalar
        W^e is not available in PySCF's j-adapted spinor basis.
    table : mapping, optional
        SAP parameter table, see ``load_sap_table``.
    c : float, optional
        Speed of light; defaults to PySCF's CODATA value.
    return_terms : bool
        If True, also return a dict with the uncontracted ingredients.

    Returns
    -------
    hcore : ndarray
        Contracted (nao, nao) or (2 nao, 2 nao) matrix in the basis of
        ``mol``.  For spinor=True this is the two-component Hamiltonian that
        can be passed to ``pyscf.x2c.UHF(mol).get_hcore``.
    terms : dict, only when return_terms=True
        s, t, v, w, ve, we, xmol, contr, h1_uncontracted.
    """
    from pyscf.x2c import x2c

    table = _resolve_table(table)
    _, xmol, contr = _x2c_helpers(mol, spinor)
    s, t, v, w, c0 = _integrals(xmol, spinor)
    c = c0 if c is None else c

    ve, we = sap_matrices(xmol, table=table, spinor=spinor)
    h1 = x2c._x2c1e_get_hcore(t, v + ve, w + we, s, c) - ve
    cc = _contract(contr, spinor, h1.shape[0])
    hcore = h1 if cc is None else np.dot(np.dot(cc.conj().T, h1), cc)

    if not return_terms:
        return hcore
    return hcore, {"s": s, "t": t, "v": v, "w": w, "ve": ve, "we": we,
                   "xmol": xmol, "contr": contr, "h1_uncontracted": h1}
