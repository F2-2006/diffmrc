"""Determinant-space state interaction for the ECP SOMF variant family.

The archived variant table was assembled from spin-summed transition
densities and a two-root model space.  ``experiments/somf_ecp/
recalc_variants_determinant.py`` recomputes it in the determinant space of
``core.spinor_ci``.  These tests pin the three statements that the
recalculation rests on, on the smallest system where all of them are
visible: Ge+ 2P in the lanl2dz valence space (eight orbitals, three
electrons, 560 determinants).

* The spin-orbit splitting of a term is the eigenvalue span of the
  determinant-space projection onto the whole term, not of a subset.  For
  the p^1 2P term the six levels must follow the Lande pattern, two at
  zero and four at 3/2 zeta.
* The archived spin-summed assembly is not a small correction: inside one
  two-root model space it is four times the determinant-space coupling,
  and over the fifteen two-root subsets of the term it takes only the two
  values zero and 1562 cm^-1, so the archived 1562.14 cm^-1 calibration is
  a property of the eigensolver's arbitrary choice inside a degenerate
  term.
* The variant family keeps its algebraic structure in the determinant
  space: on a single atom the mean field changes the 2P splitting by less
  than 0.1%, because the free-atom mean field is what the ECPso operator
  already contains.
"""
import itertools

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import ao2mo, gto, mcscf, scf  # noqa: E402

from diffmrc.core import rassi  # noqa: E402
from diffmrc.core import somf_ecp as decp  # noqa: E402
from diffmrc.core import spinor_ci as sci  # noqa: E402

CM = 219474.6313702
SPIN_HALF = [np.array([[0, 1], [1, 0]], dtype=complex) / 2,
             np.array([[0, -1j], [1j, 0]], dtype=complex) / 2,
             np.array([[1, 0], [0, -1]], dtype=complex) / 2]


def spin_orbital_integrals(h1, h2, norb):
    nso = 2 * norb
    h1so = np.zeros((nso, nso), dtype=complex)
    eri = np.zeros((nso, nso, nso, nso), dtype=complex)
    for p in range(norb):
        for q in range(norb):
            h1so[2 * p, 2 * q] = h1[p, q]
            h1so[2 * p + 1, 2 * q + 1] = h1[p, q]
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    v = h2[p, q, r, s]
                    for sp in (0, 1):
                        for sq in (0, 1):
                            eri[2 * p + sp, 2 * q + sp,
                                2 * r + sq, 2 * s + sq] = v
    return h1so, eri


def soc_one_body(hk, norb):
    nso = 2 * norb
    op = np.zeros((nso, nso), dtype=complex)
    for k in range(3):
        for p in range(norb):
            for q in range(norb):
                for a in range(2):
                    for b in range(2):
                        op[2 * p + a, 2 * q + b] += (1j * hk[k, p, q]
                                                     * SPIN_HALF[k][a, b])
    return 0.5 * (op + op.conj().T)


def _annihilate(key, x):
    if not (key >> x) & 1:
        return None
    sign = -1.0 if bin(key & ((1 << x) - 1)).count("1") % 2 else 1.0
    return key ^ (1 << x), sign


def _create(key, x):
    if (key >> x) & 1:
        return None
    sign = -1.0 if bin(key & ((1 << x) - 1)).count("1") % 2 else 1.0
    return key ^ (1 << x), sign


def spin_summed_density(ci_i, ci_j, dets, ncas):
    index = {sum(1 << x for x in d): k for k, d in enumerate(dets)}
    nso = 2 * ncas
    d = np.zeros((ncas, ncas), dtype=complex)
    for jcol, occ in enumerate(dets):
        if ci_j[jcol] == 0.0:
            continue
        key = sum(1 << x for x in occ)
        for q_so in occ:
            step = _annihilate(key, q_so)
            if step is None:
                continue
            key1, s1 = step
            for p_so in range(nso):
                if (p_so & 1) != (q_so & 1):
                    continue
                step = _create(key1, p_so)
                if step is None:
                    continue
                key2, s2 = step
                row = index.get(key2)
                if row is None or ci_i[row] == 0.0:
                    continue
                d[p_so >> 1, q_so >> 1] += (np.conj(ci_i[row]) * ci_j[jcol]
                                            * s1 * s2)
    return d


def legacy_kramers_span(energies_eh, tmat):
    n = len(energies_eh)
    h = np.zeros((2 * n, 2 * n), dtype=complex)
    for i in range(n):
        h[2 * i:2 * i + 2, 2 * i:2 * i + 2] += energies_eh[i] * np.eye(2)
        for j in range(n):
            for k in range(3):
                h[2 * i:2 * i + 2, 2 * j:2 * j + 2] += (1j * tmat[i, j, k]
                                                        * SPIN_HALF[k])
    w = np.linalg.eigvalsh(h)
    return float(w[-1] - w[0])


@pytest.fixture(scope="module")
def ge_plus():
    mol = gto.M(atom=[("Ge", (0, 0, 0))], basis={"Ge": "lanl2dz"},
                ecp={"Ge": "ecpds28mdfso"}, charge=1, spin=1,
                unit="Bohr", verbose=0)
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    ncas = mol.nao_nr()
    nele = mol.nelectron
    mc = mcscf.CASCI(mf, ncas, nele)
    mc.verbose = 0
    h1, ecore = mc.get_h1eff()
    h2 = np.asarray(mc.get_h2eff())
    if h2.ndim == 2:
        h2 = ao2mo.restore(1, h2, ncas)
    c = np.asarray(mc.mo_coeff)
    h1so, eso = spin_orbital_integrals(np.asarray(h1), h2, ncas)
    h0, dets, ecore2 = sci.spinor_ci_matrix(h1so, eso, 0, 2 * ncas, nele)
    w0, v0 = np.linalg.eigh(h0)
    term = list(range(6))
    assert w0[5] - w0[0] < 1e-8, "the lowest term must be sixfold"
    assert (w0[6] - w0[0]) * CM > 1e3
    # Resolve M_S inside the degenerate term and order by <S_z>.  Without
    # this the eigenvectors are an arbitrary basis of the six-dimensional
    # term and the two-root model space is not reproducible.
    space = rassi.DetSpace(norb=ncas, nelec=nele)
    sz = np.zeros((2 * ncas, 2 * ncas), dtype=complex)
    for p in range(ncas):
        sz[2 * p, 2 * p] = 0.5
        sz[2 * p + 1, 2 * p + 1] = -0.5
    msz = v0[:, term].conj().T @ rassi.one_body_matrix(sz, space) @ v0[:, term]
    _, u = np.linalg.eigh(0.5 * (msz + msz.conj().T))
    ms = np.real(np.diag(u.conj().T @ msz @ u))
    order = np.argsort(-ms)
    v0 = v0.copy()
    v0[:, term] = (v0[:, term] @ u)[:, order]
    hk = -decp.ecp_soc(mol)
    hk_mo = np.einsum("ap,kab,bq->kpq", c, hk, c, optimize=True)
    op = rassi.one_body_matrix(soc_one_body(hk_mo, ncas), space)
    return dict(mol=mol, ncas=ncas, nele=nele, h1=h1, h2=h2, c=c,
                dets=dets, w0=w0, v0=v0, term=term, hk=hk, hk_mo=hk_mo,
                op=op, ecore=ecore, spin_orbital=(h1so, eso))


def term_span(data, states):
    v = data["v0"][:, list(states)]
    h = v.conj().T @ data["op"] @ v + np.diag(data["w0"][list(states)])
    lev = np.linalg.eigvalsh(0.5 * (h + h.conj().T))
    return (lev - lev[0]) * CM


def test_full_term_span_follows_the_lande_pattern(ge_plus):
    """Ge+ 2P: two levels at zero (J = 1/2) and four at 3/2 zeta."""
    lev = term_span(ge_plus, ge_plus["term"])
    assert abs(lev[-1] - 2252.91) < 0.05
    assert np.allclose(lev[:2], 0.0, atol=1e-6)
    assert np.allclose(lev[2:], lev[-1], atol=1e-6)


def test_two_root_subset_underestimates_the_term(ge_plus):
    """Every two-root subset spans less than the six-state term."""
    full = term_span(ge_plus, ge_plus["term"])[-1]
    spans = [term_span(ge_plus, pr)[-1]
             for pr in itertools.combinations(ge_plus["term"], 2)]
    assert max(spans) < 0.95 * full
    assert min(spans) < 0.5 * full


def _legacy_pair_spans(data, v_term):
    spans = []
    for pr in itertools.combinations(range(v_term.shape[1]), 2):
        pr = list(pr)
        tmat = np.zeros((2, 2, 3))
        for a in range(2):
            for b in range(2):
                d = spin_summed_density(v_term[:, pr[a]], v_term[:, pr[b]],
                                        data["dets"], data["ncas"])
                for k in range(3):
                    tmat[a, b, k] = float(np.real(np.sum(d
                                                         * data["hk_mo"][k])))
        spans.append(legacy_kramers_span(data["w0"][[data["term"][i]
                                                     for i in pr]], tmat) * CM)
    return np.array(spans)


def test_legacy_assembly_is_not_a_property_of_the_term(ge_plus):
    """The archived assembly over two-root subspaces depends on the basis.

    In the eigensolver's own basis it returns only 0 or 1562.13 cm^-1; a
    rotation inside the degenerate term turns that into a continuum.  The
    archived Ge+ calibration therefore reports the basis, not the term.
    """
    v_term = ge_plus["v0"][:, ge_plus["term"]]
    base = _legacy_pair_spans(ge_plus, v_term)
    assert base.max() > 1500.0
    assert base.max() < 1570.0
    assert base.min() < 1.0
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    rot = _legacy_pair_spans(ge_plus, v_term @ q)
    assert rot.min() > 1.0
    assert rot.max() < base.max()
    assert rot.min() < 0.5 * base.max()


def test_variant_family_keeps_the_atomic_order(ge_plus):
    """On one atom the valence mean field is what ECPso already contains."""
    mol = ge_plus["mol"]
    hk = ge_plus["hk"]
    hk_mo = ge_plus["hk_mo"]
    op_ecp = ge_plus["op"]
    p = np.zeros((ge_plus["ncas"], ge_plus["ncas"]))
    for k in ge_plus["term"]:
        g = sci.spinor_ci_rdm1(ge_plus["v0"][:, k], ge_plus["dets"],
                               2 * ge_plus["ncas"])
        for a in range(ge_plus["ncas"]):
            for b in range(ge_plus["ncas"]):
                p[a, b] += (g[2 * a, 2 * b] + g[2 * a + 1, 2 * b + 1]).real / 6
    from diffmrc.core import somf as dsomf
    c = ge_plus["c"]
    p_ao = c @ p @ c.T
    p_ao = 0.5 * (p_ao + p_ao.T)
    h2 = dsomf.somf_two_electron(mol, p_ao, contract="explicit")
    nuc = decp.nuclear_soc(mol, decp.nuclear_centers(mol))
    val = decp.SOMF_ALPHA2_HALF * (nuc + h2) - decp.ecp_soc(mol)
    assert np.linalg.norm(val - hk) > 0.0
    hk_mo_val = np.einsum("ap,kab,bq->kpq", c, val, c, optimize=True)
    space = rassi.DetSpace(norb=ge_plus["ncas"], nelec=ge_plus["nele"])
    op_val = rassi.one_body_matrix(soc_one_body(hk_mo_val,
                                                ge_plus["ncas"]), space)
    v = ge_plus["v0"][:, ge_plus["term"]]
    lev = np.linalg.eigvalsh(0.5 * ((v.conj().T @ op_val @ v
                                     + np.diag(ge_plus["w0"][ge_plus["term"]]))
                                    + (v.conj().T @ op_val @ v
                                       + np.diag(ge_plus["w0"][ge_plus["term"]])).conj().T))
    span = (lev[-1] - lev[0]) * CM
    assert abs(span - 2252.91) / 2252.91 < 1e-3
