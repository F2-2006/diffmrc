"""The ASS1ST perturbed densities: the provider half of the NOON rule.

Why: the decision half (``policy.rules.noon_selection``) takes occupation
numbers on trust; this module is where they come from.  The protocol is
Khedkar & Roemelt's (JCTC 2019): the first-order perturbed wavefunction
of SC-NEVPT2 is split into the eight classes of their eq. (12), each
class contributes to the internal and external blocks of the one-particle
density (eqs. 20/21 and SI section A), and the block eigenvectors give
quasi-natural orbitals whose occupations the selection rule judges.  The
2020 sequel's IPEA form is the default for the two classes whose
effective potentials carry the fourth-order density (``S_a^(-1)``,
``S_i^(+1)``): their eps_eff is replaced by the density-weighted average
active orbital energy plus a 0.25 Eh shift (empirical, borrowed from the
Roos-Andersson CASPT2 level shift).

Design decisions, measured during planning:

- The class vectors are built explicitly: the eq.-12 operators are
  applied to the reference CI vector with the kernel's bitmap machinery
  (``core.nevpt2.m4``, read-only), so no effective-potential formulas
  and no fourth-order densities are needed anywhere.  The anchor is
  ``S_ij,ab^(0)``: its operator never touches the active space, so its
  effective potential must come out exactly zero and its denominator
  must equal the bare orbital-energy difference.
- The general denominator is ``Delta = sum(particles eps) -
  sum(holes eps) + eps_eff`` with ``eps_eff = E_v0 - <V|H_v|V>/N``,
  where ``H_v`` is the source's eq.-5 active Hamiltonian and ``E_v0``
  its reference expectation (this is the commutator expectation
  ``<V|[H_v, V]|0>/N`` written out; see Helmich-Paris & Knecht 2017,
  eqs. 16-22, whose paper also confirms only the two 3-active-index
  classes involve the fourth-order density -- the same two the IPEA
  substitution covers).
- ``h_eff`` follows the package's own production form,
  ``h + sum_c [2 (pq|cc) - (pc|cq)]`` (``core/nevpt2/impl.py``'s core
  field); the OCR of the source's eq. (9) drops the minus and the
  package is the arbiter.  ``h_eff'`` is eq. (10), ``h_eff - sum_t
  (pt|tq)``.
- The density cross terms collapse to same-pattern squares only (the
  source: "only terms that include two perturber functions of the same
  class are nonvanishing"), so the blocks need no pair machinery.

Cost: the expectation helpers are written brute force over the vector's
determinant pairs, which bounds this module to the small active spaces
the ASS1ST loop starts from; the closed contractions of SI section A
are the production route once a consumer needs them.
"""
import numpy as np

__all__ = ['e_operator', 'class_vectors', 'class_denominators',
           'PerturbedDensities', 'perturbed_densities', 'NoonProvider',
           'ass1st_noon']


def _popcount(x):
    return bin(x).count('1')


def _annihilate(det, q):
    """``a_q |det>`` with the kernel's bitmap sign rule.

    Mirror of ``core.nevpt2.m4.ann_so``: that module pulls jax at
    import time and this one is numpy-only by design, so the rule is
    copied rather than imported -- and bound back to the kernel by a
    jax-gated cross-check test so the two cannot drift.
    """
    if not (det >> q) & 1:
        return None, 0
    sign = -1 if _popcount(det & ((1 << q) - 1)) % 2 else 1
    return det & ~(1 << q), sign


def _create(det, p):
    """``a+_p |det>`` with the kernel's bitmap sign rule (mirror of
    ``core.nevpt2.m4.cre_so``; see ``_annihilate``)."""
    if (det >> p) & 1:
        return None, 0
    sign = -1 if _popcount(det & ((1 << p) - 1)) % 2 else 1
    return det | (1 << p), sign


def e_operator(det, ann, cre):
    """``E_ann^cre = a+_cre a_ann`` on one determinant: (det', sign).

    Returns ``(None, 0)`` when the determinant cannot carry the operation.
    """
    d, s1 = _annihilate(det, ann)
    if d is None:
        return None, 0
    d, s2 = _create(d, cre)
    if d is None:
        return None, 0
    return d, s1 * s2


def _chain(det, ops):
    """Apply ``(kind, so)`` operators rightmost first: (det', sign)."""
    d, s = det, 1
    for kind, so in ops:
        d, s1 = (_annihilate if kind == 'ann' else _create)(d, so)
        if d is None:
            return None, 0
        s *= s1
    return d, s


def _h_eff(h1e_mo, eri_mo, ncore):
    """``h_eff[p, q] = h + sum_c [2 (pq|cc) - (pc|cq)]`` (chemist order)."""
    h1e = np.asarray(h1e_mo, dtype=float)
    eri = np.asarray(eri_mo, dtype=float)
    out = np.array(h1e, copy=True)
    for c in range(ncore):
        out += 2.0 * eri[:, :, c, c] - eri[:, c, c, :]
    return out


def _spin_chains(ops_by_e):
    """Expand an E-operator product into its spin-orbital chains.

    ``ops_by_e`` lists the (ann, cre) index pairs in application order:
    the RIGHTMOST E of the product comes first, matching how the term
    tables below transcribe eq. (12) (right to left).  Each E carries
    its own spin, so an m-factor product yields 2**m chains; a chain is
    the flat operator sequence ``_chain`` applies.
    """
    import itertools

    chains = []
    for spins in itertools.product((0, 1), repeat=len(ops_by_e)):
        ops = []
        for (ann, cre), sigma in zip(ops_by_e, spins):
            ops.append(('ann', 2 * ann + sigma))
            ops.append(('cre', 2 * cre + sigma))
        chains.append(ops)
    return chains


def _class_pattern_terms(label, key, heff, heff_p, eri, ncore, nact, nmo):
    """The spin-expanded (weight, ops) list of one class pattern."""
    act = list(range(ncore, ncore + nact))
    virt = list(range(ncore + nact, nmo))
    terms = []
    if label == 'S_ia':
        i, a = key
        for t in act:
            for u in act:
                terms += [(eri[a, i, t, u],
                           _spin_chains([(u, t), (i, a)]))]
                terms += [(eri[a, u, t, i],
                           _spin_chains([(u, a), (i, t)]))]
        for t in act:
            terms += [(heff[a, i], _spin_chains([(i, a)]))]
    elif label == 'S_ijab':
        i, j, a, b = key
        g = (1.0 - 0.5 * (i == j)) * (1.0 - 0.5 * (a == b))
        terms += [(g * eri[i, a, j, b], _spin_chains([(j, b), (i, a)]))]
        terms += [(g * eri[i, b, j, a], _spin_chains([(j, a), (i, b)]))]
    elif label == 'S_a':
        (a,) = key
        for t in act:
            for u in act:
                for v in act:
                    terms += [(eri[a, u, t, v],
                               _spin_chains([(v, t), (u, a)]))]
        for t in act:
            terms += [(heff_p[a, t], _spin_chains([(t, a)]))]
    elif label == 'S_iab':
        i, a, b = key
        g = 1.0 - 0.5 * (a == b)
        for t in act:
            terms += [(g * eri[a, i, b, t], _spin_chains([(t, b), (i, a)]))]
            terms += [(g * eri[b, i, a, t], _spin_chains([(t, a), (i, b)]))]
    elif label == 'S_i':
        (i,) = key
        for t in act:
            for u in act:
                for v in act:
                    terms += [(eri[u, i, t, v],
                               _spin_chains([(v, t), (i, u)]))]
        for t in act:
            terms += [(heff[t, i], _spin_chains([(i, t)]))]
    elif label == 'S_ija':
        i, j, a = key
        g = 1.0 - 0.5 * (i == j)
        for t in act:
            terms += [(g * eri[i, t, j, a], _spin_chains([(i, t), (j, a)]))]
            terms += [(g * eri[i, a, j, t], _spin_chains([(j, t), (i, a)]))]
    elif label == 'S_ij':
        i, j = key
        g = 1.0 - 0.5 * (i == j)
        for t in act:
            for u in act:
                terms += [(g * eri[u, i, t, j],
                           _spin_chains([(j, t), (i, u)]))]
    elif label == 'S_ab':
        a, b = key
        g = 1.0 - 0.5 * (a == b)
        for t in act:
            for u in act:
                terms += [(g * eri[a, u, b, t],
                           _spin_chains([(t, b), (u, a)]))]
    return terms


def _class_patterns(ncore, nact, nmo):
    """Every (label, pattern-key) of the eight classes."""
    import itertools

    core = list(range(ncore))
    act = list(range(ncore, ncore + nact))
    virt = list(range(ncore + nact, nmo))
    out = []
    for i in core:
        for a in virt:
            out.append(('S_ia', (i, a)))
    for i, j in itertools.combinations_with_replacement(core, 2):
        for a, b in itertools.combinations_with_replacement(virt, 2):
            out.append(('S_ijab', (i, j, a, b)))
    for a in virt:
        out.append(('S_a', (a,)))
    for i in core:
        for a, b in itertools.combinations_with_replacement(virt, 2):
            out.append(('S_iab', (i, a, b)))
    for i in core:
        out.append(('S_i', (i,)))
    for i, j in itertools.combinations_with_replacement(core, 2):
        for a in virt:
            out.append(('S_ija', (i, j, a)))
    for i, j in itertools.combinations_with_replacement(core, 2):
        out.append(('S_ij', (i, j)))
    for a, b in itertools.combinations_with_replacement(virt, 2):
        out.append(('S_ab', (a, b)))
    return out


def class_vectors(h1e_mo, eri_mo, c_ci, dets, eps, ncore=0, nact=2):
    """The explicit SC perturber vector of every class pattern.

    Returns ``{(label, key): {determinant: amplitude}}`` over spin-orbital
    bitmaps covering core, active and external orbitals.  Determinants
    that receive no contribution simply do not appear.
    """
    h1e = np.asarray(h1e_mo, dtype=float)
    eri = np.asarray(eri_mo, dtype=float)
    c = np.asarray(c_ci, dtype=float)
    dets = [int(d) for d in dets]
    nmo = h1e.shape[0]
    heff = _h_eff(h1e, eri, ncore)
    heff_p = np.array(heff, copy=True)
    for t in range(ncore, ncore + nact):
        heff_p -= eri[:, t, t, :]
    out = {}
    for label, key in _class_patterns(ncore, nact, nmo):
        term_list = _class_pattern_terms(label, key, heff, heff_p, eri,
                                         ncore, nact, nmo)
        vec = {}
        for det, amp in zip(dets, c):
            for w, chains in term_list:
                if w == 0.0:
                    continue
                for ops in chains:
                    d2, sg = _chain(det, ops)
                    if d2 is None:
                        continue
                    vec[d2] = vec.get(d2, 0.0) + amp * w * sg
        out[(label, key)] = vec
    return out


def _op_expectation(vec, ann, cre):
    """``<V| E_ann^cre |V>`` spin summed over both spin components."""
    tot = 0.0
    for sigma in (0, 1):
        so_a, so_c = 2 * ann + sigma, 2 * cre + sigma
        for det, amp in vec.items():
            d, s = e_operator(det, so_a, so_c)
            if d is None:
                continue
            tot += amp * s * vec.get(d, 0.0)
    return tot


def _ee_expectation(vec, first, second):
    """``<V| E_{p1}^{q1} E_{p2}^{q2} |V>``, both E's spin summed."""
    import itertools

    (p1, q1), (p2, q2) = first, second
    tot = 0.0
    for s1, s2 in itertools.product((0, 1), repeat=2):
        ops = [('ann', 2 * p2 + s2), ('cre', 2 * q2 + s2),
               ('ann', 2 * p1 + s1), ('cre', 2 * q1 + s1)]
        for det, amp in vec.items():
            d, sg = _chain(det, ops)
            if d is None:
                continue
            tot += amp * sg * vec.get(d, 0.0)
    return tot


def _hv_expectation(vec, h1e_mo, eri_mo, ncore, nact):
    """``<V|H_v|V>`` with the source's eq.-5 active Hamiltonian.

    Brute force over the active index quadruple and the vector's
    determinant pairs: fine for the small spaces this module serves,
    recorded as the cost boundary in the module docstring.
    """
    h1e = np.asarray(h1e_mo, dtype=float)
    eri = np.asarray(eri_mo, dtype=float)
    heff = _h_eff(h1e, eri, ncore)
    act = list(range(ncore, ncore + nact))
    total = 0.0
    for t in act:
        for u in act:
            w = heff[t, u]
            if w != 0.0:
                total += w * _op_expectation(vec, u, t)
    for t in act:
        for u in act:
            for v in act:
                for w in act:
                    coef = eri[t, u, v, w]
                    if coef == 0.0:
                        continue
                    term = coef * _ee_expectation(vec, (u, t), (w, v))
                    if u == v:
                        term -= coef * _op_expectation(vec, w, t)
                    total += 0.5 * term
    return total


_HOLES = {'S_ia': 1, 'S_ijab': 2, 'S_a': 0, 'S_iab': 1, 'S_i': 1,
          'S_ija': 2, 'S_ij': 2, 'S_ab': 0}
_PARTICLES = {'S_ia': 1, 'S_ijab': 2, 'S_a': 1, 'S_iab': 2, 'S_i': 0,
              'S_ija': 1, 'S_ij': 0, 'S_ab': 2}


def class_denominators(h1e_mo, eri_mo, c_ci, dets, eps, ncore=0, nact=2,
                       *, ipea=True):
    """The energy denominator ``Delta = E0 - E_l^(k)`` of every pattern.

    ``Delta = sum(particles eps) - sum(holes eps) + eps_eff`` with
    ``eps_eff = E_v0 - <V|H_v|V>/N`` (the commutator expectation of the
    source, written out).  The class patterns' hole and particle counts
    are fixed by the eq.-12 operators.  With ``ipea=True`` the two
    fourth-order-density classes use the 2020 sequel's substitution
    instead: ``S_a`` gets ``eps_eff = 0.25 - eps_avg`` and ``S_i`` gets
    ``eps_avg + 0.25``, with ``eps_avg`` the density-weighted average
    active orbital energy (weights the reference occupations, normalized
    by the active electron count).
    """
    from ..analysis.ci import occupations_from_ci

    h1e = np.asarray(h1e_mo, dtype=float)
    eri = np.asarray(eri_mo, dtype=float)
    c = np.asarray(c_ci, dtype=float)
    dets = [int(d) for d in dets]
    eps = np.asarray(eps, dtype=float)
    vecs = class_vectors(h1e, eri, c, dets, eps, ncore=ncore, nact=nact)
    ref = {int(d): float(a) for d, a in zip(dets, c)}
    e_v0 = _hv_expectation(ref, h1e, eri, ncore, nact)
    n_a, n_b, _ = occupations_from_ci(c, dets, ncore=ncore, nact=nact)
    n_act = int(n_a.sum() + n_b.sum())
    eps_avg = float(((n_a + n_b) * eps[ncore:ncore + nact]).sum()
                    / n_act)
    out = {}
    for (label, key), vec in vecs.items():
        n_l = sum(a * a for a in vec.values())
        if n_l < 1e-14:
            continue
        holes = [key[0]] if _HOLES[label] == 1 else (
            list(key[:2]) if _HOLES[label] == 2 else [])
        if _PARTICLES[label] == 1:
            particles = [key[-1]]
        elif _PARTICLES[label] == 2:
            particles = list(key[-2:])
        else:
            particles = []
        if ipea and label == 'S_a':
            eps_eff = 0.25 - eps_avg
        elif ipea and label == 'S_i':
            eps_eff = eps_avg + 0.25
        else:
            eps_eff = e_v0 - _hv_expectation(vec, h1e, eri, ncore,
                                             nact) / n_l
        delta = (sum(eps[a] for a in particles)
                 - sum(eps[i] for i in holes) + eps_eff)
        out[(label, key)] = float(delta)
    return out


class PerturbedDensities:
    """The first-order blocks and the normalization behind them.

    ``d_ext`` is indexed over the external MOs, ``d_int`` over the
    active MOs; ``n1`` is the first-order wavefunction norm
    ``1 + sum_l N_l / Delta_l^2`` (main-text eq. 17).  ``weights`` maps
    each surviving class pattern to its contribution ``N_l / Delta_l^2``.
    """

    def __init__(self, d_int, d_ext, n1, weights, evidence):
        self.d_int = d_int
        self.d_ext = d_ext
        self.n1 = n1
        self.weights = weights
        self.evidence = evidence


def perturbed_densities(h1e_mo, eri_mo, c_ci, dets, eps, ncore=0, nact=2,
                        *, ipea=True, classes=None):
    """The internal and external blocks of the first-order density.

    Main-text eqs. 20/21: the cross terms collapse to same-pattern
    squares (each ``E`` conserves the external and internal hole and
    particle counts), so the blocks are

        d_ext[a, b] = sum_l <V_l| E_b^a |V_l> / Delta_l^2 / N1
        d_int[i, j] = (2 delta_ij
                       + sum_l <V_l| E_j^i |V_l> / Delta_l^2) / N1

    with the reference contribution to the external block identically
    zero (eq. 20a) and the ``2 delta`` of eq. (21a).  ``classes``, when
    given, restricts the sum to those labels (used by the tests to
    isolate one class).
    """
    h1e = np.asarray(h1e_mo, dtype=float)
    eri = np.asarray(eri_mo, dtype=float)
    c = np.asarray(c_ci, dtype=float)
    dets = [int(d) for d in dets]
    nmo = h1e.shape[0]
    eps = np.asarray(eps, dtype=float)
    virt = list(range(ncore + nact, nmo))
    act = list(range(ncore, ncore + nact))
    vecs = class_vectors(h1e, eri, c, dets, eps, ncore=ncore, nact=nact)
    denoms = class_denominators(h1e, eri, c, dets, eps, ncore=ncore,
                                nact=nact, ipea=ipea)
    weights = {}
    n1_acc = 0.0
    d_ext = np.zeros((len(virt), len(virt)))
    d_int = 2.0 * np.eye(len(act))
    for (label, key), vec in vecs.items():
        if classes is not None and label not in classes:
            continue
        delta = denoms.get((label, key))
        if delta is None:
            continue
        n_l = sum(a * a for a in vec.values())
        # The density carries the *unnormalized* vectors over 1/Delta^2
        # (main-text eq. 16); N_l belongs to the norm N1 (eq. 17) only.
        # Multiplying both by N_l/Delta^2 was measured wrong by the
        # assembly test (a factor N_l per pattern).
        weights[(label, key)] = n_l / delta ** 2
        n1_acc += n_l / delta ** 2
        inv_d2 = 1.0 / delta ** 2
        for a_i, a in enumerate(virt):
            for b_i, b in enumerate(virt):
                d_ext[a_i, b_i] += _op_expectation(vec, b, a) * inv_d2
        for i_i, i in enumerate(act):
            for j_i, j in enumerate(act):
                d_int[i_i, j_i] += _op_expectation(vec, j, i) * inv_d2
    n1 = 1.0 + n1_acc
    d_ext = 0.5 * (d_ext + d_ext.T) / n1
    d_int = 0.5 * (d_int + d_int.T) / n1
    top = sorted(weights.items(), key=lambda kv: -kv[1])[:3]
    evidence = [
        f'first-order norm N1 = {n1:.6f} over {len(weights)} class '
        f'patterns',
        f'largest weights: ' + ', '.join(
            f'{lab}{key}: {w:.4g}' for (lab, key), w in top),
    ]
    return PerturbedDensities(d_int, d_ext, n1, weights, evidence)


class NoonProvider:
    """Quasi-natural occupations per block, ready for the selection rule.

    ``noon_internal`` / ``noon_external`` are ascending eigenvalues
    (eigh order) of the two blocks; the coefficient matrices are the
    quasi-natural orbitals.  The blocks are diagonalized separately --
    the source never forms the active/active block, so the occupations
    do not sum to the electron count (their own caveat, kept).
    """

    def __init__(self, noon_internal, vectors_internal, noon_external,
                 vectors_external, densities, evidence):
        self.noon_internal = noon_internal
        self.vectors_internal = vectors_internal
        self.noon_external = noon_external
        self.vectors_external = vectors_external
        self.densities = densities
        self.evidence = evidence


def ass1st_noon(h1e_mo, eri_mo, c_ci, dets, eps, ncore=0, nact=2, *,
                ipea=True):
    """The two blocks' quasi-natural occupations of one reference.

    Plain ``numpy.linalg.eigh``: this layer carries no derivatives (the
    differentiable consumers of a selection are the chain quantities
    the margin diagnostics use, not the selection itself), so the
    degeneracy-safe decomposition of the kernel is not needed here --
    recorded as the layer boundary.
    """
    dens = perturbed_densities(h1e_mo, eri_mo, c_ci, dets, eps,
                               ncore=ncore, nact=nact, ipea=ipea)
    w_i, u_i = np.linalg.eigh(dens.d_int)
    w_e, u_e = np.linalg.eigh(dens.d_ext)
    evidence = list(dens.evidence)
    evidence.append('internal occupations '
                    + np.array2string(w_i, precision=6))
    evidence.append('external occupations '
                    + np.array2string(w_e, precision=6))
    return NoonProvider(w_i, u_i, w_e, u_e, dens, evidence)
