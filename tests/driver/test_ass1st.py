"""The ASS1ST provider: class vectors, denominators, blocks.

Transcribed from Khedkar & Roemelt (JCTC 2019, main-text eqs. 9, 10, 12,
13-17, 20-21; SI section A) with the IPEA substitution of the 2020 sequel
(average active orbital energy + 0.25 Eh for the two fourth-order-density
classes).  The model is a four-MO chain -- one core, CAS(2,2), one external
-- small enough that every expectation is computable by hand.  Pure numpy:
the reference 1-RDM values used below are analytic for this vector, so no
jax is needed anywhere in this file.

Operator convention (the kernel's, from ``core.nevpt2.m4``): every operator
product is applied right to left, annihilations ascending within themselves,
creations descending; single-operator chains are unambiguous and that is
all this module builds.  ``h_eff`` follows the package's own production
form, ``h + sum_c [2 (pq|cc) - (pc|cq)]`` (``core/nevpt2/impl.py``); the
OCR of the source's eq. (9) drops the minus and the package is the arbiter.
"""
import numpy as np
import pytest


def _two_det_model():
    """One core (MO 0), CAS(2,2) (MOs 1-2), one external (MO 3).

    Integrals are synthetic but chemist-symmetric (the 8-fold transpose
    set), so every expectation below is computable by hand.
    """
    rng = np.random.default_rng(7)
    nmo = 4
    h1 = rng.normal(size=(nmo, nmo))
    h1 = 0.5 * (h1 + h1.T)
    eri = rng.normal(size=(nmo,) * 4)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(2, 3, 0, 1)
    return h1, eri


def _two_det_vector():
    """|core, act0^2> and |core, act1^2>, weights 0.6 / 0.4.

    Determinant bitmaps over all four MOs (eight spin orbitals; the core
    is doubly occupied in both): bits 0,1 core; 2,3 act0; 4,5 act1.
    """
    dets = [0b001111, 0b110011]
    c = np.array([np.sqrt(0.6), np.sqrt(0.4)])
    return dets, c


def test_the_single_excitation_operator_matches_the_analytic_dm1():
    """<E_t^t> over both spins equals the analytic occupation.

    For this vector orbital 0 is doubly occupied with weight 0.6 and
    orbital 1 with weight 0.4: dm1 = diag(1.2, 0.8).  The value is
    computed in-test from the weights, not recalled.
    """
    from diffmrc.driver.ass1st import e_operator

    ncore = 1
    dets, c = _two_det_vector()
    targets = (1.2, 0.8)
    for t, want in enumerate(targets):
        tot = 0.0
        for sigma in (0, 1):
            so = 2 * (ncore + t) + sigma
            m = np.zeros((2, 2))
            for I, dI in enumerate(dets):
                for J, dJ in enumerate(dets):
                    d, sg = e_operator(dJ, so, so)
                    if d is not None and d == dI:
                        m[I, J] = sg
            tot += float(c @ m @ c)
        assert tot == pytest.approx(want, abs=1e-12)


def test_the_double_external_class_denominator_is_the_bare_energy_difference():
    """Class S_ij,ab^(0): its Delta carries no effective potential.

    Its operator never touches the active space, so the general
    denominator machinery's eps_eff must come out zero and Delta must
    equal eps_a + eps_b - eps_i - eps_j exactly.  This is the anchor for
    the whole denominator construction.
    """
    from diffmrc.driver.ass1st import class_denominators

    h1, eri = _two_det_model()
    eps = np.array([-1.2, -0.4, 0.3, 0.9])
    dets, c = _two_det_vector()
    out = class_denominators(h1, eri, c, dets, eps, ncore=1, nact=2)
    delta = out[('S_ijab', (0, 0, 3, 3))]
    assert delta == pytest.approx(eps[3] + eps[3] - 2 * eps[0], abs=1e-10)


def test_the_mirrored_bitmap_operators_match_the_kernel_ones():
    """The local mirrors of ``m4.ann_so``/``cre_so`` must not drift.

    The driver module copies the kernel's two bitmap rules to stay
    numpy-only (``m4`` imports jax at module level); this jax-gated
    cross-check binds the copy to the original over every determinant
    of an eight-spin-orbital space.
    """
    pytest.importorskip('jax')
    from diffmrc.core.nevpt2.m4 import ann_so, cre_so
    from diffmrc.driver.ass1st import _annihilate, _create

    for det in range(256):
        for q in range(8):
            assert _annihilate(det, q) == ann_so(det, q)
            assert _create(det, q) == cre_so(det, q)


def test_a_vanishing_perturbation_leaves_the_reference_untouched():
    """With zero integrals every class vector is empty and the blocks
    collapse to the reference: d_ext = 0, d_int = 2 I, N1 = 1.

    (A single determinant does NOT give zero blocks -- it correlates
    with the external space like an MP2 reference; that expectation is
    wrong and this test does not make it.)
    """
    from diffmrc.driver.ass1st import perturbed_densities

    _, eri = _two_det_model()
    h1 = np.zeros((4, 4))
    eri0 = np.zeros((4, 4, 4, 4))
    eps = np.array([-1.2, -0.4, 0.3, 0.9])
    dets, c = _two_det_vector()
    out = perturbed_densities(h1, eri0, c, dets, eps, ncore=1, nact=2)
    assert np.allclose(out.d_ext, 0.0, atol=1e-12)
    assert np.allclose(out.d_int, 2.0 * np.eye(2), atol=1e-12)
    assert out.n1 == pytest.approx(1.0)
    assert out.weights == {}


def test_the_block_assembly_follows_the_pattern_weights():
    """Tr(d_ext) * N1 must equal sum_l n_particles N_l / Delta_l^2.

    The vector-side trace law (<V|sum E|V> = n_particles x N_l) has its
    own test; here the assembly arithmetic is pinned -- the summation,
    the 1/Delta^2 weights and the 1/N1 normalization -- against the
    same primitives.
    """
    from diffmrc.driver.ass1st import (
        class_denominators,
        class_vectors,
        perturbed_densities,
    )

    h1, eri = _two_det_model()
    eps = np.array([-1.2, -0.4, 0.3, 0.9])
    dets, c = _two_det_vector()
    subset = ('S_ia', 'S_ijab', 'S_a', 'S_iab', 'S_ija', 'S_ab')
    out = perturbed_densities(h1, eri, c, dets, eps, ncore=1, nact=2,
                              classes=subset)
    n_particles = {'S_ia': 1, 'S_ijab': 2, 'S_a': 1, 'S_iab': 2,
                   'S_ija': 1, 'S_ab': 2}
    vecs = class_vectors(h1, eri, c, dets, eps, ncore=1, nact=2)
    denoms = class_denominators(h1, eri, c, dets, eps, ncore=1, nact=2)
    want = sum(n_particles[lab] * sum(a * a for a in vec.values())
               / denoms[(lab, key)] ** 2
               for (lab, key), vec in vecs.items()
               if lab in subset and (lab, key) in denoms)
    assert np.trace(out.d_ext) * out.n1 == pytest.approx(want, abs=1e-10)
    assert out.n1 > 1.0


def test_the_quasi_natural_occupations_feed_the_selection_rule():
    from diffmrc.driver.ass1st import ass1st_noon
    from diffmrc.policy.rules import noon_selection

    h1, eri = _two_det_model()
    eps = np.array([-1.2, -0.4, 0.3, 0.9])
    dets, c = _two_det_vector()
    prov = ass1st_noon(h1, eri, c, dets, eps, ncore=1, nact=2)
    assert prov.noon_internal.shape == (2,)
    assert prov.noon_external.shape == (1,)
    assert np.all(np.isfinite(prov.noon_internal))
    assert np.all(np.isfinite(prov.noon_external))
    d = noon_selection(
        {f'int{k}': float(v) for k, v in enumerate(prov.noon_internal)},
        {f'ext{k}': float(v) for k, v in enumerate(prov.noon_external)},
        threshold=0.05)
    assert set(d.classification) == {'int0', 'int1', 'ext0'}
    assert prov.evidence


def test_per_pattern_external_trace_equals_the_particle_count():
    """<V|sum_b E_b^b|V> = n_particles x N_l for every pattern.

    Every determinant a class vector carries has exactly the class's
    number of external electrons (E_b^b is diagonal, so the trace is a
    weighted occupation count), which makes the identity analytic.  It
    pins the vector construction independently of any denominator.
    """
    from diffmrc.driver.ass1st import class_vectors

    h1, eri = _two_det_model()
    eps = np.array([-1.2, -0.4, 0.3, 0.9])
    dets, c = _two_det_vector()
    vecs = class_vectors(h1, eri, c, dets, eps, ncore=1, nact=2)
    n_particles = {'S_ia': 1, 'S_ijab': 2, 'S_a': 1, 'S_iab': 2,
                   'S_i': 0, 'S_ija': 1, 'S_ij': 0, 'S_ab': 2}
    ext_mask = (1 << 6) | (1 << 7)          # the one external MO
    checked = 0
    for (label, key), vec in vecs.items():
        if not vec:
            continue
        n_l = sum(a * a for a in vec.values())
        tot = sum(a * a * bin(d & ext_mask).count('1')
                  for d, a in vec.items())
        assert tot == pytest.approx(n_particles[label] * n_l, abs=1e-10), \
            (label, key)
        checked += 1
    assert checked > 0


def test_the_sa_densities_match_the_single_state_case_for_one_state():
    """One state, weight 1: the averaged blocks equal the single-state
    blocks bit for bit (the SA wrapper must be an exact superset)."""
    from diffmrc.driver.ass1st import (perturbed_densities,
                                       perturbed_densities_sa)

    h1, eri = _two_det_model()
    dets, c = _two_det_vector()
    eps = np.array([-0.5, -0.3, 0.4, 1.0])
    d1 = perturbed_densities(h1, eri, c, dets, eps, ncore=1, nact=2)
    dsa = perturbed_densities_sa(h1, eri, c, dets, eps, ncore=1, nact=2)
    assert np.array_equal(dsa.d_int, d1.d_int)
    assert np.array_equal(dsa.d_ext, d1.d_ext)


def test_the_sa_densities_are_the_weighted_mean_of_the_state_blocks():
    """Two states: D_sa = w1 D1 + w2 D2, blockwise, on the normalized
    per-state blocks (Khedkar & Roemelt 2020: average, then
    diagonalize)."""
    from diffmrc.driver.ass1st import (perturbed_densities,
                                       perturbed_densities_sa)

    h1, eri = _two_det_model()
    dets, c = _two_det_vector()
    eps = np.array([-0.5, -0.3, 0.4, 1.0])
    c_open = np.array([np.sqrt(0.4), -np.sqrt(0.6)])
    c2 = np.column_stack([c, c_open])
    w = [0.7, 0.3]
    d1 = perturbed_densities(h1, eri, c, dets, eps, ncore=1, nact=2)
    d2 = perturbed_densities(h1, eri, c_open, dets, eps, ncore=1, nact=2)
    dsa = perturbed_densities_sa(h1, eri, c2, dets, eps, ncore=1, nact=2,
                                 state_weights=w)
    ref_i = 0.7 * d1.d_int + 0.3 * d2.d_int
    ref_e = 0.7 * d1.d_ext + 0.3 * d2.d_ext
    assert np.allclose(dsa.d_int, ref_i, rtol=0, atol=1e-14)
    assert np.allclose(dsa.d_ext, ref_e, rtol=0, atol=1e-14)


def test_sa_noon_produces_occupations_for_every_block():
    from diffmrc.driver.ass1st import ass1st_noon_sa

    h1, eri = _two_det_model()
    dets, c = _two_det_vector()
    eps = np.array([-0.5, -0.3, 0.4, 1.0])
    c2 = np.column_stack([c, np.array([np.sqrt(0.4), -np.sqrt(0.6)])])
    noon = ass1st_noon_sa(h1, eri, c2, dets, eps, ncore=1, nact=2)
    assert noon.noon_internal.shape == (2,)
    assert noon.noon_external.shape == (1,)
    assert np.isfinite(noon.noon_internal).all()
    assert np.isfinite(noon.noon_external).all()
    assert 'state-averaged' in ' '.join(noon.evidence)
