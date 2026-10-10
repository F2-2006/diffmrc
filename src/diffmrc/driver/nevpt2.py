"""The differentiable NEVPT2 chain, assembled in the driver layer.

The composition mirrors the validated per-class chain of the original
t-scripts (t121/t123/t132), which is the only path in this project whose
NEVPT2 numbers were checked against an independent implementation:

    CASCI in the theta-rotated basis  (provides the CI vector)
      -> canonicalize_jax             (active natural orbitals + sub-Fock
                                       diagonalization; the NEVPT2
                                       denominators are defined only for
                                       the canonical set)
      -> full-space MO integrals in the canonical basis
      -> CASCI again                  (energy and CI in that basis)
      -> nevpt2_e2_full2 with the per-spin link tables

Two facts about ``nevpt2_e2_full2`` shape this module and were verified
by reading the frozen source rather than assumed:

  * it slices ``eri_mo[nocc:, ...]`` for the external classes and hands
    the full ``eri_mo`` to ``e2_core`` for the core-class mixed blocks,
    so the integrals handed in must span the *whole* MO space -- a
    truncated ``[:ncore+nact]`` array silently drops the virtual space;
  * with ``links_a``/``links_b`` left at ``None`` the ``ncore > 0``
    branch skips the entire core-class contribution (Si, Sij, Sir,
    Sijr, Srsi, Sijrs). For any system with a frozen core -- i.e. all of
    the f-block ones -- the links are mandatory, not an optimisation.

``e2_scale`` exists for one reason: with it set to zero the joint
function reduces to the CASSCF energy function, whose nuclear gradient
is already validated against invariance to 1e-14. That is the only
"known answer" this chain has.

Two entry points share the same core (``_components_from``):

  * ``make_nevpt2_joint_fn``  -- ``(mol, theta)``, the SCF inside, for
    everything that differentiates with respect to the geometry;
  * ``make_nevpt2_theta_fn``  -- ``theta`` only, the reference computed
    once at construction, for the theta optimizer, which wraps its
    objective in ``jax.jit`` (a context this pyscfad build cannot trace
    its SCF in -- see the factory docstring).
"""
import numpy as np

# Kernel symbols are imported at module level by discipline: the t397
# trace tool missed this file while this import was function-scoped, and
# the same mistake would hide any future kernel use from a static audit
# (tests/core/test_kernel_binding.py enforces the rule). Heavy optional
# dependencies keep the deferred style -- see ``run.py`` for why.
from ..core.eigh import eigh_cluster_ift, eigh_safe

__all__ = ['make_nevpt2_joint_fn', 'make_nevpt2_theta_fn',
           'nevpt2_nuclear_gradient', 'nevpt2_hvp',
           'nevpt2_relaxed_hessian']


def _spin_links(nact, nele):
    """Per-spin CI link tables (static numpy; depend only on the space)."""
    from pyscf.fci import cistring

    na = nele // 2
    nb = nele - na
    la = cistring.gen_linkstr_index(range(nact), na)
    lb = cistring.gen_linkstr_index(range(nact), nb)
    fmt = lambda l: [[tuple(x) for x in l[s]] for s in range(len(l))]
    return fmt(la), fmt(lb)


def _casscf_solution(C0_, h_, e_, th_, *, ncore, nact, nele, nmo, pairs,
                     ci_solver='plain', eig_solver='default'):
    """The rotated, canonicalized CASSCF state at theta.

    Returns ``(E0c, C_can, ci2, eps, h1c, eric)``: the canonical-basis
    energy, orbitals and CI vector, the orbital energies, and the
    canonical-basis integrals (which the energy path reuses for the
    correlation contraction). Extracted from ``_components_from`` so that
    the energy path and the dipole path (``driver.properties``) evaluate
    the same object with the same conventions; two copies of this
    sequence would be free to drift apart, and a drift here is invisible
    in every downstream number.

    ``ci_solver`` selects the CI eigensolver, following the instruction in
    ``core.casci.cs_casci_blocked``'s own docstring: the plain solver for
    the first derivative and below (more thoroughly validated), the
    S_z-blocked solver wherever a second derivative is required (the
    plain one's ``eigh`` vjp produces NaN at the cross-S_z degeneracy --
    measured here as 0/0 for H2O/6-31G CAS(2,2), whose CI matrix is
    three-fold degenerate in the unblocked layout).
    """
    from jax import numpy as jnp

    from ..core.casci import cs_casci_blocked
    from ..core.nevpt2.impl import _casci_ci, canonicalize_jax
    from ..core.rotation import block_rotation

    nm = ncore + nact
    # ``eig_solver`` selects the eigen-kernel inside the CI solvers and the
    # canonicalization. 'default' keeps each solver's validated choice
    # (eigh_safe inside the blocked solver, plain eigh elsewhere);
    # 'lanczos' routes every kernel through core.lanczos, which is the
    # only setting whose second derivatives are finite on matrices with
    # bit-exact degenerate pairs (CO/N2 and every linear molecule).
    blk_eig = 'eigh_safe' if eig_solver == 'default' else eig_solver
    pln_eig = 'eigh' if eig_solver == 'default' else eig_solver

    def ci_solve(h1, eri):
        if ci_solver == 'blocked':
            return cs_casci_blocked(h1[:nm, :nm], eri[:nm, :nm, :nm, :nm],
                                    ncore=ncore, nact=nact, nele_act=nele,
                                    eig_solver=blk_eig)
        if ci_solver == 'plain':
            return _casci_ci(h1, eri, ncore, nact, nele, eig_solver=pln_eig)
        raise ValueError('unknown ci_solver: {!r}'.format(ci_solver))

    C2 = jnp.asarray(C0_) @ block_rotation(th_, nmo, pairs)

    h1_, eri_ = _stepwise_mo_ints(C2, h_, e_)
    _, ci = ci_solve(h1_, eri_)
    C_can, _, eps = canonicalize_jax(C2, h_, e_, ci, ncore=ncore,
                                     nact=nact, nele_act=nele,
                                     eig_solver=pln_eig)
    h1c, eric = _stepwise_mo_ints(C_can, h_, e_)
    E0c, ci2 = ci_solve(h1c, eric)
    return E0c, C_can, ci2, eps, h1c, eric


def _stepwise_mo_ints(Cx, h_, e_):
    """MO integrals by four sequential contractions.

    The four-operand einsum must be split into four sequential
    contractions. XLA lowers the one-shot form on a 96^4 array with only
    ~1e-6 absolute accuracy (measured: max|delta| = 1.28e-6 against the
    stepwise transform, activity block 8.9e-16), and that perturbation is
    amplified by the ~1e-15 gaps of the generalized Fock's degenerate
    pairs -- linear molecules have exact pi/delta degeneracies -- so the
    eigenvector basis inside the degenerate subspace comes out completely
    different (perturbation/gap ~ 1e9).  The energy is insensitive to it,
    but the theta gradient differed by 8% between the two contraction
    orders and the HVP was NaN for the one-shot form and finite for the
    stepwise one (t222/t224).
    """
    from jax import numpy as jnp

    h1 = jnp.einsum('ai,ab,bj->ij', Cx, h_, Cx)
    z = jnp.einsum('pi,pqrs->iqrs', Cx, e_)
    z = jnp.einsum('qj,iqrs->ijrs', Cx, z)
    z = jnp.einsum('rk,ijrs->ijks', Cx, z)
    eri = jnp.einsum('sl,ijks->ijkl', Cx, z)
    return h1, eri


def sa_cas_solution(h1e_mo, eri_mo, *, ncore, nact, nele, nroots,
                    eig_solver='default'):
    """State-averaged CAS solution: equal-weight mean energy and the
    per-root CI vectors.

    Thin wrapper over ``core.casci.cs_casci``'s state-average branch:
    ``nroots > 1`` with ``return_ci`` returns the equal-weight mean of
    the lowest ``nroots`` eigenvalues together with the corresponding
    eigenvectors as an ``(ndet, nroots)`` array.  Kept separate from
    ``_casscf_solution`` so the validated single-root path is untouched;
    both use the same call conventions (block slicing to ncore + nact,
    the plain eigensolver default for first derivatives).
    """
    from ..core.casci import cs_casci

    pln_eig = 'eigh' if eig_solver == 'default' else eig_solver
    nm = ncore + nact
    e_sa, vecs = cs_casci(h1e_mo[:nm, :nm], eri_mo[:nm, :nm, :nm, :nm],
                          ncore=ncore, nact=nact, nele_act=nele,
                          nroots=nroots, return_ci=True, eig_solver=pln_eig)
    # The core call returns a bare vector for nroots == 1 and an
    # (ndet, nroots) matrix otherwise; the SA interface is uniform.
    if nroots == 1:
        vecs = vecs[:, None]
    return e_sa, vecs


def canonicalize_sa_jax(C, h1e, eri, CIs, *, ncore, nact, nele, occ2_idx=(),
                        occ0_idx=(), weights=None, eig_solver='default'):
    """Average-density canonicalization for state-averaged solutions.

    The same sequence as ``core.nevpt2.impl.canonicalize_jax`` -- active
    natural orbitals, then the subspace Fock diagonalization of the
    (core + occupied-active) and (empty-active + external) blocks, then a
    CI re-solve in the canonical basis -- with the active one-particle
    density replaced by the equal-weight average over the state-averaged
    CI vectors (the density the state-averaged orbitals minimize
    against; the Fock matrix is likewise built from the average active
    density, which is the direct analogue of the core routine's use of
    the single root's density in the original basis).  Returns
    ``(C2, CIs2, eps)`` in the core routine's order.  With a single
    vector this reduces to the core routine bit-for-bit (identity test in
    ``tests/driver/test_sa_pipeline.py``); the single-root path is
    untouched.
    """
    from jax import numpy as jnp

    from ..core.nevpt2.impl import cas_dets_for2, make_dm123_sf

    eig = eigh_cluster_ift if eig_solver == 'lanczos' else eigh_safe
    nmo = C.shape[1]
    nocc = ncore + nact
    CIs = jnp.asarray(CIs)
    if CIs.ndim == 1:
        CIs = CIs[:, None]  # tolerate a bare vector (core-style nroots=1)
    nroots = int(CIs.shape[1])
    dets = cas_dets_for2(ncore, nact, nele)
    # Weighted average of the per-root active densities (equal weights
    # when none are given; a weight vector shorter than the root count is
    # extended by zero, a longer one is rejected).
    if weights is None:
        wv = jnp.full((nroots,), 1.0 / nroots)
    else:
        wv = jnp.asarray(weights, dtype=float)
        if wv.ndim != 1 or wv.shape[0] > nroots:
            raise ValueError(
                f'weights must be a vector of at most {nroots} entries, '
                f'got shape {tuple(wv.shape)}')
        if wv.shape[0] < nroots:
            wv = jnp.concatenate([wv, jnp.zeros(nroots - wv.shape[0])])
        wv = wv / jnp.sum(wv)
    dm1 = make_dm123_sf(CIs[:, 0], dets, nact, ncore)[0] * wv[0]
    for k in range(1, nroots):
        dm1 = dm1 + make_dm123_sf(CIs[:, k], dets, nact, ncore)[0] * wv[k]

    h1e_mo = jnp.einsum("ai,ab,bj->ij", C, h1e, C)
    eri_mo = jnp.einsum("pi,qj,rk,sl,pqrs->ijkl", C, C, C, C, eri)
    # 1) Active natural orbitals from the average density.  The columns
    #    are reversed because eigh returns ascending eigenvalues while
    #    the reference orders by descending occupation number.
    w_nat, U_nat = eig(dm1)
    U_nat = U_nat[:, ::-1]
    C1 = C.at[:, ncore:nocc].set(C[:, ncore:nocc] @ U_nat)
    # Re-solve the CI in the natural-orbital basis (mirrors the core
    # routine, which performs this solve as well; it enters no returned
    # quantity there, and the identity test holds both routines to the
    # same computation).
    h1e_mo1 = jnp.einsum("ai,ab,bj->ij", C1, h1e, C1)
    eri_mo1 = jnp.einsum("pi,qj,rk,sl,pqrs->ijkl", C1, C1, C1, C1, eri)
    sa_cas_solution(h1e_mo1, eri_mo1, ncore=ncore, nact=nact, nele=nele,
                    nroots=nroots, eig_solver=eig_solver)
    # 2) Generalized Fock matrix, F = h + J - K/2, in the original orbital
    #    basis and with the original (average) active density.
    D = jnp.zeros((nmo, nmo))
    for i in range(ncore):
        D = D.at[i, i].set(2.0)
    D = D.at[ncore:nocc, ncore:nocc].set(dm1)
    F = h1e_mo + jnp.einsum("pqrs,rs->pq", eri_mo, D) \
        - 0.5 * jnp.einsum("psrq,rs->pq", eri_mo, D)
    # Transform to the natural-orbital basis, rotating only the active block.
    U_full = jnp.eye(nmo).at[ncore:nocc, ncore:nocc].set(U_nat)
    F = U_full.T @ F @ U_full
    # Block diagonalization, matching the reference: only the core and
    # external blocks are rotated; the active space stays in its
    # natural-orbital basis and keeps the Fock diagonal as energies.
    occ2 = [ncore + i for i in occ2_idx]
    occ0 = [ncore + i for i in occ0_idx]
    blocks = [list(range(ncore)) + occ2, occ0 + list(range(nocc, nmo))]
    blocks = [b for b in blocks if b]
    active_idx = [ncore + i for i in range(nact)]
    C2 = C1.copy()
    eps_map = {}
    for blk in blocks:
        idx = jnp.asarray(blk, dtype=int)
        if len(blk) > 1:
            f_blk = F[idx][:, idx]
            w, u = eig(f_blk)
            C2 = C2.at[:, blk].set(C1[:, blk] @ u)
            for a, i in enumerate(blk):
                eps_map[i] = w[a]
        else:
            eps_map[blk[0]] = F[blk[0], blk[0]]
    for i in active_idx:
        eps_map[i] = F[i, i]
    eps = jnp.stack([eps_map[i] for i in range(nmo)])
    # 3) Final state-averaged CI solve in the canonical orbital basis.
    h1e_mo2 = jnp.einsum("ai,ab,bj->ij", C2, h1e, C2)
    eri_mo2 = jnp.einsum("pi,qj,rk,sl,pqrs->ijkl", C2, C2, C2, C2, eri)
    _, CIs2 = sa_cas_solution(h1e_mo2, eri_mo2, ncore=ncore, nact=nact,
                              nele=nele, nroots=nroots,
                              eig_solver=eig_solver)
    return C2, CIs2, eps


def _components_from(C0_, h_, e_, enuc, th_, *, ncore, nact, nele, nmo,
                     pairs, dets, links_a, links_b, ci_solver='plain',
                     eig_solver='default', want_e2=True):
    """``(E_casscf, E2)`` from reference quantities already in hand.

    Contains no SCF call and no Mole access -- theta and the precomputed
    arrays only. Both the joint function (which computes the reference
    from the Mole first) and the frozen-geometry theta function (which
    computes it once at construction) funnel through here, so the two
    cannot drift apart.
    """
    from jax import numpy as jnp

    from ..core.nevpt2.impl import nevpt2_e2_full2

    E0c, _C_can, ci2, eps, h1c, eric = _casscf_solution(
        C0_, h_, e_, th_, ncore=ncore, nact=nact, nele=nele, nmo=nmo,
        pairs=pairs, ci_solver=ci_solver, eig_solver=eig_solver)
    if not want_e2:
        # The caller multiplies e2 by zero; skipping the expensive
        # contraction is exact (0 * x = 0 in IEEE, and the VJP of a term
        # multiplied by zero is zero), and on actinide-sized virtual
        # spaces it is the difference between minutes and hours.
        return E0c + enuc, jnp.zeros(())
    e2, _ = nevpt2_e2_full2(h1c, eric, jnp.asarray(ci2), dets, E0c,
                            mo_energy=eps, ncore=ncore, nact=nact,
                            nocc=ncore + nact,
                            links_a=links_a, links_b=links_b)
    return E0c + enuc, e2


def make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, scf_builder,
                         e2_scale=1.0, ci_solver='plain',
                         eig_solver='default'):
    """Build ``E(mol, theta) = E_casscf + e2_scale * E2_nevpt2``.

    Parameters
    ----------
    ncore, nact, nele : int
        Active-space definition.
    nmo : int
        Number of basis functions.
    pairs : list of (int, int)
        Orbital rotation pairs; must include core<->virtual.
    scf_builder : callable
        ``mol -> mean field``. The builder declares the reference
        *flavour* only -- it constructs and configures the object
        (RHF/ROHF, X2C, ECP, damping) and the joint function runs
        ``kernel()`` on it, following ``make_joint_energy_fn``. For the
        f block the flavour is the caller's decision: a lanthanide or
        actinide run needs ROHF (open shell) with or without X2C, and an
        ECP route needs no X2C at all -- ``lambda m: pscf.ROHF(m).x2c1e()``
        versus ``lambda m: pscf.ROHF(m)``. An earlier version hard-coded
        ``pscf.RHF``, which silently excluded every open-shell system
        this project actually studies.
    e2_scale : float
        Multiplier on the correlation energy; 0 gives the CASSCF
        function exactly, which is the validation anchor.

    Returns
    -------
    callable
        ``(mol, theta) -> jax scalar``, differentiable in both.
    """
    from ..core.nevpt2.impl import cas_dets_for2

    dets = cas_dets_for2(ncore, nact, nele)
    links_a, links_b = _spin_links(nact, nele)
    kw = dict(ncore=ncore, nact=nact, nele=nele, nmo=nmo, pairs=pairs,
              dets=dets, links_a=links_a, links_b=links_b,
              ci_solver=ci_solver, eig_solver=eig_solver)

    def components(m_, th_, want_e2=True):
        """CASSCF energy and NEVPT2 correction, both differentiable."""
        mf_ = scf_builder(m_)
        mf_.kernel()
        # get_hcore carries the ECP and the X2C rotation; the raw
        # int1e_kin + int1e_nuc sum would silently drop both.
        return _components_from(mf_.mo_coeff, mf_.get_hcore(),
                                m_.intor('int2e', aosym='s1'),
                                m_.energy_nuc(), th_, want_e2=want_e2, **kw)

    def E_joint(m_, th_):
        E_cas, e2 = components(m_, th_, want_e2=(e2_scale != 0.0))
        return E_cas + e2_scale * e2

    E_joint.components = components      # the anchor test uses both
    return E_joint


def sa_vertical_excitations(states):
    """Vertical excitation energies from per-state ``(E_k, e2_k)`` entries.

    ``states`` is the list ``make_nevpt2_sa_fn`` returns; state 0 is the
    reference.  Returns one dict per state with ``dE_casscf`` and
    ``dE_nevpt2`` in the same energy unit as the input (the e2 entries
    carry the caller's ``e2_scale`` already).  Plain arithmetic on the
    component outputs -- no Mole and no differentiation here; the
    theta-derivative of these differences is whatever the components
    carry.
    """
    E0, e20 = states[0]
    return [{'dE_casscf': Ek - E0,
             'dE_nevpt2': (Ek + e2k) - (E0 + e20)}
            for Ek, e2k in states]


def sa_nevpt2_state_table(out, dets, *, tol=1e-9):
    """Physical-state table from a state-averaged NEVPT2 output.

    ``out`` is what ``make_nevpt2_sa_fn`` / ``make_nevpt2_sa_theta_fn``
    return.  The roots are collapsed with ``analysis.states.state_groups``
    (energy-degenerate spin components form one state, <S^2> labels the
    multiplicity, the lowest component with a finite NEVPT2 energy is the
    group's representative) and every group gains ``dE_casscf`` /
    ``dE_nevpt2`` -- the vertical excitations of its representative, in
    the caller's energy unit and ``e2_scale``.  This is the composition
    the state-averaged driver scripts otherwise assemble by hand; keeping
    it here makes the root table a library output rather than script
    glue.

    Returns ``(rows, groups)`` exactly as ``state_groups``, with the two
    extra keys on the groups.
    """
    from ..analysis.states import state_groups

    energies = [float(e) for e, _ in out['states']]
    usable = [bool(np.isfinite(float(e2))) for _, e2 in out['states']]
    rows, groups = state_groups(energies, out['cis'], dets, tol=tol,
                                usable=usable)
    vees = sa_vertical_excitations(out['states'])
    for g in groups:
        r = vees[g['representative']]
        g['dE_casscf'] = float(r['dE_casscf'])
        g['dE_nevpt2'] = float(r['dE_nevpt2'])
    return rows, groups


def _sa_components_from(C0_, h_, e_, enuc, th_, *, ncore, nact, nele, nmo,
                        pairs, nroots, dets, links_a, links_b,
                        state_weights=None, e2_scale=1.0,
                        eig_solver='default', want_e2=True):
    """``(E_sa, states, cis)`` from reference quantities already in hand.

    The state-averaged analogue of ``_components_from``: contains no SCF
    call and no Mole access -- theta and the precomputed arrays only.
    Both the joint factory and the frozen-geometry theta factory funnel
    through here, so the two cannot drift apart.
    """
    from jax import numpy as jnp

    from ..core.nevpt2.impl import nevpt2_e2_full2
    from ..core.rotation import block_rotation

    C2 = jnp.asarray(C0_) @ block_rotation(th_, nmo, pairs)
    h1_, eri_ = _stepwise_mo_ints(C2, h_, e_)
    E_sa_rot, CIs_rot = sa_cas_solution(h1_, eri_, ncore=ncore,
                                        nact=nact, nele=nele,
                                        nroots=nroots, eig_solver=eig_solver)
    C_can, CIs_can, eps = canonicalize_sa_jax(
        C2, h_, e_, CIs_rot, ncore=ncore, nact=nact, nele=nele,
        weights=state_weights, eig_solver=eig_solver)
    h1c, eric = _stepwise_mo_ints(C_can, h_, e_)
    # The public branch clamps to the determinant-space dimension; the
    # running-mean inversion is exact only up to that many roots, so the
    # effective count is probed first and everything below uses it.
    _, CIs_probe = sa_cas_solution(h1c, eric, ncore=ncore, nact=nact,
                                   nele=nele, nroots=nroots,
                                   eig_solver=eig_solver)
    n_eff = int(CIs_probe.shape[1])
    # Running means give the per-state energies; the largest call's
    # vectors are the canonical-basis ones the contractions consume.
    m_prev, k_prev, Ek_list, CIs = 0.0, 0, [], None
    for k in range(1, n_eff + 1):
        mk, CIs_k = sa_cas_solution(h1c, eric, ncore=ncore, nact=nact,
                                    nele=nele, nroots=k,
                                    eig_solver=eig_solver)
        Ek_list.append(k * mk - k_prev * m_prev)
        m_prev, k_prev, CIs = mk, k, CIs_k
    if state_weights is None:
        E_sa = m_prev  # equal weights: the largest running mean itself
    else:
        wv = jnp.asarray(state_weights, dtype=float)[:n_eff]
        wv = wv / jnp.sum(wv)
        E_sa = sum(wv[k] * Ek_list[k] for k in range(n_eff))
    # ``want_e2=False`` skips the per-state NEVPT2 contraction.  The
    # orbital-point search reads only ``E_sa``, which is fully computed
    # before this loop, so the skip changes neither the trajectory nor
    # the converged point -- it removes the dominant per-evaluation
    # cost from every optimizer step (the DVS scan's candidates converge
    # from trial orbitals far from the solution and pay this hundreds of
    # times per row).
    states = []
    for k in range(n_eff):
        if want_e2:
            e2k, _ = nevpt2_e2_full2(h1c, eric, CIs[:, k], dets, Ek_list[k],
                                     mo_energy=eps, ncore=ncore, nact=nact,
                                     nocc=ncore + nact, links_a=links_a,
                                     links_b=links_b)
            states.append((Ek_list[k] + enuc, e2_scale * e2k))
        else:
            states.append((Ek_list[k] + enuc, 0.0))
    # ``C`` is the canonical-basis orbital set the ``cis`` are expressed
    # in; the on-top columns (sa_mcpdft_table_from_mol) evaluate their
    # densities on exactly these orbitals.
    return {'E_sa': E_sa + enuc, 'states': states, 'cis': CIs, 'C': C_can}


def make_nevpt2_sa_theta_fn(mol, ncore, nact, nele, nmo, pairs, scf_builder,
                            nroots, e2_scale=1.0, state_weights=None,
                            eig_solver='default', c0=None):
    """Theta-only state-averaged components at a fixed geometry.

    The reference SCF and the AO integrals are computed once here and the
    returned ``components(th_) -> dict`` depends on theta alone.  The
    theta optimizer needs this form: this pyscfad build cannot trace the
    SCF (or the intor JVPs it calls) under a jit context -- the
    ``TracerArrayConversionError`` diagnostic is recorded in
    ``make_nevpt2_theta_fn``.  At a fixed geometry the frozen-reference
    form is the same function as the joint one: the SCF would run once
    either way, so no physics is dropped.

    ``c0`` replaces the SCF orbitals as the theta reference: the rotation
    chart is measured from these columns instead.  The DVS scan passes
    its test orbitals here, which is the paper's protocol -- the SA
    convergence starts from the candidate, not from the mean field.
    """
    from ..core.nevpt2.impl import cas_dets_for2

    mf_ = scf_builder(mol)
    mf_.kernel()
    C0_ = (mf_.mo_coeff if c0 is None
           else np.asarray(c0, dtype=float))
    h_ = mf_.get_hcore()
    e_ = mol.intor('int2e', aosym='s1')
    enuc = mol.energy_nuc()
    dets = cas_dets_for2(ncore, nact, nele)
    links_a, links_b = _spin_links(nact, nele)
    kw = dict(ncore=ncore, nact=nact, nele=nele, nmo=nmo, pairs=pairs,
              nroots=nroots, dets=dets, links_a=links_a, links_b=links_b,
              state_weights=state_weights, e2_scale=e2_scale,
              eig_solver=eig_solver)

    def components(th_, want_e2=True):
        return _sa_components_from(C0_, h_, e_, enuc, th_, want_e2=want_e2,
                                   **kw)

    return components


def make_nevpt2_sa_fn(ncore, nact, nele, nmo, pairs, scf_builder, nroots,
                      e2_scale=1.0, state_weights=None,
                      eig_solver='default'):
    """Build the state-averaged joint components: SA-CASSCF plus
    per-state NEVPT2.

    Returns ``components(mol, theta) -> dict`` with ``E_sa`` (the
    equal-weight state-averaged CASSCF total energy, differentiable in
    both arguments), ``states = [(E_k, e2_k)]`` -- per-state CASSCF
    total energies and NEVPT2 correlation corrections in the canonical
    basis whose orbitals the state-averaged density defines -- and
    ``cis`` (the ``(ndet, nroots)`` canonical-basis CI matrix, for spin
    and symmetry bookkeeping downstream).

    The per-state CASSCF energies are recovered from the running means of
    the public state-average branch (``M_k`` averages the lowest ``k``
    roots, so ``E_k = k M_k - (k-1) M_{k-1}``), which keeps the function
    on validated public interfaces and avoids assuming anything about the
    symmetry layout of the internal CI Hamiltonian.  The state-average
    branch is called at growing ``nroots``; the largest call also
    provides the canonical-basis vectors for the NEVPT2 contractions.
    With ``nroots=1`` the function reduces to ``make_nevpt2_joint_fn``
    (identity test in ``tests/driver/test_sa_pipeline.py``); the
    single-root path is untouched.
    """
    from jax import numpy as jnp

    from ..core.nevpt2.impl import cas_dets_for2, nevpt2_e2_full2
    from ..core.rotation import block_rotation

    dets = cas_dets_for2(ncore, nact, nele)
    links_a, links_b = _spin_links(nact, nele)

    def components(m_, th_):
        """SA-CASSCF energy and per-state NEVPT2 corrections."""
        mf_ = scf_builder(m_)
        mf_.kernel()
        # get_hcore carries the ECP and the X2C rotation; the raw
        # int1e_kin + int1e_nuc sum would silently drop both.
        C0_ = jnp.asarray(mf_.mo_coeff)
        h_ = mf_.get_hcore()
        e_ = m_.intor('int2e', aosym='s1')
        enuc = m_.energy_nuc()

        return _sa_components_from(
            C0_, h_, e_, enuc, th_, ncore=ncore, nact=nact, nele=nele,
            nmo=nmo, pairs=pairs, nroots=nroots, dets=dets,
            links_a=links_a, links_b=links_b, state_weights=state_weights,
            e2_scale=e2_scale, eig_solver=eig_solver)

    return components


def make_nevpt2_theta_fn(mol, ncore, nact, nele, nmo, pairs, scf_builder,
                         e2_scale=1.0, ci_solver='plain',
                         eig_solver='default'):
    """Theta-only joint energy at a fixed geometry (safe under jax.jit).

    The reference SCF and the AO integrals are computed once here; the
    returned function depends on theta alone. The theta optimizer needs
    this form: ``core.optimizer`` wraps its objective in ``jax.jit``, and
    this pyscfad build cannot trace the SCF under a jit context -- it
    raises ``TracerArrayConversionError`` inside ``get_occ`` (probe
    ``code/t190_probe_theta_grad.py``: ``jax.grad`` and ``jax.hessian``
    trace the SCF fine, ``jax.jit`` does not, both with the default and a
    1e-12 convergence tolerance). The same pattern is used by
    ``driver.kernel.energy_function``, which takes ``C0`` as an argument
    for exactly this reason.

    At a fixed geometry the frozen-reference form is the same function as
    the joint one: the SCF would run once either way, so no physics is
    dropped -- only the (irrelevant, here) geometry dependence.
    """
    from ..core.nevpt2.impl import cas_dets_for2

    mf_ = scf_builder(mol)
    mf_.kernel()
    C0_ = mf_.mo_coeff
    h_ = mf_.get_hcore()
    e_ = mol.intor('int2e', aosym='s1')
    enuc = mol.energy_nuc()
    dets = cas_dets_for2(ncore, nact, nele)
    links_a, links_b = _spin_links(nact, nele)
    kw = dict(ncore=ncore, nact=nact, nele=nele, nmo=nmo, pairs=pairs,
              dets=dets, links_a=links_a, links_b=links_b,
              ci_solver=ci_solver, eig_solver=eig_solver)

    def E_theta(th_):
        E_cas, e2 = _components_from(C0_, h_, e_, enuc, th_,
                                     want_e2=(e2_scale != 0.0), **kw)
        return E_cas + e2_scale * e2

    E_theta.components = lambda th_: _components_from(  # noqa: E731
        C0_, h_, e_, enuc, th_, **kw)
    return E_theta


def nevpt2_nuclear_gradient(mol, E_of):
    """dE/dR for an energy function that rebuilds its reference inside.

    The two requirements are the ones ``core.grad`` documents: the
    reference orbitals must be a function of the geometry (which
    ``E_of`` guarantees by rebuilding them), and any rotation
    parameters must be frozen (which the caller guarantees by closing
    over a fixed theta). The differentiation itself follows the
    validated route of ``core.grad.nuclear_gradient``: the pyscfad Mole
    is a pytree, and only its ``coords`` field carries the nuclear
    coordinates.

    Parameters
    ----------
    mol : pyscfad Mole
        Target geometry.
    E_of : callable(Mole) -> scalar
        The frozen-theta energy; typically ``lambda m: fn(m, theta)``.

    Returns
    -------
    numpy.ndarray, shape (natm, 3), in Eh/bohr
    """
    import jax

    g = jax.grad(E_of)(mol)
    return np.asarray(g.coords)


def nevpt2_hvp(mol, E_of, v):
    """Hessian-vector product, forward-over-reverse -- one ``jax.jvp``.

    Cheaper than assembling the matrix and exact to machine precision,
    which is what makes it the primary route (the full matrix is built
    only when a caller needs its spectrum). The direction is applied as
    a pytree tangent: the pyscfad Mole pytree's coordinate leaf is the
    (natm, 3) array, the same leaf ``core.hessian`` reads Hessians back
    out of (``H.coords.coords``).

    ``E_of`` must be built with ``ci_solver='blocked'``: a second
    derivative through the plain CI eigensolver returns NaN wherever the
    unblocked CI matrix carries a cross-S_z degeneracy (measured: NaN
    already at the CASCI stage for H2O/6-31G CAS(2,2)). This is the
    instruction in ``cs_casci_blocked``'s own docstring.

    Parameters
    ----------
    mol : pyscfad Mole
    E_of : callable(Mole) -> scalar
        The frozen-theta energy.
    v : array_like, shape (natm, 3) or (3 * natm,)
        The direction. For a diatomic, every perpendicular direction is
        an exact zero mode (rigid rotation/translation), so a probe
        meant to test anything has to include the bond stretch.

    Returns
    -------
    numpy.ndarray, shape (natm, 3)  --  H @ v
    """
    import jax
    from jax import numpy as jnp
    from jax import tree_util

    v_arr = jnp.asarray(np.asarray(v, dtype=float).reshape(-1, 3))
    coords_ref = np.asarray(mol.atom_coords())
    leaves, treedef = tree_util.tree_flatten(mol)
    idx = [i for i, l in enumerate(leaves)
           if hasattr(l, 'shape') and tuple(np.shape(l)) == coords_ref.shape
           and np.allclose(np.asarray(l), coords_ref)]
    if len(idx) != 1:
        raise ValueError('cannot locate the coordinate leaf of the Mole pytree')
    tan_leaves = [jnp.zeros_like(l) for l in leaves]
    tan_leaves[idx[0]] = v_arr
    tangent = tree_util.tree_unflatten(treedef, tan_leaves)
    _, hv = jax.jvp(jax.grad(E_of), (mol,), (tangent,))
    return np.asarray(hv.coords)


def nevpt2_relaxed_hessian(mol, theta, E_casscf, E_nevpt2, rcond=None):
    """The NEVPT2 Hessian along the CASSCF-relaxed orbital path.

    The correlation energy is a correction evaluated on the CASSCF
    reference, so the orbital response entering its second derivative is
    the *CASSCF* one -- dtheta*/dR comes from the CASSCF Lagrangian --
    while the frozen curvature and the mixed term come from the NEVPT2
    energy itself. In Park's Lagrangian this is exactly the split between
    the CASSCF Z-vector (the response operator) and the NEVPT2 source
    terms (``文献细读/细读_NEVPT2梯度_BAGEL2019.md``, Eqs. 23-28).

    Optimizing theta on the SC-NEVPT2 surface is not an alternative: that
    surface is not smooth (the SC energy lacks invariance under inactive
    rotations, and its denominators cross zero). Measured here: the
    theta optimizer walks into a singularity from the zeros start
    (E = -3.2e7, |g| = 3.6e14) *and* from the CASSCF minimum
    (E = -1.1e6, |g| = 3.4e15).

    Returns the same record shape as ``core.hessian.implicit_hessian``,
    with ``H_total = H_RR(nevpt2) - M_nevpt2^T H_theta_theta(casscf)^-1
    M_nevpt2``.
    """
    import jax

    from ..core.hessian import (DEFAULT_RCOND, ImplicitHessianResult,
                                _frozen_hessian, _mixed_derivative)
    from ..core.stationary import curvature_report

    rcond = DEFAULT_RCOND if rcond is None else float(rcond)
    theta = jax.numpy.asarray(np.asarray(theta, dtype=float))

    H_tt = np.asarray(jax.hessian(lambda t: E_casscf(mol, t))(theta))
    H_tt = 0.5 * (H_tt + H_tt.T)
    M = _mixed_derivative(E_nevpt2, mol, theta)
    n_r = M.shape[1]
    H_RR = _frozen_hessian(E_nevpt2, mol, theta, n_r)
    H_RR = 0.5 * (H_RR + H_RR.T)

    H_tt_inv = np.linalg.pinv(H_tt, rcond=rcond)
    D = -H_tt_inv @ M
    H_resp = M.T @ D
    H_total = H_RR + H_resp
    H_total = 0.5 * (H_total + H_total.T)

    return ImplicitHessianResult(
        H_tt=H_tt, H_tt_inverse=H_tt_inv,
        curvature=curvature_report(H_tt),
        mixed=M, H_RR=H_RR, H_response=H_resp, H_total=H_total,
        dtheta_dR_max=float(np.abs(D).max()) if D.size else 0.0,
        rcond=float(rcond),
    )


def sa_nevpt2_table_from_mol(mol, family, scf_builder, *, nroots,
                             max_steps=400, refine_max_iter=40,
                             e2_scale=1.0):
    """State-averaged CASSCF + per-state NEVPT2 from a molecule, one call.

    The window comes from ``driver.orbitals.auto_window`` (the frontier
    identity rules); the rest is the validated composition the M5-C
    driver scripts otherwise assemble by hand: equal-weight SA for the
    orbital point, per-state NEVPT2 on it, the physical-state table by
    ``sa_nevpt2_state_table``.  A single equal-weight pass is used; the
    between-group second pass changes tables by tenths of an eV at this
    level of theory and can be added by the caller by re-running the
    factory with ``state_weights``.

    Returns a dict with the window triple, the frontier report that
    suggested it, the space dimensions, the orbital-point gradient norm,
    the table rows and groups, and the raw ``out`` for further analysis.
    """
    from ..core.nevpt2.impl import cas_dets_for2
    from ..core.orblib import orbital_pairs
    from .orbitals import auto_window, converge_orbital_point

    nmo = mol.nao_nr()
    mf = scf_builder(mol).run(verbose=0)
    report = auto_window(mol, mf, family)
    ncore = int(report['suggested_window']['ncore'])
    nact = int(report['suggested_window']['nact'])
    nele = int(mol.nelectron - 2 * ncore)

    pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)
    dets = cas_dets_for2(ncore, nact, nele)
    comp = make_nevpt2_sa_theta_fn(mol, ncore, nact, nele, nmo, pairs,
                                   scf_builder, nroots=nroots,
                                   e2_scale=e2_scale)
    point = converge_orbital_point(lambda th: comp(th, want_e2=False)['E_sa'],
                                   len(pairs), max_steps=max_steps,
                                   refine_max_iter=refine_max_iter)
    out = comp(point.theta)
    rows, groups = sa_nevpt2_state_table(out, dets)
    return {'window': {'ncore': ncore, 'nact': nact, 'nele': nele},
            'frontier_report': report, 'nmo': nmo, 'npar': len(pairs),
            'ndet': len(dets), 'grad_norm': float(point.grad_norm),
            'rows': rows, 'groups': groups, 'out': out}


def make_sa_energy_fn(mol, scf_builder, *, ncore, nact, nele, nmo, pairs,
                      nroots, c0=None, eig_solver='default'):
    """The state-averaged energy alone, for orbital-point searches.

    The full components' graph is heavy: it contains the running-mean
    loop (one CI solve per root count) and the canonicalization (a
    further CI solve), and tracing it costs ~8 minutes per eager
    evaluation at the DVS test-orbital windows (measured, t393).  The
    orbital-point search reads only ``E_sa``, and the equal-weight mean
    is a single call to the state-average branch on the rotated
    integrals -- eigenvalues do not care which active basis they are
    computed in, so this path differs from the full one only at the
    float-noise level (~1e-12), while its graph is roughly five times
    lighter to trace and evaluate.

    Same SCF/one-time construction contract as
    ``make_nevpt2_sa_theta_fn``; ``c0`` is the trial-orbital hook the
    DVS scan uses.
    """
    from ..core.rotation import block_rotation

    mf_ = scf_builder(mol)
    mf_.kernel()
    C0_ = (mf_.mo_coeff if c0 is None else np.asarray(c0, dtype=float))
    h_ = mf_.get_hcore()
    e_ = mol.intor('int2e', aosym='s1')
    enuc = mol.energy_nuc()

    def E_sa_of(th_):
        C2 = C0_ @ block_rotation(th_, nmo, pairs)
        h1_, eri_ = _stepwise_mo_ints(C2, h_, e_)
        e_sa, _ = sa_cas_solution(h1_, eri_, ncore=ncore, nact=nact,
                                  nele=nele, nroots=nroots,
                                  eig_solver=eig_solver)
        return e_sa + enuc

    return E_sa_of


def sa_mcpdft_table(mol, scf_builder, *, ncore, nact, nele, nroots,
                    c0=None, max_steps=400, refine_max_iter=40,
                    optimize_tol=None, refine_target=None,
                    e2_scale=1.0, grid=None, grid_level=3, eval_xc=None,
                    hyb=0.0):
    """SA-CASSCF + per-state one-step MC-PDFT on an explicit window.

    The window-explicit core of ``sa_mcpdft_table_from_mol``: the
    equal-weight SA-CASSCF point -- optionally started from ``c0``, the
    DVS scan passes its test orbitals there, which is the paper's
    protocol -- then one tPBE evaluation per root from that root's own
    density, on the canonical-basis orbitals the components return.
    The per-state NEVPT2 corrections ride along.

    Returns ``window``, ``nmo``, ``npar``, ``ndet``, ``grad_norm``,
    ``rows``, ``groups`` (with ``dE_casscf``/``dE_nevpt2``/
    ``dE_mcpdft``), the raw ``out`` and ``e_tpbe``.
    """
    import numpy as np
    from jax import numpy as jnp

    from ..core.mcpdft import eot_energy, pbe_eval_xc, pi_active, w_sigma
    from ..core.nevpt2.impl import cas_dets_for2
    from ..core.orblib import orbital_pairs
    from .kernel import frozen_grid
    from .orbitals import converge_orbital_point

    nmo = mol.nao_nr()
    pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)
    dets = cas_dets_for2(ncore, nact, nele)
    comp = make_nevpt2_sa_theta_fn(mol, ncore, nact, nele, nmo, pairs,
                                   scf_builder, nroots=nroots,
                                   e2_scale=e2_scale, c0=c0)
    # The search rides the light energy path (t393: the full graph's
    # eager trace alone costs ~8 min at these windows); the full
    # components are evaluated once at the converged point below.
    E_sa_light = make_sa_energy_fn(mol, scf_builder, ncore=ncore, nact=nact,
                                   nele=nele, nmo=nmo, pairs=pairs,
                                   nroots=nroots, c0=c0)
    converge_kw = dict(max_steps=max_steps, refine_max_iter=refine_max_iter)
    # The scan's rows only need the orbital point to ~1e-7: the SA energy
    # is stationary there (second order) and the on-top column's leading
    # sensitivity to the point is ~3e-3 per radian, so a looser target
    # moves it by ~1e-10 Eh while cutting the Newton stage's evaluation
    # count.  None keeps the standard targets for every other caller.
    if optimize_tol is not None:
        converge_kw['optimize_tol'] = float(optimize_tol)
    if refine_target is not None:
        converge_kw['refine_target'] = float(refine_target)
    point = converge_orbital_point(E_sa_light, len(pairs), **converge_kw)
    out = comp(point.theta)
    rows, groups = sa_nevpt2_state_table(out, dets)

    if grid is None:
        grid = frozen_grid(mol, level=grid_level)
    if eval_xc is None:
        eval_xc = pbe_eval_xc

    C_can = np.asarray(out['C'], dtype=float)
    cis = np.asarray(out['cis'])
    Cact = jnp.asarray(C_can[:, ncore:ncore + nact])
    Ccore = jnp.asarray(C_can[:, :ncore])
    h1e = jnp.asarray(mol.intor('int1e_kin') + mol.intor('int1e_nuc'))
    eri = jnp.asarray(mol.intor('int2e', aosym='s1'))
    enuc = float(mol.energy_nuc())
    ao = jnp.asarray(grid.ao)
    ao_grad = jnp.asarray(grid.ao_grad)
    wts = jnp.asarray(grid.weights)
    acas = ao @ Cact

    def gd(D):
        return (jnp.einsum('gr,rs,gs->g', ao, D, ao),
                2.0 * jnp.einsum('gxr,rs,gs->gx', ao_grad, D, ao))

    e_tpbe = []
    for k in range(cis.shape[1]):
        ci_k = jnp.asarray(cis[:, k])
        W = w_sigma(ci_k, dets, nact, ncore)
        dm1s = jnp.einsum('spqI,I->spq', W, ci_k)
        Dact = jnp.einsum('rp,spq,tq->srt', Cact, dm1s, Cact)
        Dfull = Ccore @ Ccore.T + Dact
        rho_a, g_a = gd(Dfull[0])
        rho_b, g_b = gd(Dfull[1])
        rho_aa, _ = gd(Dact[0])
        rho_bb, _ = gd(Dact[1])
        Pi = pi_active(W, acas) + (rho_a * rho_b - rho_aa * rho_bb)
        E_ot = eot_energy(rho_a, g_a, rho_b, g_b, Pi, wts, eval_xc)
        D = Dfull[0] + Dfull[1]
        E_mcwfn = (enuc + jnp.einsum('ab,ab->', h1e, D)
                   + 0.5 * jnp.einsum('abcd,ab,cd->', eri, D, D))
        E_k = E_mcwfn + E_ot
        if hyb:
            # tPBE0: the convex combination with the state's own CASSCF
            # energy (the RDM split; see the kernel chain).
            E_k = (1.0 - hyb) * E_k + hyb * out['states'][k][0]
        e_tpbe.append(float(E_k))
    e_tpbe = np.asarray(e_tpbe)
    for g in groups:
        g['dE_mcpdft'] = float(e_tpbe[g['representative']] - e_tpbe[0])
    return {'window': {'ncore': ncore, 'nact': nact, 'nele': nele},
            'nmo': nmo, 'npar': len(pairs),
            'ndet': len(dets), 'grad_norm': float(point.grad_norm),
            'rows': rows, 'groups': groups, 'out': out,
            'e_tpbe': e_tpbe}


def sa_mcpdft_table_from_mol(mol, family, scf_builder, *, nroots,
                             max_steps=400, refine_max_iter=40,
                             e2_scale=1.0, grid=None, grid_level=3,
                             eval_xc=None, hyb=0.0):
    """State-averaged CASSCF + per-state one-step MC-PDFT table.

    The orbital point is the equal-weight SA-CASSCF solution of
    ``sa_nevpt2_table_from_mol`` -- same window rules, same convergence
    -- and on it every root gets one tPBE evaluation from its own
    density: the one-step MC-PDFT protocol of the DVS paper, whose
    tables carry exactly these mcscf/nevpt2/tpbe columns.  The per-state
    NEVPT2 corrections ride along (the SA machinery computes them for
    the orbital point anyway), so the result is the complete
    three-column table.

    The densities are evaluated on the canonical-basis orbitals the
    components return (``out['C']``), through the same primitives the
    ground-state chain uses (``core.mcpdft``; the jnp PBE kernel by
    default); the frozen grid comes from ``kernel.frozen_grid``.

    Returns the dict of ``sa_mcpdft_table`` plus ``frontier_report``;
    the groups carry ``dE_mcpdft`` next to the other two columns.
    """
    from .orbitals import auto_window

    nmo = mol.nao_nr()
    mf = scf_builder(mol).run(verbose=0)
    report = auto_window(mol, mf, family)
    ncore = int(report['suggested_window']['ncore'])
    nact = int(report['suggested_window']['nact'])
    nele = int(mol.nelectron - 2 * ncore)
    out = sa_mcpdft_table(mol, scf_builder, ncore=ncore, nact=nact,
                          nele=nele, nroots=nroots, max_steps=max_steps,
                          refine_max_iter=refine_max_iter, e2_scale=e2_scale,
                          grid=grid, grid_level=grid_level, eval_xc=eval_xc,
                          hyb=hyb)
    out['frontier_report'] = report
    return out
