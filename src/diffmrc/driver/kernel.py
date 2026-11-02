"""Assembling the numerical path the driver runs.

The pieces come from the frozen kernel, the policy layer and
``driver.orbitals``; this module decides the order and the reference
they are measured against. Only one step needs pyscfad -- the SCF that
produces the reference orbitals -- and it is isolated in its own
function so that every other piece can be tested without the kernel
installed.

The partition of the orbitals is taken from the occupations, never from
the index order. See ``partition_from_occupations`` for the measurement
that made that a rule.
"""
from dataclasses import dataclass, field

import numpy as np

from .orbitals import (converge_orbital_point,
                       converge_orbital_point_two_stage,
                       should_split_rotation_set)

__all__ = ['ActiveSpace', 'partition_from_occupations', 'energy_function',
           'search_point', 'FrozenGrid', 'frozen_grid',
           'mcpdft_energy_function', 'mcpdft_terms_function',
           'stage_one_block']

# When the analytic orbital Hessian is not finite at a symmetric stationary
# point, the curvature is taken from a symmetric difference of the analytic
# gradient. The cost is two gradient evaluations per rotation parameter, so
# the route is capped; above the cap the driver refuses and says why.
CURVATURE_FD_MAX_PARAMETERS = 300
CURVATURE_FD_STEP = 1e-5

# A point is treated as stationary when its gradient norm is at or below
# this value. The same threshold guarded the old gradient-based fallback
# retry; it is kept here because a curvature evaluated at a point the
# optimizer never reached is not a minimum certificate.
STATIONARY_GRADIENT = 1e-3

# A core orbital this close to the lowest active orbital is reported as a
# near-degenerate partner left outside the space. The value is diagnostic:
# it does not change any decision, it only makes the refusal readable.
NEAR_DEGENERATE_GAP = 0.2

# A minimum with no negative eigenvalue can still carry a zero mode, for
# example when one component of a degenerate shell was frozen outside the
# space. The smallest orbital-Hessian eigenvalue is always recorded; when
# it falls below this value the evidence says the minimum is not certified.
# The value is diagnostic: it never changes a self-check or a verdict.
FLAT_DIRECTION_TOL = 1e-6

# A reference core-active or active-virtual gap below this value means an
# exactly degenerate shell was split across the space boundary. The
# threshold is deliberately tight so that only exact degeneracy is caught:
# N2/6-31G has a core-active gap of 8.1e-3 Ha and must not be flagged.
# HF/6-31G's final [4,5] space is exactly degenerate (2.8e-15 Ha) and is
# flagged; that is the same split pi shell as HCl and HI, and HF is
# already refused by its gradient. The check is report-only until it has
# been validated as a self-check on its own.
BOUNDARY_DEGENERACY_TOL = 1e-8


def _candidate_key(record, stationary_tol=STATIONARY_GRADIENT):
    """Sort key for one optimized orbital candidate.

    A candidate only enters the minimum comparison when it is stationary.
    The tiers are, in order: a stationary minimum (no negative curvature),
    a stationary saddle, and a point the optimizer did not converge. A
    non-stationary point never outranks a stationary one, because the
    curvature there says nothing about a minimum the optimizer has not
    reached. Within the first tier the energy decides, within the second
    the number of negative eigenvalues, and within the third the gradient.
    """
    gradient = float(record['gradient'])
    n_negative = int(record['n_negative'])
    energy = float(record['energy'])
    stationary = gradient <= float(stationary_tol)
    if stationary and n_negative == 0:
        return (0, energy, n_negative, gradient)
    if stationary:
        return (1, n_negative, energy, gradient)
    return (2, gradient, energy, n_negative)


def prefer_candidate(first, second, stationary_tol=STATIONARY_GRADIENT):
    """Choose between two optimized orbital candidates, curvature first.

    The rule is fixed so that two runs over the same candidates always
    agree and a report can be read without the code:

    * a stationary candidate with no negative curvature beats any saddle;
    * between two minima the lower energy wins;
    * between two stationary saddles the one with fewer negative
      eigenvalues wins;
    * a non-stationary candidate never beats a stationary one.

    Parameters
    ----------
    first, second : dict
        Records with keys ``active``, ``n_negative``, ``min_eigenvalue``,
        ``gradient`` and ``energy``. ``n_negative`` must be an integer;
        an unavailable curvature is handled by the caller, not here.
    stationary_tol : float
        Gradient norm at or below which a candidate counts as stationary.

    Returns
    -------
    (dict, str)
        The chosen record and a reason written for the evidence list.
    """
    a = _candidate_key(first, stationary_tol)
    b = _candidate_key(second, stationary_tol)
    chosen, other = (first, second) if a <= b else (second, first)
    chosen_key, other_key = (a, b) if a <= b else (b, a)

    here, there = list(chosen['active']), list(other['active'])
    if chosen_key[0] != other_key[0]:
        if chosen_key[0] == 0:
            reason = (f'{here} was preferred over {there}: it is an orbital '
                      f'minimum (no negative eigenvalue)')
        elif chosen_key[0] == 1:
            reason = (f'{here} was preferred over {there}: it is stationary, '
                      f'while the other candidate is not')
        else:
            reason = (f'{here} was preferred over {there}: lower gradient')
    elif chosen_key[0] == 0:
        reason = (f'{here} was preferred over {there}: both are orbital '
                  f'minima and {float(chosen["energy"]):.10f} < '
                  f'{float(other["energy"]):.10f}')
    elif chosen_key[0] == 1:
        if int(chosen["n_negative"]) == int(other["n_negative"]):
            reason = (f'{here} was preferred over {there}: both are '
                      f'stationary saddles with '
                      f'{int(chosen["n_negative"])} negative eigenvalue(s), '
                      f'so the lower energy was kept '
                      f'({float(chosen["energy"]):.10f} < '
                      f'{float(other["energy"]):.10f})')
        else:
            reason = (f'{here} was preferred over {there}: fewer negative '
                      f'eigenvalues ({int(chosen["n_negative"])} vs '
                      f'{int(other["n_negative"])})')
    else:
        reason = (f'{here} was preferred over {there}: neither candidate is '
                  f'stationary, so the smaller gradient was kept '
                  f'({float(chosen["gradient"]):.2e} vs '
                  f'{float(other["gradient"]):.2e})')
    return chosen, reason


def _candidate_record(candidate):
    """The decision record of one optimized candidate.

    ``n_negative`` and ``min_eigenvalue`` are None when the curvature
    could not be evaluated; ``prefer_candidate`` is only called with
    records whose curvature is present.
    """
    report = candidate['curvature']
    return dict(
        active=list(candidate['space'].active),
        n_negative=None if report is None else int(report.n_negative),
        min_eigenvalue=None if report is None else float(report.smallest),
        gradient=float(candidate['point'].grad_norm),
        energy=float(candidate['point'].energy),
        method=candidate['curvature_method'],
    )


def _candidate_evidence(record):
    """One candidate line for the evidence list, with every decision field.

    The comparison is only checkable when all four numbers that drove it
    are in the report: negative eigenvalues, smallest eigenvalue,
    gradient and energy.
    """
    active = list(record['active'])
    tail = (f'gradient={float(record["gradient"]):.3e}, '
            f'energy={float(record["energy"]):.10f}')
    if record.get('n_negative') is None:
        return (f'candidate {active}: curvature unavailable '
                f'({record.get("method")}), {tail}')
    return (f'candidate {active}: n_negative={int(record["n_negative"])}, '
            f'min_eigenvalue={float(record["min_eigenvalue"]):.6e}, '
            f'{tail}')


def _core_active_gap(space, mo_energy):
    """Reference gap from the highest core to the lowest active orbital.

    Returns ``(core_index, active_index, gap_in_hartree)`` or None when
    either block is empty. The number is the reference-orbital estimate
    of how far a near-degenerate partner was left outside the space.
    """
    if mo_energy is None or not space.core or not space.active:
        return None
    e = np.asarray(mo_energy, dtype=float).reshape(-1)
    core_i = int(max(space.core, key=lambda i: e[i]))
    active_i = int(min(space.active, key=lambda i: e[i]))
    return core_i, active_i, float(e[active_i] - e[core_i])


def _active_virtual_gap(space, mo_energy):
    """Reference gap from the highest active to the lowest virtual orbital.

    Returns ``(active_index, virtual_index, gap_in_hartree)`` or None when
    either block is empty. Together with :func:`_core_active_gap` this
    bounds the active space on both sides, so an exact degeneracy on
    either boundary can be reported.
    """
    if mo_energy is None or not space.active or not space.virtual:
        return None
    e = np.asarray(mo_energy, dtype=float).reshape(-1)
    active_i = int(max(space.active, key=lambda i: e[i]))
    virtual_i = int(min(space.virtual, key=lambda i: e[i]))
    return active_i, virtual_i, float(e[virtual_i] - e[active_i])


def _boundary_degeneracy_evidence(degeneracy):
    """Evidence lines for an exactly degenerate shell split at a boundary.

    One line per boundary whose gap is below the report tolerance. The
    wording states both facts a reader needs: the pair of orbitals and
    the reason the energy is still invariant while the wavefunction is
    not symmetry-adapted.
    """
    lines = []
    tol = float(degeneracy.get('tol', BOUNDARY_DEGENERACY_TOL))
    for label, gap in (('core-active', degeneracy.get('core_active')),
                       ('active-virtual', degeneracy.get('active_virtual'))):
        if gap is None:
            continue
        high_i, low_i, value = gap
        if float(value) >= tol:
            continue
        lines.append(
            f'a degenerate shell is split across the space boundary: the '
            f'{label} gap between orbitals {int(high_i)} and {int(low_i)} '
            f'is {float(value):.3e} Ha (< {tol:.0e} Ha); the energy is '
            f'invariant under the omitted rotation, but the wavefunction '
            f'is not symmetry-adapted')
    return lines


def _boundary_degeneracy(space, mo_energy, tol=BOUNDARY_DEGENERACY_TOL):
    """Reference-orbital boundary check, report only.

    Both boundaries are tested. A gap below ``tol`` means the highest
    core (or lowest virtual) orbital is degenerate with the lowest (or
    highest) active orbital, so one component of a degenerate shell was
    frozen outside the space. This function only reports the condition;
    whether it becomes a self-check is a separate decision made by the
    caller. See ``driver.run.checks_from_result`` for the optional check.

    Returns
    -------
    dict
        ``core_active`` and ``active_virtual`` are None or
        ``(high_index, low_index, gap)``, ``split`` is True when either
        gap is below ``tol``, ``tol`` is carried for the record, and
        ``evidence`` holds one line per split boundary.
    """
    core_active = _core_active_gap(space, mo_energy)
    active_virtual = _active_virtual_gap(space, mo_energy)
    split = any(gap is not None and float(gap[2]) < float(tol)
                for gap in (core_active, active_virtual))
    report = dict(core_active=core_active,
                  active_virtual=active_virtual,
                  split=bool(split),
                  tol=float(tol))
    report['evidence'] = _boundary_degeneracy_evidence(report)
    return report


def _flat_direction_evidence(curvature, tol=FLAT_DIRECTION_TOL, npar=None):
    """Evidence for a zero-mode minimum that is not certified.

    A point with no negative eigenvalue can still carry a flat direction,
    for example when a degenerate shell is split across the boundary. The
    line is diagnostic and is never converted into a failing check here.
    Returns None unless ``n_negative`` is zero and the smallest eigenvalue
    is below ``tol``. ``npar`` is the number of rotation directions; zero
    means there is no curvature to certify, not a flat direction.
    """
    if npar is not None and int(npar) == 0:
        return None
    if int(curvature.n_negative) != 0:
        return None
    smallest = float(curvature.smallest)
    if smallest >= float(tol):
        return None
    return (f'flat direction, not certified: the smallest eigenvalue of '
            f'the orbital Hessian is {smallest:.3e} (< {float(tol):.0e}) '
            f'with zero negative eigenvalues; a zero mode is present and '
            f'the curvature does not exclude it')


def _non_minimum_evidence(active, curvature, gradient, gap=None,
                          stationary_tol=STATIONARY_GRADIENT):
    """Why a non-minimum result is not silently reported as a minimum.

    The text states the curvature that failed and, when the reference
    offers one, the core-active gap that points at the missing partner.
    It never claims a pass: the return value ``n_negative`` is what the
    self-check consumes, and this line is the reason a reader can check.
    """
    gradient = float(gradient)
    n_negative = int(curvature.n_negative)
    smallest = float(curvature.smallest)
    if gradient <= float(stationary_tol):
        text = (f'the selected space {list(active)} is not an orbital '
                f'minimum: {n_negative} negative eigenvalue(s) of the '
                f'orbital Hessian, smallest {smallest:.6e}, at a stationary '
                f'point (gradient {gradient:.3e})')
        if gap is not None:
            core_i, active_i, value = gap
            if abs(value) <= 1e-3:
                text += (f'; the highest core orbital ({core_i}) is '
                         f'degenerate with the lowest active orbital '
                         f'({active_i}) (gap {value:.3e} Ha), so one '
                         f'component of the degenerate pair is frozen in '
                         f'the core')
            elif value <= NEAR_DEGENERATE_GAP:
                text += (f'; the highest core orbital ({core_i}) lies only '
                         f'{value:.3f} Ha below the lowest active orbital '
                         f'({active_i}), so a near-degenerate partner is '
                         f'frozen in the core')
            else:
                text += (f'; the highest core orbital ({core_i}) lies '
                         f'{value:.3f} Ha below the lowest active orbital '
                         f'({active_i})')
        return (text + '; the active space is too small and must be '
                'expanded before the point can be certified')
    return (f'the reported point for {list(active)} is not stationary '
            f'(gradient {gradient:.3e}); it carries {n_negative} negative '
            f'eigenvalue(s) (smallest {smallest:.6e}), but curvature at a '
            f'point the optimizer did not reach is not a minimum '
            f'certificate')


def _curvature_at(energy_function, theta, npar):
    """Curvature of one optimized candidate, evaluated exactly once.

    Returns ``(report, method, error)``. ``report`` is a
    :class:`~diffmrc.core.stationary.CurvatureReport`; when the analytic
    Hessian is not finite and the size is above the gradient-difference
    cap, ``report`` is None and ``error`` carries the refusal message.

    The gradient-difference fallback is the same one the driver used
    before the fallback decision became curvature-aware: the blocked
    eigensolver's second derivative is undefined at an exact degeneracy
    while the energy and its first derivative are smooth.
    """
    import jax
    from ..core.stationary import curvature_from_gradient, curvature_report

    hessian = np.asarray(jax.hessian(energy_function)(theta))
    method = 'analytic'
    if not np.isfinite(hessian).all():
        n_nan = int(np.isnan(hessian).sum())
        if int(npar) > CURVATURE_FD_MAX_PARAMETERS:
            return None, method, (
                f'the analytic orbital Hessian at the converged point is '
                f'not finite: {n_nan} of {hessian.size} entries are NaN. '
                f'The point sits on an exact degeneracy, and the '
                f'gradient-difference route is capped at '
                f'{CURVATURE_FD_MAX_PARAMETERS} parameters while this '
                f'space has {int(npar)}. The energy and the first '
                f'derivative are not affected; obtain the curvature from '
                f'displaced orbitals if one is needed')
        hessian = curvature_from_gradient(
            jax.grad(energy_function), theta, step=CURVATURE_FD_STEP,
            max_parameters=CURVATURE_FD_MAX_PARAMETERS)
        method = 'gradient-difference'
    return curvature_report(hessian), method, None


@dataclass
class ActiveSpace:
    """The three orbital blocks, in the order the kernel expects."""

    core: list = field(default_factory=list)
    active: list = field(default_factory=list)
    virtual: list = field(default_factory=list)
    n_elec: int = 0

    @property
    def ncas(self):
        return len(self.active)

    @property
    def ncore(self):
        return len(self.core)

    @property
    def mo_order(self):
        """The column order that puts core first, then active, virtual."""
        return list(self.core) + list(self.active) + list(self.virtual)


def partition_from_occupations(occ, active_indices):
    """Split a reference occupation vector around an active set.

    Parameters
    ----------
    occ : array_like
        Occupation numbers of the reference (SCF) orbitals.
    active_indices : sequence of int
        Orbitals selected for the active space.

    Returns
    -------
    ActiveSpace

    Raises
    ------
    ValueError
        If an index falls outside the reference, or ``occ`` is not
        one-dimensional.

    Notes
    -----
    The core and virtual blocks are read off the occupations
    (``occ > 0.5`` counts as occupied), so an orbital ordering that
    departs from aufbau still yields a consistent partition. The
    electron count is the sum over the active block, which is what makes
    the ``active_electrons`` self-check meaningful: it compares this
    number against what the selection rule intended.
    """
    occ = np.asarray(occ, dtype=float)
    if occ.ndim != 1:
        raise ValueError(
            f'occ must be one-dimensional, got shape {occ.shape}')
    active = sorted({int(i) for i in active_indices})
    for i in active:
        if not 0 <= i < occ.size:
            raise ValueError(
                f'active orbital index {i} is outside 0..{occ.size - 1}'
            )
    act = set(active)
    core = [i for i in range(occ.size) if i not in act and occ[i] > 0.5]
    virtual = [i for i in range(occ.size)
               if i not in act and occ[i] <= 0.5]
    n_elec = 0
    for i in active:
        if occ[i] > 1.5:
            n_elec += 2
        elif occ[i] > 0.5:
            n_elec += 1
    return ActiveSpace(core=core, active=active, virtual=virtual,
                       n_elec=n_elec)


def _core_hamiltonian(mol):
    """One-electron Hamiltonian, including the ECP scalar term.

    ``int1e_kin + int1e_nuc`` is not the core Hamiltonian for a
    pseudopotential: the ECP scalar integral is a separate term, and
    dropping it changes both the matrix trace and the total energy on
    every ECP system. Measured on ThO: the traces differ by 150.7 Ha and
    the energy by 71 Ha. The mean-field object's ``get_hcore`` adds the
    term, and this helper brings the kernel path onto that convention
    without requiring the mean field to be passed down; a molecule
    without ``has_ecp`` returns the plain sum unchanged.
    """
    h = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    if hasattr(mol, 'has_ecp') and mol.has_ecp():
        h = h + mol.intor('ECPscalar')
    return h


def _mo_block(mol, C0, space, enuc=None, include_cv=True):
    """The static pieces shared by the energy and the CI-vector routes.

    Hoisted so that :func:`energy_function` and :func:`casci_vectors_at`
    slice and rotate the same integrals in one convention instead of
    holding two copies free to drift apart. Only the orbital rotation
    depends on the parameters; everything returned here is fixed for the
    molecule, basis and partition.

    ``include_cv=False`` builds the parameter set of the two-stage
    route's first stage: core-active and active-virtual rotations
    only. The integrals are the same; only the pair list changes, so
    callers on the same molecule should derive this block from the full
    one instead of recomputing the integrals.

    The reference columns are brought to the deterministic phase
    convention of :func:`diffmrc.core.gauge.canonical_mo_phases` before
    ``space.mo_order`` and before any rotation.  ``theta`` is measured
    from these columns, so without the convention a fixed theta would
    describe a different physical rotation whenever the SCF eigensolver
    returned a different column sign (it does, with the thread count).
    The convention is applied to the caller's ``C0`` as well, so a
    reference supplied directly is covered.
    """
    import jax
    jax.config.update('jax_enable_x64', True)
    from jax import numpy as jnp

    from ..core.gauge import canonical_mo_phases
    from ..core.orblib import orbital_pairs

    C0 = jnp.asarray(np.asarray(C0, dtype=float))
    C0 = canonical_mo_phases(C0)
    nmo = C0.shape[1]
    h1e = _core_hamiltonian(mol)
    eri = mol.intor('int2e', aosym='s1')
    if enuc is None:
        enuc = mol.energy_nuc()
    return dict(
        C0o=C0[:, space.mo_order],
        h1e=h1e,
        eri=eri,
        enuc=enuc,
        pairs=orbital_pairs(space.ncore, space.ncas, nmo,
                            include_cv=bool(include_cv)),
        nmo=nmo,
        ncore=int(space.ncore),
        ncas=int(space.ncas),
        n_elec=int(space.n_elec),
    )


def stage_one_block(block):
    """The same integrals with the core-virtual pairs removed.

    This is the parameter set the first stage of the two-stage route
    optimizes. It shares every array with `block`; only the pair list is
    replaced, so no integral is recomputed.
    """
    from ..core.orblib import orbital_pairs

    pairs = orbital_pairs(block['ncore'], block['ncas'], block['nmo'],
                          include_cv=False)
    return dict(block, pairs=pairs)


def energy_function(mol, C0, space, enuc=None, block=None):
    """The differentiable CASCI energy under orbital rotations.

    The integrals are computed once, outside the returned function, so
    that the rotation's backward pass only differentiates the einsums
    and the CI. ``space.mo_order`` is applied to ``C0`` first: the
    kernel's convention is that the first ``ncore`` orbitals are the
    frozen core, then the active block, then the virtuals.

    Parameters
    ----------
    mol : object
        Anything with ``nao_nr()``, ``intor(name)`` and ``energy_nuc()``
        -- a pyscfad Mole in production, a stand-in in tests.
    C0 : array_like
        Reference orbital coefficients, shape ``(nao, nmo)``.
    space : ActiveSpace
        The partition to use.
    enuc : float, optional
        Nuclear repulsion; taken from `mol` when omitted.

    Returns
    -------
    (E_theta, npar) : (callable, int)
        The energy as a function of the rotation parameters, and how
        many parameters there are.
    """
    from jax import numpy as jnp

    from ..core.casci import cs_casci_blocked
    from ..core.rotation import block_rotation

    block = _mo_block(mol, C0, space, enuc=enuc) if block is None else block
    C0o, h1e, eri = block['C0o'], block['h1e'], block['eri']
    enuc, pairs, nmo = block['enuc'], block['pairs'], block['nmo']
    ncore, ncas, n_elec = block['ncore'], block['ncas'], block['n_elec']

    def E_theta(t):
        C = C0o @ block_rotation(t, nmo, pairs)
        nm = ncore + ncas
        h1 = jnp.einsum('ai,ab,bj->ij', C, h1e, C)[:nm, :nm]
        e1 = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', C, C, C, C,
                        eri)[:nm, :nm, :nm, :nm]
        return cs_casci_blocked(h1, e1, ncore=ncore, nact=ncas,
                                nele_act=n_elec)[0] + enuc

    return E_theta, len(pairs)


def casci_vectors_at(mol, C0, space, theta, nroots=1, enuc=None, block=None):
    """CI vectors at one orbital rotation, shape ``(ndet, nroots)``.

    WHY THIS EXISTS

    The energy route reduces the CI to a scalar inside the differentiated
    function. The active-space verification needs the vectors themselves,
    in the determinant order the entropy extraction reads
    (``core.nevpt2.impl.cas_dets_for2``) and in the same orbital
    convention as the reported energy. The rotation and the slicing are
    therefore taken from :func:`_mo_block`, the pieces
    :func:`energy_function` uses, rather than re-derived here.

    ``nroots = 1`` returns the ground state of the same blocked solver
    the energy used. With several roots the state-average branch of
    ``core.casci.cs_casci`` returns the lowest eigenvalues in ascending
    order; for a spin multiplet the components related by spin projection
    are degenerate and are therefore the lowest ones, which is the set
    the manifold entropy merges.
    """
    from ..core.rotation import block_rotation

    block = _mo_block(mol, C0, space, enuc=enuc) if block is None else block
    C = np.asarray(block['C0o'] @ block_rotation(
        np.asarray(theta, dtype=float), block['nmo'], block['pairs']))
    nm = block['ncore'] + block['ncas']
    h1 = np.einsum('ai,ab,bj->ij', C, np.asarray(block['h1e']), C)[:nm, :nm]
    e1 = np.einsum('pi,qj,rk,sl,pqrs->ijkl', C, C, C, C,
                   np.asarray(block['eri']))[:nm, :nm, :nm, :nm]
    if int(nroots) == 1:
        from ..core.casci import cs_casci_blocked
        _, ci = cs_casci_blocked(h1, e1, ncore=block['ncore'],
                                 nact=block['ncas'],
                                 nele_act=block['n_elec'],
                                 eig_solver='eigh_safe')
        return np.asarray(ci)[:, None]
    from .nevpt2 import sa_cas_solution
    _, vecs = sa_cas_solution(h1, e1, ncore=block['ncore'],
                              nact=block['ncas'], nele=block['n_elec'],
                              nroots=int(nroots))
    return np.asarray(vecs)


@dataclass
class FrozenGrid:
    """The quadrature the on-top energy is evaluated on, frozen.

    ``ao`` is ``(ngrids, nao)`` AO values, ``ao_grad`` is
    ``(ngrids, 3, nao)`` gradient rows, ``weights`` is ``(ngrids,)``.
    """

    ao: object
    ao_grad: object
    weights: object


def frozen_grid(mol, level=3):
    """Build the frozen quadrature from a pyscf-like molecule.

    One-time and outside every differentiated path by construction: a
    later geometry still evaluates the on-top energy on this grid, so
    the grid-velocity term is left out of the derivative. The
    calibration (t382/t383) bounds that term at 7.3e-5 Eh/bohr on
    CO/6-31G at level 3, falling to 2.5e-5 at level 4 -- about three
    orders of magnitude below the wavefunction response the derivative
    is made of, and convergent with the level.

    ``eval_ao(deriv=1)`` returns the deterministic ``(4, ngrids, nao)``
    layout; it is asserted, not inferred -- a dimension-matching
    heuristic transposed it silently for nao = 4 (t348--t355).
    """
    import numpy as np
    from pyscf.dft import numint
    from pyscf.dft.gen_grid import Grids

    grid = Grids(mol)
    grid.level = int(level)
    grid.build()
    coords = np.asarray(grid.coords)
    ngr = int(coords.shape[0])
    nao = int(mol.nao_nr())
    raw = np.asarray(numint.eval_ao(mol, coords, deriv=1))
    if raw.shape != (4, ngr, nao):
        raise ValueError(
            f'eval_ao(deriv=1) returned {raw.shape}, expected '
            f'(4, {ngr}, {nao})')
    return FrozenGrid(ao=raw[0].copy(),
                      ao_grad=raw[1:4].transpose(1, 0, 2).copy(),
                      weights=np.asarray(grid.weights).copy())


def _mcpdft_chain(mol, C0, space, grid, grid_level, eval_xc, enuc,
                  hyb=0.0):
    """Assemble the on-top theta chain; shared by the two factories.

    Everything independent of the rotation parameters is computed here,
    once -- the integrals, the frozen grid, the determinant list. The
    returned closure evaluates ``(E_total, E_ot, E_mcwfn, E_cas)`` at
    one parameter vector.  The reference columns are phase-canonicalized
    before the rotation for the same reason as in :func:`_mo_block`: the
    parameters are measured from those columns.
    """
    import jax
    jax.config.update('jax_enable_x64', True)
    from jax import numpy as jnp

    from ..core.casci import cs_casci_blocked
    from ..core.gauge import canonical_mo_phases
    from ..core.mcpdft import eot_energy, pi_active, w_sigma
    from ..core.nevpt2.impl import cas_dets_for2
    from ..core.orblib import orbital_pairs
    from ..core.rotation import block_rotation

    C0 = jnp.asarray(np.asarray(C0, dtype=float))
    C0 = canonical_mo_phases(C0)
    nmo = C0.shape[1]
    C0o = C0[:, space.mo_order]
    ncore, ncas = int(space.ncore), int(space.ncas)
    n_elec = int(space.n_elec)
    h1e = _core_hamiltonian(mol)
    eri = mol.intor('int2e', aosym='s1')
    if enuc is None:
        enuc = mol.energy_nuc()
    if grid is None:
        grid = frozen_grid(mol, level=grid_level)
    if eval_xc is None:
        # The package's own jnp PBE kernel: closed-form, jit- and
        # hessian-safe. The pyscfad/libxc wrapper is custom_jvp over the
        # C library and cannot be traced under jit (t384), which the
        # driver's optimizer and curvature check both need.
        from ..core.mcpdft import pbe_eval_xc
        eval_xc = pbe_eval_xc
    pairs = orbital_pairs(ncore, ncas, nmo, include_cv=True)
    # The determinant list must be the one cs_casci_blocked builds
    # internally (combinations over the active spin orbitals, in
    # itertools order) or every RDM below would be indexed against a
    # different basis. Both come from the same generator.
    dets = cas_dets_for2(ncore, ncas, n_elec)

    h1j, erij = jnp.asarray(h1e), jnp.asarray(eri)
    enuc_j = jnp.asarray(enuc)
    ao, ao_grad = jnp.asarray(grid.ao), jnp.asarray(grid.ao_grad)
    weights = jnp.asarray(grid.weights)

    def gd(D):
        return (jnp.einsum('gr,rs,gs->g', ao, D, ao),
                2.0 * jnp.einsum('gxr,rs,gs->gx', ao_grad, D, ao))

    def terms(t):
        C = C0o @ block_rotation(t, nmo, pairs)
        nm = ncore + ncas
        Cs = C[:, :nm]
        h1m = Cs.T @ h1j @ Cs
        e1m = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', Cs, Cs, Cs, Cs, erij)
        E_cas, ci = cs_casci_blocked(h1m, e1m, ncore=ncore, nact=ncas,
                                     nele_act=n_elec)
        # Spin-resolved active 1-RDM: W[s, p, q, :] = M[(2p+s, 2q+s)]^T ci,
        # so W . ci is the spin density matrix <a^dag_p,s a_q,s> directly.
        W = w_sigma(ci, dets, ncas, ncore)
        dm1s = jnp.einsum('spqI,I->spq', W, ci)
        Cact = C[:, ncore:nm]
        Ccore = C[:, :ncore]
        Dact = jnp.einsum('rp,spq,tq->srt', Cact, dm1s, Cact)
        # Per spin: the core contributes one electron per spin per core
        # orbital, the active block its own spin density.
        Dfull = Ccore @ Ccore.T + Dact
        rho_a, g_a = gd(Dfull[0])
        rho_b, g_b = gd(Dfull[1])
        rho_aa, _ = gd(Dact[0])
        rho_bb, _ = gd(Dact[1])
        acas = jnp.einsum('gr,rp->gp', ao, Cact)
        Pi = pi_active(W, acas) + (rho_a * rho_b - rho_aa * rho_bb)
        E_ot = eot_energy(rho_a, g_a, rho_b, g_b, Pi, weights, eval_xc)
        D = Dfull[0] + Dfull[1]
        E_mcwfn = (enuc_j + jnp.einsum('ab,ab->', h1j, D)
                   + 0.5 * jnp.einsum('abcd,ab,cd->', erij, D, D))
        # Hybrid (tPBE0, hybrid = 0.25): the standard RDM split makes the
        # mixed energy the convex combination of the on-top total and the
        # CASSCF total at the same orbital point -- E_x + E_c =
        # E_CASSCF - classical (engine-confirmed to the digit, t390/t391)
        # -- so no 2-RDM convention appears anywhere.
        E_tot = (1.0 - hyb) * (E_mcwfn + E_ot) + hyb * (E_cas + enuc_j)
        return E_tot, E_ot, E_mcwfn, E_cas

    return terms, len(pairs)


def mcpdft_energy_function(mol, C0, space, grid=None, grid_level=3,
                           eval_xc=None, enuc=None, hyb=0.0):
    """The differentiable MC-PDFT (tPBE) energy under orbital rotations.

    The on-top companion of ``energy_function``: the same rotation
    chart and the same blockwise CI, with the CASCI energy replaced by
    the split the engine uses,

        E = E_mcwfn + E_ot,
        E_mcwfn = Vnn + <h | D> + 1/2 <J | D>     (full density D),
        E_ot    = the on-top functional of the translated spin densities
                  on the frozen grid.

    The conventions that took the M1/M2 measurements to pin down live in
    ``core.mcpdft``: the translation uses gradient rows
    ``(grad rho / 2)(1 +- zeta)`` with no ``grad zeta`` term (Li Manni
    et al. 2014; pyscf ``get_rho_translated``), and the on-top pair
    density is the factorized active-space form plus
    ``rho_a rho_b - rho_aa rho_bb``.

    Verified against pyscf.mcpdft (t384): H2/STO-3G CAS(2,2) reproduces
    the engine's ``e_tot``/``e_ot``/``e_mcwfn`` to 1e-16 (golden
    -1.156628479816); H2O/STO-3G with a four-orbital core to 1.6e-9;
    H2/6-31G to 1.3e-8, which is the engine's own FCI convergence floor
    -- its davidson vector converges to ~1e-7 and the on-top energy is
    first order in that density error, while our ci comes from a full
    block decomposition. The engine's ``mcpdft.CASSCF`` is the one-step
    protocol (CASSCF-driven orbitals, a single on-top evaluation; the
    DVS paper's protocol); the theta chain here minimizes the on-top
    energy itself, so its converged point is stationary for the MC-PDFT
    surface and lies below the one-step value. AD vs FD on the rotation
    gradient measures 1e-10, and the optimized point reaches
    |grad| = 1e-14 on the production pyscfad path.

    ``hyb=0.0`` is tPBE; ``hyb=0.25`` is the tPBE0 hybrid, whose
    energy at these orbitals is the convex combination
    ``(1 - hyb) E_tPBE + hyb E_CASSCF`` (the RDM split collapses the
    hybrid's E_x + E_c into the wavefunction energy; engine-confirmed to
    the digit, t390/t391).  The returned components (E_ot, E_mcwfn) stay
    unscaled; E_total carries the mix.  Excited-state tables are
    separate registered work. The size of the active space is the size
    ``cs_casci_blocked`` can diagonalize -- the same limit as
    ``energy_function``.

    Parameters
    ----------
    mol : object
        Anything with ``nao_nr()``, ``intor(name)`` and ``energy_nuc()``
        that pyscf's grid machinery accepts -- a pyscfad Mole in
        production.
    C0 : array_like
        Reference orbital coefficients, shape ``(nao, nmo)``.
    space : ActiveSpace
        The partition to use.
    grid : FrozenGrid, optional
        A frozen quadrature; built from `mol` at `grid_level` when
        omitted.
    grid_level : int
        Grid density for the automatic build (pyscf's scale; 3 is the
        pyscf mcpdft default).
    eval_xc : callable, optional
        The xc kernel, called as ``eval_xc('PBE', rho_t, spin=1,
        deriv=0)[0]``; defaults to the package's jnp PBE kernel
        (``core.mcpdft.pbe_eval_xc``, validated against libxc in t385).
        ``pyscfad.dft.libxc.eval_xc`` can be injected for cross-checks,
        but it is not jit-safe (its custom_jvp traces into the C
        library), so the driver's optimizer cannot use it.
    enuc : float, optional
        Nuclear repulsion; taken from `mol` when omitted.

    Returns
    -------
    (E_theta, npar) : (callable, int)
        The energy as a function of the rotation parameters, and how
        many parameters there are.
    """
    terms, npar = _mcpdft_chain(mol, C0, space, grid, grid_level,
                                eval_xc, enuc, hyb=hyb)

    def E_theta(t):
        return terms(t)[0]

    return E_theta, npar


def mcpdft_terms_function(mol, C0, space, grid=None, grid_level=3,
                          eval_xc=None, enuc=None, hyb=0.0):
    """The MC-PDFT theta chain with its components exposed.

    Same chain as ``mcpdft_energy_function``; the closure returns
    ``(E_total, E_ot, E_mcwfn, E_cas)`` from one forward pass, so an
    anchor script or a report can show each component against the
    engine's ``e_tot``, ``e_ot`` and ``e_mcwfn``.
    """
    return _mcpdft_chain(mol, C0, space, grid, grid_level, eval_xc, enuc,
                         hyb=hyb)


def search_point(E_theta, npar, tries=1, seed=0, stage_one=None, **kwargs):
    """Converge the orbital point, retrying from perturbed starts.

    The first start is always the reference itself (theta = 0), so a
    ``tries=1`` call is the fast level exactly. The perturbations are
    seeded, so the same call gives the same point twice -- a
    reproducible run is worth more than a fresh random draw, and the
    basin problems this project has hit are found by varying the start
    on purpose, not by accident.

    Parameters
    ----------
    E_theta : callable
        Energy as a function of the full parameter vector.
    npar : int
        Full parameter count.
    tries : int
        Number of starts, ``>= 1``.
    seed : int
        Seed for the perturbations.
    stage_one : dict, optional
        When given, each start runs the two-stage route instead of the
        single-stage one. The keys are ``E_subset``,
        ``npar_subset``, ``subset_pairs`` and ``full_pairs``; the
        starting perturbation is drawn on the subset vector.
    **kwargs
        Forwarded to the convergence routine.

    Returns
    -------
    OrbitalPoint or TwoStagePoint
        The variationally lowest of the converged points.
    """
    tries = int(tries)
    if tries < 1:
        raise ValueError(f'tries must be at least 1, got {tries}')
    sub = None if stage_one is None else dict(stage_one)
    n_start = npar if sub is None else int(sub['npar_subset'])
    rng = np.random.default_rng(seed)
    best = None
    for k in range(tries):
        start = (np.zeros(n_start) if k == 0
                 else 0.3 * rng.standard_normal(n_start))
        if sub is None:
            point = converge_orbital_point(E_theta, npar, theta0=start,
                                           **kwargs)
        else:
            point = converge_orbital_point_two_stage(
                sub['E_subset'], int(sub['npar_subset']), E_theta, npar,
                sub['subset_pairs'], sub['full_pairs'], theta0=start,
                **kwargs)
        if best is None or point.energy < best.energy:
            best = point
    return best


def _warn_if_thread_count_breaks_reproducibility():
    """Warn that a fixed-theta value can depend on the thread count.

    The SCF eigen-decomposition is threaded, and its output can depend on
    the thread count in two ways. Column signs are now canonicalized
    (``core.gauge``) in the NEVPT2, kernel, Hessian and X2C theta paths,
    which removed the measured sign-gauge dependence (H2O/6-31G CAS(2,2)
    at one fixed theta: 4.6e-2 Ha before, 8.5e-14 Ha after). An exactly
    degenerate subspace can still come back in a different orthonormal
    basis: measured at one fixed theta, N2/6-31G differs by 3.5e-2 Ha and
    HI/def2-SVP by 1.5e-1 Ha between one and four threads, with the SCF
    energies agreeing to 3e-14. A run that compares numbers across runs
    should keep one thread.
    """
    import os
    import warnings

    raw = os.environ.get('OMP_NUM_THREADS', '').strip()
    if not raw:
        return
    try:
        threads = int(raw)
    except ValueError:
        return
    if threads <= 1:
        return
    warnings.warn(
        f'OMP_NUM_THREADS={threads}: a fixed-theta value can depend on the '
        f'thread count. Column signs are canonicalized, but an exactly '
        f'degenerate subspace may return a different orthonormal basis '
        f'(measured at one fixed theta: N2/6-31G 3.5e-2 Ha, HI/def2-SVP '
        f'1.5e-1 Ha between one and four threads). Set it to 1 for any '
        f'number that will be compared or quoted.',
        RuntimeWarning, stacklevel=3)


def scf_reference(atoms, charge, spin, basis, ecp=None):
    """Run the SCF that supplies the reference orbitals.

    pyscfad is imported here, not at module import: the rest of the
    driver must be usable -- ``--check``, ``--show-defaults`` -- on a
    machine where the numerical kernel is not installed, which is what
    the command line's ImportError branch is for.

    Parameters
    ----------
    atoms : str
        Geometry in the ``'Eu 0 0 0; F 0 0 2.06'`` form.
    charge, spin : int
        Charge and ``2S``.
    ecp : str or dict, optional
        Pseudopotential definition passed to ``Mole.ecp``. A dict keyed by
        element is the safe form: a bare name is applied to every element,
        and the elements that have no ECP under that name are what make
        the reference fail to build. ``def2-*`` sets carry an ECP for the
        elements above Kr only, so an iodine/hydrogen molecule needs
        ``{'I': 'def2-svp'}``.

    Returns
    -------
    (mol, mf)
        The molecule and the converged mean-field object.
    """
    _warn_if_thread_count_breaks_reproducibility()
    try:
        from pyscfad import gto, scf
    except ImportError as exc:
        raise ImportError('pyscfad is not installed') from exc

    mol = gto.Mole()
    mol.atom = atoms
    mol.basis = basis
    if ecp is not None:
        mol.ecp = ecp
    mol.charge = charge
    mol.spin = spin
    mol.build()
    mf = (scf.ROHF(mol) if spin != 0 else scf.RHF(mol))
    mf.kernel()
    if not getattr(mf, 'converged', True):
        raise RuntimeError(
            'the SCF did not converge; every number below would rest on '
            'an unconverged reference')
    return mol, mf


def select_active_space(mol, C, occ, symbols, manual=None, params=None,
                        mo_energy=None):
    """Pick the active orbitals: the user's list, or the element rules.

    Returns the indices and the evidence behind them. The evidence is
    carried into the report because a selection nobody can read is a
    selection nobody can check.
    """
    from ..policy.rules import (candidate_from_elements,
                                candidate_from_occupations,
                                molecular_end_candidate, plan_from_elements)

    if manual is not None:
        active = sorted({int(i) for i in manual})
        return active, [f'active space given manually: {active}']

    plan = plan_from_elements(symbols)
    if plan is None:
        # No element rule was written for this composition. The occupation
        # criterion still applies, and a benchmark of six systems showed
        # that refusing outright leaves five ordinary molecules without
        # any automatic selection at all. The fallback is deliberately
        # minimal and says so in its evidence.
        fallback = candidate_from_occupations(np.asarray(occ),
                                              mo_energy=mo_energy)
        if fallback is None:
            return None, ['no rule covers this element combination, and '
                          'the reference occupations offer no frontier pair']
        return [int(i) for i in fallback.active], list(fallback.evidence)
    if plan.is_molecular:
        cand = molecular_end_candidate(mol, np.asarray(C), np.asarray(occ),
                                       center=plan.center,
                                       ligand=plan.ligand)
    else:
        cand = candidate_from_elements(mol, np.asarray(C), np.asarray(occ),
                                       center=plan.center,
                                       ligand=plan.ligand or None,
                                       center_n=plan.center_n,
                                       ligand_n=plan.ligand_n)
    if cand is None:
        return None, ['the rules refused to propose an active space']
    return [int(i) for i in cand.active], list(cand.evidence)


def _verify_active_space(mol, C, space, theta, spin, level, block=None):
    """Entanglement verification of the selected space, manifold-merged.

    WHY THIS EXISTS

    The four-state entropy criterion is defined for one state, and the
    ``M_S = +S`` component of a high-spin open shell is a single
    determinant whose open shells carry one spin each: read on that state
    alone every entropy is zero, even though the degenerate shells belong
    in the space. The manifold of ``2S + 1`` components is merged before
    the entropy is read (:func:`analysis.ci.manifold_entropy_from_ci`),
    which restores the spin-isotropic occupations the criterion measures.

    SCOPE

    The spectrum is the one the selected space can show. Orbitals left
    out of it have zero entropy by construction, so this check can state
    that the space is multireference and that its selected orbitals are
    not noise; it cannot state that the space is not truncated at the
    frontier. Boundary completeness remains the selection rule's
    responsibility.

    Returns ``(verified, report)``. ``verified`` is True when the merged
    spectrum is not single-reference. A failure to evaluate the spectrum
    is reported as unverified rather than raised: the robust level turns
    an unverified space into a refusal, and a verification that cannot
    run is exactly that.
    """
    if level == 'fast':
        return None, None
    nroots = max(1, int(abs(int(spin))) + 1)
    try:
        from ..analysis.ci import manifold_entropy_from_ci
        from ..core.nevpt2.impl import cas_dets_for2
        from ..policy.verification import Verdict, verify_candidate

        vectors = casci_vectors_at(mol, C, space, theta, nroots=nroots,
                                   block=block)
        dets = cas_dets_for2(int(space.ncore), int(space.ncas),
                             int(space.n_elec))
        s1 = manifold_entropy_from_ci(
            vectors, dets, ncore=int(space.ncore), nact=int(space.ncas))
        verdict = verify_candidate(s1, candidate=list(space.active))
        verified = verdict.verdict is not Verdict.SINGLE_REFERENCE
        return verified, dict(
            s1=[float(x) for x in s1],
            nroots=int(nroots),
            verdict=verdict.verdict.value,
            plateau_heights=[int(h) for h in verdict.plateau_heights],
            evidence=list(verdict.evidence),
        )
    except Exception as exc:  # noqa: BLE001 - the gate must report, not crash
        return False, dict(
            nroots=int(nroots),
            verdict='failed',
            evidence=[f'the manifold entropy could not be evaluated: '
                      f'{type(exc).__name__}: {exc}'],
        )


def compute_ground_state(atoms, charge, spin, active_space, params,
                         level='fast'):
    """Everything the self-checks need, from a geometry string.

    The order is a ladder of refusals: the SCF first, then the active
    space, then the electron count, and only then anything expensive. A
    step that can refuse does, so a bad input costs a parse rather than
    a calculation.

    The reported quantity follows the ``method`` parameter: the CASSCF
    value (default), or the one-step MC-PDFT energy -- tPBE evaluated
    once at the CASSCF-converged orbitals, the protocol of the DVS
    paper. The self-checks certify the wavefunction either way, so
    ``e_casscf`` is carried next to ``energy`` in the result.

    Raises
    ------
    ElectronicStructureFailure
        When a step that can refuse does: an SCF that will not converge,
        an element no rule covers, a rule that refuses, an unparseable
        active-space string, or a partition whose electron count cannot
        make the requested spin.
    """
    from ..policy.params import parse_active_space
    from .run import ElectronicStructureFailure

    def _value(key, default):
        # EffectiveParams carries __contains__ and __getitem__, not get;
        # driver/run.py reads parameters the same way.
        return params[key] if params is not None and key in params else default

    symbols = [part.split()[0] for part in atoms.split(';') if part.strip()]

    try:
        mol, mf = scf_reference(atoms, charge, spin,
                                _value('basis', '6-31g'),
                                ecp=_value('ecp', None))
    except ImportError:
        raise
    except Exception as exc:
        raise ElectronicStructureFailure(
            f'the reference SCF failed: {type(exc).__name__}: {exc}'
        ) from exc

    C = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ)

    manual = None
    if active_space is not None:
        manual = parse_active_space(str(active_space))
        if manual is None:
            raise ElectronicStructureFailure(
                f'active-space {active_space!r} is not a range like 2-4 or '
                f'a list like 2,3,4'
            )
    active, evidence = select_active_space(
        mol, C, occ, symbols, manual=manual, params=params,
        mo_energy=np.asarray(mf.mo_energy))
    if not active:
        raise ElectronicStructureFailure(
            'no active space was selected: ' + '; '.join(evidence))

    space = partition_from_occupations(occ, active)
    if space.n_elec < abs(spin) or (space.n_elec - abs(spin)) % 2 != 0:
        raise ElectronicStructureFailure(
            f'the active space holds {space.n_elec} electrons, which '
            f'cannot make spin {spin}'
        )

    import jax
    jax.config.update('jax_enable_x64', True)

    mo_energy = np.asarray(mf.mo_energy)
    tries = 1 if level == 'fast' else int(_value('multistart_tries', 1))
    optimize_tol = float(_value('optimizer_grad_tol', 1e-4))
    refine_target = float(_value('refine_target', 1e-11))
    # A space completed from degenerate partners is a guess, and the
    # frontier pair is the fallback whenever it does not hold up. The
    # first attempt therefore gets a step budget: measured on HI/def2-SVP,
    # the three-orbital space spent more than ten minutes before the check
    # could see that it had stalled, while sixty L-BFGS-B steps and twenty
    # refinement steps are enough to establish it.
    fallback_first = manual is None and len(space.active) > 2
    first_budget = (dict(max_steps=60, refine_max_iter=20)
                    if fallback_first else {})

    def _optimize(candidate_space, budget=None):
        """Optimize one candidate and take its curvature exactly once."""
        candidate_block = _mo_block(mol, C, candidate_space)
        candidate_energy, candidate_npar = energy_function(
            mol, C, candidate_space, block=candidate_block)
        # The core-virtual block makes the full parameter set ill
        # conditioned (HI/def2-SVP: the full L-BFGS-B stage stopped at
        # 3.0e-4 while the core-active + active-virtual subset reached
        # 4.3e-8). When that block dominates the rotation set
        # (should_split_rotation_set), the point is converged on the
        # subset first and then refined on the full energy.
        candidate_subset = stage_one_block(candidate_block)
        candidate_sub_energy, candidate_sub_npar = energy_function(
            mol, C, candidate_space, block=candidate_subset)
        stage_one = None
        if (0 < int(candidate_sub_npar) < int(candidate_npar)
                and should_split_rotation_set(
                    candidate_space.ncore, candidate_space.ncas,
                    int(candidate_block['nmo']))):
            stage_one = dict(
                E_subset=candidate_sub_energy,
                npar_subset=candidate_sub_npar,
                subset_pairs=candidate_subset['pairs'],
                full_pairs=candidate_block['pairs'])
        candidate_point = search_point(
            candidate_energy, candidate_npar, tries=tries,
            stage_one=stage_one, optimize_tol=optimize_tol,
            refine_target=refine_target, **(budget or {}))
        report, method, error = _curvature_at(
            candidate_energy, candidate_point.theta, candidate_npar)
        return dict(
            space=candidate_space,
            block=candidate_block,
            energy_function=candidate_energy,
            npar=int(candidate_npar),
            npar_subset=int(candidate_sub_npar),
            point=candidate_point,
            curvature=report,
            curvature_method=method,
            curvature_error=error,
            split=stage_one is not None,
        )

    selected = _optimize(space, first_budget)

    # The old retry looked only at the gradient. A completed space can be
    # an orbital saddle and still converge below 1e-4, so that gate never
    # fires; D3h ozone is the measured case. The curvature is read first
    # and decides together with the gradient: negative curvature is a
    # reason to retry the pure frontier pair, a stalled gradient remains
    # one, because a point the optimizer did not reach cannot be
    # certified by its curvature either.
    from ..policy.rules import plan_from_elements
    occupation_fallback = (manual is None
                           and plan_from_elements(symbols) is None)
    auto_completed = manual is None and len(space.active) > 2
    gradient = float(selected['point'].grad_norm)
    curvature_retry = bool(
        occupation_fallback and auto_completed
        and selected['curvature'] is not None
        and int(selected['curvature'].n_negative) > 0)
    gradient_retry = auto_completed and gradient > STATIONARY_GRADIENT
    if selected['curvature'] is None and not gradient_retry:
        raise ElectronicStructureFailure(selected['curvature_error'])
    if gradient_retry or curvature_retry:
        occupied = [i for i in range(len(occ)) if occ[i] > 0.5]
        virtual = [i for i in range(len(occ)) if occ[i] <= 0.5]
        if occupied and virtual:
            minimal = [occupied[-1], virtual[0]]
            alternative = _optimize(partition_from_occupations(occ, minimal))
            if alternative['curvature'] is None:
                raise ElectronicStructureFailure(
                    alternative['curvature_error'])
            first_record = _candidate_record(selected)
            second_record = _candidate_record(alternative)
            if first_record['n_negative'] is None:
                # Without a curvature for the completed space the old
                # gradient comparison is all that is left; the frontier
                # pair is kept only when its point is the lower one.
                if (float(alternative['point'].grad_norm)
                        < float(selected['point'].grad_norm)):
                    chosen = second_record
                    reason = (f'the curvature of {first_record["active"]} '
                              f'could not be evaluated; the frontier pair '
                              f'{second_record["active"]} has the lower '
                              f'gradient and was kept')
                else:
                    chosen = first_record
                    reason = (f'the curvature of {first_record["active"]} '
                              f'could not be evaluated; its gradient is not '
                              f'worse than {second_record["active"]}, so it '
                              f'was kept')
            else:
                chosen, reason = prefer_candidate(first_record,
                                                  second_record)
            # Both candidates are reported, not only the winner, so the
            # comparison can be checked against the numbers that drove it.
            evidence.append(_candidate_evidence(first_record))
            evidence.append(_candidate_evidence(second_record))
            evidence.append(reason)
            if chosen is second_record:
                selected = alternative

    if selected['curvature'] is None:
        # No candidate reached a point where the curvature could be
        # evaluated; report the refusal rather than dereference a missing
        # report below.
        raise ElectronicStructureFailure(selected['curvature_error'])

    space = selected['space']
    block = selected['block']
    E_theta = selected['energy_function']
    npar = selected['npar']
    point = selected['point']
    curvature = selected['curvature']
    curvature_method = selected['curvature_method']
    if selected['split']:
        nmo = int(block['nmo'])
        n_cv = int(space.ncore) * (nmo - int(space.ncore) - int(space.ncas))
        evidence.append(
            f'the core-virtual block holds {n_cv} of {int(npar)} rotation '
            f'parameters ({n_cv / selected["npar_subset"]:.1f} times the '
            f'rest); the two-stage orbital route was used')
    # Reference-orbital gaps on both sides of the space, and the smallest
    # orbital-Hessian eigenvalue, are recorded for every run. They are
    # diagnostic: neither line feeds a self-check, so no verdict changes.
    core_active_gap = _core_active_gap(space, mo_energy)
    active_virtual_gap = _active_virtual_gap(space, mo_energy)
    boundary_degeneracy = _boundary_degeneracy(space, mo_energy)
    smallest_eigenvalue = float(curvature.smallest)
    flat_direction = bool(int(curvature.n_negative) == 0
                          and int(npar) > 0
                          and smallest_eigenvalue < FLAT_DIRECTION_TOL)
    flat_evidence = _flat_direction_evidence(curvature, npar=npar)
    if flat_evidence is not None:
        evidence.append(flat_evidence)
    evidence.extend(boundary_degeneracy['evidence'])
    if int(curvature.n_negative) > 0:
        # The self-check still sees n_negative and refuses; this line is
        # the reason a reader can check, including the measured reference
        # gap that points at a partner left outside the space.
        evidence.append(_non_minimum_evidence(
            space.active, curvature, float(point.grad_norm),
            gap=core_active_gap))
    n_doubly = sum(1 for i in space.active if occ[i] > 1.5)
    energy = float(point.energy)
    e_casscf = energy
    from ..policy.methods import method_spec
    spec = method_spec(_value('method', 'casci'))
    method_data = {'method': spec.name}
    if spec.on_top:
        # One-step MC-PDFT (the DVS paper's protocol): the orbitals are
        # the CASSCF-driven ones just converged, and the on-top energy
        # is evaluated once at that point. The self-checks below still
        # certify the wavefunction -- the on-top energy is not a
        # variational expectation value, so `variational` runs on
        # `e_casscf`, which is reported alongside.  ``mcpdft0`` is the
        # 0.25 hybrid: E = 0.75 E_tPBE + 0.25 E_CASSCF at the same
        # orbitals (the RDM split of the engine's E_x + E_c; anchors
        # 8.9e-16 on H2/STO-3G, t391).
        terms, _ = mcpdft_terms_function(mol, C, space, hyb=spec.hyb)
        E_tot, E_ot, E_mcwfn, _ = (float(x) for x in terms(point.theta))
        energy = E_tot
        method_data = dict(
            method=spec.name,
            protocol=spec.ground_protocol,
            hyb=spec.hyb,
            e_casscf=e_casscf,
            e_ot=E_ot,
            e_mcwfn=E_mcwfn,
        )
    verification, entropy_report = _verify_active_space(
        mol, C, space, point.theta, spin, level, block=block)
    return dict(
        energy=energy,
        e_casscf=e_casscf,
        grad_norm=float(point.grad_norm),
        n_negative=int(curvature.n_negative),
        min_eigenvalue=smallest_eigenvalue,
        e_reference=float(mf.e_tot),
        n_elec_active=int(space.n_elec),
        n_elec_expected=int(space.n_elec),
        algorithm_data=dict(
            geometry=atoms,
            charge=charge,
            spin=spin,
            active_space=dict(active=list(space.active),
                              n_elec_active=int(space.n_elec),
                              core=list(space.core),
                              n_virtual=len(space.virtual)),
            evidence=list(evidence),
            optimizer=dict(gradient=float(point.optimizer_grad_norm),
                           reached=bool(point.optimizer_reached),
                           message=point.optimizer_message),
            two_stage=(point.stages_report()
                       if hasattr(point, 'stages_report') else None),
            stop_reason=getattr(point.stop_reason, 'value',
                                str(point.stop_reason)),
            level=level,
            multistart_tries=int(tries),
            n_doubly_active=int(n_doubly),
            curvature_method=curvature_method,
            curvature=dict(method=curvature_method,
                           n_negative=int(curvature.n_negative),
                           min_eigenvalue=smallest_eigenvalue,
                           flat_direction=flat_direction,
                           flat_tol=float(FLAT_DIRECTION_TOL)),
            orbital_gaps=dict(
                core_active=(None if core_active_gap is None
                             else float(core_active_gap[2])),
                active_virtual=(None if active_virtual_gap is None
                                else float(active_virtual_gap[2])),
                core_active_pair=(None if core_active_gap is None
                                  else [int(core_active_gap[0]),
                                        int(core_active_gap[1])]),
                active_virtual_pair=(None if active_virtual_gap is None
                                     else [int(active_virtual_gap[0]),
                                           int(active_virtual_gap[1])]),
                boundary_degeneracy_tol=float(BOUNDARY_DEGENERACY_TOL),
            ),
            boundary_degeneracy=boundary_degeneracy,
            active_space_verified=verification,
            active_space_entropy=entropy_report,
            **method_data,
        ),
    )
