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

from .orbitals import converge_orbital_point

__all__ = ['ActiveSpace', 'partition_from_occupations', 'energy_function',
           'search_point']


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


def energy_function(mol, C0, space, enuc=None):
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
    import jax
    jax.config.update('jax_enable_x64', True)
    from jax import numpy as jnp

    from ..core.casci import cs_casci_blocked
    from ..core.orblib import orbital_pairs
    from ..core.rotation import block_rotation

    C0 = jnp.asarray(np.asarray(C0, dtype=float))
    nmo = C0.shape[1]
    C0o = C0[:, space.mo_order]
    h1e = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    eri = mol.intor('int2e', aosym='s1')
    if enuc is None:
        enuc = mol.energy_nuc()
    pairs = orbital_pairs(space.ncore, space.ncas, nmo, include_cv=True)
    n_elec = int(space.n_elec)
    ncore, ncas = int(space.ncore), int(space.ncas)

    def E_theta(t):
        C = C0o @ block_rotation(t, nmo, pairs)
        nm = ncore + ncas
        h1 = jnp.einsum('ai,ab,bj->ij', C, h1e, C)[:nm, :nm]
        e1 = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', C, C, C, C,
                        eri)[:nm, :nm, :nm, :nm]
        return cs_casci_blocked(h1, e1, ncore=ncore, nact=ncas,
                                nele_act=n_elec)[0] + enuc

    return E_theta, len(pairs)


def search_point(E_theta, npar, tries=1, seed=0, **kwargs):
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
        Energy as a function of the parameters.
    npar : int
        Parameter count.
    tries : int
        Number of starts, ``>= 1``.
    seed : int
        Seed for the perturbations.
    **kwargs
        Forwarded to ``converge_orbital_point``.

    Returns
    -------
    OrbitalPoint
        The variationally lowest of the converged points.
    """
    tries = int(tries)
    if tries < 1:
        raise ValueError(f'tries must be at least 1, got {tries}')
    rng = np.random.default_rng(seed)
    best = None
    for k in range(tries):
        start = (np.zeros(npar) if k == 0
                 else 0.3 * rng.standard_normal(npar))
        point = converge_orbital_point(E_theta, npar, theta0=start, **kwargs)
        if best is None or point.energy < best.energy:
            best = point
    return best


def scf_reference(atoms, charge, spin, basis):
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

    Returns
    -------
    (mol, mf)
        The molecule and the converged mean-field object.
    """
    try:
        from pyscfad import gto, scf
    except ImportError as exc:
        raise ImportError('pyscfad is not installed') from exc

    mol = gto.Mole()
    mol.atom = atoms
    mol.basis = basis
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


def select_active_space(mol, C, occ, symbols, manual=None, params=None):
    """Pick the active orbitals: the user's list, or the element rules.

    Returns the indices and the evidence behind them. The evidence is
    carried into the report because a selection nobody can read is a
    selection nobody can check.
    """
    from ..policy.rules import (candidate_from_elements,
                                molecular_end_candidate, plan_from_elements)

    if manual is not None:
        active = sorted({int(i) for i in manual})
        return active, [f'active space given manually: {active}']

    plan = plan_from_elements(symbols)
    if plan is None:
        return None, ['no rule covers this element combination']
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


def compute_ground_state(atoms, charge, spin, active_space, params,
                         level='fast'):
    """Everything the self-checks need, from a geometry string.

    The order is a ladder of refusals: the SCF first, then the active
    space, then the electron count, and only then anything expensive. A
    step that can refuse does, so a bad input costs a parse rather than
    a calculation.

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
                                _value('basis', '6-31g'))
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
    active, evidence = select_active_space(mol, C, occ, symbols,
                                           manual=manual, params=params)
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
    from ..core.stationary import curvature_report

    E_theta, npar = energy_function(mol, C, space)
    tries = 1 if level == 'fast' else int(_value('multistart_tries', 1))
    point = search_point(
        E_theta, npar, tries=tries,
        optimize_tol=float(_value('optimizer_grad_tol', 1e-6)),
        refine_target=float(_value('refine_target', 1e-11)))

    curvature = curvature_report(np.asarray(jax.hessian(E_theta)(point.theta)))
    n_doubly = sum(1 for i in space.active if occ[i] > 1.5)
    return dict(
        energy=float(point.energy),
        grad_norm=float(point.grad_norm),
        n_negative=int(curvature.n_negative),
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
            stop_reason=getattr(point.stop_reason, 'value',
                                str(point.stop_reason)),
            level=level,
            multistart_tries=int(tries),
            n_doubly_active=int(n_doubly),
        ),
    )
