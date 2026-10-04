"""The self-checks that gate a reported result, and the run that produces it.

Four conditions, all cheap and all independent of any external program.
Each one corresponds to a failure that actually occurred in this project,
which is why these four and not others:

* ``orbital_gradient`` -- a manuscript table row stopped at 8.9e-7
* ``hessian_curvature`` -- two other rows were saddle points, with one
  and three negative eigenvalues
* ``variational`` -- an active space once left CASSCF 0.248 Ha above the
  ROHF it started from, which is impossible for an exact solver and
  therefore means the space was chosen wrongly
* ``active_electrons`` -- a label-parsing bug selected core orbitals, and
  only a downstream library assertion caught it

The thresholds are named constants and also function arguments, so a
caller can see what is enforced and can tighten it without editing the
module.
"""
from dataclasses import dataclass, field

from .state import Decision, Level, SelfCheck, next_level

__all__ = ['RunResult', 'ElectronicStructureFailure', 'checks_from_result',
           'run_from_geometry', 'GRADIENT_TOL', 'ENERGY_IDENTITY_TOL']

GRADIENT_TOL = 1e-8
ENERGY_IDENTITY_TOL = 1e-8


class ElectronicStructureFailure(RuntimeError):
    """The energy at this geometry could not be computed reliably.

    Distinct from a geometry that merely has not converged. An SCF that
    will not converge, or an active space that turned out wrong, makes
    the energy meaningless, and no downstream quantity may be built on
    it. This is the failure that stops a run rather than being optimized
    through.
    """


@dataclass
class RunResult:
    """A finished run and the decision about how far to trust it."""

    energy: float = float('nan')
    gradient: object = None
    frequencies: object = None
    decision: Decision = None
    algorithm_data: dict = field(default_factory=dict)

    @property
    def trustworthy(self):
        """True only when the state machine did not refuse."""
        return (self.decision is not None
                and self.decision.level is not Level.REFUSED)


def checks_from_result(grad_norm, n_negative, e_casscf, e_reference,
                       n_elec_active, n_elec_expected,
                       gradient_tol=GRADIENT_TOL,
                       identity_tol=ENERGY_IDENTITY_TOL,
                       active_space_verified=None):
    """Build the four self-checks from a finished calculation.

    Parameters
    ----------
    grad_norm : float
        Norm of the orbital gradient at the reported point.
    n_negative : int
        Number of negative eigenvalues of the orbital Hessian. Zero is a
        minimum; anything else is a saddle point.
    e_casscf, e_reference : float
        The CASSCF energy and the energy of the reference the orbitals
        started from. An exact solver cannot exceed its own starting
        point, so a violation means the active space is wrong.
    n_elec_active, n_elec_expected : int
        Electrons in the selected space, and how many the rule called for.
    gradient_tol, identity_tol : float
        Thresholds for the first and third checks.
    active_space_verified : bool, optional
        Whether the entanglement verification of the active space ran
        and agreed. ``None`` -- the fast level -- adds no check at all;
        ``False`` adds a failing one, because the robust level promises
        a verification this build does not yet perform and a passing
        robust result would be a claim it cannot support.

    Returns
    -------
    list of SelfCheck
        Always four, always in the same order, so a report can be
        compared across runs; a fifth when the robust level asked for
        the verification.
    """
    checks = [
        SelfCheck(
            name='orbital_gradient',
            passed=bool(grad_norm < gradient_tol),
            detail=f'orbital gradient norm {grad_norm:.3e} '
                   f'(threshold {gradient_tol:.0e})',
        ),
        SelfCheck(
            name='hessian_curvature',
            passed=bool(n_negative == 0),
            detail=f'{n_negative} negative eigenvalue(s) in the orbital '
                   f'Hessian; a minimum has none',
        ),
        SelfCheck(
            name='variational',
            passed=bool(e_casscf <= e_reference + identity_tol),
            detail=f'CASSCF {e_casscf:.10f} vs reference {e_reference:.10f} '
                   f'(difference {e_casscf - e_reference:+.3e})',
        ),
        SelfCheck(
            name='active_electrons',
            passed=bool(n_elec_active == n_elec_expected),
            detail=f'{n_elec_active} active electrons, expected '
                   f'{n_elec_expected}',
        ),
    ]
    if active_space_verified is not None:
        checks.append(SelfCheck(
            name='active_space',
            passed=bool(active_space_verified),
            detail=('the entanglement verification ran and agreed'
                    if active_space_verified else
                    'the entanglement verification did not run: it is not '
                    'implemented in this build, so the active space rests '
                    'on the selection rules alone'),
        ))
    return checks


def estimator_checks(overlap_report=None, gap_report=None,
                     shift_report=None, consistency_report=None):
    """Policy report objects as ``SelfCheck`` entries, fixed order.

    The report layer renders whatever self-checks a run carries; this is
    the one conversion through which the round-1 and round-2 estimators
    enter a report, so their names stay stable across runs.  Each entry
    fails exactly when its report says the measurement did (a vanished
    overlap, a gap beyond the convergence threshold, a correlation shift
    beyond its bound, inconsistent spaces between states).  Nothing is
    imported from the policy layer here: the reports arrive as objects
    and are read by their documented fields.
    """
    checks = []
    if overlap_report is not None:
        checks.append(SelfCheck(
            name='active_space_overlap',
            passed=bool(not overlap_report.vanished),
            detail='; '.join(overlap_report.evidence),
        ))
    if gap_report is not None:
        checks.append(SelfCheck(
            name='convergence_gap',
            passed=bool(gap_report.converged),
            detail='; '.join(gap_report.evidence),
        ))
    if shift_report is not None:
        checks.append(SelfCheck(
            name='correlation_shift',
            passed=bool(shift_report.reliable),
            detail='; '.join(shift_report.evidence),
        ))
    if consistency_report is not None:
        checks.append(SelfCheck(
            name='state_consistency',
            passed=bool(consistency_report.consistent),
            detail='; '.join(consistency_report.evidence),
        ))
    return checks


def read_xyz(path):
    """Read an xyz file into a list of ``(symbol, x, y, z)``.

    Raises
    ------
    ValueError
        If the header count and the number of coordinate lines disagree.
    """
    lines = [ln for ln in open(path, encoding='utf-8').read().splitlines()
             if ln.strip()]
    if not lines:
        raise ValueError(f'{path}: the file is empty')
    try:
        n_atoms = int(lines[0].split()[0])
    except (ValueError, IndexError):
        raise ValueError(
            f'{path}: the first line must be the number of atoms, '
            f'found {lines[0]!r}')
    body = lines[2:2 + n_atoms]
    if len(body) < n_atoms:
        raise ValueError(
            f'{path}: the header says {n_atoms} atoms but only '
            f'{len(body)} coordinate lines follow')
    atoms = []
    for line in body:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(
                f'{path}: expected "symbol x y z", found {line!r}')
        try:
            atoms.append((parts[0], float(parts[1]), float(parts[2]),
                          float(parts[3])))
        except ValueError:
            raise ValueError(
                f'{path}: coordinates must be numbers, found {line!r}')
    return atoms


def run_from_geometry(geometry_file, charge=0, spin=0, level='robust',
                      active_space=None, params=None):
    """Run one calculation from a geometry file.

    This is the sequence the command line drives. It is deliberately
    thin: read the geometry, ask the kernel for the numbers, run the
    self-checks, decide how far to trust the result. Every step that can
    refuse does, and a refusal is returned rather than raised.

    Parameters
    ----------
    geometry_file : str
        Path to an xyz file.
    charge, spin : int
        Total charge and ``2S = Nalpha - Nbeta``.
    level : str
        ``'fast'`` skips the entanglement verification, so it starts one
        rung lower on the escalation ladder.
    active_space : str, optional
        Manual specification such as ``'2-4'``; bypasses selection.
    params : mapping, optional
        Effective parameters. Defaults are used when omitted.

    Returns
    -------
    RunResult

    Notes
    -----
    The numerical layers are imported inside this function so that
    ``--check`` and ``--show-defaults`` work without them installed.
    """
    params = params or {}
    atoms = read_xyz(geometry_file)

    result = RunResult()
    result.algorithm_data['geometry'] = '; '.join(
        f'{s} {x} {y} {z}' for s, x, y, z in atoms)
    result.algorithm_data['charge'] = charge
    result.algorithm_data['spin'] = spin

    start = Level.FAST if level == 'fast' else Level.ROBUST

    try:
        computed = _compute(atoms, charge, spin, active_space, params,
                            level=level)
    except ElectronicStructureFailure as exc:
        # This one stops the run rather than escalating. The level only
        # controls how many orbital starts are tried, so no escalation
        # can repair a reference that does not exist -- and a number
        # built on an unconverged reference is exactly what the refusal
        # exists to prevent.
        result.decision = Decision(
            level=Level.REFUSED,
            checks=[SelfCheck('electronic_structure', False, str(exc))],
            reasons=[f'electronic_structure: {exc}'],
            suggestions=['check the geometry, charge, spin and basis, and '
                         'whether the SCF converges at all'],
        )
        return result

    result.energy = float(computed['energy'])
    result.frequencies = computed.get('frequencies')
    result.algorithm_data.update(computed.get('algorithm_data', {}))

    # The robust level promises the entanglement verification; the value
    # is None at the fast level, where no such promise is made.
    verification = computed.get('algorithm_data', {}).get(
        'active_space_verified')
    checks = checks_from_result(
        grad_norm=computed['grad_norm'],
        n_negative=computed['n_negative'],
        # The variational check certifies the wavefunction layer; for
        # the one-step on-top method the reported energy is the on-top
        # value, and the CASSCF value rides next to it.
        e_casscf=computed.get('e_casscf', computed['energy']),
        e_reference=computed['e_reference'],
        n_elec_active=computed['n_elec_active'],
        n_elec_expected=computed['n_elec_expected'],
        gradient_tol=float(params['gradient_tol'])
        if 'gradient_tol' in params else GRADIENT_TOL,
        identity_tol=float(params['energy_identity_tol'])
        if 'energy_identity_tol' in params else ENERGY_IDENTITY_TOL,
        active_space_verified=(None if start is Level.FAST
                               else bool(verification)),
    )
    extra = (computed.get('algorithm_data', {}) or {}).get(
        'estimator_reports')
    if extra:
        # The single injection point for the policy estimators: a
        # producer fills this mapping and the report renders the checks.
        checks = checks + estimator_checks(**extra)
    result.decision = next_level(start, checks)
    return result


def _compute(atoms, charge, spin, active_space, params, level='fast'):
    """Run the numerical kernel.

    Raises
    ------
    ImportError
        If the numerical layers are not installed.
    ElectronicStructureFailure
        If the calculation cannot produce a trustworthy energy.
    """
    try:
        from .kernel import compute_ground_state
        from ..core import casci  # noqa: F401
        from ..policy import rules  # noqa: F401
    except ImportError as exc:
        raise ImportError('the numerical kernel is not installed') from exc

    atoms_str = '; '.join(f'{s} {x} {y} {z}' for s, x, y, z in atoms)
    return compute_ground_state(atoms_str, charge, spin, active_space,
                                params, level=level)


def vertical_excitations_from_geometry(geometry_file, family, *, nroots=6,
                                       charge=0, spin=0, basis='6-31g',
                                       max_steps=400, refine_max_iter=40,
                                       method='casci'):
    """A vertical-excitation table from a geometry file, one call.

    The state-averaged companion of ``run_from_geometry``.  The active
    window is suggested by the frontier identity rules
    (``driver.orbitals.auto_window``) for one excitation family --
    ``'npi'`` (lone pair to pi*) or ``'ppi'`` (pi to pi*) -- the
    orbital point comes from the two-stage convergence, and the
    correlation columns are evaluated per state.  The frontier report
    that suggested the window is returned with the table so the
    suggestion can be checked.

    ``method='mcpdft'`` switches the table factory to
    ``sa_mcpdft_table_from_mol``: the same SA-CASSCF orbital point,
    with the per-state one-step MC-PDFT column added (the groups gain
    ``dE_mcpdft``); the default keeps the NEVPT2 table only.

    The reference is a closed-shell RHF; ``spin`` must be 0.  All
    excitation energies in the returned tables are in Hartree; the
    report layer converts to eV.  Returns the dict of
    ``driver.nevpt2.sa_nevpt2_table_from_mol`` --
    ``window``, ``frontier_report``, ``grad_norm``, ``rows``,
    ``groups`` and the raw ``out``.
    """
    from pyscfad import gto, scf as pscf

    atoms = read_xyz(geometry_file)
    mol = gto.Mole()
    mol.atom = [(s, (x, y, z)) for s, x, y, z in atoms]
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    def builder(m):
        return pscf.RHF(m)

    if method == 'mcpdft':
        from .nevpt2 import sa_mcpdft_table_from_mol
        return sa_mcpdft_table_from_mol(
            mol, family, builder, nroots=nroots, max_steps=max_steps,
            refine_max_iter=refine_max_iter)
    from .nevpt2 import sa_nevpt2_table_from_mol
    return sa_nevpt2_table_from_mol(
        mol, family, builder, nroots=nroots, max_steps=max_steps,
        refine_max_iter=refine_max_iter)
