"""The self-checks that gate a reported result.

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

The thresholds are named constants so a caller can see what is being
enforced rather than infer it from the numbers.
"""
from dataclasses import dataclass, field

from .state import Decision, Level, SelfCheck, next_level

__all__ = ['RunResult', 'checks_from_result', 'GRADIENT_TOL',
           'ENERGY_IDENTITY_TOL']

GRADIENT_TOL = 1e-8
ENERGY_IDENTITY_TOL = 1e-8


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
        return self.decision is not None and self.decision.level is not Level.REFUSED


def checks_from_result(grad_norm, n_negative, e_casscf, e_reference,
                       n_elec_active, n_elec_expected,
                       gradient_tol=GRADIENT_TOL,
                       identity_tol=ENERGY_IDENTITY_TOL):
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

    Returns
    -------
    list of SelfCheck
        Always four, always in the same order, so a report can be
        compared across runs.
    """
    return [
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
