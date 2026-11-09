"""Running a two-state scan and handing it to the crossing locator.

``core.crossing`` decides whether a set of energies, gradients and
active-space overlaps licences a crossing.  This module owns the order
in which those numbers are produced: one geometry at a time, each
geometry started from the previous geometry's orbitals so that the scan
follows a branch instead of jumping between solutions.

The evaluator is a callable, so the order can be tested without an
electronic-structure kernel.  The pyscf evaluator at the bottom of the
file is the implementation this project uses; it is imported lazily so
that importing this module does not require pyscf.

WHY BRANCH TRACKING IS PART OF THE SCAN, NOT THE LOCATOR

A scan that restarts every point from the SCF guess is a sequence of
independent calculations; the lowest solution at each geometry can
change identity between points, and the sign change of the gap then
belongs to two different pairs of states.  Starting each point from the
previous point's orbitals keeps the identity, and the active-space
overlap records whether it actually did.  The locator then refuses a
bracket that contains a change of identity (see
``core.crossing.locate_crossing``).

The two multiplicities are handled the same way: the high-spin state is
computed first and the low-spin state starts from the high-spin active
window at the same geometry, so both are described in one orbital space.
"""
from dataclasses import dataclass

import numpy as np

from ..core.crossing import CrossingReport, ScanPoint, locate_crossing

__all__ = ['CrossingScan', 'scan_two_state_crossing',
           'make_casscf_evaluator']


@dataclass(frozen=True)
class CrossingScan:
    """The points of a scan and the crossing report built from them."""

    points: tuple
    report: CrossingReport


def scan_two_state_crossing(evaluate, coordinates, *, overlap_threshold=0.90,
                            monotonic_window=2):
    """Evaluate two surfaces on ``coordinates`` and locate the crossing.

    ``evaluate(coordinate, previous)`` returns ``(ScanPoint, state)``.
    The point must carry both energies and, to make the gradient
    difference meaningful, both gradients with respect to the scan
    coordinate.  ``previous`` is the state returned by the preceding
    call (``None`` for the first point); the evaluator uses it to start
    from the previous orbitals, and may keep its own state instead.
    """
    coordinates = [float(x) for x in coordinates]
    if not coordinates:
        raise ValueError('a scan needs at least one coordinate')
    points = []
    state = None
    for x in coordinates:
        point, state = evaluate(x, state)
        points.append(point)
    report = locate_crossing(points, overlap_threshold=overlap_threshold,
                             monotonic_window=monotonic_window)
    return CrossingScan(points=tuple(points), report=report)


def _orth(mo, overlap):
    m = np.asarray(mo).T @ overlap @ np.asarray(mo)
    w, v = np.linalg.eigh(0.5 * (m + m.T))
    return np.asarray(mo) @ (v @ np.diag(w ** -0.5) @ v.T)


def make_casscf_evaluator(build_mol, spins, active, *, conv_tol=1e-9,
                          conv_tol_grad=1e-6, max_cycle=300,
                          with_gradient=True, project_gradient=None):
    """A branch-tracking CASSCF evaluator for one pair of multiplicities.

    Parameters
    ----------
    build_mol : callable
        ``build_mol(coordinate, spin)`` returns a pyscf ``Mole`` for
        that scan coordinate and multiplicity.  The caller owns the
        geometry convention, basis and charge.
    spins : sequence of two ints
        ``(high_spin, low_spin)`` as ``2S`` values.  The high-spin state
        is evaluated first at every coordinate.
    active : sequence of int
        Active orbital indices in the ROHF orbital ordering.  The same
        columns are used for both multiplicities at every geometry.
    conv_tol, conv_tol_grad : float
        CASSCF energy and orbital-gradient targets.  The gradient target
        matters for the two-start certificate: the pyscf default is the
        square root of the energy tolerance, which is too loose to call
        a point stationary.
    with_gradient : bool
        Evaluate the analytic nuclear gradient projected on the scan
        coordinate.  Without it the gradient difference falls back to a
        finite difference of the gap.
    project_gradient : callable, optional
        ``project_gradient(coordinate, cart_gradient) -> float``.  The
        default is the two-atom z-axis convention
        ``g[-1, 2] - g[0, 2]``.  A bending or multi-atom coordinate must
        supply its own projection; otherwise the reported gradient
        difference is the derivative along an axis the scan does not
        follow.

    Returns
    -------
    callable
        ``evaluate(coordinate, previous) -> (ScanPoint, state)``.
    """
    from pyscf import mcscf, scf

    spins = (int(spins[0]), int(spins[1]))
    active = tuple(int(i) for i in active)
    state = {'mo': {}, 'dm': {}}

    def _ordered(mf):
        coeff = np.asarray(mf.mo_coeff)
        occ = np.asarray(mf.mo_occ)
        act = set(active)
        core = [i for i in range(coeff.shape[1])
                if i not in act and occ[i] > 0.5]
        virt = [i for i in range(coeff.shape[1])
                if i not in act and occ[i] <= 0.5]
        return core + list(active) + virt, len(core)

    def _rohf(mol, dm0):
        if dm0 is not None:
            mf = scf.ROHF(mol)
            mf.conv_tol = 1e-9
            mf.max_cycle = 200
            mf.verbose = 0
            try:
                mf.kernel(dm0=dm0)
                if bool(getattr(mf, 'converged', True)):
                    return mf
            except Exception:                            # noqa: BLE001
                pass
        best = None
        for guess in ('minao', 'huckel'):
            mf = scf.ROHF(mol)
            mf.conv_tol = 1e-9
            mf.max_cycle = 200
            mf.verbose = 0
            if guess != 'minao':
                mf.init_guess = guess
            try:
                mf.kernel()
            except Exception:                            # noqa: BLE001
                continue
            if not bool(getattr(mf, 'converged', True)):
                continue
            if best is None or float(mf.e_tot) < float(best.e_tot):
                best = mf
        if best is None:
            raise RuntimeError('no ROHF guess converged')
        return best

    def evaluate(coordinate, previous=None):
        x = float(coordinate)
        energies = {}
        gradients = {}
        overlaps = {}
        for spin in spins:
            mol = build_mol(x, spin)
            overlap = mol.intor('int1e_ovlp')
            mf = _rohf(mol, state['dm'].get(spin))
            order, ncore = _ordered(mf)
            prev = state['mo'].get(spin)
            if prev is None and spin != spins[0]:
                prev = state['mo'].get(spins[0])
            mo0 = (_orth(prev, overlap) if prev is not None
                   else np.asarray(mf.mo_coeff)[:, order])
            ncas = len(active)
            nele = int(mol.nelectron) - 2 * ncore
            mc = mcscf.CASSCF(mf, ncas, nele)
            mc.mo_coeff = mo0
            mc.conv_tol = float(conv_tol)
            if conv_tol_grad is not None:
                mc.conv_tol_grad = float(conv_tol_grad)
            mc.max_cycle_macro = int(max_cycle)
            mc.verbose = 0
            s = 0.5 * float(spin)
            mc.fix_spin_(ss=s * (s + 1.0), shift=0.5)
            mc.kernel()
            energies[spin] = float(mc.e_tot)
            if with_gradient:
                g = np.asarray(mc.nuc_grad_method().kernel(), dtype=float)
                if project_gradient is None:
                    # Two atoms on the z axis with the first at the
                    # origin: the derivative along the scan coordinate.
                    gradients[spin] = float(g[-1, 2] - g[0, 2])
                else:
                    gradients[spin] = float(project_gradient(x, g))
            if prev is not None:
                a = np.asarray(prev)[:, ncore:ncore + ncas]
                b = np.asarray(mc.mo_coeff)[:, ncore:ncore + ncas]
                m = a.T @ overlap @ b
                overlaps[spin] = float(np.sum(m * m) / ncas)
            mo = np.asarray(mc.mo_coeff, copy=True)
            state['mo'][spin] = mo
            state['dm'][spin] = mf.make_rdm1()

        overlap_prev = None
        if overlaps:
            overlap_prev = float(min(overlaps.values()))
        point = ScanPoint(
            coordinate=x,
            energies=(energies[spins[0]], energies[spins[1]]),
            gradients=((gradients[spins[0]], gradients[spins[1]])
                       if with_gradient else None),
            overlap_previous=overlap_prev)
        return point, state

    return evaluate
