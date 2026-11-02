"""Differentiable open-shell SCF reference for f-block molecules.

WHY THIS MODULE EXISTS

pyscfad differentiates a converged SCF by unrolling the iteration by
default (``config.scf_implicit_diff = False``).  For a lanthanide
open-shell reference that path is not the derivative of the converged
solution.  Measured on YbF (stuttgart_rsc + ecpds28mwbso, spin 1,
Yb-F 3.6 Bohr), the analytic ROHF nuclear gradient is
+7.0703556e-2 Eh/Bohr on the metal; central differences of the SCF
energy reproduce it, while the unrolled autodiff gradient returns
+2.4184624e-1 Eh/Bohr (3.42 times the analytic magnitude), and the
corresponding density response is wrong
by five orders of magnitude (2.8e5 against 1.5).  The same unrolled
gradient changes from 2.8e5 to 1.4 when the SCF is started from the
converged density instead of the default guess, so it is not a
well-defined derivative of the converged solution at all.

The defective step is the DIIS extrapolation.  ``pyscfad.scf.diis``
detaches the error vectors (``ops.stop_grad``) and converts the DIIS
matrix entries to numpy, so in the differentiated iteration the
extrapolation coefficients are frozen.  The linearized iteration map is
then not the map that actually converges; its Neumann series does not
contract, and the accumulated derivative depends on the starting point
and on how many cycles were run.

What is *not* responsible was checked term by term on YbF:

* the energy expression and the four-centre integral derivatives.  The
  frozen-coefficient energy, differentiated with respect to the nuclear
  coordinates, agrees with its own central difference to 2.1e-6
  (relative), and each of the three terms (one-electron, Coulomb,
  exchange) agrees with a hand-derived analytic orbital gradient to
  3e-14.
* the ECP, which only enters the one-electron Hamiltonian.
* the ROHF open-shell coupling.  The converged point satisfies the
  stationarity conditions of the reported energy to 3e-7, so the energy
  and the Fock matrix are variationally consistent.
* the occupied-virtual separation.  The occupation vector is constant
  over the displaced geometries.

The fix is to differentiate the fixed point, not the iteration
(implicit differentiation).  pyscfad already implements the adjoint of
the ROHF fixed-point condition; it only has to be enabled.  With that
path the YbF gradient is +7.0703535e-2, a relative deviation of 3.0e-7
from the analytic ROHF gradient.

MULTI-SOLUTION REFERENCES

A lanthanide or actinide SCF can have several converged solutions close
in energy.  CeO (stuttgart_rsc + ecpds28mwbso, spin 2, Ce-O 3.4 Bohr)
has two, 0.112 Eh apart, and both carry the same occupation vector, so
an occupation guard cannot tell them apart.  The initial guess does.
``scf_reference`` therefore prices a fixed, ordered ladder of guesses
plus an optional relay density from a neighbouring geometry and returns
the lowest converged candidate.  The order is fixed and the selection
is by energy, so repeated calls at one geometry return the same branch
bit for bit.  The branch decision is made once, outside the
differentiation; ``rohf_energy`` then runs a single start from the
reference density so that every displaced geometry stays on that
branch.

MEMORY

A pyscfad ``Mole`` carries the basis exponents and contraction
coefficients as differentiable leaves.  Differentiating with respect to
them makes JAX linearize the four-centre integrals over one tangent
direction per primitive (hundreds), which allocates tens of gigabytes
for nao = 70.  Only the nuclear coordinates are differentiated here, so
``_coordinate_view`` drops those two leaves.  The YbF gradient then
needs 3.7 GB and 45 s instead of exceeding 20 GB.

INTEGRATION

This module is the differentiable counterpart of two value-only paths
in the same package:

* :func:`diffmrc.driver.kernel.scf_reference` builds a pyscfad ROHF
  reference for the downstream CASSCF chain and returns ``(mol, mf)``.
  It does not change the differentiation mode, so a gradient that
  passes through its SCF is the unrolled gradient this module replaces.
* :func:`diffmrc.driver.scf_protocol.scf_protocol` prices the same
  guess ladder with plain pyscf and returns a branch verdict.  It is not
  differentiable.

The ladder is one contract: ``GUESS_ORDER`` and ``RELAX_GRID`` are
imported from :mod:`diffmrc.driver.scf_protocol` so the two paths cannot
drift.  The implicit path is selected explicitly -- importing this
module changes nothing for an existing caller, and ``implicit_diff``
restores the process-global pyscfad flag on exit.  That flag is not
thread-safe, so do not run differentiated SCFs concurrently in one
process.
"""
from contextlib import contextmanager

import numpy as np

from .scf_protocol import GUESSES, RELAX_GRID

__all__ = ['GUESS_ORDER', 'RELAX_GRID', 'SCFReference', 'implicit_diff',
           'rohf_energy', 'rohf_gradient', 'scf_reference']

#: Fixed, ordered candidate ladder.  The order is part of the contract:
#: the same molecule must always be priced the same way.  This is the
#: same tuple as :data:`diffmrc.driver.scf_protocol.GUESSES`.
GUESS_ORDER = GUESSES


class SCFReference:
    """A converged reference and the facts needed to reuse its branch.

    Attributes
    ----------
    mf : pyscfad.scf.ROHF
        The converged mean-field object (not differentiated).
    e_tot : float
        Total energy of the selected candidate, in Hartree.
    dm : numpy.ndarray
        Density matrix of the selected candidate, used as the starting
        point of every later single-start solve.
    occ : numpy.ndarray
        Occupation vector of the selected candidate.
    guess : object
        Initial-guess key of the selected candidate, or ``'relay'``.
    spread : float
        Energy difference between the lowest and highest converged
        candidate, in Hartree.  A large spread means the molecule has
        several solutions.
    """

    __slots__ = ('mf', 'e_tot', 'dm', 'occ', 'guess', 'spread')

    def __init__(self, mf, e_tot, dm, occ, guess, spread):
        self.mf = mf
        self.e_tot = e_tot
        self.dm = dm
        self.occ = occ
        self.guess = guess
        self.spread = spread

    def __repr__(self):
        return ('SCFReference(e_tot=%.12f, guess=%r, spread=%.3e)'
                % (self.e_tot, self.guess, self.spread))


@contextmanager
def implicit_diff(enabled=True):
    """Set ``config.scf_implicit_diff`` for the duration of the block.

    Restoring the previous value matters because the flag is process
    global and other callers may rely on the unrolled path (for
    example, for a closed-shell SCF where it is adequate).
    """
    from pyscfad import config
    previous = config.scf_implicit_diff
    config.scf_implicit_diff = bool(enabled)
    try:
        yield
    finally:
        config.scf_implicit_diff = previous


def _coordinate_view(mol):
    """Return a differentiable view of ``mol`` in which only the nuclear
    coordinates are leaves.

    ``copy.copy`` is not usable here: a pyscfad ``Mole`` does not
    preserve its ``coords`` leaf through the copy protocol, and the copy
    ends up with no differentiable leaf at all.  The view is therefore
    rebuilt from the atom specification and the input's coordinate leaf
    is attached afterwards.  The basis data are rebuilt from the same
    atom list, so the integral values are unchanged.  Molecule settings
    that change the integrals (``cart``, ``symmetry``) are carried over;
    dropping ``cart`` alone would silently switch a cartesian basis to
    spherical functions.
    """
    if getattr(mol, 'coords', None) is None:
        raise TypeError(
            'scf_ad needs a pyscfad gto.Mole built with traced coordinates; '
            'a plain pyscf Mole has no coordinate leaf to differentiate')
    if getattr(mol, 'exp', None) is None and getattr(mol, 'ctr_coeff',
                                                     None) is None:
        return mol
    from pyscfad import gto
    view = gto.Mole()
    view.atom = mol.atom
    view.basis = mol.basis
    view.ecp = mol.ecp
    view.unit = mol.unit
    view.spin = mol.spin
    view.charge = mol.charge
    view.cart = bool(getattr(mol, 'cart', False))
    view.symmetry = bool(getattr(mol, 'symmetry', False))
    if getattr(mol, 'max_memory', None) is not None:
        view.max_memory = mol.max_memory
    view.verbose = 0
    view.build(trace_exp=False, trace_ctr_coeff=False)
    view.coords = mol.coords
    return view


def _build(mf_class, mol):
    mf = mf_class(mol)
    mf.verbose = 0
    return mf


def _solve(mol, guess=None, dm0=None, damp=None, level_shift=None,
           conv_tol=1e-10, max_cycle=400, failures=None, label=''):
    """One SCF attempt.  Returns ``None`` when it does not converge.

    ``failures`` (optional list) receives a one-line reason per rejected
    candidate.  A candidate that throws is data, not a stop: the ladder
    moves on, and the caller reports every reason if nothing converges.
    """
    from pyscfad import scf
    mf = _build(scf.ROHF, mol)
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    if guess is not None:
        mf.init_guess = guess
    if damp is not None:
        mf.damp = damp
    if level_shift is not None:
        mf.level_shift = level_shift
    try:
        mf.kernel(dm0=dm0)
    except Exception as exc:  # noqa: BLE001 - a failed candidate is data
        if failures is not None:
            failures.append('%s: %s: %s'
                            % (label or 'candidate', type(exc).__name__,
                               str(exc)[:200]))
        return None
    if not mf.converged:
        if failures is not None:
            failures.append('%s: not converged' % (label or 'candidate',))
        return None
    return mf


def scf_reference(mol, guesses=GUESS_ORDER, dm0_guess=None, conv_tol=1e-10,
                  max_cycle=400):
    """Lowest converged candidate from a fixed ladder of starts.

    ``dm0_guess`` (optional) joins the ladder as the ``'relay'``
    candidate: a density from a neighbouring geometry, offered as one
    more starting point and selected only if it is the lowest converged
    candidate.  This is the adiabatic-continuation handle: pass the
    reference density of the previous geometry and the branch does not
    jump between neighbouring points when the ladder itself is
    ambiguous.

    Raises ``RuntimeError`` when no candidate converges; the message
    carries one line per rejected candidate (exception or
    non-convergence) so a failure can be diagnosed without rerunning the
    ladder.  A caller that needs a reference has no honest fallback; a
    silent unconverged reference is the failure this module exists to
    prevent.
    """
    candidates = []
    failures = []
    for guess in guesses:
        label = 'guess=%r' % (guess,)
        mf = _solve(mol, guess=guess, conv_tol=conv_tol,
                    max_cycle=max_cycle, failures=failures, label=label)
        if mf is not None:
            candidates.append((guess, mf))
    if dm0_guess is not None:
        mf = _solve(mol, dm0=dm0_guess, conv_tol=conv_tol,
                    max_cycle=max_cycle, failures=failures, label='relay')
        if mf is not None:
            candidates.append(('relay', mf))

    if not candidates:
        for damp, shift in RELAX_GRID:
            label = 'damp=%g/shift=%g' % (damp, shift)
            mf = _solve(mol, conv_tol=conv_tol, max_cycle=max_cycle,
                        damp=damp, level_shift=shift, failures=failures,
                        label=label)
            if mf is not None:
                candidates.append((('damp', damp, shift), mf))

    if not candidates:
        detail = '; '.join(failures[-8:]) if failures else 'no start ran'
        raise RuntimeError(
            'no ROHF candidate converged; refusing to return a reference '
            '(%s)' % detail)

    energies = [float(np.asarray(mf.e_tot)) for _, mf in candidates]
    best = int(np.argmin(energies))
    guess, mf = candidates[best]
    return SCFReference(
        mf=mf,
        e_tot=energies[best],
        dm=np.asarray(mf.make_rdm1()),
        occ=np.asarray(mf.mo_occ),
        guess=guess,
        spread=max(energies) - min(energies),
    )


def rohf_energy(mol, dm0=None, conv_tol=1e-10, max_cycle=200):
    """Differentiable ROHF total energy of ``mol``.

    The energy is the value of a converged ROHF fixed point; the
    derivative is obtained from the adjoint of the fixed-point
    condition (implicit differentiation), not from unrolling the SCF
    iteration.  ``dm0`` should be the density of a previously selected
    reference so that every call stays on one branch.

    The molecule is taken in a coordinate-only view internally, so the
    caller may pass an ordinary pyscfad ``Mole``; any basis leaves it
    carries receive no gradient by construction.  ROHF is used for every
    spin, including ``spin = 0``, where it is equivalent to RHF.
    """
    from pyscfad import scf

    view = _coordinate_view(mol)
    with implicit_diff(True):
        mf = _build(scf.ROHF, view)
        mf.conv_tol = conv_tol
        mf.max_cycle = max_cycle
        mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError('ROHF did not converge; the energy has no '
                           'fixed point to differentiate')
    return mf.e_tot


def rohf_gradient(mol, dm0=None, conv_tol=1e-10, max_cycle=200):
    """Nuclear gradient of :func:`rohf_energy`.

    Convenience wrapper for the common one-shot call; the same result
    comes from ``jax.grad(rohf_energy)(mol)``.
    """
    import jax
    view = _coordinate_view(mol)
    gradient = jax.grad(lambda m: rohf_energy(m, dm0=dm0, conv_tol=conv_tol,
                                              max_cycle=max_cycle))(view)
    return np.asarray(gradient.coords)
