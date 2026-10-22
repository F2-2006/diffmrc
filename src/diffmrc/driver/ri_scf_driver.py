"""RI-J Coulomb inside the SCF iteration (the RIJONX route).

WHY THIS MODULE EXISTS

``driver.ri_scf`` supplies the RI-J Coulomb matrix and its analytic
fixed-coefficient nuclear gradient, and ``core.ri`` supplies the
three-center factors and their geometry derivatives.  Neither module
runs a self-consistent field: the energy and gradient are evaluated at a
density handed in by the caller.  This module closes that gap by
subclassing the pyscfad Hartree-Fock class and replacing the Coulomb
half of ``get_jk`` with the RI-J contraction.  Exchange stays exact
(four-center), so the method is RIJONX, not RI-JK and not RIJCOSX.

WHAT IS REPLACED AND WHAT IS NOT

    get_jk(mol, dm) -> (vj_RI, vk_exact)
    E = E_nuc + Tr(D h) + 1/2 Tr(D J_RI) - 1/4 Tr(D K_exact)

The exact-K branch is the pyscfad ``get_jk`` with ``with_j=False``.  It
still forms and caches the dense ``nao**4`` ERI tensor and, under
geometry differentiation, still goes through pyscfad's dense
``_fill_grad0`` derivative path.  That is the boundary of this route:
the RI-J approximation removes the Coulomb four-center work, not the
exchange four-center work.  RI-JK (no analytic gradient) and RIJCOSX
(fixed-grid prototype only) are out of scope.

DIFFERENTIABILITY

pyscfad differentiates the SCF energy through the implicit function
theorem.  A Coulomb matrix built from host numpy factors would have a
zero tangent with respect to the nuclear coordinates, and the resulting
gradient would silently miss every RI-factor derivative.  The Fock
assembly is therefore wrapped in a ``jax.custom_jvp`` whose coordinate
tangent is the analytic RI derivative of ``core.ri`` (``dB``, ``dS``,
``dS^{-1/2}``).  The tangent with respect to the density is the linear
RI-J map.  With this JVP, ``mf.energy_grad()`` reproduces central
differences of the converged RIJONX energy; see the tests.

The fixed-coefficient gradient of ``driver.ri_scf`` is *not* the SCF
gradient even at a converged density.  The full gradient carries the
orbital-response term, which for a converged RHF density reduces to the
overlap-derivative (energy-weighted density) term.  That term is
supplied here by :func:`rijonx_scf_gradient`, which combines the
skeleton gradient of ``ri_scf.rijonx_gradient`` with the overlap term
in the same way pyscf's ``grad.rhf`` does.

OPEN SHELL AND ECP

``RIJONX_RHF`` handles closed-shell densities.  ``RIJONX_ROHF`` reuses
the same ``get_jk``; the pyscfad ROHF layer calls it with a
``(2, nao, nao)`` density and the RI-J map is applied spin by spin.
Only the closed-shell explicit gradient is implemented; open-shell
gradients are available through ``mf.energy_grad()`` only and have not
been checked against finite differences here.  ECP molecules work as
long as the caller builds the molecule with an explicit ``ecp``
argument; the auxiliary basis is built from the same atom list.

MEMORY

The SCF working set is ``nao**2 * naux`` for the RI-J factors plus the
dense ``nao**4`` exact-K tensor (owned by pyscfad).  The coordinate
tangent materializes one ``(3, nao, nao, naux)`` factor-derivative array
per atom, sequentially.

GEOMETRY CHANGES

Build a fresh ``Mole`` and mean-field object per geometry.  The RI
factor cache is keyed on the built ``Mole._env``, and pyscfad's own
``_eri`` cache and integral path are not reliably refreshed when an
already-built molecule is edited in place; both sides assume a new
molecule.  ``rijonx_scf`` always constructs a fresh object.
"""
from __future__ import annotations

import numpy as np

from ..core import ri as dri
from . import ri_scf

try:  # pyscfad is optional at import time; the CLI must import without it
    import jax
    from jax import numpy as jnp
    from pyscfad.scf import hf as _ad_hf
    from pyscfad.scf import rohf as _ad_rohf
except ImportError:  # pragma: no cover - exercised only without pyscfad
    jax = None
    jnp = None
    _ad_hf = None
    _ad_rohf = None

__all__ = [
    'RIJONX_RHF',
    'RIJONX_ROHF',
    'rijonx_scf',
    'rijonx_scf_gradient',
]


def _pyscf_view(mol):
    """A plain pyscf ``Mole`` view for the host integral calls."""
    to_pyscf = getattr(mol, 'to_pyscf', None)
    return to_pyscf() if to_pyscf is not None else mol


def _geometry_key(mol):
    """Cache key that changes when the molecule is rebuilt at a new geometry.

    pyscfad keeps the built coordinates in ``_env`` (a numpy array) and
    the differentiable copy in ``coords``.  ``_env`` is the linearization
    point, so it is the right key for the host-side factors: inside a
    differentiation the tangent is handled by the custom JVP, not by
    rebuilding the factors.
    """
    env = getattr(mol, '_env', None)
    if env is None:
        return None
    try:
        return np.asarray(env).tobytes()
    except (TypeError, ValueError):
        return None


class _RIState:
    """Host RI factors at one geometry plus their lazy derivative tables."""

    def __init__(self, mol, auxmol, factors):
        self.mol = mol
        self.auxmol = auxmol
        self.factors = factors
        self.nao = int(factors.nao)
        self.naux = int(factors.naux)
        self.natm = int(mol.natm)
        self._dw = {}
        self.fock = self._make_fock()

    def dw(self, atom):
        """``dW/dR_atom`` with ``W = B S^{-1/2}``, shape ``(3, nao, nao, naux)``."""
        atom = int(atom)
        table = self._dw.get(atom)
        if table is None:
            d = dri.ri_factors_deriv(self.mol, self.auxmol, atom,
                                     self.factors)
            table = (np.einsum('cuvP,PQ->cuvQ', d.db, self.factors.jhalf)
                     + np.tensordot(self.factors.b, d.djhalf,
                                    axes=([2], [1])).transpose(2, 0, 1, 3))
            self._dw[atom] = table
        return table

    def _make_fock(self):
        """The RI-J Fock map with an analytic JVP.

        The primal is ``J = W D W^T`` for a density that may carry a
        leading spin/batch axis.  The density tangent is the same linear
        map.  The coordinate tangent contracts the per-atom factor
        derivatives with the coordinate tangent::

            dJ = dW z + W dz,   z = W^T D,   dz = dW^T D.
        """
        nao, naux = self.nao, self.naux
        w = np.asarray(self.factors.b @ self.factors.jhalf)
        wf = jnp.asarray(w.reshape(nao * nao, naux))
        w3 = jnp.asarray(w)

        def assemble(dm):
            n = dm.shape[-1]
            flat = jnp.reshape(dm, (-1, n * n))
            return jnp.reshape(flat @ wf @ wf.T, dm.shape)

        @jax.custom_jvp
        def fock(mol, dm):
            return assemble(dm)

        @fock.defjvp
        def _fock_jvp(primals, tangents):
            mol, dm = primals
            mol_t, dm_t = tangents
            primal = assemble(dm)
            tangent = assemble(dm_t)
            coords = getattr(mol_t, 'coords', None)
            if coords is not None:
                n = dm.shape[-1]
                flat = jnp.reshape(dm, (-1, n * n))
                z = flat @ wf
                d3 = jnp.reshape(dm, (-1, n, n))
                for atom in range(self.natm):
                    dw = self.dw(atom)               # (3, nao, nao, naux)
                    term1 = jnp.einsum('cuvP,xP->xcuv', dw, z)
                    dz = jnp.einsum('cuvP,xuv->xcP', dw, d3)
                    term2 = jnp.einsum('uvP,xcP->xcuv', w3, dz)
                    dj = (term1 + term2) * coords[atom].reshape(1, 3, 1, 1)
                    tangent = tangent + jnp.sum(dj, axis=1).reshape(dm.shape)
            return primal, tangent

        return fock


class _RIJONXMixin:
    """``get_jk`` with RI-J Coulomb and exact exchange."""

    _ignore_attr = ['_ri_state', '_ri_key', '_ri_jk_calls']
    auxbasis = ri_scf.RI_SCF_DEFAULT_AUX
    ri_rcond = dri.DEFAULT_RCOND

    def _init_rijonx(self, auxbasis=None, rcond=dri.DEFAULT_RCOND):
        if auxbasis is not None:
            self.auxbasis = auxbasis
        self.ri_rcond = float(rcond)
        self._ri_state = None
        self._ri_key = None
        self._ri_jk_calls = 0

    def ri_state(self, mol=None):
        """RI factors for the current geometry, rebuilt when the geometry moves."""
        mol = self.mol if mol is None else mol
        key = _geometry_key(mol)
        state = getattr(self, '_ri_state', None)
        if state is None or getattr(self, '_ri_key', None) != key:
            pmol = _pyscf_view(mol)
            auxmol, _label = dri.make_auxmol(pmol, auxbasis=self.auxbasis)
            factors = dri.ri_factors(pmol, auxmol, rcond=self.ri_rcond)
            state = _RIState(pmol, auxmol, factors)
            self._ri_state = state
            self._ri_key = key
        return state

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        """J from RI factors, K from the dense exact four-center integrals.

        ``dm`` may carry a leading spin axis; the RI-J map is linear and
        is applied to each density.  ``omega`` (range-separated Coulomb)
        is refused: the RI-J factors here are built with the full
        Coulomb operator.
        """
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if omega:
            raise NotImplementedError(
                'range-separated RI-J is not implemented; the RI factors '
                'are built for the full Coulomb operator')
        self._ri_jk_calls += 1
        vj = vk = None
        if with_j:
            vj = self.ri_state(mol).fock(mol, dm)
        if with_k:
            _, vk = super().get_jk(mol, dm, hermi=hermi, with_j=False,
                                   with_k=True)
        return vj, vk


if _ad_hf is not None:

    class RIJONX_RHF(_RIJONXMixin, _ad_hf.RHF):
        """Restricted Hartree-Fock with the RIJONX Coulomb route."""

        def __init__(self, mol, auxbasis=None, rcond=dri.DEFAULT_RCOND):
            _ad_hf.RHF.__init__(self, mol)
            self._init_rijonx(auxbasis, rcond)

    class RIJONX_ROHF(_RIJONXMixin, _ad_rohf.ROHF):
        """Restricted open-shell Hartree-Fock with the RIJONX Coulomb route."""

        def __init__(self, mol, auxbasis=None, rcond=dri.DEFAULT_RCOND):
            _ad_rohf.ROHF.__init__(self, mol)
            self._init_rijonx(auxbasis, rcond)

else:  # pragma: no cover - exercised only without pyscfad

    RIJONX_RHF = None
    RIJONX_ROHF = None


def rijonx_scf(mol, auxbasis=None, rcond=dri.DEFAULT_RCOND, **kernel_kwargs):
    """Run a converged RIJONX SCF and return the mean-field object.

    The class is chosen from ``mol.spin`` (``RIJONX_ROHF`` for a
    non-zero spin).  A non-converged run raises instead of returning a
    reference every downstream number would rest on.
    """
    if _ad_hf is None:
        raise ImportError('pyscfad is not installed')
    spin = int(getattr(mol, 'spin', 0))
    cls = RIJONX_ROHF if spin != 0 else RIJONX_RHF
    mf = cls(mol, auxbasis=auxbasis, rcond=rcond)
    mf.kernel(**kernel_kwargs)
    if not getattr(mf, 'converged', True):
        raise RuntimeError(
            'the RIJONX SCF did not converge; every number below would '
            'rest on an unconverged reference')
    return mf


def rijonx_scf_gradient(mf, auxbasis=None, rcond=None):
    """Full nuclear gradient of the converged closed-shell RIJONX energy.

    The fixed-coefficient (skeleton) part comes from
    :func:`diffmrc.driver.ri_scf.rijonx_gradient`: the RI-J derivative
    is analytic (three-center factors) and the exchange derivative is
    the dense four-center ``int2e_ip1``/``ip2`` contraction.  The
    orbital-response part is the overlap-derivative term

        -2 sum_{mu in atom} W_{mu nu} dS_{mu nu}/dR,   W = 2 C_occ eps C_occ^T,

    which is the same completion pyscf's ``grad.rhf`` applies for a
    converged RHF density.  Without it the skeleton gradient differs
    from the SCF gradient by O(1) Eh/Bohr even at convergence.

    Returns
    -------
    ndarray, (natm, 3)
        Eh / Bohr.  Closed-shell occupations only; the exchange
        derivative is validation scale (dense ``nao**4`` tensors).
    """
    mol = mf.mol
    pmol = _pyscf_view(mol)
    c = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ)
    if occ.ndim != 1:
        raise NotImplementedError(
            'only closed-shell (single occupation vector) gradients are '
            'implemented; use mf.energy_grad() for open-shell systems')
    if np.any((occ != 0.0) & (occ != 2.0)):
        raise NotImplementedError(
            'fractional or single occupations are not supported')
    nocc = int(np.count_nonzero(occ))
    aux = auxbasis if auxbasis is not None else getattr(
        mf, 'auxbasis', ri_scf.RI_SCF_DEFAULT_AUX)
    rcond = dri.DEFAULT_RCOND if rcond is None else float(rcond)
    grad = ri_scf.rijonx_gradient(pmol, c[:, :nocc], aux=aux, rcond=rcond)

    # pyscf.grad.rhf.get_ovlp returns -int1e_ipovlp; keep the same sign.
    s1 = -pmol.intor('int1e_ipovlp', comp=3)
    e_mo = np.asarray(mf.mo_energy)
    if e_mo.ndim != 1:
        raise NotImplementedError('open-shell orbital energies')
    dm1e = (c[:, :nocc] * e_mo[:nocc]) @ c[:, :nocc].T * 2.0
    slices = pmol.aoslice_by_atom()
    for atom in range(int(pmol.natm)):
        p0, p1 = int(slices[atom][2]), int(slices[atom][3])
        grad[atom] -= 2.0 * np.einsum('xij,ij->x', s1[:, p0:p1],
                                      dm1e[p0:p1])
    return grad
