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

``RIJONX_RHF`` handles closed-shell densities.  ``RIJONX_ROHF`` and
``RIJONX_UHF`` reuse the same ``get_jk``; the pyscfad open-shell layer
calls it with a ``(2, nao, nao)`` density and the RI-J map is applied
spin by spin.  Both open-shell layouts have an explicit analytic
gradient: the skeleton part sees the total density for the RI-J and
core terms and the spin densities for exact exchange, and the response
term is the energy-weighted density built from the RIJONX Fock.  The
explicit routes are checked against ``mf.energy_grad()`` and central
differences in the tests.  ECP molecules work as long as the caller
builds the molecule with an explicit ``ecp`` argument; the auxiliary
basis is built from the same atom list.

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
    from pyscfad.scf import uhf as _ad_uhf
except ImportError:  # pragma: no cover - exercised only without pyscfad
    jax = None
    jnp = None
    _ad_hf = None
    _ad_rohf = None
    _ad_uhf = None

__all__ = [
    'RIJONX_RHF',
    'RIJONX_ROHF',
    'RIJONX_UHF',
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

    class RIJONX_UHF(_RIJONXMixin, _ad_uhf.UHF):
        """Unrestricted Hartree-Fock with the RIJONX Coulomb route."""

        def __init__(self, mol, auxbasis=None, rcond=dri.DEFAULT_RCOND):
            _ad_uhf.UHF.__init__(self, mol)
            self._init_rijonx(auxbasis, rcond)

else:  # pragma: no cover - exercised only without pyscfad

    RIJONX_RHF = None
    RIJONX_ROHF = None
    RIJONX_UHF = None


def rijonx_scf(mol, auxbasis=None, rcond=dri.DEFAULT_RCOND, uhf=False,
               **kernel_kwargs):
    """Run a converged RIJONX SCF and return the mean-field object.

    The class is chosen from ``mol.spin`` (``RIJONX_ROHF`` for a
    non-zero spin, or ``RIJONX_UHF`` when ``uhf`` is true).  A
    non-converged run raises instead of returning a reference every
    downstream number would rest on.
    """
    if _ad_hf is None:
        raise ImportError('pyscfad is not installed')
    spin = int(getattr(mol, 'spin', 0))
    if spin == 0:
        cls = RIJONX_RHF
    else:
        cls = RIJONX_UHF if uhf else RIJONX_ROHF
    mf = cls(mol, auxbasis=auxbasis, rcond=rcond)
    mf.kernel(**kernel_kwargs)
    if not getattr(mf, 'converged', True):
        raise RuntimeError(
            'the RIJONX SCF did not converge; every number below would '
            'rest on an unconverged reference')
    return mf


def _spin_occupations(mf):
    """Spin layout of a converged reference, with integer occupations.

    A restricted open-shell reference carries one coefficient matrix
    and one occupation vector (2, 1, 0); an unrestricted reference
    carries one matrix and one 0/1 occupation vector per spin.  The
    density returned by ``make_rdm1`` disambiguates the two layouts
    before the coefficient shape is inspected, because a closed-shell
    RHF object also passes the coefficient test.
    """
    c = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ)
    dm = np.asarray(mf.make_rdm1())
    if dm.ndim != 3 or dm.shape[0] != 2:
        raise NotImplementedError(
            f'expected a spin-stacked density (2, nao, nao), got {dm.shape}')
    if c.ndim == 2 and occ.ndim == 1:
        c_a = c_b = c
        occ_a = np.asarray(occ > 0.0, dtype=float)
        occ_b = np.asarray(occ == 2.0, dtype=float)
        restricted = True
    elif c.ndim == 3 and occ.ndim == 2:
        c_a, c_b = c[0], c[1]
        occ_a = np.asarray(occ[0] > 0.0, dtype=float)
        occ_b = np.asarray(occ[1] > 0.0, dtype=float)
        restricted = False
    else:
        raise NotImplementedError(
            'unrecognized spin layout: mo_coeff ndim '
            f'{c.ndim}, mo_occ ndim {occ.ndim}')
    for label, o in (('alpha', occ_a), ('beta', occ_b)):
        if np.any((o != 0.0) & (o != 1.0)):
            raise NotImplementedError(
                f'fractional {label} occupations are not supported')
    return c_a, c_b, occ_a, occ_b, dm, restricted


def _rijonx_open_shell_gradient(mf, auxbasis=None, rcond=None):
    """Explicit gradient of a converged ROHF or UHF RIJONX energy.

    The skeleton part is assembled from the spin-summed energy

        E = E_nuc + Tr(D_tot h) + 1/2 Tr(D_tot J_RI[D_tot])
            - 1/2 sum_sigma Tr(D_sigma K[D_sigma]),

    so the RI-J term and the core Hamiltonian see the total density
    while the exact exchange sees each spin density separately.  The
    orbital-response completion is the overlap term

        -2 sum_{mu in atom} W_{mu nu} dS_{mu nu}/dR,

    with the energy-weighted density W that pyscf's ROHF/UHF gradient
    modules use: ``P_a f_a P_a + P_b f_b P_b`` for ROHF and the
    occupation-weighted orbital-energy densities for UHF.  The Fock
    matrices come from ``mf.get_veff``, i.e. from the same RI-J plus
    exact-K assembly the SCF converged on, so the response is taken on
    the RIJONX energy surface rather than on the exact-Coulomb one.
    """
    mol = mf.mol
    pmol = _pyscf_view(mol)
    c_a, c_b, occ_a, occ_b, dm, restricted = _spin_occupations(mf)
    nao = int(pmol.nao_nr())
    if dm.shape[1:] != (nao, nao):
        raise ValueError(
            f'spin density shape {dm.shape} disagrees with nao={nao}')
    d_tot = dm[0] + dm[1]
    aux = auxbasis if auxbasis is not None else getattr(
        mf, 'auxbasis', ri_scf.RI_SCF_DEFAULT_AUX)
    rcond = dri.DEFAULT_RCOND if rcond is None else float(rcond)

    grad = ri_scf.ri_j_gradient(pmol, density=d_tot, aux=aux, rcond=rcond)
    grad += ri_scf.exact_k_gradient_spin(
        pmol, density_alpha=dm[0], density_beta=dm[1])
    grad += ri_scf.core_hamiltonian_gradient(pmol, density=d_tot)
    grad += ri_scf.nuclear_repulsion_gradient(pmol)

    h = pmol.intor('int1e_kin') + pmol.intor('int1e_nuc')
    if pmol.has_ecp():
        h = h + pmol.intor('ECPscalar')
    vhf = np.asarray(mf.get_veff(dm=jnp.asarray(dm)))
    if vhf.shape != (2, nao, nao):
        raise RuntimeError(
            f'expected a spin-stacked RIJONX potential, got {vhf.shape}')
    if restricted:
        p_a = c_a[:, occ_a > 0] @ c_a[:, occ_a > 0].T
        p_b = c_b[:, occ_b > 0] @ c_b[:, occ_b > 0].T
        f_a = h + vhf[0]
        f_b = h + vhf[1]
        dm1e = p_a @ f_a @ p_a + p_b @ f_b @ p_b
    else:
        e_mo = np.asarray(mf.mo_energy)
        if e_mo.ndim != 2:
            raise NotImplementedError(
                'unrestricted gradients need spin-resolved orbital energies')
        ca_o = c_a[:, occ_a > 0]
        cb_o = c_b[:, occ_b > 0]
        dm1e = ((ca_o * e_mo[0][occ_a > 0]) @ ca_o.T
                + (cb_o * e_mo[1][occ_b > 0]) @ cb_o.T)

    # pyscf.grad.rhf.get_ovlp returns -int1e_ipovlp; keep the same sign.
    s1 = -pmol.intor('int1e_ipovlp', comp=3)
    slices = pmol.aoslice_by_atom()
    for atom in range(int(pmol.natm)):
        p0, p1 = int(slices[atom][2]), int(slices[atom][3])
        grad[atom] -= 2.0 * np.einsum('xij,ij->x', s1[:, p0:p1],
                                      dm1e[p0:p1])
    return grad


def rijonx_scf_gradient(mf, auxbasis=None, rcond=None):
    """Full nuclear gradient of the converged RIJONX energy.

    The fixed-coefficient (skeleton) part comes from
    :func:`diffmrc.driver.ri_scf.rijonx_gradient`: the RI-J derivative
    is analytic (three-center factors) and the exchange derivative is
    the dense four-center ``int2e_ip1``/``ip2`` contraction.  The
    orbital-response part is the overlap-derivative term

        -2 sum_{mu in atom} W_{mu nu} dS_{mu nu}/dR,

    with ``W = 2 C_occ eps C_occ^T`` for a closed-shell RHF reference
    and the spin-resolved energy-weighted density for an open-shell
    reference (see :func:`_rijonx_open_shell_gradient`).  Without it the
    skeleton gradient differs from the SCF gradient by O(1) Eh/Bohr
    even at convergence.

    Both the ROHF and the UHF layouts are accepted; RHF keeps its own
    branch so the closed-shell path is bit-for-bit unchanged.

    Returns
    -------
    ndarray, (natm, 3)
        Eh / Bohr.  The exchange derivative is validation scale (dense
        ``nao**4`` tensors).
    """
    mol = mf.mol
    pmol = _pyscf_view(mol)
    c = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ)
    closed = bool(occ.ndim == 1
                  and np.all((occ == 0.0) | (occ == 2.0)))
    if not closed:
        return _rijonx_open_shell_gradient(mf, auxbasis=auxbasis,
                                           rcond=rcond)
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
