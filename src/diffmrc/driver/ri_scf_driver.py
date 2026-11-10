"""RI-J Coulomb inside the SCF iteration (the RIJONX and RIJK routes).

WHY THIS MODULE EXISTS

``driver.ri_scf`` supplies the RI-J Coulomb matrix, the RI-K exchange
matrix and their analytic fixed-coefficient nuclear gradients, and
``core.ri`` supplies the three-center factors and their geometry
derivatives.  Neither module runs a self-consistent field: the energy
and gradient are evaluated at a density handed in by the caller.  This
module closes that gap by subclassing the pyscfad Hartree-Fock class and
replacing the Coulomb and, optionally, the exchange half of ``get_jk``.

WHAT IS REPLACED AND WHAT IS NOT

    get_jk(mol, dm) -> (vj_RI, vk)
    E = E_nuc + Tr(D h) + 1/2 Tr(D J_RI) - 1/4 Tr(D K)

``K`` is the dense exact four-center exchange by default (the RIJONX
route).  With ``ri_k=True`` it is the RI-K exchange of
``ri_scf.ri_k_fock``, built from the same three-center factors as the
Coulomb term (the RIJK route).  The exact-K branch is the pyscfad
``get_jk`` with ``with_j=False``: it still forms and caches the dense
``nao**4`` ERI tensor and, under geometry differentiation, still goes
through pyscfad's dense ``_fill_grad0`` derivative path.  RIJCOSX
(fixed-grid prototype only) remains out of scope.

DIFFERENTIABILITY

pyscfad differentiates the SCF energy through the implicit function
theorem.  A Coulomb or exchange matrix built from host numpy factors
would have a zero tangent with respect to the nuclear coordinates, and
the resulting gradient would silently miss every RI-factor derivative.
Both Fock assemblies are therefore wrapped in ``jax.custom_jvp`` whose
coordinate tangent is the analytic RI derivative of ``core.ri``
(``dB``, ``dS``, ``dS^{-1/2}``).  The tangent with respect to the
density is the same linear map in both cases.  With these JVPs,
``mf.energy_grad()`` reproduces central differences of the converged
energy; see the tests.

The fixed-coefficient gradient of ``driver.ri_scf`` is *not* the SCF
gradient even at a converged density.  The full gradient carries the
orbital-response term, which for a converged RHF density reduces to the
overlap-derivative (energy-weighted density) term.  That term is
supplied here by :func:`rijonx_scf_gradient`, which combines the
skeleton gradient of ``ri_scf.rijonx_gradient`` or
``ri_scf.rijk_gradient`` with the overlap term in the same way pyscf's
``grad.rhf`` does.

OPEN SHELL AND ECP

``RIJONX_RHF`` handles closed-shell densities.  ``RIJONX_ROHF`` and
``RIJONX_UHF`` reuse the same ``get_jk``; the pyscfad open-shell layer
calls it with a ``(2, nao, nao)`` density and both RI maps are applied
spin by spin.  Both open-shell layouts have an explicit analytic
gradient: the skeleton part sees the total density for the RI-J and
core terms and the spin densities for exchange (exact or RI-K), and the
response term is the energy-weighted density built from the RIJONX
Fock.  The explicit routes are checked against ``mf.energy_grad()`` and
central differences in the tests.  ECP molecules work as long as the
caller builds the molecule with an explicit ``ecp`` argument; the
auxiliary basis is built from the same atom list.

PLAIN-PYSCF VARIANT

``HostRIJK_RHF``/``ROHF``/``UHF`` run the same ``get_jk`` on plain
pyscf (no pyscfad), with numpy maps instead of the custom-JVP ones.
They exist so that the RI-K SCF loop can be exercised and validated
where pyscfad is unavailable, and they are the reference the pyscfad
classes are compared against.

MEMORY

The SCF working set is ``nao**2 * naux`` for the RI factors.  The RIJK
route adds one more ``nao**2 * naux`` intermediate for the exchange
half-transform and never allocates a ``nao**4`` array; the RIJONX route
adds the dense ``nao**4`` exact-K tensor (owned by pyscfad).  The
coordinate tangent materializes one ``(3, nao, nao, naux)``
factor-derivative array per atom, sequentially.

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

from pyscf import scf as _host_scf

__all__ = [
    'RIJONX_RHF',
    'RIJONX_ROHF',
    'RIJONX_UHF',
    'HostRIJK_RHF',
    'HostRIJK_ROHF',
    'HostRIJK_UHF',
    'rijonx_scf',
    'rijk_scf',
    'rijonx_scf_gradient',
    'rijk_scf_gradient',
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
        # The custom-JVP maps exist only when JAX/pyscfad is importable;
        # the plain-pyscf classes use the numpy maps below.
        self.fock = self._make_fock() if jnp is not None else None
        self.fock_k = self._make_fock_k() if jnp is not None else None

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
        w = np.asarray(dri.ri_weight_matrix(self.factors))
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

    def _make_fock_k(self):
        """The RI-K exchange map with an analytic JVP.

        The primal is ``K = sum_P W_P D W_P``, evaluated as two
        half-transforms through the auxiliary index (``T = W D`` then
        ``K = sum_P T_P W_P``).  The density tangent is the same linear
        map; the coordinate tangent is

            dK = sum_P (dW_P D W_P + W_P D dW_P),

        which is assembled from the per-atom factor derivatives, one
        atom at a time.
        """
        w = np.asarray(dri.ri_weight_matrix(self.factors))
        wT = jnp.asarray(np.transpose(w, (2, 0, 1)))   # (naux, nao, nao)

        def assemble(dm):
            n = dm.shape[-1]
            flat = jnp.reshape(dm, (-1, n, n))
            t = jnp.matmul(wT[None], flat[:, None])    # W_P D, batched
            k = jnp.matmul(t, wT[None]).sum(axis=1)    # sum_P W_P D W_P
            k = 0.5 * (k + jnp.swapaxes(k, -1, -2))
            return jnp.reshape(k, dm.shape)

        @jax.custom_jvp
        def fock_k(mol, dm):
            return assemble(dm)

        @fock_k.defjvp
        def _fock_k_jvp(primals, tangents):
            mol, dm = primals
            mol_t, dm_t = tangents
            primal = assemble(dm)
            tangent = assemble(dm_t)
            coords = getattr(mol_t, 'coords', None)
            if coords is not None:
                n = dm.shape[-1]
                flat = jnp.reshape(dm, (-1, n, n))
                z = jnp.matmul(flat[:, None], wT[None])   # D W_P, batched
                for atom in range(self.natm):
                    dT = jnp.asarray(
                        np.transpose(self.dw(atom), (0, 3, 1, 2)))
                    t1 = jnp.matmul(dT[:, None], z[None, :]).sum(axis=2)
                    dk = t1 + jnp.swapaxes(t1, -1, -2)    # (3, batch, nao, nao)
                    tangent = tangent + jnp.tensordot(
                        coords[atom], dk, axes=(0, 0)).reshape(dm.shape)
            return primal, tangent

        return fock_k

    def fock_np(self, dm):
        """Host numpy RI-J matrix for this geometry (plain-pyscf path)."""
        return ri_scf.ri_j_fock(self.mol, dm, factors=self.factors)

    def fock_k_np(self, dm):
        """Host numpy RI-K matrix for this geometry (plain-pyscf path)."""
        return ri_scf.ri_k_fock(self.mol, dm, factors=self.factors)


class _RIJONXMixin:
    """``get_jk`` with RI-J Coulomb and exact or RI-K exchange.

    ``ri_k`` selects the exchange route: ``False`` keeps the dense exact
    four-center exchange (RIJONX), ``True`` replaces it by the RI-K
    contraction (RIJK).  ``_ri_host`` selects numpy maps instead of the
    custom-JVP maps; it is set by the plain-pyscf subclasses only.
    """

    _ignore_attr = ['_ri_state', '_ri_key', '_ri_jk_calls']
    auxbasis = ri_scf.RI_SCF_DEFAULT_AUX
    ri_rcond = dri.DEFAULT_RCOND
    ri_k = False
    _ri_host = False

    def _init_rijonx(self, auxbasis=None, rcond=dri.DEFAULT_RCOND,
                     ri_k=False):
        if auxbasis is not None:
            self.auxbasis = auxbasis
        self.ri_rcond = float(rcond)
        self.ri_k = bool(ri_k)
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
        """J from RI factors, K from exact four-center or RI-K integrals.

        ``dm`` may carry a leading spin axis; both RI maps are linear and
        are applied to each density.  ``omega`` (range-separated Coulomb)
        is refused: the RI factors here are built with the full Coulomb
        operator.
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
        state = self.ri_state(mol) if (with_j or (with_k and self.ri_k)) \
            else None
        if with_j:
            vj = state.fock_np(dm) if self._ri_host else state.fock(mol, dm)
        if with_k:
            if not self.ri_k:
                _, vk = super().get_jk(mol, dm, hermi=hermi, with_j=False,
                                       with_k=True)
            elif self._ri_host:
                vk = state.fock_k_np(dm)
            else:
                vk = state.fock_k(mol, dm)
        return vj, vk


if _ad_hf is not None:

    class RIJONX_RHF(_RIJONXMixin, _ad_hf.RHF):
        """Restricted Hartree-Fock with the RIJONX Coulomb route.

        ``ri_k=True`` switches the exchange to RI-K (RIJK route).
        """

        def __init__(self, mol, auxbasis=None, rcond=dri.DEFAULT_RCOND,
                     ri_k=False):
            _ad_hf.RHF.__init__(self, mol)
            self._init_rijonx(auxbasis, rcond, ri_k=ri_k)

    class RIJONX_ROHF(_RIJONXMixin, _ad_rohf.ROHF):
        """Restricted open-shell Hartree-Fock with the RIJONX Coulomb route."""

        def __init__(self, mol, auxbasis=None, rcond=dri.DEFAULT_RCOND,
                     ri_k=False):
            _ad_rohf.ROHF.__init__(self, mol)
            self._init_rijonx(auxbasis, rcond, ri_k=ri_k)

    class RIJONX_UHF(_RIJONXMixin, _ad_uhf.UHF):
        """Unrestricted Hartree-Fock with the RIJONX Coulomb route."""

        def __init__(self, mol, auxbasis=None, rcond=dri.DEFAULT_RCOND,
                     ri_k=False):
            _ad_uhf.UHF.__init__(self, mol)
            self._init_rijonx(auxbasis, rcond, ri_k=ri_k)

else:  # pragma: no cover - exercised only without pyscfad

    RIJONX_RHF = None
    RIJONX_ROHF = None
    RIJONX_UHF = None


class HostRIJK_RHF(_RIJONXMixin, _host_scf.hf.RHF):
    """Plain-pyscf RHF with RI-J Coulomb and RI-K exchange.

    The numpy maps are used instead of the custom-JVP maps, so this class
    runs without pyscfad.  It exists to exercise and validate the RI-K
    SCF loop where pyscfad is unavailable and to serve as the numerical
    reference for the pyscfad classes.  ``ri_k=False`` gives the
    plain-pyscf RIJONX control.
    """

    _ri_host = True

    def __init__(self, mol, auxbasis=None, rcond=dri.DEFAULT_RCOND,
                 ri_k=True):
        _host_scf.hf.RHF.__init__(self, mol)
        self._init_rijonx(auxbasis, rcond, ri_k=ri_k)


class HostRIJK_ROHF(_RIJONXMixin, _host_scf.rohf.ROHF):
    """Plain-pyscf ROHF with RI-J Coulomb and RI-K exchange."""

    _ri_host = True

    def __init__(self, mol, auxbasis=None, rcond=dri.DEFAULT_RCOND,
                 ri_k=True):
        _host_scf.rohf.ROHF.__init__(self, mol)
        self._init_rijonx(auxbasis, rcond, ri_k=ri_k)


class HostRIJK_UHF(_RIJONXMixin, _host_scf.uhf.UHF):
    """Plain-pyscf UHF with RI-J Coulomb and RI-K exchange."""

    _ri_host = True

    def __init__(self, mol, auxbasis=None, rcond=dri.DEFAULT_RCOND,
                 ri_k=True):
        _host_scf.uhf.UHF.__init__(self, mol)
        self._init_rijonx(auxbasis, rcond, ri_k=ri_k)


def rijonx_scf(mol, auxbasis=None, rcond=dri.DEFAULT_RCOND, uhf=False,
               ri_k=False, **kernel_kwargs):
    """Run a converged RIJONX (or RIJK) SCF and return the mean-field object.

    The class is chosen from ``mol.spin`` (``RIJONX_ROHF`` for a
    non-zero spin, or ``RIJONX_UHF`` when ``uhf`` is true).  With
    ``ri_k=True`` the exchange is the RI-K contraction instead of the
    dense exact four-center exchange; the method is then RIJK.  A
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
    mf = cls(mol, auxbasis=auxbasis, rcond=rcond, ri_k=ri_k)
    mf.kernel(**kernel_kwargs)
    if not getattr(mf, 'converged', True):
        raise RuntimeError(
            'the RIJONX SCF did not converge; every number below would '
            'rest on an unconverged reference')
    return mf


def rijk_scf(mol, auxbasis=None, rcond=dri.DEFAULT_RCOND, uhf=False,
             **kernel_kwargs):
    """Run a converged RIJK SCF (RI-J Coulomb plus RI-K exchange)."""
    return rijonx_scf(mol, auxbasis=auxbasis, rcond=rcond, uhf=uhf,
                      ri_k=True, **kernel_kwargs)


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


def _rijonx_open_shell_gradient(mf, auxbasis=None, rcond=None, ri_k=False):
    """Explicit gradient of a converged ROHF or UHF RIJONX energy.

    The skeleton part is assembled from the spin-summed energy

        E = E_nuc + Tr(D_tot h) + 1/2 Tr(D_tot J_RI[D_tot])
            - 1/2 sum_sigma Tr(D_sigma K[D_sigma]),

    so the RI-J term and the core Hamiltonian see the total density
    while the exchange (exact or RI-K, selected by ``ri_k``) sees each
    spin density separately.  The orbital-response completion is the
    overlap term

        -2 sum_{mu in atom} W_{mu nu} dS_{mu nu}/dR,

    with the energy-weighted density W that pyscf's ROHF/UHF gradient
    modules use: ``P_a f_a P_a + P_b f_b P_b`` for ROHF and the
    occupation-weighted orbital-energy densities for UHF.  The Fock
    matrices come from ``mf.get_veff``, i.e. from the same RI-J plus
    exchange assembly the SCF converged on, so the response is taken on
    the RIJONX/RIJK energy surface rather than on the exact-Coulomb one.
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
    if ri_k:
        grad += ri_scf.ri_k_gradient_spin(
            pmol, density_alpha=dm[0], density_beta=dm[1], aux=aux,
            rcond=rcond)
    else:
        grad += ri_scf.exact_k_gradient_spin(
            pmol, density_alpha=dm[0], density_beta=dm[1])
    grad += ri_scf.core_hamiltonian_gradient(pmol, density=d_tot)
    grad += ri_scf.nuclear_repulsion_gradient(pmol)

    h = pmol.intor('int1e_kin') + pmol.intor('int1e_nuc')
    if pmol.has_ecp():
        h = h + pmol.intor('ECPscalar')
    dm_arg = jnp.asarray(dm) if jnp is not None else np.asarray(dm)
    vhf = np.asarray(mf.get_veff(dm=dm_arg))
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


def rijonx_scf_gradient(mf, auxbasis=None, rcond=None, ri_k=None):
    """Full nuclear gradient of a converged RIJONX or RIJK energy.

    The fixed-coefficient (skeleton) part comes from
    :func:`diffmrc.driver.ri_scf.rijonx_gradient` (exact four-center
    exchange) or :func:`diffmrc.driver.ri_scf.rijk_gradient` (RI-K
    exchange); ``ri_k=None`` takes the route the mean-field object
    converged on.  In both cases the RI-J derivative is analytic
    (three-center factors); the exact-exchange derivative is the dense
    four-center ``int2e_ip1``/``ip2`` contraction, the RI-K derivative
    uses only the three-center factor derivatives.  The orbital-response
    part is the overlap-derivative term

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
        Eh / Bohr.  With exact exchange the derivative is validation
        scale (dense ``nao**4`` tensors); with RI-K it is production
        scale (``nao**2 * naux`` working set).
    """
    if ri_k is None:
        ri_k = bool(getattr(mf, 'ri_k', False))
    mol = mf.mol
    pmol = _pyscf_view(mol)
    c = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ)
    closed = bool(occ.ndim == 1
                  and np.all((occ == 0.0) | (occ == 2.0)))
    if not closed:
        return _rijonx_open_shell_gradient(mf, auxbasis=auxbasis,
                                           rcond=rcond, ri_k=ri_k)
    if np.any((occ != 0.0) & (occ != 2.0)):
        raise NotImplementedError(
            'fractional or single occupations are not supported')
    nocc = int(np.count_nonzero(occ))
    aux = auxbasis if auxbasis is not None else getattr(
        mf, 'auxbasis', ri_scf.RI_SCF_DEFAULT_AUX)
    rcond = dri.DEFAULT_RCOND if rcond is None else float(rcond)
    if ri_k:
        grad = ri_scf.rijk_gradient(pmol, c[:, :nocc], aux=aux, rcond=rcond)
    else:
        grad = ri_scf.rijonx_gradient(pmol, c[:, :nocc], aux=aux,
                                      rcond=rcond)

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


def rijk_scf_gradient(mf, auxbasis=None, rcond=None):
    """Full nuclear gradient of a converged RIJK energy.

    Convenience wrapper around :func:`rijonx_scf_gradient` with the
    RI-K exchange route.  Only ``nao**2 * naux`` arrays are formed; the
    dense ``nao**4`` exchange-derivative tensors are never allocated.
    """
    return rijonx_scf_gradient(mf, auxbasis=auxbasis, rcond=rcond, ri_k=True)
