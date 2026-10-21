"""Differentiable resolution of identity (RI/DF) infrastructure.

WHY THIS MODULE EXISTS

The differentiable multireference chain currently materializes four-center
AO integrals from ``mol.intor('int2e', aosym='s1')``.  The nuclear
derivative of that route goes through pyscfad's ``_fill_grad0``, which
allocates ``nctr * nao**4`` doubles in one block (197 GB and OOM at
nao=175; 51.3 GiB at nao=96).  The RI approximation

    (mu nu | lam sig) ~= sum_{P Q} (mu nu | P) S^{-1}_{P Q} (Q | lam sig),
    S_{P Q} = (P | Q),

replaces the four-center tensor by two three-center tensors.  Its nuclear
derivative needs only the three-center derivative integrals, whose memory
is ``3 * nao**2 * naux`` per atom instead of ``nctr * nao**4``.

AXIS AND SIGN CONVENTIONS (fixed here; verified against central
differences with pyscf 2.14.0 on 101, h = 1e-5 Bohr, see
``tests/core/test_ri.py`` and the development validation script):

* ``B[mu, nu, P] = (mu nu | P)``, shape ``(nao, nao, naux)``.  Both AO
  indices come first; the auxiliary index is last.
* ``S[P, Q] = (P | Q)``, shape ``(naux, naux)``.
* ``eri_ao[mu, nu, lam, sig] = (mu nu | lam sig)`` in chemist notation.
* ``aux_e2(mol, auxmol, intor='int3c2e_ip1')`` returns an array with a
  MINUS sign relative to the nuclear derivative of the FIRST AO center:

      ip1[c, mu, nu, P] = - d(mu nu | P) / dR_{c(mu)}[c].

* The derivative with respect to the SECOND AO center is ip1 with the AO
  pair axes exchanged:

      d/dR_{c(nu)} = -ip1.transpose(0, 2, 1, 3).

* ``aux_e2(..., intor='int3c2e_ip2')`` differentiates the AUXILIARY
  center, not the second AO center (pyscf's own ``df.grad.rhf`` labels
  the same array "(i,j|d/dX P)"):

      ip2[c, mu, nu, P] = - d(mu nu | P) / dR_{c(P)}[c].

  libcint exposes no ``int3c2e_ip3`` (undefined symbol on 101), so the
  auxiliary-center derivative has to come from this ip2.
* ``int2c2e_ip1`` / ``int2c2e_ip2`` differentiate the first / second
  auxiliary center with the same minus sign:

      dS/dR_A = -(ip1[P on A] + ip2[Q on A]).

The minus sign is the libcint convention: ``ip1`` is the derivative with
respect to the electronic coordinate, which is minus the nuclear
derivative.  All signs above were confirmed by isolated finite
differences: displacing the molecule with the auxiliary basis fixed
gives the AO-center terms to 8e-11, displacing the auxiliary basis with
the molecule fixed gives the auxiliary term to 8e-11, and the same
decomposition reproduces the full displacement to 5e-11.

INTEGRATION WITH THE JAX CHAIN

The host functions (``ri_factors``, ``ri_factors_deriv``) call pyscf and
return numpy arrays.  The pure-JAX function ``ri_eri_from_factors(b,
jhalf)`` maps the raw three-center tensor ``b`` and the inverse square
root of the metric ``jhalf`` to the RI four-center tensor and carries an
explicit ``custom_jvp``.  The intended nuclear-gradient recipe is:

    factors = ri_factors(mol, auxmol)                 # host, numpy
    E = energy_from_factors(factors.b, factors.jhalf) # pure JAX chain
    dE_db, dE_djhalf = jax.grad(E, argnums=(0, 1))(factors.b, factors.jhalf)
    for atom in range(mol.natm):
        d = ri_factors_deriv(mol, auxmol, atom, factors)
        dE_dR[atom] = sum(dE_db * d.db) + sum(dE_djhalf * d.djhalf)

This never touches pyscfad's ``_fill_grad0``: the only derivative
integrals are the three-center ip1/ip2 arrays and the metric ip1/ip2
arrays, all of size ``nao**2 * naux``.  ``ri_eri_deriv_contract``
implements the same contraction for a caller that already holds
``gamma = dE/d(eri_ao)`` instead of the factor gradients.

The derivative of ``jhalf`` with respect to the metric is analytic
(Daleckii-Krein divided differences, degeneracy-safe) and is exposed as
``inv_sqrt_metric_jvp``.  Do not differentiate ``jnp.linalg.eigh`` inside
a JAX chain for this purpose: the eigh vjp is singular at degenerate
eigenvalues and the auxiliary metric can carry them.

MEMORY

``ri_eri_ao`` still constructs the dense ``nao**4`` tensor; it is the
physical object the CI layer consumes.  The derivative path
(``ri_factors_deriv`` + ``ri_eri_deriv_contract`` or the factor-gradient
recipe) is the one that avoids the memory wall.  ``ri_eri_deriv``
materializes the full ``(3, nao, nao, nao, nao)`` derivative and is a
validation tool only; it refuses to allocate above ``max_bytes``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
from jax import numpy as jnp
from pyscf import df
from pyscf.gto.mole import ATOM_OF

__all__ = [
    'AUX_BASIS_PRIORITY',
    'MetricEig',
    'RIFactors',
    'RIFactorDerivs',
    'make_auxmol',
    'ri_three_center',
    'ri_metric',
    'metric_eig',
    'inv_sqrt_metric',
    'inv_sqrt_metric_jvp',
    'ri_factors',
    'ri_eri_ao',
    'ri_eri_mo',
    'ri_three_center_deriv',
    'ri_metric_deriv',
    'ri_factors_deriv',
    'ri_eri_deriv',
    'ri_eri_deriv_contract',
    'ri_eri_from_factors',
    'ri_eri_jvp',
]

#: Auxiliary basis sets tried in order when the caller does not name one.
AUX_BASIS_PRIORITY = ('def2-universal-jkfit', 'cc-pvdz-ri')

#: Default reciprocal condition number for truncating the metric.
DEFAULT_RCOND = 1e-12


# ---------------------------------------------------------------------------
# Auxiliary basis
# ---------------------------------------------------------------------------
def make_auxmol(mol, auxbasis=None):
    """Build the auxiliary molecule and report the basis actually used.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Built AO molecule.
    auxbasis : str or None
        Explicit auxiliary basis.  ``None`` tries ``AUX_BASIS_PRIORITY``
        and falls back to ``pyscf.df.addons.make_auxmol(mol)`` (which
        reuses the orbital basis) when no fitting basis covers every
        element.

    Returns
    -------
    auxmol : pyscf.gto.Mole
    used : str
        Human-readable label of the auxiliary basis.

    A basis that silently produces zero functions counts as a failure,
    not as a success: element coverage gaps in the fitting sets surface
    that way in some pyscf versions.
    """
    if auxbasis is not None:
        auxmol = df.addons.make_auxmol(mol, auxbasis=auxbasis)
        if auxmol.nao_nr() == 0:
            raise ValueError(
                f'auxiliary basis {auxbasis!r} produced zero functions')
        return auxmol, str(auxbasis)

    failures = []
    for name in AUX_BASIS_PRIORITY:
        try:
            auxmol = df.addons.make_auxmol(mol, auxbasis=name)
        except Exception as exc:  # element not covered, malformed name, ...
            failures.append(f'{name}: {type(exc).__name__}')
            continue
        if auxmol.nao_nr() > 0:
            return auxmol, name
        failures.append(f'{name}: zero functions')
    auxmol = df.addons.make_auxmol(mol)
    if auxmol.nao_nr() == 0:
        raise ValueError(
            'no auxiliary basis available; tried ' + '; '.join(failures)
            + '; fallback to the orbital basis also produced zero functions')
    return auxmol, 'orbital-basis fallback (' + '; '.join(failures) + ')'


# ---------------------------------------------------------------------------
# Integral and metric assembly
# ---------------------------------------------------------------------------
def ri_three_center(mol, auxmol):
    """Raw three-center integrals ``B[mu, nu, P] = (mu nu | P)``.

    Uses ``pyscf.df.incore.aux_e2`` (the concatenated-molecule path used by
    the pyscf DF module itself); ``mol.intor('int3c2e_sph')`` is NOT
    equivalent because it puts the orbital basis in the third slot too.
    """
    return df.incore.aux_e2(mol, auxmol, intor='int3c2e_sph', aosym='s1')


def ri_metric(auxmol):
    """Auxiliary metric ``S[P, Q] = (P | Q)``, full matrix."""
    s = auxmol.intor('int2c2e_sph', aosym='s1')
    return 0.5 * (s + s.T)


@dataclass(frozen=True)
class MetricEig:
    """Eigendecomposition of the auxiliary metric (ascending eigenvalues)."""
    w: np.ndarray
    u: np.ndarray
    keep: np.ndarray
    rcond: float
    cond: float
    n_dropped: int


def metric_eig(s, rcond=DEFAULT_RCOND):
    """Eigendecompose a symmetric metric and flag the retained subspace.

    Eigenvalues ``<= rcond * max(w)`` are dropped; their inverse square
    root is set to zero (a truncation, not a reflection).
    """
    s = np.asarray(s)
    s = 0.5 * (s + s.T)
    w, u = np.linalg.eigh(s)
    if w[-1] <= 0.0:
        raise ValueError('auxiliary metric is not positive definite')
    keep = w > rcond * w[-1]
    cond = float(w[-1] / w[0]) if w[0] > 0.0 else np.inf
    return MetricEig(w=w, u=u, keep=keep, rcond=rcond, cond=cond,
                     n_dropped=int((~keep).sum()))


def inv_sqrt_metric(s, rcond=DEFAULT_RCOND, eig=None):
    """Symmetric inverse square root ``S^{-1/2}`` with an rcond truncation.

    Returns ``(jhalf, info)`` when ``return_info`` is not requested; the
    info dictionary carries ``cond``, ``n_dropped``, ``wmin``, ``wmax``
    and the ``rcond`` actually used.
    """
    eig = metric_eig(s, rcond) if eig is None else eig
    scale = np.zeros_like(eig.w)
    scale[eig.keep] = 1.0 / np.sqrt(eig.w[eig.keep])
    jhalf = (eig.u * scale) @ eig.u.T
    jhalf = 0.5 * (jhalf + jhalf.T)
    return jhalf


def _inv_sqrt_deriv_from_eig(eig, ds):
    """Derivative of the truncated S^{-1/2} map, Daleckii-Krein form.

    For f(x) = x^{-1/2} the divided difference simplifies to

        F_ij = -1 / (sqrt(w_i) sqrt(w_j) (sqrt(w_i) + sqrt(w_j))),

    which is finite at w_i = w_j (limit -1/(2 w^{3/2})).  The formula
    therefore survives degenerate metric eigenvalues where the naive
    ``(f_i - f_j) / (w_i - w_j)`` overflows.  Directions dropped by
    ``rcond`` contribute zero derivative.
    """
    w = eig.w[eig.keep]
    u = eig.u[:, eig.keep]
    sq = np.sqrt(w)
    f = -1.0 / (sq[:, None] * sq[None, :] * (sq[:, None] + sq[None, :]))
    dst = u.T @ np.asarray(ds) @ u
    dst = 0.5 * (dst + dst.T)
    out = u @ (f * dst) @ u.T
    return 0.5 * (out + out.T)


def inv_sqrt_metric_jvp(s, ds, rcond=DEFAULT_RCOND, eig=None):
    """Directional derivative of :func:`inv_sqrt_metric` along ``ds``.

    ``ds`` may have a leading Cartesian axis, ``(3, naux, naux)``; the
    result has the same shape.
    """
    eig = metric_eig(s, rcond) if eig is None else eig
    ds = np.asarray(ds)
    if ds.ndim == 2:
        return _inv_sqrt_deriv_from_eig(eig, ds)
    return np.stack([_inv_sqrt_deriv_from_eig(eig, d) for d in ds])


# ---------------------------------------------------------------------------
# Factor container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RIFactors:
    """Host-side RI factors at one geometry.

    Attributes
    ----------
    b : ndarray, (nao, nao, naux)
        Raw three-center integrals.
    s : ndarray, (naux, naux)
        Auxiliary metric.
    jhalf : ndarray, (naux, naux)
        Truncated symmetric inverse square root of ``s``.
    eig : MetricEig
        Eigendecomposition reused by the derivative.
    auxbasis : str
        Basis actually used for the auxiliary molecule.
    rcond : float
        Truncation threshold.
    """
    b: np.ndarray
    s: np.ndarray
    jhalf: np.ndarray
    eig: MetricEig
    auxbasis: str
    rcond: float

    @property
    def nao(self):
        return self.b.shape[0]

    @property
    def naux(self):
        return self.b.shape[2]

    @property
    def cond(self):
        return self.eig.cond


@dataclass(frozen=True)
class RIFactorDerivs:
    """Nuclear derivatives of the RI factors for one atom.

    Attributes
    ----------
    db : ndarray, (3, nao, nao, naux)
        ``d(mu nu | P) / dR_atom`` in Eh / Bohr.
    ds : ndarray, (3, naux, naux)
        ``d(P | Q) / dR_atom``.
    djhalf : ndarray, (3, naux, naux)
        ``d S^{-1/2} / dR_atom``.
    """
    db: np.ndarray
    ds: np.ndarray
    djhalf: np.ndarray


def ri_factors(mol, auxmol, auxbasis=None, rcond=DEFAULT_RCOND):
    """Assemble the host-side RI factors at the current geometry."""
    b = ri_three_center(mol, auxmol)
    s = ri_metric(auxmol)
    eig = metric_eig(s, rcond)
    jhalf = inv_sqrt_metric(s, rcond, eig=eig)
    label = auxbasis if auxbasis is not None else _auxbasis_label(mol, auxmol)
    return RIFactors(b=b, s=s, jhalf=jhalf, eig=eig, auxbasis=label,
                     rcond=rcond)


def _auxbasis_label(mol, auxmol):
    """Best-effort label for a pre-built auxiliary molecule."""
    name = getattr(auxmol, 'basis', None)
    if isinstance(name, str):
        return name
    return 'custom auxiliary molecule'


# ---------------------------------------------------------------------------
# Four-center assembly
# ---------------------------------------------------------------------------
def ri_eri_ao(mol, auxmol, factors=None, rcond=DEFAULT_RCOND):
    """RI approximation to the dense AO ERI, ``(nao, nao, nao, nao)``.

    The contraction is done in two steps, ``w = B S^{-1/2}`` followed by
    ``eri = w w^T`` as a single BLAS GEMM over the pair index.  A
    one-step multi-operand einsum is deliberately avoided: on this stack
    a four-operand einsum produced ~1e-6 errors at 96**4 and an 8%
    gradient discrepancy between algebraically equivalent chains.

    Memory: the output alone is ``8 * nao**4`` bytes (7.5 GB at nao=175).
    """
    f = ri_factors(mol, auxmol, rcond=rcond) if factors is None else factors
    w = np.matmul(f.b, f.jhalf)
    wf = w.reshape(f.nao * f.nao, f.naux)
    eri = wf @ wf.T
    return eri.reshape(f.nao, f.nao, f.nao, f.nao)


def ri_eri_mo(C, b, jhalf=None, full=True):
    """Transform the RI three-center tensor to the MO basis.

    Parameters
    ----------
    C : ndarray, (nao, norb)
        Orbital coefficients; columns outside ``norb`` are ignored.
    b : ndarray, (nao, nao, naux)
        Raw three-center integrals (as returned by ``ri_three_center``).
    jhalf : ndarray, (naux, naux) or None
        Inverse square root of the metric.  Pass ``None`` only if ``b``
        has already been multiplied by it.
    full : bool
        If True also return the four-center MO ERI, built as one GEMM
        over the pair index.  If False return ``None`` in its place, for
        callers that only need the three-center tensor.

    Returns
    -------
    b_mo : ndarray, (norb, norb, naux)
        ``b_mo[i, j, P] = sum_{mu nu} C[mu, i] C[nu, j] B[mu, nu, P]``
        (after the metric factor).
    eri_mo : ndarray, (norb, norb, norb, norb) or None
        Chemist-notation MO ERI ``(i j | k l) = sum_P b_mo[i,j,P]
        b_mo[k,l,P]``.
    """
    c = np.asarray(C)
    nao, norb = c.shape
    if b.shape[0] != nao or b.shape[1] != nao:
        raise ValueError('C and b disagree on nao')
    w = np.asarray(b) if jhalf is None else np.matmul(b, jhalf)
    # Two contractions, one AO index at a time; no multi-operand einsum.
    t = np.matmul(c.T, w.reshape(nao, nao * w.shape[2]))
    t = t.reshape(norb, nao, w.shape[2])                 # t[i, nu, P]
    b_mo = np.tensordot(t, c, axes=([1], [0])).transpose(0, 2, 1)
    if not full:
        return b_mo, None
    bmf = b_mo.reshape(norb * norb, w.shape[2])
    eri_mo = (bmf @ bmf.T).reshape(norb, norb, norb, norb)
    return b_mo, eri_mo


# ---------------------------------------------------------------------------
# Nuclear derivatives
# ---------------------------------------------------------------------------
def atom_ao_mask(mol, atom_idx):
    """Boolean mask of the AO functions sitting on ``atom_idx``."""
    nao = mol.nao_nr()
    mask = np.zeros(nao, dtype=bool)
    loc = mol.ao_loc_nr()
    for ib in range(mol.nbas):
        if int(mol._bas[ib, ATOM_OF]) == int(atom_idx):
            mask[loc[ib]:loc[ib + 1]] = True
    return mask


def ri_three_center_deriv(mol, auxmol, atom_idx):
    """``d(mu nu | P) / dR_atom``, shape ``(3, nao, nao, naux)``.

    Contributions from the two AO centers and the auxiliary center are
    assembled separately; see the module docstring for the verified
    conventions.
    """
    ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1', aosym='s1')
    ip2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2', aosym='s1')
    m_ao = atom_ao_mask(mol, atom_idx)
    m_aux = atom_ao_mask(auxmol, atom_idx)
    first = ip1 * m_ao[:, None, None]
    second = ip1.transpose(0, 2, 1, 3) * m_ao[None, :, None]
    aux = ip2 * m_aux[None, None, :]
    return -(first + second + aux)


def ri_metric_deriv(auxmol, atom_idx):
    """``d(P | Q) / dR_atom``, shape ``(3, naux, naux)``."""
    ip1 = auxmol.intor('int2c2e_ip1', aosym='s1')
    ip2 = auxmol.intor('int2c2e_ip2', aosym='s1')
    m_aux = atom_ao_mask(auxmol, atom_idx)
    out = -(ip1 * m_aux[:, None] + ip2 * m_aux[None, :])
    return 0.5 * (out + out.transpose(0, 2, 1))


def ri_factors_deriv(mol, auxmol, atom_idx, factors):
    """Nuclear derivative of every RI factor for one atom."""
    db = ri_three_center_deriv(mol, auxmol, atom_idx)
    ds = ri_metric_deriv(auxmol, atom_idx)
    dj = inv_sqrt_metric_jvp(factors.s, ds, eig=factors.eig)
    return RIFactorDerivs(db=db, ds=ds, djhalf=dj)


def ri_eri_deriv(mol, auxmol, atom_idx, factors=None, rcond=DEFAULT_RCOND,
                 max_bytes=4.0e9):
    """Materialized ``d(ERI_RI)/dR_atom``, shape ``(3, nao, nao, nao, nao)``.

    This is a validation tool.  Production callers use
    ``ri_factors_deriv`` plus either the factor-gradient recipe
    (``ri_eri_from_factors`` and JAX autodiff) or
    ``ri_eri_deriv_contract``.  The full array is ``24 * nao**4`` bytes;
    the guard refuses to allocate more than ``max_bytes``.
    """
    f = ri_factors(mol, auxmol, rcond=rcond) if factors is None else factors
    nao, naux = f.nao, f.naux
    need = 3 * 8 * nao ** 4
    if need > max_bytes:
        raise MemoryError(
            f'ri_eri_deriv would allocate {need / 1e9:.2f} GB '
            f'(3 x {nao}^4 doubles); use ri_factors_deriv with '
            'ri_eri_deriv_contract instead, or raise max_bytes explicitly')
    d = ri_factors_deriv(mol, auxmol, atom_idx, f)
    w = np.matmul(f.b, f.jhalf)
    wf = w.reshape(nao * nao, naux)
    dbj = np.matmul(d.db, f.jhalf)
    bdj = np.tensordot(f.b, d.djhalf, axes=([2], [1])).transpose(2, 0, 1, 3)
    dw = dbj + bdj
    out = np.empty((3, nao, nao, nao, nao), dtype=w.dtype)
    for c in range(3):
        a = dw[c].reshape(nao * nao, naux) @ wf.T
        a = a.reshape(nao, nao, nao, nao)
        # pair exchange (mu nu) <-> (lam sig)
        out[c] = a + a.transpose(2, 3, 0, 1)
    return out


def ri_eri_deriv_contract(gamma, factors, derivs, per_axis=False,
                          return_terms=False):
    """Contract ``gamma = dE/d(eri_ao)`` with one atom's RI derivative.

    ``gamma`` is the pair-symmetric four-index tensor in chemist notation
    ``(mu nu | lam sig)``; it may be given as ``(nao, nao, nao, nao)`` or
    flattened ``(nao*nao, nao*nao)``.  With ``per_axis=False`` (default)
    the return value is the scalar ``sum(gamma * d(ERI_RI)/dR_atom)``
    (Eh / Bohr).  With ``per_axis=True`` it is the length-3 vector of
    Cartesian components, which is what a nuclear gradient needs.

    Cost: one ``(nao**2, nao**2) @ (nao**2, naux)`` GEMM plus
    ``nao**2 * naux**2`` work.  Memory: ``nao**2 * naux``, never
    ``nao**4`` beyond the caller's own ``gamma``.
    """
    nao, naux = factors.nao, factors.naux
    g = np.asarray(gamma).reshape(nao * nao, nao * nao)
    w = np.matmul(factors.b, factors.jhalf)
    wf = w.reshape(nao * nao, naux)
    z = (g @ wf).reshape(nao, nao, naux)
    zf = z.reshape(nao * nao, naux)
    # dW = dB J + B dJ
    u = z @ factors.jhalf                       # Z J
    bf = factors.b.reshape(nao * nao, naux)
    m = bf.T @ zf                               # M[Q, P] = sum_uv B[uv,Q] Z[uv,P]
    if per_axis:
        term_b = 2.0 * np.sum(u[None, :, :, :] * derivs.db, axis=(1, 2, 3))
        term_j = 2.0 * np.sum(m[None, :, :] * derivs.djhalf, axis=(1, 2))
        total = term_b + term_j
    else:
        term_b = 2.0 * float(np.sum(u * derivs.db))
        term_j = 2.0 * float(np.sum(m * derivs.djhalf))
        total = term_b + term_j
    if return_terms:
        return total, {'b': term_b, 'metric': term_j}
    return total


# ---------------------------------------------------------------------------
# JAX interface
# ---------------------------------------------------------------------------
@jax.custom_jvp
def ri_eri_from_factors(b, jhalf):
    """RI four-center tensor as a pure-JAX function of the factors.

    ``b`` is the raw three-center tensor ``(nao, nao, naux)`` and
    ``jhalf`` the metric inverse square root ``(naux, naux)``.  The JVP
    is the analytic RI derivative with the factor tangents ``db`` and
    ``djhalf`` supplied by :func:`ri_factors_deriv`; no eigh derivative
    and no four-center derivative integral is taken inside JAX.

    The contraction is two-step (``w = b jhalf``, then one GEMM), which
    keeps the arithmetic identical to the numpy path.
    """
    w = jnp.matmul(b, jhalf)
    nao, naux = b.shape[0], b.shape[2]
    wf = jnp.reshape(w, (nao * nao, naux))
    eri = jnp.matmul(wf, jnp.transpose(wf))
    return jnp.reshape(eri, (nao, nao, nao, nao))


@ri_eri_from_factors.defjvp
def _ri_eri_from_factors_jvp(primals, tangents):
    b, jhalf = primals
    db, djhalf = tangents
    w = jnp.matmul(b, jhalf)
    dw = jnp.matmul(db, jhalf) + jnp.matmul(b, djhalf)
    nao, naux = b.shape[0], b.shape[2]
    wf = jnp.reshape(w, (nao * nao, naux))
    dwf = jnp.reshape(dw, (nao * nao, naux))
    eri = jnp.matmul(wf, jnp.transpose(wf))
    deri = jnp.matmul(dwf, jnp.transpose(wf))
    deri = deri + jnp.transpose(deri)   # pair exchange
    shape = (nao, nao, nao, nao)
    return jnp.reshape(eri, shape), jnp.reshape(deri, shape)


def ri_eri_jvp(b, jhalf, db, djhalf):
    """Convenience wrapper: value and directional derivative of the RI ERI."""
    return jax.jvp(ri_eri_from_factors, (b, jhalf), (db, djhalf))
