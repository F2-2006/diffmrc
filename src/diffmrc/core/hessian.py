"""Implicit-function second derivatives of the orbital-rotation energy.

The orbital parameters theta are defined by a stationarity condition,
g(theta*(R), R) = dE/dtheta = 0, so the relaxed energy as a function of
the geometry is E*(R) = E(theta*(R), R). Differentiating the condition
gives the orbital response

    dtheta*/dR = -H_tt^-1 M,     M = d2E/dtheta dR,  H_tt = d2E/dtheta2

and the second derivative of the relaxed energy is

    d2E*/dR2 = H_RR + M^T (-H_tt^-1 M)

with H_RR the Hessian at frozen theta. The second term is the response
contribution; dropping it gives the frozen approximation, which this
project measured to be off by 1e-5 in the frequencies, and which moves
every frequency in the same direction once included.

Why a pseudo-inverse, not a solve
---------------------------------

The orbital Hessian is singular along directions that leave the energy
invariant -- rotations within a block, for instance. For N2 it reaches a
condition number of 2.6e15. A direct solve inverts that near-null
direction and amplifies whatever the right-hand side has there; the
pseudo-inverse drops it, which is correct because the gradient component
along an exact zero-curvature direction is zero and contributes nothing
to dtheta*/dR.

The cutoff is relative (``rcond``), matching ``numpy.linalg.pinv``. An
earlier script in this project used an absolute singular-value threshold
of 1e-3; on a matrix with condition number 1e12 the two retain five and
two singular values respectively and produce solutions differing by
99.9%. Whether that difference reaches a physical quantity depends on
the component of M along the discarded directions, which
``experiments/verify_rcond_choice.py`` measures on a real system rather
than assuming.

Conventions that matter
-----------------------

* ``theta`` must be ``stop_gradient``-ed in the frozen Hessian, or the
  frozen term silently acquires the orbital response and the two routes
  stop being distinguishable.
* Every Hessian is symmetrized as ``0.5*(H + H.T)``: an AD Hessian is
  symmetric only to round-off, and both ``eigvalsh`` and ``pinv`` assume
  exact symmetry.
* ``M`` is deliberately left unsymmetrized. Symmetry is a property of
  the assembled ``H_total``, not of ``M``.
* The SCF columns are phase-canonicalized before the rotation
  (``core.gauge.canonical_mo_phases``).  ``theta`` is measured from
  those columns; without the convention a frozen theta would describe a
  different physical rotation at every geometry whose eigensolver chose
  a different column sign, and the frozen and mixed terms would be
  assembled in different charts.
* The ``.coords`` attribute of a differentiated Mole is nested
  differently depending on whether the outermost operation is ``grad``
  or ``jacfwd``/``hessian`` of a ``grad``. Taking the wrong level is a
  silent shape error, so the two helpers below are kept separate.
"""
from dataclasses import dataclass

import numpy as np
import jax

from .adint import int2e_s1
from .gauge import canonical_mo_phases
from .orblib import orbital_pairs  # noqa: F401  (re-exported for callers)
from .rotation import block_rotation

__all__ = ['ImplicitHessianResult', 'ImplicitHessianHVPResult',
           'ImplicitHessianHVPOps', 'make_joint_energy_fn',
           'make_theta_energy_fn', 'implicit_hessian',
           'implicit_hessian_columnwise', 'implicit_hessian_hvp',
           'mole_coordinate_tangent']

DEFAULT_RCOND = 1e-9


@dataclass
class ImplicitHessianResult:
    """The assembled second derivative and the diagnostics behind it."""

    H_tt: np.ndarray
    H_tt_inverse: np.ndarray
    curvature: object
    mixed: np.ndarray
    H_RR: np.ndarray
    H_response: np.ndarray
    H_total: np.ndarray
    dtheta_dR_max: float
    rcond: float


@dataclass
class ImplicitHessianHVPResult:
    """A relaxed Hessian assembled without ever forming H_tt, M or H_RR.

    ``H_total`` is the only dense object (``3 natm`` squared).  The
    response solve is a matrix-free least-squares solution of
    ``H_tt x = M v``; ``lsmr_*`` records how each column converged, so a
    reader can judge the result instead of trusting the solver.

    ``deflation`` carries the report of the near-null basis the solve
    used: how many directions were deflated, whether the basis was
    certified, and how many operator products it cost.  ``None`` means
    the solve ran on the full operator.
    """

    H_total: np.ndarray
    H_tt_diagonal: np.ndarray
    hvp: object
    rcond: float
    n_par: int
    n_geom: int
    lsmr_iterations: np.ndarray
    lsmr_istop: np.ndarray
    lsmr_residual: np.ndarray
    maxiter: object
    atol: float
    btol: float
    dtheta_dR_max: float
    deflation: object = None
    solver_reason: object = None
    total_iterations: int = 0
    relative_residual: object = None
    total_residual: object = None


def make_joint_energy_fn(ncore, nact, nele, nmo, pairs, pscf,
                         use_blocked=True, scf=None):
    """Build E(theta, mol), differentiable in both arguments.

    The reference orbitals and the integrals are recomputed inside the
    function from the Mole. Holding them fixed across a displaced
    geometry represents a different set of orbitals; this project
    recorded the resulting error as 4e-3 Eh/bohr for the mildest case.

    Parameters
    ----------
    ncore, nact, nele : int
        Active-space definition.
    nmo : int
        Number of basis functions.
    pairs : list of (int, int)
        Orbital rotation pairs. Must include core<->virtual.
    pscf : module
        ``pyscfad.scf``, injected to avoid a hard dependency here.
    scf : callable or None
        Mean-field constructor ``mol -> mf``; ``None`` means
        ``pscf.RHF``.  Pass the same constructor to
        :func:`make_theta_energy_fn` when the two surfaces must agree
        to tighter than the default SCF tolerance.

    Returns
    -------
    callable
        ``(mol, theta) -> energy``.
    """
    from .casci import cs_casci_blocked

    scf_of = pscf.RHF if scf is None else scf

    def E_joint(m_, th_):
        mf_ = scf_of(m_)
        mf_.kernel()
        # theta is measured from the SCF columns.  Canonicalize them
        # before the rotation so that a frozen theta describes the same
        # physical rotation at every displaced geometry, whatever sign
        # the eigensolver returned there.
        C_ = canonical_mo_phases(mf_.mo_coeff)
        h_ = m_.intor('int1e_kin') + m_.intor('int1e_nuc')
        e_ = int2e_s1(m_)
        C2 = C_ @ block_rotation(th_, nmo, pairs)
        nm = ncore + nact
        h1 = jax.numpy.einsum('ai,ab,bj->ij', C2, h_, C2)[:nm, :nm]
        e1 = jax.numpy.einsum('pi,qj,rk,sl,pqrs->ijkl', C2, C2, C2, C2,
                              e_)[:nm, :nm, :nm, :nm]
        return cs_casci_blocked(h1, e1, ncore=ncore, nact=nact,
                                nele_act=nele)[0] + m_.energy_nuc()

    return E_joint


def make_theta_energy_fn(mol, ncore, nact, nele, nmo, pairs, pscf,
                         scf=None):
    """Theta-only energy at one geometry, with the SCF outside the graph.

    The orbital Hessian ``H_tt`` is a derivative at fixed nuclear
    geometry, so its reference determinant and AO integrals are
    constants of that derivative.  Rebuilding the SCF inside every
    tangent multiplies the cost of the blocked and matrix-free routes
    by the SCF cost without changing any number.  This function
    evaluates the SCF once and returns ``theta -> energy``.

    Use the same ``scf`` constructor as :func:`make_joint_energy_fn`;
    with different SCF tolerances the two surfaces differ at that
    tolerance, which shows up in ``H_tt`` at the same order.

    Parameters
    ----------
    mol : pyscfad Mole
        The fixed reference geometry.
    ncore, nact, nele, nmo, pairs :
        Active-space definition, as in :func:`make_joint_energy_fn`.
    pscf : module
        ``pyscfad.scf``.
    scf : callable or None
        Mean-field constructor ``mol -> mf``.

    Returns
    -------
    callable
        ``theta -> energy``.
    """
    from .casci import cs_casci_blocked

    scf_of = pscf.RHF if scf is None else scf
    mf_ = scf_of(mol)
    mf_.kernel()
    # Same convention as `make_joint_energy_fn`: the theta chart is
    # measured from the canonical columns, not from the eigensolver's
    # arbitrary sign.
    C_ = canonical_mo_phases(mf_.mo_coeff)
    h_ = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    e_ = mol.intor('int2e', aosym='s1')
    en_ = mol.energy_nuc()
    nm = ncore + nact

    def E_theta(th_):
        C2 = C_ @ block_rotation(th_, nmo, pairs)
        h1 = jax.numpy.einsum('ai,ab,bj->ij', C2, h_, C2)[:nm, :nm]
        e1 = jax.numpy.einsum('pi,qj,rk,sl,pqrs->ijkl', C2, C2, C2, C2,
                              e_)[:nm, :nm, :nm, :nm]
        return cs_casci_blocked(h1, e1, ncore=ncore, nact=nact,
                                nele_act=nele)[0] + en_

    return E_theta


def _mixed_derivative(E_joint, mol, theta):
    """M = d2E/dtheta dR, taken as jacfwd of the theta-gradient."""
    g_of = jax.grad(E_joint, argnums=1)
    M = jax.jacfwd(g_of, argnums=0)(mol, theta)
    n_par = int(np.asarray(theta).size)
    return np.asarray(M.coords).reshape(n_par, -1)


def _frozen_hessian(E_joint, mol, theta, n_par):
    """H_RR = d2E/dR2 with theta held fixed."""
    th_sg = jax.lax.stop_gradient(theta)

    def E_frozen(m_):
        return E_joint(m_, th_sg)

    H = jax.jacfwd(jax.grad(E_frozen))(mol)
    return np.asarray(H.coords.coords).reshape(n_par, n_par)


def implicit_hessian(mol, theta, E_joint, rcond=DEFAULT_RCOND):
    """Assemble the relaxed-geometry Hessian at a stationary theta.

    Parameters
    ----------
    mol : pyscfad Mole
        Reference geometry.
    theta : array_like
        Orbital parameters at the stationary point for that geometry.
    E_joint : callable
        ``(mol, theta) -> energy``, as built by `make_joint_energy_fn`.
    rcond : float
        Relative cutoff for the pseudo-inverse of the orbital Hessian.

    Returns
    -------
    ImplicitHessianResult
        The total Hessian plus every intermediate, so a caller can see
        how much of it came from the orbital response rather than only
        the total.
    """
    from .stationary import curvature_report

    theta = jax.numpy.asarray(np.asarray(theta, dtype=float))
    n_par = int(np.asarray(theta).size)

    H_tt = np.asarray(jax.hessian(lambda t: E_joint(mol, t))(theta))
    H_tt = 0.5 * (H_tt + H_tt.T)

    M = _mixed_derivative(E_joint, mol, theta)
    n_r = M.shape[1]

    H_RR = _frozen_hessian(E_joint, mol, theta, n_r)
    H_RR = 0.5 * (H_RR + H_RR.T)

    H_tt_inv = np.linalg.pinv(H_tt, rcond=rcond)
    D = -H_tt_inv @ M
    H_resp = M.T @ D
    H_total = H_RR + H_resp
    H_total = 0.5 * (H_total + H_total.T)

    return ImplicitHessianResult(
        H_tt=H_tt, H_tt_inverse=H_tt_inv,
        curvature=curvature_report(H_tt),
        mixed=M, H_RR=H_RR, H_response=H_resp, H_total=H_total,
        dtheta_dR_max=float(np.abs(D).max()) if D.size else 0.0,
        rcond=float(rcond),
    )


# ---------------------------------------------------------------------------
# Second-derivative cost: blocked assembly and matrix-free products
# ---------------------------------------------------------------------------
def mole_coordinate_tangent(mol, v):
    """Pytree tangent whose coordinate leaf is ``v``.

    A pyscfad ``Mole`` is a pytree with several differentiable leaves
    (``coords``, ``exp``, ``ctr_coeff``, ``r0``).  A geometry direction
    touches only the coordinate leaf, so every other leaf is zeroed.
    The leaf is located by shape and value rather than by position,
    because the leaf order is a property of the pyscfad build rather
    than of this package.
    """
    from jax import numpy as jnp
    from jax import tree_util

    v_arr = jnp.asarray(np.asarray(v, dtype=float).reshape(-1, 3))
    coords_ref = np.asarray(mol.coords, dtype=float)
    if v_arr.shape != coords_ref.shape:
        raise ValueError(
            f'tangent shape {v_arr.shape} disagrees with coordinates '
            f'{coords_ref.shape}')
    leaves, treedef = tree_util.tree_flatten(mol)
    idx = [i for i, leaf in enumerate(leaves)
           if hasattr(leaf, 'shape')
           and tuple(np.shape(leaf)) == coords_ref.shape
           and np.allclose(np.asarray(leaf), coords_ref)]
    if len(idx) != 1:
        raise ValueError(
            'cannot locate the coordinate leaf of the Mole pytree: '
            f'{len(idx)} candidates')
    tan_leaves = [jnp.zeros_like(leaf) for leaf in leaves]
    tan_leaves[idx[0]] = v_arr
    return tree_util.tree_unflatten(treedef, tan_leaves)


def _coordinate_size(mol):
    """Number of geometric degrees of freedom carried by ``mol``."""
    coords = getattr(mol, 'coords', None)
    if coords is not None:
        return int(np.asarray(coords).size)
    return int(np.asarray(mol.atom_coords()).size)


def _tangent_coords(tangent):
    """Flatten the coordinate leaf of a tangent pytree."""
    return np.asarray(tangent.coords).reshape(-1)


def _tangent_of(mol, v, tangent_fn):
    if tangent_fn is None:
        return mole_coordinate_tangent(mol, v)
    return tangent_fn(mol, v)


def implicit_hessian_columnwise(mol, theta, E_joint, rcond=DEFAULT_RCOND,
                                tangent_fn=None, E_tt=None):
    """Assemble the implicit Hessian one column at a time.

    ``jax.hessian`` and ``jacfwd`` broadcast every intermediate over all
    tangent directions at once.  At ``n_par`` around 500 that broadcast
    is the 402 GB allocation recorded in ``driver.orbitals.hessp_of``;
    the resulting matrix is small, the working set is not.  This route
    touches one tangent per ``jax.jvp``, so the peak working set is one
    column's intermediates and the assembled matrices themselves.  The
    answer is the same object as :func:`implicit_hessian`; only the
    traversal order differs.

    ``tangent_fn`` exists for the surrogate tests: it turns a flat
    direction into a pytree tangent for the object standing in for the
    Mole.  The default is :func:`mole_coordinate_tangent`.

    ``E_tt`` is the theta-only energy at the reference geometry (see
    :func:`make_theta_energy_fn`).  When given, the ``H_tt`` columns
    are taken from it, which removes the SCF from the differentiated
    graph; the mixed and frozen blocks still use ``E_joint`` because
    they differentiate the reference orbitals with respect to geometry.
    """
    from .stationary import curvature_report

    theta = jax.numpy.asarray(np.asarray(theta, dtype=float))
    n_par = int(np.asarray(theta).size)
    n_r = _coordinate_size(mol)

    g_theta = jax.grad(E_joint, argnums=1)
    g_mixed = lambda m: g_theta(m, theta)
    if E_tt is None:
        g_tt = lambda t: g_theta(mol, t)
    else:
        g_tt = jax.grad(E_tt)
    th_sg = jax.lax.stop_gradient(theta)
    g_frozen = jax.grad(lambda m: E_joint(m, th_sg))

    H_tt = np.zeros((n_par, n_par))
    for j in range(n_par):
        e = jax.numpy.zeros((n_par,)).at[j].set(1.0)
        H_tt[:, j] = np.asarray(
            jax.jvp(g_tt, (theta,), (e,))[1]).reshape(-1)
    H_tt = 0.5 * (H_tt + H_tt.T)

    M = np.zeros((n_par, n_r))
    H_RR = np.zeros((n_r, n_r))
    for j in range(n_r):
        v = np.zeros(n_r)
        v[j] = 1.0
        tan = _tangent_of(mol, v, tangent_fn)
        M[:, j] = np.asarray(
            jax.jvp(g_mixed, (mol,), (tan,))[1]).reshape(-1)
        H_RR[:, j] = _tangent_coords(
            jax.jvp(g_frozen, (mol,), (tan,))[1])
    H_RR = 0.5 * (H_RR + H_RR.T)

    H_tt_inv = np.linalg.pinv(H_tt, rcond=rcond)
    D = -H_tt_inv @ M
    H_resp = M.T @ D
    H_total = H_RR + H_resp
    H_total = 0.5 * (H_total + H_total.T)

    return ImplicitHessianResult(
        H_tt=H_tt, H_tt_inverse=H_tt_inv,
        curvature=curvature_report(H_tt),
        mixed=M, H_RR=H_RR, H_response=H_resp, H_total=H_total,
        dtheta_dR_max=float(np.abs(D).max()) if D.size else 0.0,
        rcond=float(rcond),
    )


@dataclass
class ImplicitHessianHVPOps:
    """Matrix-free operators behind the implicit Hessian.

    Every block is a Hessian of the same joint energy, so each is
    reached through one forward-over-reverse product per application:

        H_tt u  = jvp_theta (grad_theta E) u
        M v     = jvp_R     (grad_theta E) v
        M^T x   = vjp_R     (grad_theta E) x
        H_RR v  = jvp_R     (grad_R E)     v

    The response solve uses LSMR on ``H_tt``.  LSMR returns the
    minimum-norm least-squares solution for a singular or indefinite
    symmetric operator, which is the same role the relative-cutoff
    pseudo-inverse plays in the dense route.  Nothing of size
    ``n_par**2`` or ``n_par * n_geom`` is formed.

    The solve has no iteration cap.  It stops on the relative residual
    (``rtol``) or on the normal-equation criterion (``atol``), with a
    wall-clock guard and a stagnation guard so that a hopeless system
    terminates.  ``maxiter`` remains available for a caller who wants a
    hard budget, but it is no longer the control.

    ``deflate`` builds a near-null basis once (see
    :func:`core.lsmr.near_null_basis`) and hands the small-eigenvalue
    block to an exact pseudo-inverse action, so LSMR only sees the
    complement.  The basis is used only when it certifies; otherwise
    the solve falls back to the full operator.
    """

    E_joint: object
    mol: object
    theta: object
    rcond: float = DEFAULT_RCOND
    tangent_fn: object = None
    E_tt: object = None
    atol: float = 1e-12
    btol: float = 1e-12
    maxiter: int = None
    deflate: bool = True
    deflate_cond_target: float = 100.0
    deflate_tol_rel: float = 1e-8
    deflate_max_size: int = None
    deflate_max_rounds: int = 30
    stall_iters: int = 40
    time_limit: float = None
    fallback: bool = True

    def __post_init__(self):
        self.theta = jax.numpy.asarray(
            np.asarray(self.theta, dtype=float))
        self._g_theta = jax.grad(self.E_joint, argnums=1)
        if self.E_tt is None:
            self._g_tt = jax.grad(
                lambda t: self.E_joint(self.mol, t))
            self._htt_jit = None
        else:
            self._g_tt = jax.grad(self.E_tt)
            # E_tt has no SCF in its graph, so the H_tt product can be
            # jitted.  The LSMR loop calls it once per iteration; without
            # the compiled executable every call re-traces the CASCI
            # graph, which dominated the measured wall time and let the
            # working set grow across columns.
            self._htt_jit = jax.jit(
                lambda u: jax.jvp(self._g_tt, (self.theta,), (u,))[1])
        self._basis = None
        self._ritz = None
        self._diag = None
        self.deflation_info = None

    @property
    def n_par(self):
        return int(np.asarray(self.theta).size)

    @property
    def n_geom(self):
        return _coordinate_size(self.mol)

    def h_tt(self, u):
        u = jax.numpy.asarray(u, dtype=float)
        if self._htt_jit is not None:
            return np.asarray(self._htt_jit(u)).reshape(-1)
        return np.asarray(
            jax.jvp(self._g_tt, (self.theta,), (u,))[1]).reshape(-1)

    def mixed(self, v):
        g = lambda m: self._g_theta(m, self.theta)
        tan = _tangent_of(self.mol, np.asarray(v, dtype=float),
                          self.tangent_fn)
        return np.asarray(jax.jvp(g, (self.mol,), (tan,))[1]).reshape(-1)

    def mixed_transpose(self, x):
        g = lambda m: self._g_theta(m, self.theta)
        _, pullback = jax.vjp(g, self.mol)
        cot = pullback(jax.numpy.asarray(x, dtype=float))[0]
        return _tangent_coords(cot)

    def frozen(self, v):
        th_sg = jax.lax.stop_gradient(self.theta)
        g = jax.grad(lambda m: self.E_joint(m, th_sg))
        tan = _tangent_of(self.mol, np.asarray(v, dtype=float),
                          self.tangent_fn)
        return _tangent_coords(jax.jvp(g, (self.mol,), (tan,))[1])

    def prepare_deflation(self, diag=None):
        """Build (once) the near-null basis used by every column solve.

        The basis costs ``n_par`` products for the diagonal when it is
        not supplied, plus the greedy expansion; it is shared by all
        ``n_geom`` columns, so its cost does not grow with the number
        of displacements.
        """
        from .lsmr import near_null_basis

        if not self.deflate:
            self.deflation_info = dict(certified=False, reason='disabled',
                                       n_wanted=0, block_size=0,
                                       matvecs=0)
            return None
        if diag is None:
            diag = self.h_tt_diagonal()
        self._diag = np.asarray(diag, dtype=float)
        basis, ritz, info = near_null_basis(
            self.h_tt, self.n_par, diag=self._diag,
            cond_target=self.deflate_cond_target, tol_rel=self.deflate_tol_rel,
            max_size=self.deflate_max_size, max_rounds=self.deflate_max_rounds)
        self.deflation_info = info
        if info.get('certified'):
            self._basis, self._ritz = basis, ritz
        else:
            self._basis, self._ritz = None, None
        return info

    def solve(self, b):
        """Minimum-norm least-squares action of ``pinv(H_tt)`` on ``b``."""
        from .lsmr import deflated_solve, lsmr_solve

        b = np.asarray(b, dtype=float).reshape(-1)
        if self._basis is not None:
            x, info = deflated_solve(
                self.h_tt, b, self._basis, self._ritz, rcond=self.rcond,
                rtol=float(self.btol), atol=float(self.atol),
                maxiter=self.maxiter, time_limit=self.time_limit,
                stall_iters=self.stall_iters, fallback=self.fallback,
                lambda_max=(self.deflation_info or {}).get('lambda_max'))
            return x, info
        x, info = lsmr_solve(
            self.h_tt, b, rtol=float(self.btol), atol=float(self.atol),
            maxiter=self.maxiter, time_limit=self.time_limit,
            stall_iters=self.stall_iters)
        return x, info

    def matvec(self, v):
        """One relaxed Hessian-vector product ``H_total @ v``.

        Uses whatever basis ``prepare_deflation`` installed; a single
        product does not build one, because the basis costs ``n_par``
        products and only pays off across the column loop.
        """
        x, _ = self.solve(self.mixed(v))
        return self.frozen(v) - self.mixed_transpose(x)

    def dense_columns(self):
        """Assemble the dense Hessian by columns, keeping the solves."""
        n_r = self.n_geom
        H = np.zeros((n_r, n_r))
        X = np.zeros((self.n_par, n_r))
        iters = np.zeros(n_r, dtype=int)      # LSMR iterations per column
        istop = np.zeros(n_r, dtype=int)      # 1 residual, 2 normal eqs, ...
        normr = np.zeros(n_r)                 # relative residual (projected)
        infos = []
        for j in range(n_r):
            v = np.zeros(n_r)
            v[j] = 1.0
            b = self.mixed(v)
            x, info = self.solve(b)
            X[:, j] = x
            iters[j] = int(info.get('itn', info.get('iterations', 0)))
            istop[j] = int(info.get('istop', 0))
            normr[j] = info.get('relative_residual', info.get('normr', np.nan))
            infos.append(info)
            H[:, j] = self.frozen(v) - self.mixed_transpose(x)
        self.column_info = infos
        return H, X, iters, istop, normr

    def h_tt_diagonal(self):
        if self._diag is not None:
            return np.array(self._diag)
        diag = np.zeros(self.n_par)
        for j in range(self.n_par):
            e = np.zeros(self.n_par)
            e[j] = 1.0
            diag[j] = self.h_tt(e)[j]
        return diag


def implicit_hessian_hvp(mol, theta, E_joint, rcond=DEFAULT_RCOND,
                         tangent_fn=None, E_tt=None, atol=1e-12,
                         btol=1e-12, maxiter=None, with_ht_diagonal=True,
                         deflate=True, deflate_cond_target=100.0,
                         deflate_tol_rel=1e-8, deflate_max_size=None,
                         deflate_max_rounds=30, stall_iters=40,
                         time_limit=None, fallback=True):
    """Relaxed Hessian assembled without forming any large matrix.

    The three blocks of the implicit route are used only through
    Hessian-vector products; the response equation is solved with LSMR.
    The only dense object is the ``(3 natm)**2`` result.  Memory
    therefore scales as ``n_par`` and ``3 natm`` vectors rather than as
    their outer products, and the ``n_par``-tangent broadcast of
    ``jax.hessian`` never happens.

    Cost is the trade: one LSMR solve per geometric column, each
    iteration one ``H_tt`` product.  The ``lsmr_*`` fields record the
    iteration count and residual of every column.

    Parameters
    ----------
    mol, theta, E_joint :
        As in :func:`implicit_hessian`.
    rcond : float
        Recorded for comparison with the dense pseudo-inverse cutoff.
        LSMR does not use it directly; the least-squares solution drops
        only the exactly null directions.
    E_tt : callable or None
        Theta-only energy at the reference geometry.  When given, every
        ``H_tt`` product is taken on this fixed-reference surface, so
        the SCF leaves the inner loop; the mixed and frozen blocks
        still use ``E_joint``.  Build it with
        :func:`make_theta_energy_fn` and the same SCF constructor as
        ``E_joint``.  When ``None``, every product goes through
        ``E_joint`` and the SCF is rebuilt per iteration.
    atol, btol : float
        LSMR stopping tolerances: ``btol`` bounds ``||r|| / ||b||``,
        ``atol`` bounds the normal-equation residual, which is the
        criterion that fires on a singular operator whose right-hand
        side has a null-space component.
    maxiter : int or None
        Optional hard iteration cap per column.  ``None`` (the default)
        means no cap; the solve stops on the tolerance, on the
        wall-clock guard or on the stagnation guard.  The cap that used
        to be ``8 * n_par`` is gone.
    with_ht_diagonal : bool
        Record ``diag(H_tt)`` for a curvature sanity check.  This costs
        ``n_par`` products, and the same array seeds the deflation
        basis, so it is not wasted when ``deflate`` is on.
    deflate : bool
        Build a near-null basis once and solve the small-|lambda| block
        exactly, leaving LSMR the complement.  A basis that does not
        certify is not used.
    deflate_cond_target : float
        Deflate until the complement has condition number about this.
    deflate_tol_rel : float
        Certification threshold on the Ritz residual, relative to
        ``lambda_max``.  Measured on the two test systems: 1e-4 leaves
        the complement as ill-conditioned as the original operator (the
        CO Hessian came out wrong by 6e4), 1e-6 is already exact; the
        default 1e-8 keeps a margin.
    deflate_max_size, deflate_max_rounds : int or None
        Caps on the basis block and its growth rounds.
    stall_iters : int
        Stagnation guard passed to the solver.
    time_limit : float or None
        Wall-clock guard per column, in seconds.
    fallback : bool
        Re-solve with plain LSMR when the deflated route has not
        converged, so the answer is the unaccelerated one rather than a
        wrong one.
    """
    ops = ImplicitHessianHVPOps(
        E_joint=E_joint, mol=mol, theta=theta, rcond=rcond,
        tangent_fn=tangent_fn, E_tt=E_tt, atol=atol, btol=btol,
        maxiter=maxiter, deflate=deflate,
        deflate_cond_target=deflate_cond_target,
        deflate_tol_rel=deflate_tol_rel, deflate_max_size=deflate_max_size,
        deflate_max_rounds=deflate_max_rounds, stall_iters=stall_iters,
        time_limit=time_limit, fallback=fallback)
    # The diagonal seeds the deflation basis; when neither the curvature
    # report nor deflation needs it, the n_par products are not spent.
    if deflate or with_ht_diagonal:
        diag = ops.h_tt_diagonal()
    else:
        diag = None
    ops.prepare_deflation(diag=diag)
    H, X, iters, istop, normr = ops.dense_columns()
    H = 0.5 * (H + H.T)
    if not with_ht_diagonal:
        diag = np.zeros(ops.n_par)
    reasons = [info.get('reason') for info in getattr(ops, 'column_info', [])]
    total_res = [info.get('total_residual') for info in
                 getattr(ops, 'column_info', [])]
    return ImplicitHessianHVPResult(
        H_total=H, H_tt_diagonal=diag, hvp=ops.matvec, rcond=float(rcond),
        n_par=ops.n_par, n_geom=ops.n_geom,
        lsmr_iterations=iters, lsmr_istop=istop, lsmr_residual=normr,
        maxiter=ops.maxiter, atol=float(ops.atol),
        btol=float(ops.btol),
        dtheta_dR_max=float(np.abs(X).max()) if X.size else 0.0,
        deflation=ops.deflation_info,
        solver_reason=reasons,
        total_iterations=int(np.sum(iters)),
        relative_residual=normr.copy(),
        total_residual=np.asarray(
            [v if v is not None else np.nan for v in total_res]),
    )
