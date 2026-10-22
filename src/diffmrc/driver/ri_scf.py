"""RI-J Coulomb route for the SCF stage.

WHY THIS MODULE EXISTS

``core.ri`` supplies the three-center factors, the auxiliary metric and
their nuclear derivatives.  The multireference chain already consumes
them for the NEVPT2 integrals, but the reference SCF still forms the
four-center Coulomb tensor through pyscfad's ``int2e`` path, whose
geometry derivative allocates ``nctr * nao**4`` doubles in one block
(55 GB at nao=96, 2033 GB extrapolated at nao=175).  This module adds
the missing SCF-stage piece: the Coulomb (J) term evaluated from the RI
factors, with an analytic fixed-coefficient nuclear gradient.

ENERGY EXPRESSIONS (closed shell, chemist notation)

    D_{mu nu}          = sum_i occ_i C_{mu i} C_{nu i}   (occ = 2 default)
    E_J                = 1/2 sum D_{mu nu} D_{lam sig} (mu nu | lam sig)_RI
                       = 1/2 sum_P z_P^2,
      z_P              = sum_{mu nu} D_{mu nu} W_{mu nu P},
      W                = B S^{-1/2},   B_{mu nu P} = (mu nu | P),
      S_{P Q}          = (P | Q).
    E_x (exact)        = -1/4 sum D_{mu nu} D_{lam sig} (mu lam | nu sig)
    E_1e               = sum_{mu nu} D_{mu nu} h_{mu nu}
    E_nuc              = nuclear repulsion.

``rijonx_energy`` returns ``E_nuc + E_1e + E_J(RI) + E_x(exact)``.  The
combination is the RIJONX route: the Coulomb term is fitted, the
exchange term is not.  It is not RI-JK and it is not RIJCOSX.

WHAT THE GRADIENTS MEAN

Every gradient here is a *fixed-coefficient* (skeleton) gradient: the
MO coefficients are held fixed at their reference values and only the
integrals and the nuclear repulsion move.  At a converged SCF density
the orbital-response term vanishes, so the skeleton gradient equals the
full SCF gradient to within the SCF convergence threshold; away from
stationarity the difference is the response term and this module does
not supply it.  This is stated in every public gradient docstring
because a skeleton gradient reported as an SCF gradient would be a
silent error of the same class as an unconverged reference.

EXCHANGE HANDLING

``exact_k_fock`` / ``exact_k_gradient`` use the dense four-center
integrals.  RI-J removes the J part of the four-center work and of the
``_fill_grad0`` allocation, but the exact-K part still needs the dense
tensor and its derivative.  Removing the K part requires RI-JK (no
analytic gradient is offered by ORCA either) or RIJCOSX; neither is
implemented here.  ``cosx_exchange_matrix`` is a fixed-grid prototype,
not a production COSX.

MEMORY

``ri_j_energy`` / ``ri_j_gradient`` never form a four-center array: the
working set is ``nao**2 * naux``.  The exact-K functions are
validation-scale and guarded by ``max_bytes``; the gradient assembles a
``3 * nao**4`` derivative array and a ``nao**4`` density outer product.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import df, gto
from pyscf.gto.mole import ATOM_OF

from ..core import ri as dri

__all__ = [
    'RI_SCF_DEFAULT_AUX',
    'density_from_coeffs',
    'ri_j_fock',
    'ri_j_energy',
    'ri_j_energy_from_density',
    'ri_j_gradient',
    'exact_k_fock',
    'exact_k_energy',
    'exact_k_gradient',
    'core_hamiltonian_energy',
    'core_hamiltonian_gradient',
    'nuclear_repulsion_gradient',
    'rijonx_energy',
    'rijonx_gradient',
    'ri_j_hessian_fd',
    'CosxGrid',
    'make_cosx_grid',
    'cosx_exchange_matrix',
    'cosx_exchange_energy',
]

#: Auxiliary basis used when the caller does not name one.  It is passed
#: to ``core.ri.make_auxmol``, which falls back to the orbital basis for
#: elements the fitting set does not cover.
RI_SCF_DEFAULT_AUX = 'def2-universal-jkfit'

#: Default memory guard for the validation-scale exact-K path.
DEFAULT_MAX_BYTES = 4.0e9


# ---------------------------------------------------------------------------
# Density and auxiliary-molecule helpers
# ---------------------------------------------------------------------------
def density_from_coeffs(c, occ=None):
    """Closed-shell AO density ``D = C diag(occ) C^T``.

    Parameters
    ----------
    c : array_like, (nao, nocc)
        Occupied orbital coefficients.  Columns are the occupied MOs.
    occ : array_like, (nocc,) or None
        Occupation numbers.  ``None`` means two electrons per column
        (restricted closed shell).

    Returns
    -------
    ndarray, (nao, nao)
        Symmetric density matrix, trace = sum(occ).
    """
    c = np.asarray(c, dtype=float)
    if c.ndim != 2:
        raise ValueError(f'c must be two-dimensional, got shape {c.shape}')
    if occ is None:
        occ = np.full(c.shape[1], 2.0)
    else:
        occ = np.asarray(occ, dtype=float).reshape(-1)
        if occ.size != c.shape[1]:
            raise ValueError(
                f'occ has {occ.size} entries for {c.shape[1]} columns')
    d = (c * occ[None, :]) @ c.T
    return 0.5 * (d + d.T)


def _as_auxmol(mol, aux):
    """Accept an auxiliary Mole, a basis name, or ``None``."""
    if aux is None:
        return dri.make_auxmol(mol, auxbasis=RI_SCF_DEFAULT_AUX)[0]
    if isinstance(aux, str):
        return dri.make_auxmol(mol, auxbasis=aux)[0]
    return aux


def _guard(need_bytes, limit, what):
    if limit is not None and need_bytes > limit:
        raise MemoryError(
            f'{what} would allocate about {need_bytes / 1e9:.2f} GB; '
            f'raise max_bytes explicitly or use a smaller system')


def _sym(a):
    return 0.5 * (a + a.T)


# ---------------------------------------------------------------------------
# RI-J Coulomb
# ---------------------------------------------------------------------------
def ri_j_fock(mol, dm, aux=None, factors=None, rcond=dri.DEFAULT_RCOND):
    """RI-J Coulomb matrix ``J_{mu nu}`` for an AO density ``dm``.

    ``J = W diag(z) W^T`` with ``z_P = sum_{mu nu} dm_{mu nu} W_{mu nu P}``
    and ``W = B S^{-1/2}``.  No four-center array is formed.
    """
    dm = np.asarray(dm, dtype=float)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError(f'dm must be square, got shape {dm.shape}')
    if factors is None:
        factors = dri.ri_factors(mol, _as_auxmol(mol, aux), rcond=rcond)
    if dm.shape != (factors.nao, factors.nao):
        raise ValueError(
            f'dm shape {dm.shape} disagrees with nao={factors.nao}')
    w = np.matmul(factors.b, factors.jhalf)
    nao, naux = factors.nao, factors.naux
    wf = w.reshape(nao * nao, naux)
    z = wf.T @ dm.reshape(nao * nao)
    return _sym((wf @ z).reshape(nao, nao))


def ri_j_energy_from_density(mol, dm, aux=None, factors=None,
                             rcond=dri.DEFAULT_RCOND):
    """``E_J = 1/2 Tr(dm J)`` on the RI Coulomb energy surface."""
    dm = np.asarray(dm, dtype=float)
    j = ri_j_fock(mol, dm, aux=aux, factors=factors, rcond=rcond)
    return 0.5 * float(np.sum(dm * j))


def ri_j_energy(mol, C, aux=None, occ=None, factors=None,
                rcond=dri.DEFAULT_RCOND):
    """RI-J Coulomb energy for the determinant built from ``C``.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        AO molecule.
    C : array_like, (nao, nocc)
        Occupied MO coefficients (a full square MO matrix is accepted;
        only the columns with nonzero occupation contribute).
    aux : Mole, str, or None
        Auxiliary molecule, auxiliary basis name, or ``None`` for
        ``RI_SCF_DEFAULT_AUX``.
    occ : array_like or None
        Occupation numbers; ``None`` means 2.0 per column.
    """
    dm = density_from_coeffs(C, occ)
    return ri_j_energy_from_density(mol, dm, aux=aux, factors=factors,
                                    rcond=rcond)


def ri_j_gradient(mol, C, aux=None, occ=None, factors=None,
                  rcond=dri.DEFAULT_RCOND, atoms=None):
    """Fixed-coefficient nuclear gradient of the RI-J energy.

    The derivative of

        E_J = 1/2 sum_P z_P^2,   z_P = sum_{mu nu} D_{mu nu} W_{mu nu P}

    at fixed ``D`` is assembled from the analytic factor derivatives
    ``dB`` and ``dS^{-1/2}`` of ``core.ri``:

        dE/dR_A = sum_Q (sum_{mu nu} D_{mu nu} dB_{mu nu Q}) (S^{-1/2} z)_Q
                + sum_{Q P} m_Q dS^{-1/2}_{Q P} z_P,
        m_Q     = sum_{mu nu} D_{mu nu} B_{mu nu Q}.

    Returns
    -------
    ndarray, (natm, 3)
        Gradient in Eh / Bohr.  ``atoms`` selects a subset of rows; the
        returned array then has ``len(atoms)`` rows in that order.
    """
    dm = density_from_coeffs(C, occ)
    if factors is not None and aux is None:
        raise ValueError(
            'pass the auxiliary molecule used to build factors; otherwise '
            'the factor derivatives would be taken with a different metric')
    auxmol = _as_auxmol(mol, aux)
    if factors is None:
        factors = dri.ri_factors(mol, auxmol, rcond=rcond)
    if dm.shape != (factors.nao, factors.nao):
        raise ValueError(
            f'density shape {dm.shape} disagrees with nao={factors.nao}')

    nao, naux = factors.nao, factors.naux
    w = np.matmul(factors.b, factors.jhalf)
    wf = w.reshape(nao * nao, naux)
    bf = factors.b.reshape(nao * nao, naux)
    dflat = dm.reshape(nao * nao)
    z = wf.T @ dflat
    m_q = bf.T @ dflat
    jz = factors.jhalf @ z

    rows = list(range(mol.natm) if atoms is None else atoms)
    grad = np.zeros((len(rows), 3))
    for out_i, atom in enumerate(rows):
        d = dri.ri_factors_deriv(mol, auxmol, int(atom), factors)
        term_b = np.tensordot(d.db, dm, axes=([1, 2], [0, 1])) @ jz
        term_j = np.einsum('cQP,Q,P->c', d.djhalf, m_q, z)
        grad[out_i] = term_b + term_j
    return grad


# ---------------------------------------------------------------------------
# Exact exchange (dense four-center; validation scale)
# ---------------------------------------------------------------------------
def exact_k_fock(mol, dm, eri=None, max_bytes=DEFAULT_MAX_BYTES):
    """Exact exchange matrix ``K_{mu nu} = sum_{lam sig} dm_{lam sig}
    (mu lam | nu sig)`` from the dense four-center tensor."""
    dm = np.asarray(dm, dtype=float)
    nao = int(mol.nao_nr())
    if dm.shape != (nao, nao):
        raise ValueError(f'dm shape {dm.shape} disagrees with nao={nao}')
    if eri is None:
        _guard(8.0 * nao ** 4, max_bytes, 'exact_k_fock (nao**4 ERI)')
        eri = mol.intor('int2e', aosym='s1')
    return _sym(np.einsum('mlns,ls->mn', eri, dm))


def exact_k_energy(mol, C, occ=None, eri=None,
                   max_bytes=DEFAULT_MAX_BYTES):
    """Exact exchange energy ``-1/4 Tr(D K)`` for the determinant ``C``."""
    dm = density_from_coeffs(C, occ)
    k = exact_k_fock(mol, dm, eri=eri, max_bytes=max_bytes)
    return -0.25 * float(np.sum(dm * k))


def exact_k_gradient(mol, C, occ=None, ip1=None, ip2=None,
                     max_bytes=DEFAULT_MAX_BYTES):
    """Fixed-coefficient nuclear gradient of the exact exchange energy.

    The exchange integral is ``I[mu nu lam sig] = (mu lam | nu sig)``.
    With ``E_x = -1/4 sum G4[mu nu lam sig] I[mu nu lam sig]`` and
    ``G4 = D (x) D`` the derivative of the dense ERI tensor is

        d(ab|cd)/dR_A = -[ ip1[a,b,c,d] m_a + ip1[b,a,c,d] m_b
                          + ip2[a,b,c,d] m_c + ip2[a,b,c,d->d,c] m_d ],

    where ``ip1``/``ip2`` are pyscf's ``int2e_ip1``/``int2e_ip2``.  The
    convention was verified against central differences of the full
    four-index tensor (H2O/6-31G, h=1e-5 Bohr, max deviation 2e-11).
    The exchange index order maps ``(a,b,c,d) = (mu, lam, nu, sig)``.

    Memory: ``ip1`` and ``ip2`` are ``3 * nao**4`` each and the
    per-atom derivative array is another ``3 * nao**4``.  This is a
    validation-scale routine; ``max_bytes`` refuses large systems.
    """
    dm = density_from_coeffs(C, occ)
    nao = int(mol.nao_nr())
    if dm.shape != (nao, nao):
        raise ValueError(f'density shape {dm.shape} disagrees with nao={nao}')
    if ip1 is None or ip2 is None:
        _guard(10.0 * 8.0 * nao ** 4, max_bytes,
               'exact_k_gradient (two derivative tensors plus contraction)')
        ip1 = mol.intor('int2e_ip1', aosym='s1')
        ip2 = mol.intor('int2e_ip2', aosym='s1')
    g4 = np.einsum('mn,ls->mnls', dm, dm)
    grad = np.zeros((mol.natm, 3))
    ip1_b = ip1.transpose(0, 2, 1, 3, 4)     # ip1[b, a, c, d] as (a,b,c,d)
    ip2_d = ip2.transpose(0, 1, 2, 4, 3)     # ip2[a, b, d, c] as (a,b,c,d)
    for atom in range(mol.natm):
        mask = dri.atom_ao_mask(mol, atom)
        m0 = mask[:, None, None, None]
        m1 = mask[None, :, None, None]
        m2 = mask[None, None, :, None]
        m3 = mask[None, None, None, :]
        di = -(ip1 * m0 + ip1_b * m1 + ip2 * m2 + ip2_d * m3)
        grad[atom] = -0.25 * np.einsum('cmlns,mnls->c', di, g4)
    return grad


# ---------------------------------------------------------------------------
# Core Hamiltonian and nuclear repulsion
# ---------------------------------------------------------------------------
def core_hamiltonian_energy(mol, C, occ=None):
    """``Tr(D h)`` with ``h = T + V`` (plus the ECP scalar term when present)."""
    dm = density_from_coeffs(C, occ)
    h = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    if mol.has_ecp():
        h = h + mol.intor('ECPscalar')
    return float(np.sum(dm * h))


def core_hamiltonian_gradient(mol, C, occ=None):
    """Fixed-coefficient nuclear gradient of ``Tr(D h)``.

    Follows pyscf's ``grad.rhf.hcore_generator`` construction: the
    basis-derivative part is ``-int1e_ipkin - int1e_ipnuc`` on the
    atom's first-index block plus its transpose, and the explicit
    nuclear-potential derivative is the ``int1e_iprinv`` term with the
    atom's charge (ECP corrections included when present).
    """
    dm = density_from_coeffs(C, occ)
    natm = int(mol.natm)
    charges = -np.asarray(mol.atom_charges(), dtype=float)
    slices = mol.aoslice_by_atom()
    h1 = -(mol.intor('int1e_ipkin', comp=3)
           + mol.intor('int1e_ipnuc', comp=3))
    with_ecp = bool(mol.has_ecp())
    ecp_atoms = (set(int(x) for x in mol._ecpbas[:, ATOM_OF])
                 if with_ecp else set())
    if with_ecp:
        h1 = h1 - mol.intor('ECPscalar_ipnuc', comp=3)
    grad = np.zeros((natm, 3))
    for atom in range(natm):
        p0, p1 = int(slices[atom][2]), int(slices[atom][3])
        with mol.with_rinv_at_nucleus(atom):
            v = mol.intor('int1e_iprinv', comp=3) * charges[atom]
            if with_ecp and atom in ecp_atoms:
                v = v + mol.intor('ECPscalar_iprinv', comp=3)
        v[:, p0:p1] += h1[:, p0:p1]
        hder = v + v.transpose(0, 2, 1)
        grad[atom] = np.einsum('xij,ij->x', hder, dm)
    return grad


def nuclear_repulsion_gradient(mol):
    """Gradient of the nuclear repulsion energy, Eh / Bohr."""
    z = np.asarray(mol.atom_charges(), dtype=float)
    r = np.asarray(mol.atom_coords(), dtype=float)
    dr = r[:, None, :] - r
    dist = np.linalg.norm(dr, axis=2)
    np.fill_diagonal(dist, 1.0e100)
    rinv = 1.0 / dist
    return np.einsum('i,j,ijx,ij->ix', -z, z, dr, rinv ** 3)


# ---------------------------------------------------------------------------
# RIJONX assembly
# ---------------------------------------------------------------------------
def rijonx_energy(mol, C, aux=None, occ=None, factors=None,
                  rcond=dri.DEFAULT_RCOND, eri=None,
                  max_bytes=DEFAULT_MAX_BYTES):
    """``E_nuc + Tr(Dh) + E_J(RI) + E_x(exact)``.

    ``eri`` is accepted by the caller to avoid rebuilding the dense
    four-center tensor when it is already held.
    """
    dm = density_from_coeffs(C, occ)
    e_nuc = float(mol.energy_nuc())
    e_h = core_hamiltonian_energy(mol, C, occ=occ)
    e_j = ri_j_energy_from_density(mol, dm, aux=aux, factors=factors,
                                   rcond=rcond)
    e_x = exact_k_energy(mol, C, occ=occ, eri=eri, max_bytes=max_bytes)
    return e_nuc + e_h + e_j + e_x


def rijonx_gradient(mol, C, aux=None, occ=None, factors=None,
                    rcond=dri.DEFAULT_RCOND, return_parts=False):
    """Fixed-coefficient gradient of :func:`rijonx_energy`.

    Returns ``(natm, 3)`` in Eh / Bohr, or ``(gradient, parts)`` when
    ``return_parts`` is true.  ``parts`` holds ``'j'`` (RI), ``'k'``
    (exact, dense), ``'h'`` (core) and ``'nuc'``.
    """
    g_j = ri_j_gradient(mol, C, aux=aux, occ=occ, factors=factors,
                        rcond=rcond)
    g_k = exact_k_gradient(mol, C, occ=occ)
    g_h = core_hamiltonian_gradient(mol, C, occ=occ)
    g_n = nuclear_repulsion_gradient(mol)
    total = g_j + g_k + g_h + g_n
    if return_parts:
        return total, {'j': g_j, 'k': g_k, 'h': g_h, 'nuc': g_n}
    return total


# ---------------------------------------------------------------------------
# Second-derivative layering: finite difference of the analytic gradient
# ---------------------------------------------------------------------------
def _displaced_mol(mol, atom, axis, step):
    """A copy of ``mol`` displaced by ``step`` Bohr along one axis."""
    coords = np.array(mol.atom_coords(), dtype=float)
    coords[atom, axis] += step
    symbols = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
    return gto.M(atom=[(symbols[i], coords[i]) for i in range(mol.natm)],
                 basis=mol.basis, unit='Bohr', verbose=0)


def ri_j_hessian_fd(mol, C, auxbasis=None, step=1.0e-4):
    """Finite-difference Hessian of the RI-J surface at fixed ``C``.

    This is the first rung of the second-derivative ladder: the
    analytic first derivative is differenced geometrically, the
    auxiliary basis is rebuilt at every displacement, and the orbital
    coefficients stay fixed.  It is a prototype with ``6 * natm``
    gradient evaluations (each of which loops over ``natm`` factor
    derivatives); it is not the production Hessian route.

    Returns
    -------
    ndarray, (3*natm, 3*natm)
        Hessian in Eh / Bohr^2, row/column order ``(atom, axis)``.
    """
    natm = int(mol.natm)
    hess = np.zeros((3 * natm, 3 * natm))
    for atom in range(natm):
        for axis in range(3):
            plus = _displaced_mol(mol, atom, axis, step)
            minus = _displaced_mol(mol, atom, axis, -step)
            g_plus = ri_j_gradient(plus, C, aux=auxbasis)
            g_minus = ri_j_gradient(minus, C, aux=auxbasis)
            dg = (g_plus - g_minus) / (2.0 * step)
            hess[3 * atom + axis, :] = dg.reshape(-1)
    return 0.5 * (hess + hess.T)


# ---------------------------------------------------------------------------
# Fixed-grid exchange prototype (semi-numerical limit)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CosxGrid:
    """A frozen quadrature: coordinates, weights and AO values."""
    coords: np.ndarray
    weights: np.ndarray
    ao: np.ndarray
    level: int


def make_cosx_grid(mol, level=3):
    """Build and freeze a pyscf grid with AO values on its points."""
    from pyscf.dft import numint
    from pyscf.dft.gen_grid import Grids

    grid = Grids(mol)
    grid.level = int(level)
    grid.build()
    coords = np.asarray(grid.coords, dtype=float)
    weights = np.asarray(grid.weights, dtype=float)
    ao = np.asarray(numint.eval_ao(mol, coords), dtype=float)
    return CosxGrid(coords=coords, weights=weights, ao=ao, level=int(level))


def cosx_exchange_matrix(mol, C, grid=None, level=3, occ=None,
                         symmetrize=True):
    """Fixed-grid exchange matrix; semi-numerical prototype.

    The exchange integral is factorized over one grid point,

        (mu lam | nu sig) ~= sum_g w_g X_mu(g) X_lam(g) (nu sig | r_g),

    with the exact point-charge integral ``(nu sig | r_g)`` from
    pyscf's ``int1e_grids``.  In matrix form

        F = D X^T,     G_g = A_g F_g,     K = sum_g w_g X_g G_g^T,

    where ``A_g = V_g`` is the Coulomb kernel on the grid and ``X`` the
    AO values.  This is the *semi-numerical limit* that a fitted COSX
    matrix approximates; it is not ORCA's overlap-fitted COSX.  The
    three-step structure (X, A, F, G, K) is the one a COSX
    implementation needs, and the returned diagnostics report the
    asymmetry before symmetrization.

    Returns
    -------
    (K, info) : (ndarray, dict)
        ``info`` carries ``ngrid``, ``level`` and ``asym`` (max
        ``|K - K^T|`` before symmetrization).
    """
    dm = density_from_coeffs(C, occ)
    grid = make_cosx_grid(mol, level) if grid is None else grid
    x = grid.ao
    w = grid.weights
    nao = int(mol.nao_nr())
    if x.shape[1] != nao:
        raise ValueError(
            f'grid AO shape {x.shape} disagrees with nao={nao}')
    v = mol.intor('int1e_grids', grids=grid.coords)   # (ngrid, nao, nao)
    f = x @ dm                                        # (ngrid, nao)
    g = np.einsum('gvs,gs->gv', v, f)                 # (ngrid, nao)
    k = (x * w[:, None]).T @ g
    info = {'ngrid': int(w.size), 'level': int(grid.level),
            'asym': float(np.abs(k - k.T).max())}
    if symmetrize:
        k = _sym(k)
    return k, info


def cosx_exchange_energy(mol, C, grid=None, level=3, occ=None,
                         return_info=False):
    """Exchange energy ``-1/4 Tr(D K)`` on the fixed-grid prototype.

    This value is a quadrature approximation to the exact exchange; the
    measured error on H2O/6-31G is 2.0e-5 (level 1), 1.4e-6 (level 2),
    4.4e-8 (level 3) and -4.6e-8 Ha (level 4).  The grid is frozen at
    the geometry it was built for; a geometry-dependent grid adds a
    grid-velocity term that this prototype does not carry.
    """
    dm = density_from_coeffs(C, occ)
    k, info = cosx_exchange_matrix(mol, C, grid=grid, level=level,
                                   occ=occ, symmetrize=True)
    e = -0.25 * float(np.sum(dm * k))
    if return_info:
        return e, info
    return e
