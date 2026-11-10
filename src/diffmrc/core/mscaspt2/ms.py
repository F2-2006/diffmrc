"""Multistate CASPT2 with a selectable zeroth-order Hamiltonian.

PURPOSE
-------
Implements the MS-CASPT2 and XMS-CASPT2 effective Hamiltonians on a
determinant model space whose zeroth-order operator can be chosen as the
state-averaged or the state-specific generalized Fock matrix.

The zeroth-order choices follow Park et al., Chem. Rev. 2020, 120,
5878, eqs 11/16 (MS-CASPT2, diagonal projection of the Fock operator in
the reference space) and eq 17 (XMS-CASPT2, full ``P f P + Q f Q``).
The default MS-CASPT2 path also includes the reference-space coupling
term of Shiozaki, Woywod, Werner, Phys. Chem. Chem. Phys. 2013, 15, 262,
eq 1,

    <Omega|H0 + E_shift|Psi_N^(1)>
        - sum_M <Omega|Psi_M^(1)> <M|H0|N> + <Omega|H|N> = 0,

with the full reference-space projection ``H0 = P f P + Q f Q`` (S13
eq 3) and the first-order wave function of every state expanded in the
union of the contracted functions generated from all references (the
MS-MR contraction of S13 Section 2.1).  In the Fock eigenbasis the
off-diagonal ``<M|H0|N>`` vanish and the equations decouple (S13
eqs 5-7); in the original reference basis they are solved as one
Sylvester system.  The effective Hamiltonian is the symmetrized form of
S13 eq 2 (equivalently Park and Shiozaki, J. Chem. Theory Comput. 2017,
13, 3643, eq 8) with the level shift set to zero:

    H^eff_MN = <M|H|N> + 1/2 [ <Psi_M^(1)|H|N> + <M|H|Psi_N^(1)> ].

The coupled system is assembled with the release CASPT2 kernels
``_class_function_vectors``, ``_h0_action`` and ``_sc_kernel``.  The
union functions are held as sparse rows over their determinant support,
so ``S = V V^T`` and ``A = V H0 V^T`` come from sparse algebra instead
of a dense ``n_fn^2`` function-pair loop.  The single-state limit
reduces exactly to the release contracted solver.  Setting
``reference_coupling=False`` retains the legacy diagonal-projection
scheme (CR20 eq 16), kept for the rotation-invariance contrast.

The solver and the Slater-Condon kernel are imported lazily so that
importing this subpackage stays numpy-only.

PUBLIC INTERFACE
----------------
Densities and Fock matrices in the model space:
    ``state_averaged_density``, ``state_focks``, ``averaged_fock``,
    ``reference_fock_matrix``.
Contracted space and amplitudes:
    ``union_function_vectors``, ``coupled_reference_system``,
    ``solve_coupled_amplitudes``, ``solve_states``,
    ``first_order_vectors``, ``coupling_matrix``, ``build_heff``.
Scheme entry points:
    ``solve_ms``, ``solve_xms``.
Utility: ``random_unitary``.

``civs`` are CI vectors over the determinant list ``dets``, one row per
reference state.  ``fock_mode='sa'`` selects the state-averaged Fock
operator (MS-MR and XMS), ``fock_mode='state'`` the state-specific
operator (SS-SR).  Energies are in Eh and exclude nuclear repulsion.

CONVENTIONS
-----------
* ``H_eff`` is built from the reference-space Hamiltonian
  ``H_ref = C H_model C^T`` plus the symmetrized first-order coupling.
* With ``reference_coupling=True`` the amplitudes solve
  ``A C - S C F = -W`` over the union contraction basis and ``F`` is the
  full reference-space Fock matrix, not only its diagonal.  With
  ``reference_coupling=False`` the legacy per-state equations
  ``(A_M - F_MM S_M) x_M = -W_M`` are used instead.
* ``reference_fock_matrix`` returns ``F_MN = <M|f|N>`` in the reference
  basis; ``solve_xms`` diagonalizes it and rotates the references onto
  its eigenbasis (Chem. Rev. 2020 eq 18).
* With ``fock_mode='state'`` and ``symmetrize_fock=True`` the
  state-specific elements enter as ``(f^M + f^N)/2``, the SS-SR
  construction of Park, J. Chem. Theory Comput. 2019, 15, 3960,
  Supporting Information eq S4.

VALIDATED SCOPE
---------------
H2O CAS(2,2)/6-31G, 13 basis functions, 4 determinants, three equally
weighted singlet SA-CASSCF references at r_OH = 0.96, 1.30, 1.80 A:

* single-state limit: every scheme collapses onto the contracted solver;
  the returned E2 is bitwise identical (difference 0.0 Eh at all three
  geometries) and the assembled H_eff diagonal differs by at most
  2.8e-14 Eh from the Rayleigh-quotient round-off;
* reference-space coupling (S13 eq 1): over four random reference
  rotations (seeds 7/11/23/37) the coupled MS-MR spectrum moves by at
  most 3e-12 eV at all three geometries, the same round-off floor as
  XMS, whereas the legacy diagonal-projection MS-MR spectrum moves by
  0.28 to 1.31 eV.  The sparse union assembly and a dense function-pair
  assembly of the same system give identical spectra and the same
  regularized rank (2006 at r_OH = 0.96 A);
* Hermiticity: ``max|H_eff - H_eff^T|`` at or below 6.1e-15 Eh;
* the diagonal identity ``diag(H_eff) = diag(H_ref) + E2`` holds to at
  most 1.4e-14 Eh.

The CH2/STO-3G CAS(2,2) two-state case is the unit-test fixture.

BOUNDARIES
----------
* The reference-space coupling is implemented for the state-averaged
  Fock operator only (S13 eq 3 uses one ``f``).  The state-specific
  (SS-SR) path keeps the legacy diagonal projection and is not
  rotation invariant; that is a property of the state-specific zeroth
  order, not an implementation defect.
* ``E_shift = 0``.  The level-shift terms of S13 eqs 1-2 and the
  invariant overlap correction ``-E_shift <Psi_M^(1)|Psi_N^(1)>`` are
  not implemented.
* The union contraction has ``n_fn = n_state * n_fn_state`` functions
  and a singular overlap.  ``S`` is diagonalized densely, so the
  regularized rank is limited by memory; the determinant representation
  of ``V`` and ``H0`` is sparse and scales with the number of
  determinants reached by the contraction, not with ``n_fn^2``.
  N2 CAS(6,6)/6-31G (n_fn = 8052 per state) is assessed in the
  development report; it is not part of the validated scope.
* No IPEA shift, no real or imaginary level shift, no gradients, no
  spin-orbit coupling.  The model space is ``S_z = 0`` only.
* Equal state-averaging weights only; a weight scan was not performed.
* The literature formulas were used through the secondary presentations
  CR20, S13, PS17 and the P19 Supporting Information; the original XMS
  papers (S11, G11) were not available in full text.
"""
from __future__ import annotations

import numpy as np

from .model import ci_1rdm, embed_density, state_fock, transition_1rdm


def _contracted_solver():
    """The release contracted CASPT2 solver (lazy, keeps imports light)."""
    from ..caspt2 import caspt2_contracted_e2_coupled
    return caspt2_contracted_e2_coupled


def _reference_pairs(civs, dets):
    """Per-state determinant/coefficient lists, truncated at 1e-14."""
    civs = np.asarray(civs, dtype=float)
    return [[(int(d), float(c)) for d, c in zip(dets, ci) if abs(c) > 1e-14]
            for ci in civs]


def union_function_vectors(civs, dets, ncore, nact, nmo,
                           with_labels=False):
    """Union of the release contracted functions over all references.

    Shiozaki, Woywod and Werner, Phys. Chem. Chem. Phys. 2013, 15, 262,
    Section 2.1: with internally contracted functions the first-order
    wave function of every state is expanded in the union of the
    functions generated from all references (the MS-MR contraction).
    The per-reference lists come from the release kernel
    ``_class_function_vectors``; no operator or class rule is
    reimplemented here.

    ``with_labels=True`` also returns one ``(reference, class)`` label per
    function, which is what the block preconditioner of the matrix-free
    path groups on.
    """
    from ..caspt2 import _class_function_vectors

    functions = []
    labels = []
    for iref, ref in enumerate(_reference_pairs(civs, dets)):
        for name, vec in _class_function_vectors(ref, ncore, nact, nmo):
            functions.append((name, vec))
            labels.append((iref, name))
    if with_labels:
        return functions, labels
    return functions


def coupled_reference_system(h1e_mo, eri_mo, civs, dets, *, ncore, nact,
                             nmo=None, fock=None, rtol=1e-8):
    """Assemble the S13 eq. 1 system over the union contraction basis.

    The amplitude equation of Shiozaki et al. 2013, eq 1 (E_shift = 0),

        <Omega|H0|Psi_N^(1)> - sum_M <Omega|Psi_M^(1)> <M|H0|N>
            + <Omega|H|N> = 0,

    is projected on the union of the contracted functions of all
    references.  With ``A[i, j] = <Omega_i|H0|Omega_j>``,
    ``S[i, j] = <Omega_i|Omega_j>``, ``W[i, N] = <Omega_i|H|N>`` and
    ``F[M, N] = <M|H0|N>`` the projected equations are

        A C - S C F = -W.

    ``S`` is singular.  Its eigenpairs below ``rtol * max(ev_max, 1)``
    are dropped, the same regularization as the release contracted
    solver, and ``T`` maps the regularized amplitudes onto the union
    coefficients, ``T^T S T = 1``:

        (T^T A T) Y - Y F = -T^T W,    C = T Y.

    The union functions are rows of the sparse matrix ``V`` over their
    determinant support, so ``S = V V^T`` and ``A = V H0 V^T`` are
    assembled by sparse algebra.  The two-body Hamiltonian elements come
    from the release Slater-Condon kernel ``_sc_kernel`` and the
    one-body zeroth-order action from ``_h0_action``.

    ``fock`` is the full ``nmo x nmo`` generalized Fock matrix used for
    both ``H0`` and ``F``.  Returns a dict with ``functions``,
    ``index`` (determinant -> column), ``V``, ``S``, ``A``, ``W``,
    ``T``, ``h0q``, ``wq``, ``ev``, ``keep`` and ``nd``.
    """
    import scipy.sparse as sp

    from ..caspt2 import _h0_action, _sc_kernel

    civs = np.asarray(civs, dtype=float)
    if nmo is None:
        nmo = np.asarray(h1e_mo).shape[0]
    if fock is None:
        raise ValueError('the reference-space coupling requires a Fock '
                         'matrix')
    functions = union_function_vectors(civs, dets, ncore, nact, nmo)

    index = {}
    for _, vec in functions:
        for dd in vec:
            if dd not in index:
                index[dd] = len(index)
    nd = len(index)
    rows, cols, vals = [], [], []
    for i, (_, vec) in enumerate(functions):
        for dd, c in vec.items():
            rows.append(i)
            cols.append(index[dd])
            vals.append(c)
    V = sp.csr_matrix((vals, (rows, cols)), shape=(len(functions), nd))

    h0_rows, h0_cols, h0_vals = [], [], []
    cache = {}
    for dd, ddi in index.items():
        for tgt, val in _h0_action({dd: 1.0}, fock, nmo, cache).items():
            j = index.get(tgt)
            if j is not None:
                h0_rows.append(ddi)
                h0_cols.append(j)
                h0_vals.append(val)
    H0 = sp.csr_matrix((h0_vals, (h0_rows, h0_cols)), shape=(nd, nd))

    S = np.asarray((V @ V.T).toarray())
    S = 0.5 * (S + S.T)
    A = V @ H0 @ V.T

    h_elem, _ = _sc_kernel(h1e_mo, eri_mo)
    refs = _reference_pairs(civs, dets)
    Wd = np.zeros((nd, civs.shape[0]))
    for N, ref in enumerate(refs):
        for dd, ddi in index.items():
            Wd[ddi, N] = sum(c * h_elem(J, dd) for J, c in ref
                             if (J ^ dd).bit_count() <= 4)
    W = np.asarray(V @ Wd)

    ev, evec = np.linalg.eigh(S)
    ev_max = max(float(ev.max()), 1.0)
    keep = ev > rtol * ev_max
    if not np.any(keep):
        raise np.linalg.LinAlgError('empty overlap spectrum')
    T = evec[:, keep] / np.sqrt(ev[keep])
    h0q = T.T @ np.asarray(A @ T)
    wq = T.T @ W
    return {'functions': functions, 'index': index, 'V': V, 'S': S,
            'A': A, 'H0': H0, 'W': W, 'Wd': Wd, 'T': T, 'h0q': h0q,
            'wq': wq, 'ev': ev, 'keep': keep, 'nd': nd}


def solve_coupled_amplitudes(h0q, wq, reference_fock):
    """Solve ``h0q Y - Y F = -wq`` for the coupled amplitudes.

    ``F`` is real symmetric and is diagonalized, which turns the
    Sylvester equation into one shifted linear solve per reference.
    Returns ``(Y, cross, overlap)`` with
    ``cross = Y^T wq = <Psi_M^(1)|H|N>`` and
    ``overlap = Y^T Y = <Psi_M^(1)|Psi_N^(1)>``.
    """
    reference_fock = np.asarray(reference_fock, dtype=float)
    ef, U = np.linalg.eigh(0.5 * (reference_fock + reference_fock.T))
    wp = wq @ U
    yp = np.zeros_like(wp)
    eye = np.eye(h0q.shape[0])
    for j in range(wp.shape[1]):
        matrix = h0q - ef[j] * eye
        try:
            yp[:, j] = np.linalg.solve(matrix, -wp[:, j])
        except np.linalg.LinAlgError:
            yp[:, j] = np.linalg.lstsq(matrix, -wp[:, j], rcond=1e-12)[0]
    Y = yp @ U.T
    return Y, Y.T @ wq, Y.T @ Y


def state_averaged_density(civs, dets, norb):
    """Equal-weight state-averaged 1-RDM."""
    civs = np.asarray(civs, dtype=float)
    D = np.zeros((norb, norb))
    for ci in civs:
        D += ci_1rdm(ci, dets, norb)
    return D / civs.shape[0]


def state_focks(h1e_mo, eri_mo, civs, dets, norb, nmo=None):
    """State-specific generalized Fock matrices."""
    if nmo is None:
        nmo = np.asarray(h1e_mo).shape[0]
    return [state_fock(h1e_mo, eri_mo,
                       embed_density(ci_1rdm(ci, dets, norb), nmo))
            for ci in np.asarray(civs, dtype=float)]


def averaged_fock(h1e_mo, eri_mo, civs, dets, norb, nmo=None):
    """Generalized Fock matrix from the equal-weight state-averaged density."""
    if nmo is None:
        nmo = np.asarray(h1e_mo).shape[0]
    D = state_averaged_density(civs, dets, norb)
    return state_fock(h1e_mo, eri_mo, embed_density(D, nmo))


def reference_fock_matrix(focks, civs, dets, norb, symmetrize=False):
    """Fock matrix in the reference-state basis, ``F_MN``.

    ``focks`` is either one matrix shared by all states (state-averaged
    Fock) or one matrix per state.  With ``symmetrize=True`` the
    state-specific elements are averaged as ``(f^M + f^N)/2``, the
    SS-SR construction of Park, J. Chem. Theory Comput. 2019, 15, 3960,
    Supporting Information eq S4.
    """
    civs = np.asarray(civs, dtype=float)
    n = civs.shape[0]
    shared = np.ndim(focks) == 2
    F = np.zeros((n, n))
    for M in range(n):
        for N in range(n):
            G = transition_1rdm(civs[M], civs[N], dets, norb)
            if shared:
                fM = fN = np.asarray(focks)[:norb, :norb]
            elif symmetrize:
                fM = np.asarray(focks[M])[:norb, :norb]
                fN = np.asarray(focks[N])[:norb, :norb]
            else:
                fM = np.asarray(focks[M])[:norb, :norb]
                fN = fM
            F[M, N] = 0.5 * np.sum((fM + fN) * G)
    return 0.5 * (F + F.T)


def solve_states(h1e_mo, eri_mo, civs, dets, *, ncore, nact, nele,
                 focks=None, rtol=1e-8):
    """One contracted CASPT2 solve per reference state.

    Returns a list of slimmed solver records (amplitudes and contracted
    functions only) plus the per-state second-order energies.
    """
    solver = _contracted_solver()
    civs = np.asarray(civs, dtype=float)
    if focks is None:
        focks = [None] * civs.shape[0]
    elif np.ndim(focks) == 2:
        focks = [focks] * civs.shape[0]
    records = []
    e2 = np.zeros(civs.shape[0])
    for M, ci in enumerate(civs):
        e2[M], _, system = solver(
            h1e_mo, eri_mo, ci, dets, ncore=ncore, nact=nact, nele=nele,
            fock=focks[M], rtol=rtol, return_system=True)
        records.append({'x': np.asarray(system['x'], dtype=float),
                        'functions': system['functions'],
                        'E0': float(system['E0'])})
    return records, e2


def first_order_vectors(records):
    """Collect ``|Psi_M^(1)> = sum_i x_i |Omega_i^M>`` as determinant maps."""
    psis = []
    for rec in records:
        psi = {}
        for i, (_, vec) in enumerate(rec['functions']):
            xi = rec['x'][i]
            if xi == 0.0:
                continue
            for dd, c in vec.items():
                psi[dd] = psi.get(dd, 0.0) + xi * c
        psis.append(psi)
    return psis


def coupling_matrix(psis, civs, h1e_mo, eri_mo, dets):
    """``cross_MN = <Psi_M^(1)|H|N>`` for all state pairs.

    The matrix elements come from ``diffmrc.core.caspt2._sc_kernel``,
    the same kernel the contracted solver uses for ``W``.  Its convention
    (zero for identical determinants) is kept so that
    ``H_eff[M, M] = <M|H|M> + E2_M`` holds exactly.
    """
    from ..caspt2 import _sc_kernel

    civs = np.asarray(civs, dtype=float)
    n = civs.shape[0]
    h_elem, _ = _sc_kernel(h1e_mo, eri_mo)
    refs = [[(int(d), float(c)) for d, c in zip(dets, ci) if abs(c) > 1e-14]
            for ci in civs]
    support = set()
    for psi in psis:
        support.update(psi.keys())

    g = []
    for N in range(n):
        ref = refs[N]
        gN = {}
        for dd in support:
            acc = 0.0
            for J, c in ref:
                if (J ^ dd).bit_count() <= 4:
                    acc += c * h_elem(J, dd)
            gN[dd] = acc
        g.append(gN)

    cross = np.zeros((n, n))
    for M, psi in enumerate(psis):
        for N in range(n):
            gN = g[N]
            cross[M, N] = sum(c * gN[dd] for dd, c in psi.items())
    return cross


def _build_heff_matrix_free(h1e_mo, eri_mo, civs, dets, *, H_model, f,
                            ncore, nact, norb, rtol, time_limit,
                            return_psis):
    """The coupled union path without the dense overlap and H0 matrices."""
    from .matrix_free import (coupled_reference_system_mf,
                              solve_coupled_amplitudes_mf)

    civs = np.asarray(civs, dtype=float)
    n = civs.shape[0]
    nmo = np.asarray(h1e_mo).shape[0]
    system = coupled_reference_system_mf(
        h1e_mo, eri_mo, civs, dets, ncore=ncore, nact=nact, nmo=nmo,
        fock=f)
    F = reference_fock_matrix(f, civs, dets, norb)
    amplitudes, cross, overlap, info = solve_coupled_amplitudes_mf(
        system['V'], system['H0'], system['W'], F, rtol=rtol,
        time_limit=time_limit, h0_diag=system['h0_diag'],
        labels=system['labels'])
    records = [{'x': amplitudes[:, N].copy(),
                'functions': system['functions'],
                'E0': float(F[N, N])} for N in range(n)]
    psis = first_order_vectors(records) if return_psis else None
    Href = civs @ H_model @ civs.T
    H_eff = Href + 0.5 * (cross + cross.T)
    energies = np.linalg.eigvalsh(H_eff)
    return {
        'H_eff': H_eff,
        'H_ref': Href,
        'cross': cross,
        'energies': energies,
        'e2': np.diag(cross).copy(),
        'E0': np.diag(F).copy(),
        'reference_fock': F,
        'civs': civs,
        'records': records,
        'psis': psis,
        'amplitudes': amplitudes,
        'overlap': overlap,
        'coupled_system': system,
        'matrix_free': True,
        'mf_info': info,
    }


def build_heff(h1e_mo, eri_mo, civs, dets, *, ncore, nact, nele,
               H_model=None, det_index=None, fock_mode='sa', focks=None,
               symmetrize_fock=False, reference_coupling=False, rtol=1e-8,
               matrix_free=False, matrix_free_rtol=1e-10,
               matrix_free_time_limit=None, return_psis=True):
    """Build and diagonalize the symmetrized effective Hamiltonian.

    ``fock_mode='sa'`` uses one state-averaged Fock operator for every
    state (the MS-MR and XMS choice); ``fock_mode='state'`` uses the
    state-specific operator of each reference (the SS-SR choice).

    ``reference_coupling=True`` adds the reference-space coupling term
    of Shiozaki et al. 2013, eq 1, and expands the first-order space in
    the union of the contracted functions of all references.  It
    requires ``fock_mode='sa'`` (S13 eq 3 defines a single Fock
    operator).  With one reference state the coupling term is empty and
    the release solver is used unchanged, so the single-state limit stays
    bitwise identical.

    ``matrix_free=True`` solves the same coupled system without forming
    the ``n_fn x n_fn`` overlap and zeroth-order matrices: the shifted
    systems ``(A - E_j S) c_j = -W_j`` are handed to LSMR with the
    operator actions ``S c = V (V^T c)`` and ``A c = V (H0 (V^T c))``
    (see :mod:`diffmrc.core.mscaspt2.matrix_free`).  ``rtol`` keeps its
    meaning as the overlap-regularization threshold of the dense route
    and is not used on this path; ``matrix_free_rtol`` is the relative
    residual the iterative solve is driven to and
    ``matrix_free_time_limit`` its wall-clock guard.  ``return_psis``
    costs one determinant dictionary per state over the union support
    and can be switched off on systems where that does not fit.
    """
    civs = np.asarray(civs, dtype=float)
    n = civs.shape[0]
    norb = ncore + nact
    if H_model is None:
        from .model import model_hamiltonian
        H_model = model_hamiltonian(h1e_mo, eri_mo, dets, ncore, nact, nele)
    if det_index is None:
        det_index = {d: i for i, d in enumerate(dets)}

    if reference_coupling and fock_mode != 'sa':
        raise ValueError("reference_coupling requires fock_mode='sa'")
    if matrix_free and not (reference_coupling and n > 1):
        raise ValueError('matrix_free requires the coupled union path '
                         '(reference_coupling=True and at least two '
                         'reference states)')

    if reference_coupling and n > 1:
        f = averaged_fock(h1e_mo, eri_mo, civs, dets, norb)
        if focks is not None:
            if np.ndim(focks) != 2:
                raise ValueError('reference_coupling requires one shared '
                                 'Fock matrix')
            f = np.asarray(focks, dtype=float)
        if matrix_free:
            return _build_heff_matrix_free(
                h1e_mo, eri_mo, civs, dets, H_model=H_model, f=f,
                ncore=ncore, nact=nact, norb=norb,
                rtol=matrix_free_rtol, time_limit=matrix_free_time_limit,
                return_psis=return_psis)
        system = coupled_reference_system(
            h1e_mo, eri_mo, civs, dets, ncore=ncore, nact=nact,
            nmo=np.asarray(h1e_mo).shape[0], fock=f, rtol=rtol)
        F = reference_fock_matrix(f, civs, dets, norb)
        Y, cross, overlap = solve_coupled_amplitudes(system['h0q'],
                                                     system['wq'], F)
        amplitudes = system['T'] @ Y
        records = [{'x': amplitudes[:, N].copy(),
                    'functions': system['functions'],
                    'E0': float(F[N, N])} for N in range(n)]
        psis = first_order_vectors(records)
        Href = civs @ H_model @ civs.T
        H_eff = Href + 0.5 * (cross + cross.T)
        energies = np.linalg.eigvalsh(H_eff)
        return {
            'H_eff': H_eff,
            'H_ref': Href,
            'cross': cross,
            'energies': energies,
            'e2': np.diag(cross).copy(),
            'E0': np.diag(F).copy(),
            'reference_fock': F,
            'civs': civs,
            'records': records,
            'psis': psis,
            'amplitudes': amplitudes,
            'overlap': overlap,
            'coupled_system': system,
        }

    if fock_mode == 'sa':
        f = averaged_fock(h1e_mo, eri_mo, civs, dets, norb)
        solve_focks = f
        ref_focks = f
    elif fock_mode == 'state':
        solve_focks = state_focks(h1e_mo, eri_mo, civs, dets, norb)
        ref_focks = solve_focks
    else:
        raise ValueError("fock_mode must be 'sa' or 'state'")

    if focks is not None:
        solve_focks = focks
        ref_focks = focks

    records, e2 = solve_states(h1e_mo, eri_mo, civs, dets, ncore=ncore,
                               nact=nact, nele=nele, focks=solve_focks,
                               rtol=rtol)
    psis = first_order_vectors(records)
    cross = coupling_matrix(psis, civs, h1e_mo, eri_mo, dets)
    Href = civs @ H_model @ civs.T
    H_eff = Href + 0.5 * (cross + cross.T)
    energies = np.linalg.eigvalsh(H_eff)
    F = reference_fock_matrix(ref_focks, civs, dets, norb,
                              symmetrize=(fock_mode == 'state'
                                          and symmetrize_fock))
    return {
        'H_eff': H_eff,
        'H_ref': Href,
        'cross': cross,
        'energies': energies,
        'e2': e2,
        'E0': np.array([r['E0'] for r in records]),
        'reference_fock': F,
        'civs': civs,
        'records': records,
        'psis': psis,
    }


def solve_ms(h1e_mo, eri_mo, civs, dets, *, reference_coupling=True,
             **kwargs):
    """MS-CASPT2.

    With the default ``reference_coupling=True`` the amplitudes include
    the S13 eq. 1 reference-space coupling term and the first-order
    space is the union of the contracted functions of all references;
    the eigenvalue spectrum is then invariant under unitary rotations of
    the reference states.  ``reference_coupling=False`` reproduces the
    legacy CR20 eq. 16 diagonal-projection scheme, which is not
    invariant and is kept for the contrast in the rotation test.
    """
    out = build_heff(h1e_mo, eri_mo, civs, dets,
                     reference_coupling=reference_coupling, **kwargs)
    out['scheme'] = ('MS-CASPT2 (S13 eq. 1)' if reference_coupling
                     else 'MS-CASPT2 (diagonal projection)')
    return out


def solve_xms(h1e_mo, eri_mo, civs, dets, *, ncore, nact, nele,
              fock_mode='sa', rtol=1e-8, reference_coupling=False,
              **kwargs):
    """XMS-CASPT2: rotate the references onto the Fock eigenbasis first.

    The rotation is the ``P f P`` diagonalization of Chem. Rev. 2020 eq
    18.  With ``fock_mode='state'`` the state-specific Fock elements are
    symmetrized first (Park 2019 Supporting Information eq S4).

    ``reference_coupling=True`` additionally expands the first-order
    space in the union of the contracted functions of all references
    (the S13 eq. 1 MS-MR contraction).  The rotated reference basis
    makes ``F`` diagonal, so the coupled equations decouple; the
    resulting spectrum coincides with ``solve_ms`` with the coupling
    term, which is the basis-rotation invariance of the method.
    """
    civs = np.asarray(civs, dtype=float)
    n = civs.shape[0]
    norb = ncore + nact
    if fock_mode == 'sa':
        ref_focks = averaged_fock(h1e_mo, eri_mo, civs, dets, norb)
    elif fock_mode == 'state':
        ref_focks = state_focks(h1e_mo, eri_mo, civs, dets, norb)
    else:
        raise ValueError("fock_mode must be 'sa' or 'state'")

    F = reference_fock_matrix(ref_focks, civs, dets, norb,
                              symmetrize=(fock_mode == 'state'))
    fock_eigvals, U = np.linalg.eigh(F)
    civs_rot = U.T @ civs
    out = build_heff(h1e_mo, eri_mo, civs_rot, dets, ncore=ncore,
                     nact=nact, nele=nele, fock_mode=fock_mode,
                     symmetrize_fock=(fock_mode == 'state'),
                     reference_coupling=reference_coupling, rtol=rtol,
                     **kwargs)
    out['scheme'] = 'XMS-CASPT2'
    out['rotation'] = U
    out['fock_model'] = F
    out['fock_eigvals'] = fock_eigvals
    out['civs_input'] = civs
    return out


def random_unitary(n, seed=0):
    """Haar-ish orthogonal matrix for the invariance check."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.diag(R))
