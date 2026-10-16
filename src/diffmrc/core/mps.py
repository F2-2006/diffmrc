"""MPS/DMRG core module (graduated increment; grows with the DMRG line).

Current member: ``pin_degenerate_columns`` -- the deterministic
resolution of exactly degenerate MO column pairs that column-referenced
active-space windows require.

Graduation record (2026-10-16): the function first lived in the T5
prototype driver (``code/t416_t5_paper_systems.py``); the measurements
behind it are recorded there and in the T5 report.  In short: pyscf RHF
orbitals are unique only up to arbitrary rotations inside exactly
degenerate irreps (N2/6-31G at the calibration geometry: the pi_u pair
at columns (5,6), the pi_g pair at (7,8), and two higher pairs), so
different processes return different representatives; CAS(2,2) on the
recorded start window {6,7} scattered over 0.02 Ha across fresh builds
although the RHF energy matched to 3e-13, because the occupied pi_u
mixture decides which combination sits in the frozen core.  With the
pinned basis, the recorded reference value of the exhaustive-window
best (electronic energy -132.4777614802 = total -108.9053220847)
reproduces to 8e-11.  The prototype now forwards here (single
implementation).

The module is intentionally numpy-only: no jax, no pyscf imports; the
pyscf-facing extraction (mol/mf -> arrays) stays at the call sites.
"""
import numpy as np

__all__ = ['pin_degenerate_columns']


def pin_degenerate_columns(mo_coeff, mo_energy, x2_ao, deg_tol=1e-7):
    """Deterministic column basis for (near-)degenerate MO groups.

    Every group of columns whose energies agree within ``deg_tol`` is
    rotated into the eigenbasis of the ``x2_ao`` operator, ordered by
    DESCENDING expectation value; then each column's phase is fixed by
    making its largest-magnitude entry positive.  Non-degenerate columns
    are returned untouched (bitwise).

    Parameters
    ----------
    mo_coeff : (nao, nmo) array
        Column-ordered MO coefficients in the AO basis.
    mo_energy : (nmo,) array
        Orbital energies (the degeneracy-grouping key).
    x2_ao : (nao, nao) array
        The x^2 operator in the same AO basis.  Inside a pi pair its
        restriction has a real eigenvalue gap, which is what makes the
        pinned axes stable against ~1e-12 density noise; for linear
        molecules it also distinguishes the two pair components.
    deg_tol : float
        Energy window treated as degenerate.

    Returns
    -------
    (C_pinned, n_rotated)
        The pinned coefficient matrix (a new array) and the number of
        columns re-rotated.

    Invariants
    ----------
    * a window containing a degenerate group whole is invariant under
      the rotation (the span is preserved, so its CI energies do not
      move);
    * split groups become deterministic -- any single representative of
      a degenerate pair is symmetry-equivalent, but the process-to-
      process arbitrariness is removed;
    * pinning is idempotent and invariant to an input rotation applied
      inside a degenerate group (the x^2 axes and the sign rule pin the
      output uniquely).
    """
    eps = np.asarray(mo_energy, dtype=float)
    C = np.array(mo_coeff, dtype=float, copy=True)
    x2 = np.asarray(x2_ao, dtype=float)
    k, n = 0, len(eps)
    n_rot = 0
    while k < n:
        j = k + 1
        while j < n and abs(eps[j] - eps[k]) < deg_tol:
            j += 1
        if j - k > 1:
            G = C[:, k:j]
            _b, V = np.linalg.eigh(G.T @ x2 @ G)
            C[:, k:j] = G @ V[:, ::-1]          # descending <x^2>
            n_rot += j - k
        k = j
    for c in range(n):
        imax = int(np.argmax(np.abs(C[:, c])))
        if C[imax, c] < 0:
            C[:, c] = -C[:, c]
    return C, n_rot
