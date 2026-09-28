"""Spin recognition and physical-state grouping over eigensolver roots.

The determinant convention is the kernel's, followed not interpreted
(same wording as ``ci.py``): a determinant is an integer bitmap, bit
``s`` is spin orbital ``s``, the alpha orbital of active orbital ``j``
sits at bit ``2*(ncore+j)`` and the beta at the next bit.  Everything
here is plain NumPy over the bitmap and the CI vector -- <S^2> is applied
determinant by determinant, so no density matrix and no kernel import are
needed.

Why this exists: a state-averaged solve returns *roots*, not states.  A
triplet contributes three degenerate roots (M = +1, 0, -1), so a raw root
table over-counts states and mis-weights them if handed directly to a
state-averaging weight builder.  ``state_groups`` collapses degenerate
roots into physical states carrying <S^2>/multiplicity labels, and
``select_groups`` extracts the lowest states of a requested spin -- the
two pieces that let a driver pick excitation targets by rule instead of
by eye.

Cost: ``spin_square`` is O(ndet * nv^2) per root in the current
Python-level accumulation (nv the number of singly occupied orbitals);
at the determinant counts the bitmask route already serves this is
comfortable, and both callers above evaluate it once per root.
"""
import numpy as np

__all__ = ['det_abs_sz', 'spin_square', 'state_groups', 'select_groups']

# Two <S^2> values closer than this within one energy-degenerate group are
# the same spin; farther apart the degeneracy is spin-impure.
_S2_TOL = 1e-6


def _spin_excess(det):
    """n_alpha - n_beta = 2*M of a determinant bitmap."""
    det = int(det)
    if det < 0:
        raise ValueError(f'det={det} is not a bitmap')
    width = det.bit_length()
    even = int('01' * ((width + 1) // 2), 2) if width else 0
    na = (det & even).bit_count()
    return 2 * na - det.bit_count()


def det_abs_sz(det):
    """|M| = |n_alpha - n_beta| / 2 of a determinant bitmap."""
    return abs(_spin_excess(det)) / 2.0


def _below(bitmap, x):
    return (bitmap & ((1 << x) - 1)).bit_count()


def _excitation_sign(bitmap, q, p):
    """Parity of ``a^dag_p a_q`` on ``bitmap`` (q occupied, p empty).

    Textbook Jordan-Wigner parity: the removal and the insertion each
    count the occupied spin orbitals below their index, with q's own
    removal no longer counted when p sits above it.
    """
    lo, hi = (q, p) if q < p else (p, q)
    parity = _below(bitmap, lo) + _below(bitmap, hi) - (1 if p > q else 0)
    return -1.0 if parity & 1 else 1.0


def spin_square(c_ci, dets):
    """<S^2> = <c|S_-S_+ + S_z^2 + S_z|c> of a normalized CI vector.

    Applied determinant by determinant with the textbook ladder
    operators: S_+ raises the beta electron of a spatial orbital into its
    free alpha partner, S_- lowers it back, both sums of parity-weighted
    single excitations.  No spin adaptation and no Clebsch-Gordan
    algebra; the value is exact for the given expansion, so a
    contaminated vector shows it directly (half singlet, half triplet
    reads 1, not 0 or 2).  Determinants reached by the operators that
    are absent from ``dets`` are dropped, which is the right truncation
    when ``dets`` spans a fixed M block or the full space.
    """
    c = np.asarray(c_ci, dtype=float)
    dets = [int(d) for d in dets]
    if c.ndim != 1:
        raise ValueError(f'c_ci must be one-dimensional, got shape '
                         f'{c.shape}')
    if c.shape[0] != len(dets):
        raise ValueError(f'c_ci has {c.shape[0]} coefficients for '
                         f'{len(dets)} determinants')
    index = {d: i for i, d in enumerate(dets)}
    if len(index) != len(dets):
        raise ValueError('determinants must be distinct')
    s2c = np.zeros_like(c)
    for j, d in enumerate(dets):
        cj = c[j]
        if cj == 0.0:
            continue
        m = _spin_excess(d) / 2.0
        s2c[j] += cj * (m * m + m)
        for p in range((d.bit_length() + 1) // 2):
            beta, alpha = 2 * p + 1, 2 * p
            if not (d >> beta) & 1 or (d >> alpha) & 1:
                continue
            s1 = _excitation_sign(d, beta, alpha)
            d1 = (d ^ (1 << beta)) | (1 << alpha)
            for r in range((d1.bit_length() + 1) // 2):
                ralpha, rbeta = 2 * r, 2 * r + 1
                if not (d1 >> ralpha) & 1 or (d1 >> rbeta) & 1:
                    continue
                k = index.get((d1 ^ (1 << ralpha)) | (1 << rbeta))
                if k is not None:
                    s2c[k] += cj * s1 * _excitation_sign(d1, ralpha, rbeta)
    return float(c @ s2c)


def _multiplicity_from_s2(s2):
    """2S+1 when <S^2> sits on a spin eigenvalue S(S+1), else None."""
    mult = float(np.sqrt(1.0 + 4.0 * max(s2, 0.0)))
    nearest = round(mult)
    if abs(mult - nearest) > 1e-6 or nearest < 1:
        return None
    return int(nearest)


def state_groups(energies, civecs, dets, *, tol=1e-9, coef_tol=1e-6,
                 usable=None):
    """Collapse eigensolver roots into physical states with spin labels.

    ``civecs`` has one column per root (``(ndet, nroots)``).  Roots whose
    energies coincide within ``tol`` form one group -- the M = +1, 0, -1
    components of a triplet are one state, not three.  A group whose
    members disagree on <S^2> (beyond ``_S2_TOL``) is degenerate but
    spin-impure, and carries ``multiplicity=None`` rather than a guess.

    ``usable`` marks roots whose derived observables are finite: a
    per-component contraction may be undefined (for example NEVPT2 of a
    pure |M| = 1 component), and the group's ``representative`` is then
    its lowest usable root, falling back to the lowest root overall.

    Returns ``(rows, groups)``.  Rows carry per-root ``E``, ``s2``,
    ``multiplicity`` and the set of |M| values over the coefficients
    above ``coef_tol``; groups are in energy order and carry the merged
    spin labels (``s2``/``multiplicity``/``szs``), the member roots, and
    the representative.
    """
    E = np.asarray(energies, dtype=float)
    C = np.asarray(civecs, dtype=float)
    if C.ndim != 2:
        raise ValueError(f'civecs must be (ndet, nroots), got shape '
                         f'{C.shape}')
    if C.shape[1] != E.shape[0]:
        raise ValueError(f'{C.shape[1]} CI columns for {E.shape[0]} '
                         f'energies')
    if usable is None:
        usable = np.ones(E.shape[0], dtype=bool)
    else:
        usable = np.asarray(usable, dtype=bool)
        if usable.shape != E.shape:
            raise ValueError(f'usable has shape {usable.shape}, expected '
                             f'{E.shape}')

    rows = []
    for k in range(E.shape[0]):
        col = C[:, k]
        s2 = spin_square(col, dets)
        nz = np.where(np.abs(col) > coef_tol)[0]
        rows.append({
            'root': k,
            'E': float(E[k]),
            's2': s2,
            'multiplicity': _multiplicity_from_s2(s2),
            'szs': sorted({det_abs_sz(dets[i]) for i in nz}),
        })

    groups = []
    for r in sorted(rows, key=lambda row: (row['E'], row['root'])):
        if groups and r['E'] - groups[-1]['E'] <= tol:
            groups[-1]['roots'].append(r['root'])
        else:
            groups.append({'E': r['E'], 'roots': [r['root']]})
    by_root = {r['root']: r for r in rows}
    for g in groups:
        g['szs'] = sorted({sz for k in g['roots']
                           for sz in by_root[k]['szs']})
        s2s = [by_root[k]['s2'] for k in g['roots']]
        if max(s2s) - min(s2s) <= _S2_TOL:
            g['s2'] = float(np.mean(s2s))
            g['multiplicity'] = _multiplicity_from_s2(g['s2'])
        else:
            g['s2'] = None
            g['multiplicity'] = None
        pick = next((k for k in g['roots'] if usable[k]), g['roots'][0])
        g['representative'] = pick
        g['usable'] = any(usable[k] for k in g['roots'])
    return rows, groups


def select_groups(groups, *, multiplicity, n=1):
    """The ``n`` lowest-energy groups of the given multiplicity.

    ``n=None`` returns all of them.  Groups are already in energy order,
    so this is a stable filter; an empty list means the requested spin is
    absent from the table (the caller decides whether that is an error).
    """
    picked = [g for g in groups if g['multiplicity'] == multiplicity]
    return picked if n is None else picked[:n]
