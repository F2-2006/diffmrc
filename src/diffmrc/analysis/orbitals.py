"""Orbital identity and frontier-window suggestion (analysis layer).

The M5-C batches exposed what a naive ``(HOMO, LUMO)`` window costs.
On a linear two-pi molecule the degenerate occupied pair makes the
"LUMO" a different component than the HOMO -- an x -> y pair whose
exchange integral vanishes -- and on an amide the same window lands on
the conjugated pi/pi* instead of the lone pair/pi*.  Both were measured
as 0.5-2 eV deviations before the window was widened (t317/t321).

This module turns the by-eye audit (t320) into rules.  Everything is
plain NumPy over the coefficient matrix and the AO labels: per-axis p
weight profiles of one molecular orbital over the heavy elements, a
pi/lone classification from the profile shape, a degeneracy map of the
frontier energies, and a suggested active window.  The chemistry
conventions are the caller's; no kernel import.
"""
import numpy as np

__all__ = ['heavy_atoms', 'p_axis_profile', 'classify_orbital',
           'degenerate_pairs', 'frontier_report']

# An MO counts as single-axis (pi-like along that axis) when the leading
# axis carries this share of its total p weight; it counts as atom-centred
# (a lone pair rather than a delocalized pi) when one element carries this
# share of that axis.
_AXIS_TOL = 0.9
_CENTRE_TOL = 0.8
_DEGENERATE_TOL = 1e-6


def heavy_atoms(labels):
    """The heavy-atom keys ("0 C", "1 O", ...) present in ``labels``."""
    keys = []
    for lab in labels:
        parts = lab.split()
        if len(parts) >= 3 and parts[1] != 'H':
            key = parts[0] + ' ' + parts[1]
            if key not in keys:
                keys.append(key)
    return keys


def p_axis_profile(C, labels, k, atoms):
    """Per-atom, per-axis p weight of MO ``k``.

    ``C`` is the coefficient matrix (AO x MO), ``labels`` the AO labels
    in the kernel's own format ("0 C 2px", "1 O 3pz", ...), ``atoms``
    the heavy-atom keys from :func:`heavy_atoms`.  Profile keys are
    atoms, not elements: a C-C pi must not read as one "centre".
    Returns ``{atom: {'x': w, 'y': w, 'z': w}}`` with squared
    coefficients, summed over every p shell of that atom and axis.
    """
    C = np.asarray(C)
    profile = {}
    for key in atoms:
        ax = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        for i, lab in enumerate(labels):
            parts = lab.split()
            if (len(parts) >= 3 and parts[0] + ' ' + parts[1] == key
                    and parts[-1][-2:-1] == 'p'):
                ax[parts[-1][-1]] += float(C[i, k] ** 2)
        profile[key] = ax
    return profile


def classify_orbital(profile, *, axis_tol=_AXIS_TOL, centre_tol=_CENTRE_TOL):
    """Identity of an orbital from its axis profile.

    Returns a dict with ``axis`` (the leading p axis, or None), ``kind``
    ('pi', 'lone', 'other'), ``fraction`` (leading-axis share of the
    total p weight) and ``centre`` (the dominant element, or None).

    A delocalized single-axis orbital is a pi; a single-axis orbital
    sitting on one atom is a lone pair; anything else is 'other' (a
    sigma or a mixed object, not a frontier window member).
    """
    tot = sum(w for ax in profile.values() for w in ax.values())
    if tot <= 1e-12:
        return {'axis': None, 'kind': 'other', 'fraction': 0.0,
                'centre': None}
    axis_w = {a: sum(profile[el][a] for el in profile) for a in 'xyz'}
    axis = max(axis_w, key=axis_w.get)
    fraction = axis_w[axis] / tot
    el_w = {el: profile[el][axis] for el in profile}
    centre = max(el_w, key=el_w.get)
    centre_share = el_w[centre] / axis_w[axis] if axis_w[axis] else 0.0
    if fraction < axis_tol:
        return {'axis': None, 'kind': 'other', 'fraction': fraction,
                'centre': None}
    kind = 'lone' if centre_share >= centre_tol else 'pi'
    return {'axis': axis, 'kind': kind, 'fraction': fraction,
            'centre': centre if kind == 'lone' else None}


def degenerate_pairs(mo_energy, tol=_DEGENERATE_TOL, nocc=None):
    """Index pairs whose energies coincide within ``tol``.

    With ``nocc`` given, only pairs entirely on one side of the HOMO /
    LUMO border are returned (an occupied pair or a virtual pair) --
    the frontier pattern of a linear two-pi molecule.
    """
    e = np.asarray(mo_energy, dtype=float)
    pairs = [(i, j) for i in range(len(e)) for j in range(i + 1, len(e))
             if abs(e[i] - e[j]) <= tol]
    if nocc is not None:
        pairs = [(i, j) for i, j in pairs
                 if (j < nocc) or (i >= nocc)]
    return pairs


def frontier_report(C, labels, mo_energy, nocc, atoms, *,
                    target='ppi', nelectron=None):
    """Identity of the frontier orbitals and a suggested active window.

    ``target`` is the excitation family the caller wants to describe
    ('ppi' or 'npi').  Returns a dict with the HOMO-1/HOMO/LUMO
    identities, whether the occupied frontier is a degenerate pi pair,
    whether the naive window crosses components, the index of a lone
    pair below the HOMO when the target needs one, and the suggested
    ``(ncore, nact, nele)`` triple -- a suggestion for the caller, not
    a decision: everything that entered it is returned alongside.
    ``nele`` follows the scripts' general rule ``nelectron - 2*ncore``
    and is None unless ``nelectron`` is supplied.
    """
    e = np.asarray(mo_energy, dtype=float)
    homo, lumo = nocc - 1, nocc
    ids = {k: classify_orbital(p_axis_profile(C, labels, k, atoms))
           for k in (max(0, nocc - 2), homo, lumo)}
    occ_pairs = degenerate_pairs(e, nocc=nocc)
    homo_pair = any(homo in p for p in occ_pairs)
    cross = bool(homo_pair and ids[homo]['axis'] is not None
                 and ids[lumo]['axis'] is not None
                 and ids[homo]['axis'] != ids[lumo]['axis'])
    report = {
        'homo_minus_1': ids[max(0, nocc - 2)],
        'homo': ids[homo],
        'lumo': ids[lumo],
        'occupied_degenerate_pairs': occ_pairs,
        'homo_in_degenerate_pair': homo_pair,
        'cross_component': cross,
        'lone_below_homo': None,
    }
    if target == 'npi' and ids[homo]['kind'] != 'lone':
        # walk down until an atom-centred single-axis orbital shows up
        for k in range(nocc - 2, -1, -1):
            idk = classify_orbital(p_axis_profile(C, labels, k, atoms))
            if idk['kind'] == 'lone':
                report['lone_below_homo'] = k
                break
    if target == 'npi' and ids[homo]['kind'] == 'lone':
        ncore = nocc - 1                       # the (n, pi*) pair is adjacent
    elif target == 'npi' and report['lone_below_homo'] is not None:
        ncore = report['lone_below_homo']      # reach the lone pair and pi*
    elif cross:
        ncore = nocc - 2                       # let the CI pick components
    else:
        ncore = nocc - 1
    nact = homo - ncore + 2                    # the window [ncore .. nocc]
    report['suggested_window'] = {
        'ncore': int(ncore), 'nact': int(nact),
        'nele': None if nelectron is None else int(nelectron - 2 * ncore),
    }
    return report
