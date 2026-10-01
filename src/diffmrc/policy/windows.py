"""Frontier window suggestions as a policy rule.

The division of labour with ``analysis.orbitals.frontier_report`` follows
the layer contract, and mirrors ``policy.rules.noon_selection``: the
analysis layer reports the *facts* about the frontier (identities, the
degeneracy map, whether the naive pair crosses components, the lone pair
below the pi frontier) together with its default suggestion; this module
turns the same facts into a *rule* whose parameters the caller can set --
the window sizes per pattern and the family being described.  The default
parameters reproduce the analysis layer's suggestion exactly, which the
tests pin.

Nothing here imports the kernel; the input is the plain report mapping,
so the rule is testable on hand-built dictionaries.
"""

__all__ = ['suggest_window']


def suggest_window(frontier, *, family='ppi', nelectron=None,
                   standard_window=2, cross_window=3,
                   reach_window=3):
    """The window triple for one excitation family, from a frontier report.

    Parameters
    ----------
    frontier : mapping
        The ``analysis.orbitals.frontier_report`` output (or anything with
        the same keys: ``homo``, ``lumo``, ``homo_in_degenerate_pair``,
        ``cross_component``, ``lone_below_homo``, and ``nocc`` when
        available -- the report does not carry ``nocc``, so the caller
        passes the same inputs in the default path; the rule itself needs
        only the categorical fields plus the electron bookkeeping below).
    family : str
        ``'npi'`` or ``'ppi'``.
    nelectron : int, optional
        Total electron count; when given, the returned ``nele`` follows
        the general rule ``nelectron - 2*ncore``.
    standard_window, cross_window, reach_window : int
        The active-space sizes for the three patterns: an adjacent
        (lone pair, pi*) or (pi, pi*) pair; a frontier whose naive pair
        crosses components of a degenerate pair; a lone pair reached from
        below the pi frontier.  The defaults reproduce the analysis
        layer's suggestion.

    Returns
    -------
    dict
        ``{'ncore': int or None, 'nact': int, 'nele': int or None,
        'pattern': str}``.  ``ncore`` is None when the report does not
        carry enough information (no ``homo_index``) -- in that case the
        caller resolves it with its own space bookkeeping, exactly as
        ``driver.orbitals.auto_window`` does today.  The ``pattern`` names
        which branch fired, for the record.
    """
    homo = frontier.get('homo') or {}
    cross = bool(frontier.get('cross_component'))
    lone_below = frontier.get('lone_below_homo')

    if family == 'npi' and homo.get('kind') == 'lone':
        nact, pattern = standard_window, 'adjacent'
    elif family == 'npi' and lone_below is not None:
        nact, pattern = reach_window, 'reach'
    elif cross:
        nact, pattern = cross_window, 'cross'
    else:
        nact, pattern = standard_window, 'standard'

    result = {'nact': int(nact), 'pattern': pattern,
              'ncore': None, 'nele': None}
    nocc = frontier.get('nocc')
    if nocc is not None:
        if pattern == 'adjacent' or pattern == 'standard':
            ncore = int(nocc) - 1
        elif pattern == 'cross':
            ncore = int(nocc) - 2
        else:  # reach
            ncore = int(lone_below)
        result['ncore'] = ncore
        if nelectron is not None:
            result['nele'] = int(nelectron - 2 * ncore)
    return result
