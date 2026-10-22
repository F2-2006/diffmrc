"""AVAS active spaces with target-shell rules and space diagnostics.

The atomic-valence active space (AVAS, Sayfutyarova et al. 2017) turns a
target set of atomic orbitals into a set of active molecular orbitals
without a wavefunction: project the target subspace onto the occupied and
virtual MO blocks in the overlap metric, rotate each block by the
eigenvectors of the projected overlap, and keep the rotated slots whose
eigenvalue is large.  The result is a deterministic starting guess for a
CASSCF, and the deterministic counterpart of the entropy selectors in
:mod:`diffmrc.policy.entropy_selection`: structure proposes, the spectrum
can still falsify.

Construction
------------
Let ``X`` be the ``nao x ntarget`` selector of the target AOs and
``sigma = X^T S X`` their overlap.  For one MO block ``C_b`` the projected
overlap is

    O_b = C_b^T S X sigma^-1 X^T S C_b,

a symmetric matrix with eigenvalues in [0, 1]; eigenvalue ``lambda_i`` is
the fraction of target character carried by the rotated orbital ``i``.
The implementation obtains ``O_b`` through the thin SVD of the whitened
overlap

    Y_b = sigma^-1/2 X^T S C_b = U diag(s) V^T,
    lambda = s^2,    C_b,rotated = C_b V,

which is the same decomposition with one fewer matrix product and gives
the rotation directly.  The occupied and virtual blocks rotate
independently; singly occupied orbitals of an open-shell reference are
kept unchanged and active (the source's option 3).  Threshold mode keeps
every slot above ``threshold``; the ``nact``/``nele`` mode keeps a fixed
number of slots and ignores the threshold (the ``nel``/``norb``
specification of the ORCA implementation).

Target rules
------------
- ``'valence-f'``: for every lanthanide or actinide atom present, take the
  f shell with the lowest principal quantum number in the basis.  One f
  shell carries seven orbitals; that is the ``7f`` of the ORCA
  ``AVAS(Valence-F)`` rule.  ECP bases carry one f shell, all-electron ANO
  bases carry several (6f, 7f, 8f), and in both the first one is the
  valence shell the rule wants.
- ``'double-d'``: for every 3d metal present (Sc-Zn), take up to two d
  shells, lowest principal quantum number first.  ORCA's
  ``AVAS(Double-D)`` adds the 4d shell to the 3d shell; when the basis has
  no second d shell the rule degrades to the single d shell and says so.
- explicit ``elements`` and ``shells``: a per-element shell list such as
  ``{'O': ['2s', '2p'], 'H': ['1s']}``.  A shell token without a principal
  number matches every shell of that angular momentum.

Diagnostics
-----------
Two checks travel with the selection, both from the ORCA CASSCF manual
(3.13.7) and both heuristics rather than identity checks:

- *Occupation window*: a CASSCF natural occupation outside [0.02, 1.98]
  means the orbital is nearly uncorrelated and probably does not belong in
  the active space.  The check needs correlated natural occupations, so
  the AVAS construction itself can only expose the function
  (:func:`occupation_diagnostics`); a caller that has a 1-RDM passes the
  occupations through ``active_occupations``.
- *Orbital-energy gaps*: the active space should be separated from the
  inactive and external spaces by about 0.2 Eh.  The rotated orbitals have
  no canonical eigenvalues of their own, so the diagnotic uses the
  diagonal Fock expectation of each rotated orbital, built from the
  canonical ``mo_energy`` as ``sum_k V_ki^2 eps_k``.  ``Act-Int`` is the
  lowest active energy minus the highest inactive one; ``Ext-Act`` is the
  lowest external energy minus the highest active one.

Scope
-----
This module is a front end.  It adds target rules, the ``nact``/``nele``
mode, and the diagnostics on top of the closed-shell construction in
:mod:`diffmrc.policy.projection`; that module and its
``avas_selection(C, S, target_aos, occ)`` signature are unchanged.  The
two routes are cross-checked in the tests on a case where both apply.
"""
from dataclasses import dataclass, field

import numpy as np

from .projection import AVAS_DEFAULT_THRESHOLD
from .rules import LANTHANIDES

__all__ = ['AVAS_DEFAULT_THRESHOLD', 'AVAS_GAP_MIN',
           'AVAS_OCCUPATION_LOW', 'AVAS_OCCUPATION_HIGH',
           'ACTINIDES', 'THREE_D_METALS',
           'TargetSubspace', 'OccupationDiagnostics', 'GapDiagnostics',
           'AvasDiagnostics', 'target_ao_indices', 'target_electron_count',
           'occupation_diagnostics', 'orbital_gap_diagnostics',
           'avas_selection']

#: Natural-occupation window of an active orbital (ORCA CASSCF manual
#: 3.13.7: occupations in 0.02-1.98, values near 0 or 2 mean weak
#: coupling).
AVAS_OCCUPATION_LOW = 0.02
AVAS_OCCUPATION_HIGH = 1.98

#: Active-space separation from the inactive and external spaces, in Eh.
#: The manual says "about 0.2 Eh"; the value is a warning line, not a
#: selection criterion.
AVAS_GAP_MIN = 0.2

#: The actinides.  Promethium is absent from LANTHANIDES in ``rules`` for
#: the same reason it is present: it is radioactive.  Both lists exist so
#: that a user who supplies the element gets a rule rather than a silent
#: miss.
ACTINIDES = ('Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
             'Es', 'Fm', 'Md', 'No', 'Lr')

#: First-row transition metals, the target set of AVAS(Double-D).
THREE_D_METALS = ('Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn')

_ANGULAR = 'spdfghik'


@dataclass
class TargetSubspace:
    """The atomic-orbital indices a target rule selected, and why."""

    indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    rule: str = ''
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


@dataclass
class OccupationDiagnostics:
    """Which active orbitals fall outside the natural-occupation window."""

    occupations: np.ndarray = field(default_factory=lambda: np.array([]))
    low: float = AVAS_OCCUPATION_LOW
    high: float = AVAS_OCCUPATION_HIGH
    weak: list = field(default_factory=list)
    strong: list = field(default_factory=list)
    invalid: list = field(default_factory=list)
    ok: bool = True
    evidence: list = field(default_factory=list)


@dataclass
class GapDiagnostics:
    """Act-Int and Ext-Act gaps of an active space, in Eh."""

    act_int: float = None
    ext_act: float = None
    gap_min: float = AVAS_GAP_MIN
    ok: bool = True
    energies: np.ndarray = field(default_factory=lambda: np.array([]))
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


@dataclass
class AvasDiagnostics:
    """Everything that produced an AVAS selection, and the checks on it.

    ``active`` indexes columns of ``C_rotated``.  ``ncore`` counts the
    doubly occupied orbitals left outside the active space, so a caller
    builds a CASSCF as ``ncore`` core, the ``active`` columns, and the
    rest virtual.
    """

    active: list = field(default_factory=list)
    nact: int = 0
    nele: int = 0
    ncore: int = 0
    nexternal: int = 0
    target_aos: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    target_rule: str = ''
    target_electrons: float = 0.0
    C_rotated: np.ndarray = field(default_factory=lambda: np.array([]))
    blocks: dict = field(default_factory=dict)
    overlap_eigenvalues: dict = field(default_factory=dict)
    singular_values: dict = field(default_factory=dict)
    scf_occupations: np.ndarray = field(default_factory=lambda: np.array([]))
    occupation: object = None
    gap: object = None
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _parse_ao_label(label):
    """Split a PySCF AO label into (atom, element, principal, letter).

    ``"0 Ce 4f-3"`` becomes ``(0, 'Ce', 4, 'f')``; ``"3 O 2pz"`` becomes
    ``(3, 'O', 2, 'p')``.  The principal number is the leading digit run
    of the shell field, which ANO labels carry as ``6f`` and ECP labels as
    ``4f``.  Returns None for a label with fewer than three fields or no
    angular-momentum letter.
    """
    parts = str(label).split()
    if len(parts) < 3:
        return None
    try:
        atom = int(parts[0])
    except ValueError:
        return None
    element = parts[1]
    shell = parts[2]
    i = 0
    while i < len(shell) and shell[i].isdigit():
        i += 1
    pqn = int(shell[:i]) if i else None
    letter = None
    for ch in shell[i:]:
        if ch.lower() in _ANGULAR:
            letter = ch.lower()
            break
    if letter is None:
        return None
    return atom, element, pqn, letter


def _symbols_from_mol(mol):
    """Atom symbols in index order, from a PySCF-like molecule."""
    if hasattr(mol, 'atom_symbol'):
        return [str(mol.atom_symbol(i)) for i in range(int(mol.natm))]
    raise ValueError('cannot read atom symbols from mol; pass symbols=...')


def _parse_shell_token(token):
    """``'2p'`` -> (2, 'p'); ``'f'`` -> (None, 'f')."""
    token = str(token).strip().lower()
    i = 0
    while i < len(token) and token[i].isdigit():
        i += 1
    pqn = int(token[:i]) if i else None
    letter = token[i:]
    if letter not in _ANGULAR or not letter:
        raise ValueError(f'shell token {token!r} is not an angular-momentum '
                         f'letter with an optional principal number')
    return pqn, letter


def target_ao_indices(mol, *, rule=None, elements=None, shells=None,
                      labels=None, symbols=None):
    """Resolve a target AO subspace from a molecule and a shell rule.

    Parameters
    ----------
    mol : pyscf-like
        Provides ``ao_labels()``; ``atom_symbol(i)`` is read unless
        `symbols` is supplied.
    rule : str, optional
        ``'valence-f'`` or ``'double-d'``.  Mutually exclusive with
        `shells`.
    elements : sequence of str, optional
        Restrict the rule to these element symbols.  The default is every
        atom the rule applies to: lanthanides plus actinides for
        ``'valence-f'``, 3d metals for ``'double-d'``.
    shells : dict or sequence, optional
        Per-element shell tokens.  A dict maps an element to its tokens,
        e.g. ``{'O': ['2s', '2p'], 'H': ['1s']}``; a flat sequence applies
        to every element in `elements` (or every atom).  This route is the
        explicit form and does not use `rule`.
    labels, symbols : optional
        Override the molecule's own AO labels and atom symbols.  The
        overrides exist so the rule can be tested without a PySCF
        installation.

    Returns
    -------
    TargetSubspace
        Ascending AO indices; the evidence names the shell chosen for
        every target atom.  An empty match raises ValueError -- a target
        rule that finds nothing is a caller error, not a refusal, because
        every rule here is asked for a specific shell.
    """
    if rule is not None and shells is not None:
        raise ValueError('give either rule=... or shells=..., not both')
    if rule is None and shells is None:
        raise ValueError('give a target rule: rule=... or shells=...')
    labels = (list(mol.ao_labels()) if labels is None else list(labels))
    symbols = (_symbols_from_mol(mol) if symbols is None else list(symbols))
    records = []
    for i, label in enumerate(labels):
        parsed = _parse_ao_label(label)
        if parsed is None:
            continue
        atom, element, pqn, letter = parsed
        if 0 <= atom < len(symbols):
            records.append((i, atom, element, pqn, letter))
    if not records:
        raise ValueError('no AO label carries an element and an angular '
                         'momentum; cannot resolve a target shell')

    evidence, warnings, chosen = [], [], []
    if rule == 'valence-f':
        if elements is None:
            target_elements = [s for s in dict.fromkeys(symbols)
                               if s in LANTHANIDES or s in ACTINIDES]
        else:
            target_elements = list(elements)
            missing = [e for e in target_elements if e not in symbols]
            if missing:
                warnings.append(f'target element(s) {missing} are not in '
                                f'the molecule')
        if not target_elements:
            raise ValueError('the Valence-F rule found no lanthanide or '
                             'actinide atom; name targets with elements=...')
        for atom in range(len(symbols)):
            if symbols[atom] not in target_elements:
                continue
            f_records = [r for r in records
                         if r[1] == atom and r[4] == 'f']
            if not f_records:
                warnings.append(f'atom {atom} {symbols[atom]} has no f '
                                f'shell in this basis; the Valence-F rule '
                                f'contributes nothing for it')
                continue
            pqns = sorted({r[3] for r in f_records if r[3] is not None})
            if not pqns:
                warnings.append(f'atom {atom} {symbols[atom]} has f AOs '
                                f'without a principal number; taking all '
                                f'of them')
                picked = f_records
            else:
                picked = [r for r in f_records if r[3] == pqns[0]]
            chosen.extend(r[0] for r in picked)
            if len(picked) != 7:
                warnings.append(
                    f'atom {atom} {symbols[atom]}: expected 7 f orbitals '
                    f'for the Valence-F rule, found {len(picked)} in the '
                    f'lowest f shell')
            evidence.append(
                f'atom {atom} {symbols[atom]}: f shell '
                f'{pqns[0] if pqns else "?"}, {len(picked)} AO(s) '
                f'(lowest f shell present; the Valence-F rule takes the '
                f'seven valence f orbitals)')
        rule_name = 'valence-f'
    elif rule == 'double-d':
        if elements is None:
            target_elements = [s for s in dict.fromkeys(symbols)
                               if s in THREE_D_METALS]
        else:
            target_elements = list(elements)
            missing = [e for e in target_elements if e not in symbols]
            if missing:
                warnings.append(f'target element(s) {missing} are not in '
                                f'the molecule')
        if not target_elements:
            raise ValueError('the Double-D rule found no 3d metal; name '
                             'targets with elements=...')
        for atom in range(len(symbols)):
            if symbols[atom] not in target_elements:
                continue
            d_records = [r for r in records
                         if r[1] == atom and r[4] == 'd']
            if not d_records:
                warnings.append(f'atom {atom} {symbols[atom]} has no d '
                                f'shell in this basis; the Double-D rule '
                                f'contributes nothing for it')
                continue
            pqns = sorted({r[3] for r in d_records if r[3] is not None})
            picked_shells = pqns[:2] if pqns else []
            picked = ([r for r in d_records if r[3] in picked_shells]
                      if picked_shells else d_records)
            if len(picked_shells) < 2:
                warnings.append(
                    f'atom {atom} {symbols[atom]}: Double-D asks for two d '
                    f'shells, the basis has {len(picked_shells) or 1}; '
                    f'using what is present')
            chosen.extend(r[0] for r in picked)
            evidence.append(
                f'atom {atom} {symbols[atom]}: d shells {picked_shells}, '
                f'{len(picked)} AO(s)')
        rule_name = 'double-d'
    else:
        if isinstance(shells, dict):
            spec = {str(e): list(tokens) for e, tokens in shells.items()}
            if elements is not None:
                keep = set(elements)
                spec = {e: tokens for e, tokens in spec.items()
                        if e in keep}
            allowed = set(spec)
        else:
            allowed = set(symbols) if elements is None else set(elements)
            tokens = list(shells)
            spec = {e: tokens for e in allowed}
        wanted = {}
        for element, tokens in spec.items():
            wanted[element] = [_parse_shell_token(t) for t in tokens]
        missing = [e for e in allowed if e not in symbols]
        if missing:
            warnings.append(f'target element(s) {missing} are not in the '
                            f'molecule')
        for i, atom, element, pqn, letter in records:
            if element not in wanted:
                continue
            for want_pqn, want_letter in wanted[element]:
                if letter != want_letter:
                    continue
                if want_pqn is not None and pqn != want_pqn:
                    continue
                chosen.append(i)
                break
        rule_name = 'shells'
        evidence.append(f'explicit shell target: '
                        f'{ {e: [str(t) for t in ts] for e, ts in spec.items()} }')
    if not chosen:
        raise ValueError(f'the target rule {rule_name!r} matched no AOs')
    indices = np.array(sorted({int(i) for i in chosen}), dtype=int)
    evidence.append(f'target subspace: {indices.size} AO(s)')
    return TargetSubspace(indices, rule_name, evidence, warnings)


def _target_metric(S, target):
    """Selector and inverse square root of the target AO overlap.

    Returns ``X`` (``nao x ntarget``) and ``sigma^-1/2`` for
    ``sigma = X^T S X``.  A linearly dependent target set is refused
    rather than whitened into noise.
    """
    nao = S.shape[0]
    X = np.zeros((nao, target.size))
    X[target, np.arange(target.size)] = 1.0
    sigma = X.T @ S @ X
    sigma = 0.5 * (sigma + sigma.T)
    w, U = np.linalg.eigh(sigma)
    scale = max(1.0, float(w[-1]))
    if w[0] <= 1e-12 * scale:
        raise ValueError(
            f'the target AOs are linearly dependent in the overlap metric '
            f'(smallest eigenvalue {w[0]:.3e} of {w[-1]:.3e}); remove the '
            f'duplicate target functions')
    return X, (U * w ** -0.5) @ U.T


def _whitened_overlap_svd(S, C_block, target):
    """Overlap eigenvalues (descending) and rotation of one MO block.

    ``Y = sigma^-1/2 X^T S C_block`` is the target overlap whitened by
    the target-space metric; its SVD gives ``lambda = s^2`` (the fraction
    of target character, in [0, 1]) and the rotation ``V`` whose columns
    are the rotated orbitals in ``C_block @ V``.
    """
    X, sigma_inv_sqrt = _target_metric(S, target)
    Y = sigma_inv_sqrt @ (X.T @ S @ C_block)
    # The block usually has more orbitals than the target has functions
    # (a 7-AO f target against a large virtual block).  A thin SVD then
    # returns only min(ntarget, nblock) singular values, and its V is not
    # the square rotation the block needs; the full SVD fills the null
    # space, whose directions carry zero target character.
    nb = C_block.shape[1]
    if Y.shape[0] >= nb:
        _, s, Vt = np.linalg.svd(Y, full_matrices=False)
        lam = s * s
    else:
        _, s, Vt = np.linalg.svd(Y, full_matrices=True)
        lam = np.zeros(nb)
        lam[:s.size] = s * s
    lam = np.clip(lam, 0.0, 1.0)
    return lam, Vt.T


def target_electron_count(S, C, occ, target):
    """Number of mean-field electrons in the target AO subspace.

    With the AO density ``D = C diag(occ) C^T`` the count is the trace of
    the density projected onto the target subspace in the target's own
    metric,

        N_target = Tr(sigma^-1/2 X^T S D S X sigma^-1/2).

    For a Valence-F target this is the mean-field f occupation: about 14
    for a filled 4f shell, near zero for an empty one, and a non-integer
    value in between when the SCF solution mixes the target with ligand
    orbitals.  The number is a diagnostic, not an integer count -- the
    electron count of a CAS is the caller's choice.

    Parameters
    ----------
    S : array_like
        AO overlap, shape ``(nao, nao)``.
    C : array_like
        MO coefficients, shape ``(nao, nmo)``.
    occ : array_like
        MO occupations, one per MO.
    target : sequence of int
        Target AO indices.

    Returns
    -------
    float
    """
    S = np.asarray(S, dtype=float)
    C = np.asarray(C, dtype=float)
    occ = np.asarray(occ, dtype=float).reshape(-1)
    if C.ndim != 2 or S.shape != (C.shape[0], C.shape[0]):
        raise ValueError(f'C {C.shape} and S {S.shape} do not describe '
                         f'the same AO space')
    if occ.size != C.shape[1]:
        raise ValueError(f'occ has {occ.size} entries for {C.shape[1]} '
                         f'orbitals')
    target = np.array(sorted({int(i) for i in target}), dtype=int)
    if target.size == 0 or target[0] < 0 or target[-1] >= C.shape[0]:
        raise ValueError(f'target AOs {target.tolist()} outside the '
                         f'{C.shape[0]}-AO space')
    X, sigma_inv_sqrt = _target_metric(S, target)
    density = (C * occ) @ C.T
    projected = (sigma_inv_sqrt @ (X.T @ S @ density @ S @ X)
                 @ sigma_inv_sqrt)
    return float(np.trace(projected))


def _classify_occupations(occ, tol=1e-4):
    """Split occupations into doubly occupied, singly occupied, empty.

    AVAS is a mean-field construction: an occupation vector that is not
    within `tol` of 0, 1 or 2 belongs to a correlated density and is
    refused rather than rotated.
    """
    occ = np.asarray(occ, dtype=float).reshape(-1)
    if not np.isfinite(occ).all():
        raise ValueError('the occupation vector contains non-finite values')
    nearest = np.rint(occ)
    off = np.abs(occ - nearest) > tol
    if off.any():
        bad = [(int(i), float(occ[i])) for i in np.where(off)[0][:8]]
        raise ValueError(
            f'AVAS needs mean-field occupations within {tol:g} of 0, 1 or '
            f'2; these entries are fractional: {bad}')
    return (np.where(nearest == 2)[0], np.where(nearest == 1)[0],
            np.where(nearest == 0)[0])


def _default_occupations(mol, nmo):
    """Closed/open-shell aufbau occupations from ``mol.nelectron``/``spin``.

    The singly occupied orbitals are assumed to follow the doubly
    occupied ones; that is the ordering a restricted open-shell
    calculation produces for the targets this default serves.  A caller
    whose solver ordered them differently must pass ``occ``.
    """
    nelec = int(mol.nelectron)
    spin = int(getattr(mol, 'spin', 0))
    if (nelec - spin) % 2:
        raise ValueError(f'nelectron {nelec} and spin {spin} are '
                         f'inconsistent')
    n_double = (nelec - spin) // 2
    n_single = spin
    if n_double + n_single > nmo:
        raise ValueError(f'{nelec} electrons with spin {spin} do not fit '
                         f'in {nmo} orbitals')
    occ = np.zeros(nmo)
    occ[:n_double] = 2.0
    occ[n_double:n_double + n_single] = 1.0
    return occ


def occupation_diagnostics(occupations, low=AVAS_OCCUPATION_LOW,
                           high=AVAS_OCCUPATION_HIGH):
    """Check active-orbital occupations against the [low, high] window.

    The window is the ORCA CASSCF manual's guidance (3.13.7): an active
    natural occupation in [0.02, 1.98] carries real correlation; values
    near 0 or 2 mean the orbital is nearly uncorrelated and probably does
    not need to be active.  Occupations outside [0, 2] by more than
    rounding are reported separately as invalid input.

    Parameters
    ----------
    occupations : array_like
        Natural occupations of the active orbitals, one per orbital.
    low, high : float
        Window bounds.

    Returns
    -------
    OccupationDiagnostics
    """
    occ = np.asarray(occupations, dtype=float).reshape(-1)
    if occ.size == 0:
        raise ValueError('cannot diagnose an empty occupation vector')
    if not np.isfinite(occ).all():
        raise ValueError('the occupation vector contains non-finite values')
    if not 0.0 <= low < high <= 2.0:
        raise ValueError(f'occupation window [{low}, {high}] is not inside '
                         f'[0, 2]')
    weak = [int(i) for i in np.where(occ < low)[0]]
    strong = [int(i) for i in np.where(occ > high)[0]]
    invalid = [int(i) for i in np.where((occ < -1e-8) | (occ > 2.0 + 1e-8))[0]]
    evidence = [
        f'occupation window [{low}, {high}] over {occ.size} orbital(s), '
        f'range {float(occ.min()):.4f} to {float(occ.max()):.4f}',
    ]
    if weak:
        evidence.append(f'{len(weak)} nearly empty orbital(s) {weak}: '
                        f'occupation below {low}; the correlation is weak '
                        f'enough that the orbital can leave the active '
                        f'space')
    if strong:
        evidence.append(f'{len(strong)} nearly doubly occupied orbital(s) '
                        f'{strong}: occupation above {high}; the '
                        f'correlation is weak enough that the orbital can '
                        f'leave the active space')
    if invalid:
        evidence.append(f'{len(invalid)} occupation(s) outside [0, 2] '
                        f'{invalid}: the density is not physical')
    return OccupationDiagnostics(occ, low, high, weak, strong, invalid,
                                 not weak and not strong, evidence)


def orbital_gap_diagnostics(energies, core, active, external,
                            gap_min=AVAS_GAP_MIN):
    """Act-Int and Ext-Act gaps of an active space.

    ``Act-Int = min(active) - max(core)`` and
    ``Ext-Act = min(external) - max(active)`` in the supplied orbital
    energies.  A side with no orbitals (no core, no external) has no gap
    and does not fail the check.  The energies must be comparable: for
    rotated AVAS orbitals they are diagonal Fock expectations, not
    canonical eigenvalues.

    Parameters
    ----------
    energies : array_like
        One energy per orbital, in the same index space as the lists.
    core, active, external : sequence of int
        Disjoint orbital index lists.
    gap_min : float
        Warning line in Eh; the manual's "about 0.2 Eh".

    Returns
    -------
    GapDiagnostics
    """
    e = np.asarray(energies, dtype=float).reshape(-1)
    if not np.isfinite(e).all():
        raise ValueError('the orbital energies contain non-finite values')
    lists = {'core': [int(i) for i in core],
             'active': [int(i) for i in active],
             'external': [int(i) for i in external]}
    seen = set()
    for name, idx in lists.items():
        if any(i < 0 or i >= e.size for i in idx):
            raise ValueError(f'{name} indices {idx} outside the '
                             f'{e.size}-orbital energy vector')
        overlap = seen.intersection(idx)
        if overlap:
            raise ValueError(f'{name} repeats indices already assigned: '
                             f'{sorted(overlap)}')
        seen.update(idx)
    core_i, active_i, external_i = (lists['core'], lists['active'],
                                    lists['external'])
    act_int = (float(e[active_i].min() - e[core_i].max())
               if active_i and core_i else None)
    ext_act = (float(e[external_i].min() - e[active_i].max())
               if active_i and external_i else None)
    evidence, warnings = [], []
    for name, value in (('Act-Int', act_int), ('Ext-Act', ext_act)):
        if value is None:
            evidence.append(f'{name}: not applicable (one side is empty)')
        elif value < gap_min:
            warnings.append(f'{name} {value:.4f} Eh is below the '
                            f'{gap_min:g} Eh separation line; the active '
                            f'space touches the neighbouring space in '
                            f'orbital energy')
            evidence.append(f'{name} {value:.4f} Eh (below {gap_min:g})')
        else:
            evidence.append(f'{name} {value:.4f} Eh (at or above '
                            f'{gap_min:g})')
    evidence.append(f'gap line {gap_min:g} Eh is the ORCA manual\'s '
                    f'guidance, a heuristic for the canonical ordering; '
                    f'the rotated AVAS orbitals use diagonal Fock '
                    f'expectations instead')
    return GapDiagnostics(act_int, ext_act, gap_min,
                          not warnings, e, evidence, warnings)


def avas_selection(mol, C, target_AO=None, nact=None, nele=None, *,
                   occ=None, S=None, labels=None, symbols=None,
                   threshold=AVAS_DEFAULT_THRESHOLD,
                   rule=None, elements=None, shells=None,
                   mo_energy=None, active_occupations=None,
                   gap_min=AVAS_GAP_MIN,
                   occupation_window=(AVAS_OCCUPATION_LOW,
                                      AVAS_OCCUPATION_HIGH)):
    """Select an active space by projecting target AOs onto the MO blocks.

    Parameters
    ----------
    mol : pyscf-like
        Provides ``nelectron``, ``spin``, ``ao_labels()``, and
        ``intor('int1e_ovlp')`` unless `S` is supplied.
    C : array_like
        MO coefficients, shape ``(nao, nmo)``.
    target_AO : sequence of int, str, or None
        Explicit target AO indices, or a rule name (``'valence-f'``,
        ``'double-d'``), or None to resolve `rule`/`shells`.
    nact, nele : int, optional
        Active orbital and electron counts.  Give both or neither.  Both
        given is the ``nel``/``norb`` mode: the SVD threshold is ignored
        and the leading slots are taken, ``nele`` split as ``nele/2``
        occupied-derived orbitals plus the rest virtual-derived (singly
        occupied orbitals are always active).
    occ : array_like, optional
        Mean-field occupations, one per MO.  Default: aufbau occupations
        from ``mol.nelectron`` and ``mol.spin``.
    S : array_like, optional
        AO overlap; default ``mol.intor('int1e_ovlp')``.
    labels, symbols : optional
        Overrides for the molecule's AO labels and atom symbols, for
        target-rule resolution without PySCF.
    threshold : float
        Overlap eigenvalue above which a rotated slot is active in
        threshold mode.  The default is the source's 0.05.
    rule, elements, shells
        Forwarded to :func:`target_ao_indices` when `target_AO` is None.
    mo_energy : array_like, optional
        Canonical MO energies; enables the Act-Int/Ext-Act diagnostics.
    active_occupations : array_like, optional
        Correlated natural occupations of the selected active orbitals;
        enables the 0.02-1.98 check.  Without them that check is not run,
        because the mean-field occupations are 0/1/2 by construction.
    gap_min : float
        Gap warning line in Eh.
    occupation_window : (float, float)
        Natural-occupation window.

    Returns
    -------
    (list[int], AvasDiagnostics)
        The active column indices of ``diagnostics.C_rotated`` and the
        full record.  ``ncore`` in the record is the number of doubly
        occupied orbitals left outside the active space.

    Raises
    ------
    ValueError
        For an inconsistent ``nact``/``nele`` pair, fractional
        occupations, a linearly dependent target set, or a target rule
        that matches nothing.
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2:
        raise ValueError(f'C must be two-dimensional, got shape {C.shape}')
    nao, nmo = C.shape
    if S is None:
        S = np.asarray(mol.intor('int1e_ovlp'), dtype=float)
    else:
        S = np.asarray(S, dtype=float)
    if S.shape != (nao, nao):
        raise ValueError(f'S shape {S.shape} does not match the {nao} AO '
                         f'rows of C')
    if occ is None:
        occ = _default_occupations(mol, nmo)
    else:
        occ = np.asarray(occ, dtype=float).reshape(-1)
    if occ.size != nmo:
        raise ValueError(f'occ has {occ.size} entries for {nmo} orbitals')
    double, single, empty = _classify_occupations(occ)
    if (nact is None) != (nele is None):
        raise ValueError('nact and nele must be given together (the AVAS '
                         'nel/norb mode); give both or neither')
    if nact is None and not 0.0 < threshold < 1.0:
        raise ValueError(f'threshold {threshold} outside (0, 1)')

    warnings, evidence = [], []
    if isinstance(target_AO, str):
        if rule is not None:
            raise ValueError('target_AO names a rule and rule=... is also '
                             'given; give one')
        rule = target_AO
        target_AO = None
    if target_AO is None:
        subspace = target_ao_indices(mol, rule=rule, elements=elements,
                                     shells=shells, labels=labels,
                                     symbols=symbols)
        target = subspace.indices
        target_rule = subspace.rule
        evidence.extend(subspace.evidence)
        warnings.extend(subspace.warnings)
    else:
        if rule is not None or elements is not None or shells is not None:
            raise ValueError('target_AO is an explicit index list; do not '
                             'also give rule=..., elements=... or '
                             'shells=...')
        target = np.array(sorted({int(i) for i in target_AO}), dtype=int)
        if target.size == 0 or target[0] < 0 or target[-1] >= nao:
            raise ValueError(f'target AOs {target.tolist()} outside the '
                             f'{nao}-AO space')
        target_rule = 'explicit'
        evidence.append(f'explicit target: {target.size} AO(s)')
    target_elec = target_electron_count(S, C, occ, target)
    evidence.append(
        f'target subspace holds {target_elec:.4f} mean-field electron(s)')

    lam_d, V_d = (_whitened_overlap_svd(S, C[:, double], target)
                  if double.size else (np.zeros(0), np.zeros((0, 0))))
    lam_v, V_v = (_whitened_overlap_svd(S, C[:, empty], target)
                  if empty.size else (np.zeros(0), np.zeros((0, 0))))

    if nact is None:
        keep_d = lam_d > threshold
        keep_v = lam_v > threshold
        n_d = int(keep_d.sum())
        n_v = int(keep_v.sum())
        evidence.append(
            f'threshold mode: {n_d} of {double.size} occupied-derived '
            f'slot(s) and {n_v} of {empty.size} virtual-derived slot(s) '
            f'above {threshold:g}')
    else:
        nact = int(nact)
        nele = int(nele)
        n_s = single.size
        if nele < n_s:
            raise ValueError(f'nele {nele} is smaller than the {n_s} '
                             f'electron(s) in singly occupied orbitals, '
                             f'which are always active')
        if (nele - n_s) % 2:
            raise ValueError(f'nele {nele} minus {n_s} singly occupied '
                             f'electron(s) is odd; the remaining electrons '
                             f'must come in doubly occupied pairs')
        n_d = (nele - n_s) // 2
        n_v = nact - n_s - n_d
        if not 0 <= n_d <= double.size:
            raise ValueError(f'nact/nele want {n_d} occupied-derived '
                             f'orbitals, the occupied block has '
                             f'{double.size}')
        if not 0 <= n_v <= empty.size:
            raise ValueError(f'nact/nele want {n_v} virtual-derived '
                             f'orbitals, the virtual block has {empty.size}')
        keep_d = np.zeros(double.size, dtype=bool)
        keep_d[:n_d] = True
        keep_v = np.zeros(empty.size, dtype=bool)
        keep_v[:n_v] = True
        evidence.append(
            f'nact/nele mode: {n_d} occupied-derived, {n_v} '
            f'virtual-derived, {n_s} singly occupied; threshold ignored')

    C_rot = np.array(C, dtype=float, copy=True)
    if double.size:
        C_rot[:, double] = C[:, double] @ V_d
    if empty.size:
        C_rot[:, empty] = C[:, empty] @ V_v
    active = sorted(double[keep_d].tolist() + empty[keep_v].tolist()
                    + single.tolist())
    ncore = int(double.size - n_d)
    n_external = int(empty.size - n_v)
    n_ele = int(2 * n_d + single.size)
    if nact is None:
        nact = len(active)
    evidence.append(
        f'selection: {len(active)} active orbital(s) {active}; '
        f'{ncore} inactive core, {n_external} external; {n_ele} active '
        f'electron(s)')
    if len(active) != nact:
        raise ValueError(f'internal inconsistency: {len(active)} active '
                         f'orbitals for nact={nact}')
    if nele is not None and n_ele != int(nele):
        raise ValueError(f'internal inconsistency: {n_ele} active '
                         f'electrons for nele={nele}')

    gap = None
    if mo_energy is not None:
        eps = np.asarray(mo_energy, dtype=float).reshape(-1)
        if eps.size != nmo:
            raise ValueError(f'mo_energy has {eps.size} entries for {nmo} '
                             f'orbitals')
        eps_rot = eps.copy()
        if double.size:
            eps_rot[double] = np.einsum('ki,k,ki->i', V_d, eps[double], V_d)
        if empty.size:
            eps_rot[empty] = np.einsum('ki,k,ki->i', V_v, eps[empty], V_v)
        core_idx = double[~keep_d].tolist()
        external_idx = empty[~keep_v].tolist()
        gap = orbital_gap_diagnostics(eps_rot, core_idx, active,
                                      external_idx, gap_min)
        evidence.extend(gap.evidence)
        warnings.extend(gap.warnings)
    else:
        evidence.append('no mo_energy supplied: the Act-Int/Ext-Act gap '
                        'diagnostic was not run')

    occupation = None
    if active_occupations is not None:
        occupation = occupation_diagnostics(active_occupations,
                                            *occupation_window)
        if occupation.occupations.size != len(active):
            raise ValueError(
                f'active_occupations has '
                f'{occupation.occupations.size} entries for {len(active)} '
                f'active orbitals')
        evidence.extend(occupation.evidence)
    else:
        evidence.append(
            f'occupation window {occupation_window[0]:g}-'
            f'{occupation_window[1]:g} applies to correlated natural '
            f'occupations; pass active_occupations=... after a CASSCF to '
            f'run it (the mean-field occupations are 0/1/2)')

    diag = AvasDiagnostics(
        active=active, nact=int(nact), nele=int(n_ele), ncore=ncore,
        nexternal=n_external, target_aos=target, target_rule=target_rule,
        target_electrons=target_elec, C_rotated=C_rot,
        blocks={'doubly_occupied': double.tolist(),
                'singly_occupied': single.tolist(),
                'empty': empty.tolist()},
        overlap_eigenvalues={'doubly_occupied': lam_d,
                             'empty': lam_v},
        singular_values={'doubly_occupied': np.sqrt(lam_d),
                         'empty': np.sqrt(lam_v)},
        scf_occupations=np.asarray(occ, dtype=float),
        occupation=occupation, gap=gap, evidence=evidence,
        warnings=warnings)
    return active, diag
