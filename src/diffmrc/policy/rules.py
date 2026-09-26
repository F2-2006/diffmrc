"""Chemical rules that propose an active space from molecular structure.

The rules read the occupation vector and the orbital composition. They
never assume a particular orbital ordering: an index-based rule depends
on which SCF solution was found, and a different initial guess reorders
the orbitals, so such a rule silently selects a different space on the
next run. This project has already been bitten by that.

Every rule can refuse. A refusal is a result, not an error: it says the
structure does not match what the rule knows how to handle, and the
caller is expected to escalate or to report that it cannot decide.

What the composition weights can and cannot do
---------------------------------------------

The weights are useful *within a set of candidates*, for labelling them,
and only if the right quantity is compared. Both simpler tests fail on
the measured data, and the way they fail is worth recording.

An *absolute* weight cannot be a threshold: europium's ANO basis spreads
f character over seven contractions, so an overlap-weighted "europium f"
sum exceeds 0.5 for 48 of the 76 orbitals of EuF.

A *ratio* between two shells fails differently. One orbital of EuF has f
weight 0.0039 and p weight 0.0000, giving an f fraction of exactly 1.0
-- a confident label computed from two near-zero numbers.

Taking the shell with the largest weight avoids both. The eight singly
occupied orbitals of EuF's molecular end (measured 2026-09-22) come out
as

    orbitals 32, 33, 34, 35, 37, 38 :  Eu f, 0.997 to 1.000
    orbital  36                     :  Eu s, 0.866
    orbital  39                     :  Eu d, 0.997

which is the chemistry: six f orbitals, the 6s, and the 5d.

Two consequences follow, and both contradict what this project assumed
before the measurement. First, the molecular end has *six* f orbitals
rather than the seven a free europium atom shows -- the seventh f
electron is elsewhere, and the molecular orbital list simply does not
have seven orbitals that are mostly f. Second, the occupation pattern
cannot identify any of this: all eight orbitals are singly occupied.

So a rule for the molecular end has to label by dominant composition and
then accept whatever count comes out, rather than assume a count and
hunt for orbitals to fill it. The atom-end rules below still work on
their own terms.

A second failure, measured the same day, has the same shape. Ranking
*all* orbitals by ligand-p weight and taking the top three puts three
unoccupied antibonding orbitals into the active space, because the
weights of those are higher than the weights of the occupied bonding
pair. On EuF the top three come out as orbitals 175, 58 and 59, all
empty, while the occupied fluorine 2p orbitals sit at ranks seven and
eight. **Ranking without a partition by occupation is ranking the wrong
set.** The molecular-end rule therefore partitions first and ranks only
within the occupied block.
"""
from dataclasses import dataclass, field

import numpy as np

from .composition import element_shell_weights
from .occupation import select_valence, split_by_occupation

__all__ = ['SHELL_RULES', 'Candidate', 'SelectionPlan',
           'candidate_from_elements', 'plan_from_elements',
           'dominant_shell', 'molecular_end_candidate',
           'DipoleChoice', 'dipole_rule']

# The shells the molecular-end rule inspects, in the order they are
# offered to the labelling. The order does not affect the result -- the
# largest weight wins -- but fixing it keeps the evidence reproducible.
_ANGULAR_MOMENTA = ('s', 'p', 'd', 'f')

def dominant_shell(candidates, weights_by_shell, min_weight=0.5):
    """Label each candidate by the shell it mostly belongs to.

    Two simpler tests both fail on real data, and the way they fail is
    worth recording.

    An *absolute* weight cannot be a threshold: a lanthanide's ANO basis
    spreads f character over seven contractions, so an overlap-weighted
    "europium f" sum exceeds 0.5 for most orbitals of EuF.

    A *ratio* between two shells fails differently. On EuF's molecular
    end one orbital has f weight 0.0039 and p weight 0.0000, giving an
    f fraction of exactly 1.0 and a confident but meaningless label --
    the ratio of two near-zero numbers.

    Taking the shell with the largest weight avoids both. The same eight
    orbitals come out as six with f at 0.997 or above, one with s at
    0.866, and one with d at 0.997: clean, and matching what the
    chemistry says they are.

    Parameters
    ----------
    candidates : sequence of int
        Orbital indices to label.
    weights_by_shell : mapping
        Shell name to per-orbital weight array.
    min_weight : float
        A candidate whose largest weight falls below this is labelled
        None -- it is mixed, and calling it one thing or another would
        be arbitrary.

    Returns
    -------
    dict
        Candidate index to shell name, or to None when no shell
        dominates. A caller must handle the None entries rather than
        assuming every candidate found a home.
    """
    candidates = [int(i) for i in candidates]
    out = {}
    for i in candidates:
        best_shell, best = None, 0.0
        for name, weights in weights_by_shell.items():
            w = float(np.asarray(weights, dtype=float)[i])
            if w > best:
                best_shell, best = name, w
        out[i] = best_shell if best >= min_weight else None
    return out

# The lanthanides, whose half-filled f shell is what the f_shell rule
# looks for. Promethium is included: it is radioactive and rarely used,
# but a user who supplies it deserves a rule rather than a silent miss.
LANTHANIDES = (
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
    'Er', 'Tm', 'Yb', 'Lu',
)

# Elements the main-group valence rule can handle: the p block plus the
# light main-group elements whose valence shell is an s shell.
MAIN_GROUP = (
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
)


@dataclass
class SelectionPlan:
    """Which rule to apply, decided from the elements present.

    The ``end`` field is the decision that matters most to a caller: the
    two ends need different rules, and feeding one the other's rule is
    how the earlier attempts went wrong. A bare lanthanide atom shows
    seven singly occupied f orbitals; a lanthanide inside a molecule
    shows six, with its 6s and 5d taking two of the eight open-shell
    slots. ``center_n`` and ``ligand_n`` are the atom end's counts and
    are not a prediction for the molecular end -- there the count comes
    out of ``molecular_end_candidate``'s labelling.
    """

    center: str = ''
    ligand: str = ''
    end: str = 'atom'
    center_n: int = 7
    ligand_n: int = 3
    evidence: list = field(default_factory=list)

    @property
    def has_ligand(self):
        return bool(self.ligand)

    @property
    def is_molecular(self):
        return self.end == 'molecular'


def plan_from_elements(symbols):
    """Decide which selection rule the element list calls for.

    The command line receives an xyz file and nothing else, so the
    decision of *which* chemical rule applies has to be made from the
    elements present. Doing it here keeps the decision testable on a
    plain list of symbols.

    The rule is deliberately narrow: one lanthanide centre, optionally
    with one main-group ligand. Anything else returns None, because a
    rule that guesses at a system it was not written for is worse than
    no rule at all.

    Parameters
    ----------
    symbols : sequence of str
        Element symbols, one per atom.

    Returns
    -------
    SelectionPlan or None
        None when the composition does not match a rule this package
        knows. A caller must treat None as a refusal.
    """
    symbols = [str(s).capitalize() for s in symbols]
    if not symbols:
        return None

    centres = sorted({s for s in symbols if s in LANTHANIDES})
    if len(centres) != 1:
        # Zero centres is an ordinary main-group molecule, which this
        # package has no rule for; two centres is a different problem
        # again, and the single-centre rule does not describe it.
        return None
    center = centres[0]

    others = sorted({s for s in symbols if s != center})
    if not others:
        return SelectionPlan(center=center, center_n=7, ligand='',
                             ligand_n=0, end='atom',
                             evidence=[f'{center}: bare lanthanide atom'])
    if len(others) > 1:
        return None
    ligand = others[0]
    if ligand not in MAIN_GROUP:
        return None

    return SelectionPlan(
        center=center, ligand=ligand, end='molecular',
        center_n=7, ligand_n=3,
        evidence=[f'centre {center} with ligand {ligand}: the molecular '
                  f'end, where the centre shows six f orbitals rather than '
                  f'the atom\'s seven'],
    )

SHELL_RULES = {
    'lanthanide_f': dict(
        kind='f_shell',
        take='singly_occupied',
        n_orbitals=7,
        description='half-filled f shell of a lanthanide centre',
    ),
    'main_group_p': dict(
        kind='valence_p',
        take='valence_shell',
        n_orbitals=3,
        n_electrons=5,
        description='valence p shell of a main-group atom',
    ),
}


@dataclass
class Candidate:
    """A proposed active space together with the evidence for it."""

    active: list = field(default_factory=list)
    per_element: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)


def candidate_from_elements(mol, C, occ, center, ligand=None,
                            center_n=7, ligand_n=3):
    """Propose an active space from occupation structure and composition.

    A lanthanide centre contributes its half-filled f shell, identified
    as the singly occupied orbitals. A main-group ligand contributes its
    valence shell. Nothing is matched by index or by principal quantum
    number.

    Parameters
    ----------
    mol : pyscf-like
        Provides ``ao_labels()`` and ``intor('int1e_ovlp')``.
    C : array_like
        Molecular orbital coefficients.
    occ : array_like
        Occupation numbers.
    center : str
        Symbol of the metal centre.
    ligand : str, optional
        Symbol of a main-group ligand, if present. When omitted, the
        centre alone is treated, and the ligand is ignored.
    center_n : int
        Number of orbitals the centre's shell should contribute.
    ligand_n : int
        Number of orbitals the ligand's valence shell should contribute.

    Returns
    -------
    Candidate or None
        None when the structure does not match the rule. A caller must
        treat None as a refusal, not as an empty selection.
    """
    occ = np.asarray(occ, dtype=float)
    partition = split_by_occupation(occ)
    evidence = []

    if ligand is None:
        if len(partition.singly) != center_n:
            return None
        evidence.append(
            f'{center}: {len(partition.singly)} singly occupied orbitals '
            f'match the {SHELL_RULES["lanthanide_f"]["description"]}'
        )
        w = element_shell_weights(mol, C, center, 'f')
        if w.n_ao:
            selected = w.weights[partition.singly]
            evidence.append(
                f'{center}: f weight over the selection ranges '
                f'{float(np.min(selected)):.3f} to '
                f'{float(np.max(selected)):.3f} ({w.n_ao} f functions)'
            )
        else:
            evidence.append(f'{center}: no f functions found in the basis')
        return Candidate(
            active=sorted(partition.singly),
            per_element={center: dict(n_orbitals=len(partition.singly),
                                      n_ao_f=w.n_ao)},
            evidence=evidence,
        )

    picked = select_valence(occ, ligand_n)
    if picked is None:
        return None
    n_doubly = sum(1 for i in picked if occ[i] > 1.5)
    evidence.append(
        f'{ligand}: valence shell from {len(picked) - n_doubly} singly and '
        f'{n_doubly} doubly occupied orbitals'
    )
    w = element_shell_weights(mol, C, ligand, 'p')
    if w.n_ao == 0:
        evidence.append(f'{ligand}: no p functions found in the basis')
    else:
        selected = w.weights[picked]
        evidence.append(
            f'{ligand}: p weight over the selection ranges '
            f'{float(np.min(selected)):.3f} to '
            f'{float(np.max(selected)):.3f} ({w.n_ao} p functions)'
        )
    return Candidate(
        active=sorted(picked),
        per_element={ligand: dict(n_orbitals=len(picked), n_ao_p=w.n_ao)},
        evidence=evidence,
    )


def _shell_weight_arrays(mol, C, elements):
    """Overlap-weighted per-orbital weights for every element and shell.

    Only shells the basis actually contains are included, so an absent
    shell cannot contribute a zero-weight array that looks like a real
    measurement of zero.
    """
    out = {}
    for element in elements:
        for letter in _ANGULAR_MOMENTA:
            w = element_shell_weights(mol, C, element, letter)
            if w.n_ao:
                out[f'{element}-{letter}'] = w.weights
    return out


def molecular_end_candidate(mol, C, occ, center, ligand, min_weight=0.5,
                            ligand_doubly=None):
    """Propose an active space for a lanthanide centre inside a molecule.

    This is the rule the molecular end needs and the atom-end rules do
    not provide. Two things separate the two cases, and both were
    measured on EuF rather than assumed.

    The occupation pattern does not identify the centre's shell. At the
    molecular end all eight open-shell orbitals of EuF are singly
    occupied, so counting them gives eight and says nothing about which
    are f. Only the composition separates them: six are Eu f, one is the
    Eu 6s, one is the Eu 5d.

    The count is *six* f orbitals, not the seven a free europium atom
    shows. A rule that assumes seven and then hunts for orbitals to fill
    the count has to drop one of the eight, and which one it drops is
    decided by an arbitrary tie-break. The rule here does the opposite:
    it labels each orbital by its dominant composition and accepts
    whatever count comes out.

    Parameters
    ----------
    mol : pyscf-like
        Provides ``ao_labels()`` and ``intor('int1e_ovlp')``.
    C : array_like
        Molecular orbital coefficients.
    occ : array_like
        Occupation numbers.
    center : str
        Symbol of the metal centre.
    ligand : str
        Symbol of the main-group ligand.
    min_weight : float
        Smallest dominant weight that counts as an identification. An
        orbital whose largest weight falls below it is mixed, and the
        rule refuses rather than guess.
    ligand_doubly : int, optional
        How many of the ligand's doubly occupied valence orbitals to
        take, strongest ligand character first. ``None`` takes all of
        them, which is the labelling-driven answer; an integer asks for
        a smaller space explicitly.

    Returns
    -------
    Candidate or None
        None when the structure does not match: no open shell, no
        ligand-dominated doubly occupied orbital, more orbitals requested
        than the labelling found, or a mixed orbital that could not be
        labelled. A caller must treat None as a refusal.
    """
    occ = np.asarray(occ, dtype=float)
    partition = split_by_occupation(occ)
    if not partition.singly or not partition.doubly:
        return None

    shells = _shell_weight_arrays(mol, C, (center, ligand))
    if not shells:
        return None

    candidates = partition.singly + partition.doubly
    labels = dominant_shell(candidates, shells, min_weight=min_weight)
    unlabelled = sorted(i for i in candidates if labels[i] is None)
    if unlabelled:
        # One mixed orbital is enough to make the whole count unreliable,
        # and the count is the entire point of this rule.
        return None

    ligand_p = f'{ligand}-p'
    if ligand_p not in shells:
        return None
    found = [i for i in partition.doubly if labels[i] == ligand_p]
    if not found:
        return None

    if ligand_doubly is None:
        donor = found
    elif ligand_doubly < 0 or ligand_doubly > len(found):
        return None
    else:
        donor = sorted(found,
                       key=lambda i: -float(shells[ligand_p][i]))[:ligand_doubly]

    counts = {}
    for i in partition.singly:
        counts[labels[i]] = counts.get(labels[i], 0) + 1
    summed = ', '.join(f'{name} x{n}' for name, n in sorted(counts.items()))
    evidence = [
        f'{center}: {len(partition.singly)} singly occupied orbitals, '
        f'labelled by dominant composition as {summed}'
    ]
    if len(donor) == len(found):
        evidence.append(
            f'{ligand}: all {len(found)} doubly occupied orbitals whose '
            f'dominant composition is {ligand} p'
        )
    else:
        evidence.append(
            f'{ligand}: {len(donor)} of {len(found)} doubly occupied '
            f'{ligand} p orbitals, strongest character first'
        )

    return Candidate(
        active=sorted(partition.singly + donor),
        per_element={center: dict(n_orbitals=len(partition.singly)),
                     ligand: dict(n_orbitals=len(donor))},
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Property-distance rules.  Their input is a computed property of each
# candidate space rather than the molecular structure, so they read the
# arrays a caller has already produced -- the dipole values from
# driver.properties, in the rule below.

@dataclass
class DipoleChoice:
    """The winning candidate, or the refusal and its reason."""

    winner: object = None
    distances: dict = field(default_factory=dict)
    variant: str = 'gdm'
    evidence: list = field(default_factory=list)


def dipole_rule(dipoles, reference, variant='gdm', min_margin=None):
    """Pick the candidate whose dipole is closest to an external reference.

    The protocol is Kaufold and Dong's (JCTC 2023; the 2026 update moves
    the selection runs to CASCI orbitals and gives the projection variant
    the plain-distance form): compute the dipole of each candidate space
    and take the one nearest an outside reference -- experiment, or a
    calculation at a level the candidate is meant to reproduce.  Two
    boundaries belong to the caller, not this function: the reference
    must come from outside the candidate set, and charged systems are
    outside the protocol because the dipole is origin dependent.

    This rule consumes only dipole *values*.  The chain that produces
    them (``driver.properties``) is differentiable in the orbital
    parameters, so a caller can additionally read the sensitivity of a
    near-tie to re-optimization -- the one quantity the original protocol
    does not have -- but that diagnostic is deliberately not part of the
    decision here.

    Parameters
    ----------
    dipoles : mapping
        Candidate label to a length-3 dipole vector, any unit as long as
        all share it. Labels are opaque.
    reference : array_like
        The external reference dipole, length 3, same unit.
    variant : str
        ``'gdm'``  D_n = ||mu_n - mu_ref||;
        ``'vgdm'`` D_n = |mu_n . mu_hat_ref - ||mu_ref|||, the 2023
        projection form, which measures only the component along the
        reference direction.
    min_margin : float, optional
        When given, a winner that beats the runner-up by less than this
        margin is refused -- the distances are inside their own noise.
        ``None`` keeps the original protocol's plain argmin.

    Returns
    -------
    DipoleChoice
        ``winner`` is None when nothing could be decided; the distances
        computed so far and an evidence trail are always present.
    """
    if variant not in ('gdm', 'vgdm'):
        raise ValueError(f'unknown variant {variant!r}; expected gdm or '
                         f'vgdm')
    ref = np.asarray(reference, dtype=float).reshape(-1)
    if ref.shape != (3,):
        raise ValueError(f'reference must have three components, got '
                         f'{ref.shape}')
    if not np.isfinite(ref).all() or not np.any(ref):
        return DipoleChoice(
            None, {}, variant,
            [f'reference dipole {ref.tolist()} is zero or non-finite; '
             f'the distance to it is not meaningful'])
    ref_norm = float(np.linalg.norm(ref))

    distances = {}
    for label, mu in dict(dipoles).items():
        mu = np.asarray(mu, dtype=float).reshape(-1)
        if mu.shape != (3,) or not np.isfinite(mu).all():
            return DipoleChoice(
                None, distances, variant,
                [f'{label}: malformed dipole {mu.tolist()}; refusing the '
                 f'whole selection rather than sorting a NaN'])
        if variant == 'gdm':
            distances[label] = float(np.linalg.norm(mu - ref))
        else:
            distances[label] = abs(float(mu @ (ref / ref_norm))
                                   - ref_norm)
    if not distances:
        return DipoleChoice(None, {}, variant,
                            ['no candidate dipoles supplied'])

    order = sorted(distances, key=distances.get)
    winner = order[0]
    evidence = [f'{label}: D = {distances[label]:.6g}' for label in order]
    if (min_margin is not None and len(order) > 1
            and distances[order[1]] - distances[winner] < min_margin):
        margin = distances[order[1]] - distances[winner]
        evidence.append(
            f'margin {margin:.6g} between {winner} and {order[1]} is '
            f'below {min_margin:g}; refusing to choose')
        return DipoleChoice(None, distances, variant, evidence)
    evidence.append(f'winner {winner} with D = {distances[winner]:.6g}')
    return DipoleChoice(winner, distances, variant, evidence)
