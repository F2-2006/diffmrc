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

The weights are useful *within a set of candidates*, for ranking them
against each other. They are not useful as an absolute test, and the
measurements show why: europium's ANO basis carries 49 f-type functions
across seven contractions, so an overlap-weighted sum over "europium f"
is nonzero for 48 of the 76 orbitals of EuF. Taken as a threshold that
is meaningless.

Ranked inside the eight singly occupied orbitals of EuF, the same
quantity separates cleanly. Measured on the molecular end (t176,
2026-09-22), the fraction f/(f + p) is

    orbitals 32, 33, 34, 35, 37, 38, 39 :  1.0000, 1.0000, ... 0.9989
    orbital  36                        :  0.0124

Seven orbitals at 0.999 or above and one at 0.012 -- an eightyfold gap,
and exactly the seven the f shell should have. So the rule is: rank the
candidates by composition and take the ones that stand apart, refusing
when they do not.

This matters because the occupation pattern is not reliable here. The
molecular end of EuF has eight singly occupied orbitals, not the seven
the free atom shows, and the valence rule that works on a free main
group atom finds no doubly-occupied pair to take. The extra singly
occupied orbital is the ligand's, and only the composition tells them
apart.
"""
from dataclasses import dataclass, field

import numpy as np

from .composition import element_shell_weights
from .occupation import select_valence, split_by_occupation

__all__ = ['SHELL_RULES', 'Candidate', 'SelectionPlan', 'SplitByComposition',
           'candidate_from_elements', 'plan_from_elements',
           'split_by_composition']

# How far apart the two groups must sit for a composition split to be
# trusted. Measured on EuF's molecular end the ratio is 0.9989 / 0.0124,
# about 80; the threshold below leaves ample room without accepting a
# split that is really a continuum.
COMPOSITION_MIN_RATIO = 4.0


@dataclass
class SplitByComposition:
    """Candidates separated into two groups by their composition."""

    first: list = field(default_factory=list)
    second: list = field(default_factory=list)
    ratio: float = 0.0
    scores: dict = field(default_factory=dict)


def split_by_composition(candidates, weights_a, weights_b,
                         min_ratio=COMPOSITION_MIN_RATIO):
    """Separate candidates by which of two shells they belong to.

    The fraction ``a / (a + b)`` is computed for each candidate and the
    list is cut where the largest gap falls, provided that gap is wide
    enough to mean something. Comparing the two shells against each
    other is what makes this work where an absolute weight does not: a
    lanthanide's f functions spread over many contractions give every
    orbital a nonzero f weight, but the *ratio* still separates the
    orbitals that are mostly f from those that are mostly something
    else.

    Parameters
    ----------
    candidates : sequence of int
        Orbital indices to separate.
    weights_a, weights_b : array_like
        Per-orbital weights on the two shells, indexed over the full
        orbital set.
    min_ratio : float
        Smallest acceptable ratio between the lowest score in the first
        group and the highest in the second. Below it the split is
        treated as arbitrary and None is returned.

    Returns
    -------
    SplitByComposition or None
        None when the candidates cannot be separated: fewer than two of
        them, or no gap wide enough to trust. A caller must treat None
        as a refusal.
    """
    candidates = [int(i) for i in candidates]
    if len(candidates) < 2:
        return None
    wa = np.asarray(weights_a, dtype=float)
    wb = np.asarray(weights_b, dtype=float)
    scores = {i: float(wa[i]) / max(float(wa[i]) + float(wb[i]), 1e-30)
              for i in candidates}

    # Cut at the largest gap in the sorted scores.
    order = sorted(candidates, key=lambda i: -scores[i])
    if any(scores[order[k]] == scores[order[k + 1]]
           for k in range(len(order) - 1)):
        pass                      # ties are handled by the ratio test below
    gaps = [(scores[order[k]] - scores[order[k + 1]], k + 1)
            for k in range(len(order) - 1)]
    gap, cut = max(gaps)
    first, second = order[:cut], order[cut:]

    if not first or not second:
        return None
    ratio = scores[first[-1]] / max(scores[second[0]], 1e-30)
    if ratio < min_ratio:
        return None
    return SplitByComposition(first=sorted(first), second=sorted(second),
                              ratio=float(ratio), scores=scores)

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
    """Which rule to apply, decided from the elements present."""

    center: str = ''
    ligand: str = ''
    center_n: int = 7
    ligand_n: int = 3
    evidence: list = field(default_factory=list)

    @property
    def has_ligand(self):
        return bool(self.ligand)


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
                             ligand_n=0,
                             evidence=[f'{center}: bare lanthanide atom'])
    if len(others) > 1:
        return None
    ligand = others[0]
    if ligand not in MAIN_GROUP:
        return None

    return SelectionPlan(
        center=center, ligand=ligand, center_n=7, ligand_n=3,
        evidence=[f'centre {center} with ligand {ligand}'],
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
