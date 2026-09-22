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
"""
from dataclasses import dataclass, field

import numpy as np

from .composition import element_shell_weights
from .occupation import select_valence, split_by_occupation

__all__ = ['SHELL_RULES', 'Candidate', 'SelectionPlan',
           'candidate_from_elements', 'plan_from_elements',
           'dominant_shell']

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
