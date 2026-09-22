"""Chemical rules that propose an active space from molecular structure.

The rules read the occupation vector and the orbital composition. They
never assume a particular orbital ordering: an index-based rule depends
on which SCF solution was found, and a different initial guess reorders
the orbitals, so such a rule silently selects a different space on the
next run. This project has already been bitten by that.

Every rule can refuse. A refusal is a result, not an error: it says the
structure does not match what the rule knows how to handle, and the
caller is expected to escalate or to report that it cannot decide.
"""
from dataclasses import dataclass, field

import numpy as np

from .composition import element_shell_weights
from .occupation import select_valence, split_by_occupation

__all__ = ['SHELL_RULES', 'Candidate', 'candidate_from_elements']

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
