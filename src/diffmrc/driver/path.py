"""Per-geometry orbital signatures for the path-consistent mapping.

The Bensberg-Reiher criterion (JPCL 2023, eq. 1) needs two shape-only
quantities per orbital: the kinetic expectation ``<psi| -grad^2/2 |psi>``
and the populations on shell-resolved intrinsic atomic orbitals (Knizia
2013), chosen over Mulliken because IAO populations are basis-
insensitive and keep the shells apart.  Both are cheap: one diagonal of
``C^T T C`` and one overlap matrix against the IAO columns of
``pyscf.lo.iao``, grouped into shells (1s, 2s, 2p, 3d, ...) by the
minimal-basis labels of the same construction.

Measured interface (server pyscf 2.14.0, H2O/STO-3G): ``iao.iao(mol,
orbocc)`` returns ``(nao, n_minao)`` with the columns in the minimal
basis's own order -- a plain ``basis='minao'`` rebuild of the same
molecule reproduces the column count and labels exactly (7 = 7 for
water), which is what the shell grouping relies on.  The test that the
oxygen 1s orbital's population lands on the (O, 1s) shell at > 0.9 is
the empirical pin of that association.

Boundaries: the minimal auxiliary basis stops short of the f block
(measured in the sibling project), so the signatures serve main-group
and 3d targets; open shells are refused because the IAO reference is
the closed-shell occupied block.
"""
import re

import numpy as np

__all__ = ['orbital_signatures']


def _minao_shell_labels(mol):
    """Shell labels of the minimal basis, in IAO column order.

    The IAO matrix's columns follow the minimal-basis functions of the
    same molecule (measured: equal counts, minimal-basis order); this
    rebuild reproduces that molecule and names each column's shell.  The
    grouping only needs the (atom, shell) partition, so the order inside
    a shell does not matter -- only the boundaries, which the caller
    checks by the column count.
    """
    from pyscf import gto

    m0 = gto.Mole()
    m0.atom = mol.atom
    m0.unit = mol.unit
    m0.charge = mol.charge
    m0.cart = mol.cart
    m0.basis = 'minao'
    m0.build()
    labels = []
    for lab in m0.ao_labels():
        parts = lab.split()
        atom = int(parts[0])
        shell = re.sub(r'[xyz]$', '', parts[2]) if len(parts) > 2 else '?'
        labels.append((atom, shell))
    return labels, m0.nao


def orbital_signatures(mol, C):
    """``(kin, q, shell_labels)`` of the orbitals ``C`` at one geometry.

    ``kin[i] = (C^T T C)_ii`` with T the kinetic integral; ``q[i, a]``
    is the summed squared overlap of orbital i with every IAO column of
    shell a, ``sum_m |<iao_m | psi_i>|^2``.  The IAOs are built from the
    occupied block of ``C`` (the first ``nelectron // 2`` columns), the
    same closed-shell reference the localizer uses.
    """
    from pyscf.lo import iao as _iao

    if mol.spin:
        raise ValueError(f'orbital signatures build their IAO reference '
                         f'from the closed-shell occupied block; spin='
                         f'{mol.spin} is out of scope')
    C = np.asarray(C, dtype=float)
    T = np.asarray(mol.intor('int1e_kin'), dtype=float)
    S = np.asarray(mol.intor('int1e_ovlp'), dtype=float)
    nocc = int(mol.nelectron) // 2
    if C.shape[1] < nocc:
        raise ValueError(f'{C.shape[1]} orbitals for {nocc} occupied '
                         f'slots')
    c_iao = np.asarray(_iao.iao(mol, C[:, :nocc]), dtype=float)
    labels, n_minao = _minao_shell_labels(mol)
    if c_iao.shape[1] != n_minao:
        raise ValueError(
            f'iao returned {c_iao.shape[1]} columns but the minimal '
            f'basis has {n_minao} functions; the column-shell '
            f'association is unknown -- re-pin it (see the module '
            f'docstring) before using the signatures')
    ov = c_iao.T @ S @ C                        # (n_minao, nmo)
    shells = {}
    for col, key in enumerate(labels):
        shells.setdefault(key, []).append(col)
    shell_labels = sorted(shells)
    q = np.zeros((C.shape[1], len(shell_labels)))
    for a, key in enumerate(shell_labels):
        cols = shells[key]
        q[:, a] = (ov[cols] ** 2).sum(axis=0)
    kin = np.einsum('pi,pq,qi->i', C, T, C)
    return kin, q, shell_labels
