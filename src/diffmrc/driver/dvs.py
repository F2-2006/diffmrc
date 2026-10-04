"""The DVS scan: (lambda, N) test orbitals, one-shot MC-PDFT rows.

The discrete-variational-selection construction of Wardzala, King and
Gagliardi (JPCA 2025, doi 10.1021/acs.jpca.4c08597): candidate orbital
sets from diagonalizing ``G(lambda) = F - lambda K``, active windows
from the APC chooser (their equations (4)-(6); pyscf's ``mcscf.apc`` is
the authors' implementation), each candidate carried through an
equal-weight SA-CASSCF initialized from its own orbitals and one tPBE
evaluation per state.  The selection is the candidate with the lowest
mean on-top energy; the scan keeps every row, so the stability of the
choice can be read off rather than trusted.

The construction mirrors the worked reproduction script t369 (which
carried the pyscf engine through the full ZnH_VE1 protocol); the rows
here run the same protocol through this package's own SA-CASSCF and
on-top columns.  Windows whose determinant count exceeds what the
in-stack CI diagonalizes are refused with the count in the row rather
than attempted -- the paper's ZnH/CuCl (12,12)/(14,14) windows are of
that size and stay engine-side (registered).
"""
import math

import numpy as np

__all__ = ['dvs_scan']


def _apc_entropies(orbs, occ_lab, f_m, k_m, n):
    """APC entropies with ``n`` virtuals removed (t369's reconstruction).

    pyscf's ``apc.APC`` class drives its removal loop through a full SCF
    at every step, which the scan cannot afford per candidate; this
    reproduces the same removal rule (the largest-entropy virtual is
    removed, ``n`` times) against the supplied F/K and then applies the
    class's own entropy overrides, so the chooser sees the same numbers
    the reference protocol would produce.

    Uses the private ``APC._apc`` of pyscf 2.14 -- pinned by the t369
    reproduction (the same call produced the ZnH protocol rows); a
    pyscf upgrade that renames it fails loudly here, not silently.
    """
    from pyscf.mcscf import apc as apcmod

    obj = apcmod.APC.__new__(apcmod.APC)
    obj.eps = 1e-3
    obj.n = n
    occ = occ_lab.copy()
    removed = []
    for _ in range(int(n)):
        if (occ == 0).sum() == 0:
            break
        ents = obj._apc(orbs, occ, f_m, k_m)
        virt = np.where(occ == 0)[0]
        to_remove = virt[int(np.argmax(ents[virt]))]
        removed.append(to_remove)
        occ[to_remove] = 1
    ents = obj._apc(orbs, occ, f_m, k_m)
    maxs = np.max(ents)
    for i, o in enumerate(np.where(occ_lab == 1)[0]):
        ents[o] = maxs + 2 * obj.eps - i * obj.eps * 1e-2
    for i, o in enumerate(removed):
        ents[o] = maxs + obj.eps - i * obj.eps * 1e-2
    return ents


def _test_orbitals(mf, lam):
    """The ``G(lambda) = F - lambda K`` orbital set.

    Returns ``(C_trial, f_trial, k_trial, occupancy labels)``: the trial
    orbitals as AO columns with F/K expressed in them, and the mean-field
    occupations in that basis rounded to integers -- the partition the
    chooser reads.  For open-shell mean fields the K matrix is the sum
    over spin channels (the same object t369 handed the chooser).
    """
    S = np.asarray(mf.mol.intor('int1e_ovlp'))
    f_ao = np.asarray(mf.get_fock())
    k = mf.get_k()
    k_ao = np.sum(k, axis=0) if np.ndim(k) == 3 else np.asarray(k)
    D = mf.make_rdm1()
    D = np.sum(D, axis=0) if np.ndim(D) == 3 else np.asarray(D)
    C = np.asarray(mf.mo_coeff)
    f_mo = C.T @ f_ao @ C
    k_mo = C.T @ k_ao @ C
    D_mo = C.T @ S @ D @ S @ C
    G = f_mo - float(lam) * k_mo
    _, U = np.linalg.eigh(0.5 * (G + G.T))
    C_trial = C @ U
    f_t = U.T @ f_mo @ U
    k_t = U.T @ k_mo @ U
    occ_lab = np.clip(np.round(np.diag(U.T @ D_mo @ U)), 0, 2).astype(int)
    return C_trial, f_t, k_t, occ_lab


def dvs_scan(mol, *, lam_grid=(0.0, 0.5, 1.0, 1.5, 2.0),
             n_grid=(4, 8), nroots=2, max_size=(12, 12), max_ndet=2000,
             max_steps=400, refine_max_iter=40, optimize_tol=None,
             refine_target=None, grid=None, grid_level=3,
             eval_xc=None, scf_builder=None, hyb=0.0):
    """Run the (lambda, N) DVS scan on one molecule.

    Every (lambda, N) pair is carried through the one-step protocol:
    trial orbitals, APC window, SA-CASSCF from the trial set, per-state
    one-shot tPBE.  ``max_ndet`` guards the in-stack CI: a window whose
    determinant count exceeds it is recorded with an ``error`` string
    instead of attempted.

    Returns ``{'mf_energy', 'candidates', 'selection', 'stability'}``.
    Each candidate row carries ``lam``, ``n_remove``, ``window``,
    ``npar``, ``ndet``, ``grad_norm``, ``e_tpbe`` (per state),
    ``mean_tpbe`` and the first excited group's three dE columns -- or
    ``error``.  ``selection`` is the row with the lowest mean tPBE;
    ``stability`` is the gap to the runner-up (itself useful: a
    selection that wins by 1e-9 has not selected anything).
    """
    from pyscf import scf
    from pyscf.mcscf import apc as apcmod

    from .nevpt2 import sa_mcpdft_table

    if scf_builder is None:
        scf_builder = scf.ROHF if mol.spin else scf.RHF
    mf = scf_builder(mol)
    mf.kernel()

    candidates = []
    for lam in lam_grid:
        C_trial, f_t, k_t, occ_lab = _test_orbitals(mf, lam)
        for n_remove in n_grid:
            row = {'lam': float(lam), 'n_remove': int(n_remove)}
            try:
                ents = _apc_entropies(C_trial, occ_lab, f_t, k_t, n_remove)
                chooser = apcmod.Chooser(C_trial, occ_lab, ents,
                                         max_size=max_size, fixed=False,
                                         verbose=0)
                nact, nactel, casorbs, _ = chooser.kernel()
                nact = int(nact)
                nele = (int(nactel) if np.isscalar(nactel)
                        else int(np.sum(nactel)))
                ncore = (int(mol.nelectron) - nele) // 2
                row['window'] = {'ncore': ncore, 'nact': nact, 'nele': nele}
                ndet = math.comb(2 * nact, nele)
                if ndet > int(max_ndet):
                    row['error'] = (f'ndet {ndet} exceeds max_ndet '
                                    f'{int(max_ndet)}; the paper-scale '
                                    f'windows stay engine-side')
                    candidates.append(row)
                    continue
                table = sa_mcpdft_table(
                    mol, scf_builder, ncore=ncore, nact=nact, nele=nele,
                    nroots=nroots, c0=np.asarray(casorbs),
                    max_steps=max_steps, refine_max_iter=refine_max_iter,
                    optimize_tol=optimize_tol, refine_target=refine_target,
                    grid=grid, grid_level=grid_level, eval_xc=eval_xc,
                    hyb=hyb)
                e_tpbe = [float(x) for x in np.atleast_1d(table['e_tpbe'])]
                row.update(npar=int(table['npar']), ndet=int(table['ndet']),
                           grad_norm=float(table['grad_norm']),
                           e_tpbe=e_tpbe,
                           mean_tpbe=float(np.mean(e_tpbe)),
                           # the canonical-basis orbitals: the anchor
                           # scripts evaluate the engine on exactly these,
                           # and a caller may want the winner's orbitals
                           C=np.asarray(table['out']['C'], dtype=float))
                if len(table['groups']) > 1:
                    g1 = table['groups'][1]
                    row.update(dE_casscf=float(g1['dE_casscf']),
                               dE_nevpt2=float(g1['dE_nevpt2']),
                               dE_mcpdft=float(g1['dE_mcpdft']))
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            candidates.append(row)

    selection, stability = _select_candidate(candidates)
    return {'mf_energy': float(mf.e_tot), 'candidates': candidates,
            'selection': selection, 'stability': stability}


def _select_candidate(candidates):
    """The discrete-variational pick: lowest mean on-top energy.

    Returns ``(selection, stability)``; ``stability`` is the gap to the
    runner-up (a winner by 1e-9 has not selected anything).  Rows that
    were refused (``error``) do not take part.  Pure arithmetic on the
    row dicts, so the selection rule is testable without a scan.
    """
    good = [r for r in candidates if 'mean_tpbe' in r]
    selection = stability = None
    if good:
        ordered = sorted(good, key=lambda r: r['mean_tpbe'])
        best = ordered[0]
        selection = {'lam': best['lam'], 'n_remove': best['n_remove'],
                     'window': dict(best['window']),
                     'mean_tpbe': best['mean_tpbe']}
        if len(ordered) > 1:
            stability = float(ordered[1]['mean_tpbe'] - best['mean_tpbe'])
    return selection, stability
