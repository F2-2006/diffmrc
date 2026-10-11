"""The SCF convergence protocol for multi-solution references.

Why this exists (graduated from the f-block experiment scripts on
2026-10-11; every chain in that line imported the script copy, which is
how a fact-standard capability ended up living in an experiment file):

A lanthanide or actinide molecule can have several SCF solutions close
enough that an independent guess, started at a neighbouring geometry,
lands in the wrong one; the reported energy then carries a silent
0.7 Ha-class error (measured: a 27 mHa basin switch and a 0.67 Ha
branch switch in one geometry scan).  The protocol prices several
candidates on the same variational terms and keeps the lowest
*converged* one:

S1: five independent guesses x plain SCF at strict tolerance
    (``None``, atom, 1e, huckel, minao), plus an optional relay
    candidate: a density matrix from a neighbouring geometry offered as
    one more starting point -- not trusted, only offered, and it wins
    only if it is the lowest converged candidate.  A relay that does
    not converge from the strict density is retried on the damping grid
    (the branch-locking failure of 2026-10-11 converged only with
    damping help).
S2: damping x level-shift grid, tried only when S1 converges nothing
    (the relaxation strategy measured earlier on the europium atom,
    generalized).
Result: ``(mf, E_scf, reliable, spread, stage)`` -- when nothing
converged, the caller decides (an orbital-optimization fallback or a
refusal), never a silent dirty reference.
"""
from pyscf import scf

__all__ = ['GUESSES', 'RELAX_GRID', 'scf_protocol']

#: kcal/mol per Hartree, for the candidate-spread diagnostic.
_H2K = 627.5094740631

GUESSES = (None, 'atom', '1e', 'huckel', 'minao')
RELAX_GRID = ((0.5, 1.0), (0.3, 0.5), (0.7, 2.0))


def _scf_run(mol, spin, g, damp, shift, conv_tol, max_cycle, dm0=None):
    """One SCF attempt; ``dm0`` is an optional density-matrix start."""
    m = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
    m.max_cycle = max_cycle
    m.conv_tol = conv_tol
    if g:
        m.init_guess = g
    if damp is not None:
        m.damp = damp
    if shift is not None:
        m.level_shift = shift
    m = m.x2c1e()
    try:
        m.kernel(dm0=dm0)
    except Exception:
        return None
    return m


def scf_protocol(mol, spin, tag, dm0_guess=None):
    """S1 + S2 grid; lowest converged candidate wins.

    Returns ``(mf, E_scf, reliable, spread, stage)``.  When nothing
    converged the caller decides (orbital-optimization fallback or a
    refusal), never a silent dirty CASCI.

    ``dm0_guess`` (optional) joins S1 as one more candidate under the
    key 'relay'; it competes on the same variational terms as the
    independent guesses.
    """
    cands = {}
    if dm0_guess is not None:
        m = _scf_run(mol, spin, None, None, None, 1e-9, 300,
                     dm0=dm0_guess)
        if m is not None:
            cands['relay'] = m
        if m is None or not m.converged:
            for damp, shift in RELAX_GRID:
                m2 = _scf_run(mol, spin, None, damp, shift, 1e-7, 500,
                              dm0=dm0_guess)
                if m2 is not None:
                    cands[f'relay:{damp}'] = m2
                    if m2.converged:
                        break
    for g in GUESSES:
        m = _scf_run(mol, spin, g, None, None, 1e-9, 300)
        if m is not None:
            cands[f'g:{g}'] = m
    good = {k: m for k, m in cands.items() if m.converged}
    if not good:
        for damp, shift in RELAX_GRID:
            for g in (None, 'atom'):
                k = f'relax:{g}/{damp}'
                m = _scf_run(mol, spin, g, damp, shift, 1e-7, 500)
                if m is not None:
                    cands[k] = m
                    if m.converged:
                        good[k] = m
    if good:
        k = min(good, key=lambda k: good[k].e_tot)
        spread = ((max(m.e_tot for m in good.values())
                   - min(m.e_tot for m in good.values())) * _H2K)
        return good[k], float(good[k].e_tot), True, spread, k
    if not cands:
        return None, None, False, float('nan'), 'none'
    k = min(cands, key=lambda k: cands[k].e_tot)
    return cands[k], float(cands[k].e_tot), False, float('nan'), k
