"""Actinide validation of the differentiable NEVPT2 chain (Phase 0, Task 8).

Cases (one invocation each; run in parallel as separate jobs):

    tho_ecp   ThO, Stuttgart ECP on Th + ano on O, RHF, frontier active space
    uf_ecp    UF,  Stuttgart ECP on U + ano on F, ROHF (spin from parity)
    tho_x2c   ThO all-electron, scalar-relativistic X2C builder (heavy; run alone)

What is checked, and one boundary that had to be redrawn:

  [1] translational invariance of the gradient (sum of forces = 0)
  [2] rotational invariance (torque = 0)
  [3] an energy cross-check against PySCF's own CASCI + SC-NEVPT2 on the
      same ECP molecule. The Task-3 anchor (our scale-0 gradient against
      ``core.hessian.make_joint_energy_fn``) is NOT usable on these
      routes: that frozen function hard-codes
      ``int1e_kin + int1e_nuc`` and ``pscf.RHF``, so it silently drops
      the ECP (and would drop an X2C rotation). Comparing against it
      would compare two different Hamiltonians.
  [4] one Hessian-vector product against a short finite-difference
      ladder (two deltas; monotone improvement is the check)

Boundary for the record, in the code and in every quotation of these
numbers: scalar-relativistic only. Spin-orbit coupling is out of scope
here (roadmap P4) -- for actinides that boundary is not cosmetic.

Usage:  PYTHONPATH=src python experiments/verify_nevpt2_an.py <case> [NT]
"""
import json
import os
import sys
import time

CASE = sys.argv[1] if len(sys.argv) > 1 else 'tho_ecp'
NT = sys.argv[2] if len(sys.argv) > 2 else '4'
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = NT
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["MKL_CBWR"] = "COMPATIBLE"

import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from pyscfad import gto as pgto, scf as pscf  # noqa: E402

from diffmrc.core.orblib import orbital_pairs  # noqa: E402
from diffmrc.driver.nevpt2 import (make_nevpt2_joint_fn,  # noqa: E402
                                   nevpt2_hvp, nevpt2_nuclear_gradient)

CASES = {
    'tho_ecp': dict(atom='Th 0 0 0; O 0 0 1.85',
                    basis={'Th': 'stuttgart_rsc', 'O': 'ano'},
                    ecp={'Th': 'stuttgart_rsc'}, spin=0, x2c=False),
    'uf_ecp': dict(atom='U 0 0 0; F 0 0 2.03',
                   basis={'U': 'stuttgart_rsc', 'F': 'ano'},
                   ecp={'U': 'stuttgart_rsc'}, spin=None, x2c=False),
    'tho_x2c': dict(atom='Th 0 0 0; O 0 0 1.85',
                    basis={'Th': 'ano', 'O': 'ano'},
                    ecp=None, spin=0, x2c=True),
}


def build(case):
    cfg = CASES[case]
    mol = pgto.Mole()
    mol.atom = cfg['atom']
    mol.basis = cfg['basis']
    if cfg['ecp']:
        mol.ecp = cfg['ecp']
    if cfg['spin'] is not None:
        mol.spin = cfg['spin']
        mol.build()
        return mol, cfg
    # parity fixes the spin. The count is only known after a build, and a
    # wrong parity makes build raise -- so try the even build and flip on
    # the consistency error rather than guessing.
    mol.spin = 0
    try:
        mol.build()
    except RuntimeError:
        mol.spin = 1
        mol.build()
    print('  spin from parity: nelec={} -> spin={}'.format(
        mol.nelectron, mol.spin), flush=True)
    return mol, cfg


def make_builder(mol, cfg):
    if mol.nelectron % 2 == 0:
        base = pscf.RHF
    else:
        base = pscf.ROHF

    def builder(m):
        mf = base(m)
        mf.init_guess = 'atom'
        mf.max_cycle = 200
        mf.conv_tol = 1e-9
        if cfg['x2c']:
            mf = mf.x2c1e()
        return mf
    return builder


def frontier_space(mf):
    occ = np.asarray(mf.mo_occ)
    nocc = int((occ > 0).sum())
    # The window keeps the four highest occupied orbitals active. On the
    # discovered ThO orbital ladder (mo 15/16 exactly degenerate at
    # -0.484128) a three-orbital window splits that pair across the
    # core/active boundary -- precisely the inactive degeneracy that
    # SC-NEVPT2 lacks invariance against. Four orbitals put the pair
    # safely inside the active space.
    ncore = nocc - 4
    nact = 8
    nele = int(round(occ[:ncore + nact].sum() - 2 * ncore))
    return ncore, nact, nele


def main():
    t_start = time.time()
    mol, cfg = build(CASE)
    nmo = mol.nao_nr()
    print('=' * 86, flush=True)
    print('NEVPT2 actinide validation -- {}  nao={}  nelec={}  spin={}'.format(
        CASE, nmo, mol.nelectron, mol.spin), flush=True)
    print('  boundary: scalar-relativistic only; SOC out of scope (P4)',
          flush=True)
    print('=' * 86, flush=True)

    builder = make_builder(mol, cfg)
    mf0 = builder(mol)
    mf0.kernel()
    print('  SCF E = {:.10f}  conv={}  [{:.0f}s]'.format(
        float(mf0.e_tot), bool(mf0.converged), time.time() - t_start),
        flush=True)

    ncore, nact, nele = frontier_space(mf0)
    pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)
    print('  frontier CAS({},{})  ncore={}  NPAR={}'.format(
        nele, nact, ncore, len(pairs)), flush=True)

    fn = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, builder,
                              e2_scale=1.0, ci_solver='blocked')
    fn0 = make_nevpt2_joint_fn(ncore, nact, nele, nmo, pairs, builder,
                               e2_scale=0.0, ci_solver='blocked')
    theta = jnp.zeros(len(pairs))
    E_of = lambda m_: fn(m_, theta)       # noqa: E731
    E_of0 = lambda m_: fn0(m_, theta)     # noqa: E731

    out = dict(case=CASE, nao=nmo, nelec=mol.nelectron, spin=int(mol.spin),
               ncore=ncore, nact=nact, nele=nele, npair=len(pairs),
               scf=float(mf0.e_tot))

    # ---- [1][2] invariance of the gradient ----
    t0 = time.time()
    g = np.asarray(nevpt2_nuclear_gradient(mol, E_of))
    s_trans = float(np.abs(g.sum(axis=0)).max())
    R = np.asarray(mol.atom_coords())
    s_rot = float(np.abs(np.cross(R, g).sum(axis=0)).max())
    ok12 = s_trans < 1e-7 and s_rot < 1e-5
    print('  [1] translation |sum g| = {:.3e} (thr 1e-7)'.format(s_trans),
          flush=True)
    print('  [2] rotation    |torque| = {:.3e} (thr 1e-5)  {}  [{:.0f}s]'.format(
        s_rot, 'PASS' if ok12 else 'FAIL', time.time() - t0), flush=True)
    out['invariance'] = dict(sum_g=s_trans, torque=s_rot, ok=ok12)
    out['gradient'] = g.tolist()

    # ---- [3] energy cross-check vs PySCF CASCI + SC-NEVPT2 (same ECP) ----
    t0 = time.time()
    try:
        from pyscf import gto as gto2, scf as scf2, mcscf, mrpt
        m2 = gto2.M(atom=cfg['atom'], basis=cfg['basis'],
                    ecp=cfg['ecp'], spin=int(mol.spin), verbose=0)
        mf2 = (scf2.RHF(m2) if m2.nelectron % 2 == 0 else scf2.ROHF(m2))
        mf2.kernel()
        mc = mcscf.CASCI(mf2, nact, nele)
        mc.ncore = ncore
        mc.verbose = 0
        mc.kernel()
        e2_ref = float(mrpt.NEVPT(mc).kernel())
        E_ours, e2_ours = fn.components(mol, theta)
        E_ours = float(E_ours)
        e2_ours = float(e2_ours)
        dE = abs((E_ours + e2_ours) - (float(mc.e_tot) + e2_ref))
        ok3 = dE < 1e-6
        print('  [3] E(ours)  = {:.10f}   e2(ours)  = {:.10f}'.format(
            E_ours + e2_ours, e2_ours), flush=True)
        print('  [3] E(pyscf) = {:.10f}   e2(pyscf) = {:.10f}'.format(
            float(mc.e_tot) + e2_ref, e2_ref), flush=True)
        print('  [3] |dE| = {:.3e} (thr 1e-6)  {}  [{:.0f}s]'.format(
            dE, 'PASS' if ok3 else 'FAIL', time.time() - t0), flush=True)
        out['crosscheck'] = dict(E_ours=E_ours, e2_ours=e2_ours,
                                 E_pyscf=float(mc.e_tot) + e2_ref,
                                 e2_pyscf=e2_ref, dE=dE, ok=ok3)
    except Exception as ex:  # noqa: BLE001 -- record, do not mask [1][2]
        print('  [3] PySCF 对拍不可用: ' + repr(ex)[:120], flush=True)
        out['crosscheck'] = dict(error=repr(ex)[:200])

    # ---- [4] one HVP against a short FD ladder ----
    t0 = time.time()
    v = np.zeros(mol.natm * 3)
    v[2] = -1.0
    v[5] = 1.0
    hvp = np.asarray(nevpt2_hvp(mol, E_of, v)).reshape(-1)
    errs = {}
    for d in (3e-3, 1e-3):
        m1 = mol.copy()
        m1.set_geom_(np.asarray(mol.atom_coords()) + d * v.reshape(-1, 3),
                     'Bohr')
        m2 = mol.copy()
        m2.set_geom_(np.asarray(mol.atom_coords()) - d * v.reshape(-1, 3),
                     'Bohr')
        g1 = np.asarray(nevpt2_nuclear_gradient(m1, E_of))
        g2 = np.asarray(nevpt2_nuclear_gradient(m2, E_of))
        errs[d] = float(np.abs(((g1 - g2) / (2 * d)).reshape(-1) - hvp).max())
    ok4 = errs[1e-3] < errs[3e-3]
    print('  [4] HVP(stretch) = {}'.format(
        np.array2string(hvp, precision=6, suppress_small=True)), flush=True)
    print('  [4] |FD-HVP| d=3e-3: {:.3e}   d=1e-3: {:.3e}   单调 {}  [{:.0f}s]'.format(
        errs[3e-3], errs[1e-3], ok4, time.time() - t0), flush=True)
    out['hvp'] = dict(values=hvp.tolist(),
                      err_3e3=errs[3e-3], err_1e3=errs[1e-3], ok=ok4)

    outp = 'results/verify_nevpt2_an_{}.json'.format(CASE)
    json.dump(out, open(outp, 'w', encoding='utf-8'), indent=1, default=str)
    print('>>> {}   total {:.0f}s'.format(outp, time.time() - t_start),
          flush=True)
    print('verify_nevpt2_an_{}_DONE'.format(CASE), flush=True)


if __name__ == '__main__':
    main()
