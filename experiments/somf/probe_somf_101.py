#!/usr/bin/env python
"""Probe: SOMF two-electron mean-field operator via pyscf integral kernels.

Purpose: establish, before writing ``diffmrc.core.somf``, that

  (a) libcint exposes the two-electron SOC kernel as ``int2e_p1vxp1`` and
      pyscf's ``pyscf.prop.gtensor.uhf.get_jk`` returns the direct (J) and
      exchange (K, both permutations) contractions of that kernel,
  (b) the SOMF operator h2 = J[P] - 1.5 K[P] with the reference density P
      reproduces the ORCA 6.1.1 SOMF splittings on OH and OF,
  (c) the density choice (state-averaged vs root-0) does not dominate.

Reference values (ORCA 6.1.1, same geometry / basis / CAS / roots):
  OH: one-electron 205.88 cm^-1, SOMF 128.83 cm^-1
  OF: one-electron 272.06 cm^-1, SOMF 176.12 cm^-1

Run on 101 (pyscfad214 env):
  JAX_PLATFORMS=cpu python probe_somf_101.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ.setdefault('MKL_CBWR', 'COMPATIBLE')

import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.data import nist

ALPHA2 = float(nist.ALPHA) ** 2
CM = 219474.6313632
SPIN = 0.5 * np.array([[[0, 1], [1, 0]],
                       [[0, -1j], [1j, 0]],
                       [[1, 0], [0, -1]]], dtype=complex)

SYS = {
    'oh': dict(atom=[('O', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 1.8324))],
               spin=1, ncore=1, nact=6),
    'of': dict(atom=[('O', (0.0, 0.0, 0.0)), ('F', (0.0, 0.0, 2.4))],
               spin=1, ncore=2, nact=8),
}


def soc_jk(mol, dm):
    """J and K contractions of the int2e_p1vxp1 kernel for density ``dm``."""
    from pyscf.prop.gtensor.uhf import get_jk
    vj, vk = get_jk(mol, (0.5 * dm, 0.5 * dm))
    return vj[0] + vj[1], vk[0] + vk[1]


def si_splitting(es, T, pref):
    n = len(es)
    H = np.zeros((2 * n, 2 * n), dtype=complex)
    for i in range(n):
        H[2 * i, 2 * i] = es[i]
        H[2 * i + 1, 2 * i + 1] = es[i]
        for j in range(n):
            for k in range(3):
                H[2 * i:2 * i + 2, 2 * j:2 * j + 2] += pref * T[i, j, k] * SPIN[k]
    w = np.linalg.eigvalsh(H)
    return float((w[2] - w[0]) * CM), float(np.abs(H - H.conj().T).max())


def run(name):
    spec = SYS[name]
    mol = gto.M(atom=spec['atom'], basis='6-31g', unit='Bohr',
                spin=spec['spin'], verbose=0)
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    mc = mcscf.CASSCF(mf, spec['nact'], mol.nelectron - 2 * spec['ncore'])
    mc.verbose = 0
    mc.conv_tol = 1e-10
    mc.max_cycle = 200
    mc.state_average_([0.5, 0.5])
    mc.fcisolver.nroots = 2
    mc.kernel()
    es = np.asarray(mc.fcisolver.e_states)[:2]
    C = np.asarray(mc.mo_coeff)
    act = slice(spec['ncore'], spec['ncore'] + spec['nact'])
    Ca = C[:, act]
    ci = np.asarray(mc.ci)

    h1 = mol.intor_asymmetric('int1e_pnucxp', 3)
    h1_sym = mol.intor('int1e_pnucxp')
    print(f'[{name}] nao={mol.nao_nr()}  nelec={mol.nelectron}  '
          f'es={es}  gap_cm={(es[1]-es[0])*CM:.3e}')
    print(f'[{name}] intor_asymmetric vs intor: {np.abs(h1-h1_sym).max():.2e}')

    dm_sa = np.asarray(mc.make_rdm1())
    dm0 = np.asarray(mc.make_rdm1(fake_ci=ci[0]))
    dm1 = np.asarray(mc.make_rdm1(fake_ci=ci[1]))
    for tag, dm in (('sa', dm_sa), ('root0', dm0), ('root1', dm1)):
        print(f'[{name}] dm[{tag}] trace={np.trace(dm):.10f} '
              f'|dm-dmT|={np.abs(dm-dm.T).max():.2e}')

    j_sa, k_sa = soc_jk(mol, dm_sa)
    j0, k0 = soc_jk(mol, dm0)
    j1, k1 = soc_jk(mol, dm1)
    print(f'[{name}] linearity J: |J0+J1-2Jsa|={np.abs(j0+j1-2*j_sa).max():.2e}')
    print(f'[{name}] linearity K: |K0+K1-2Ksa|={np.abs(k0+k1-2*k_sa).max():.2e}')

    # transition densities and operator transformation
    D = {}
    for i in range(2):
        for j in range(2):
            D[i, j] = np.asarray(mc.fcisolver.trans_rdm1(
                ci[i], ci[j], spec['nact'], mc.nelecas))

    def split(h_ao, tag):
        hmo = np.einsum('ap,cab,bq->cpq', Ca, h_ao, Ca, optimize=True)
        T = np.zeros((2, 2, 3))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    T[i, j, k] = float(np.sum(D[i, j] * hmo[k]))
        s, herm = si_splitting(es, T, 1j * ALPHA2 / 2.0)
        print(f'[{name}] {tag}: splitting={s:.4f} cm^-1  herm={herm:.2e}  '
              f'T01=({T[0,1,0]:.6e},{T[0,1,1]:.6e},{T[0,1,2]:.6e})')
        return s, T, hmo

    split(h1, 'bare-1e')
    for tag, (j, k) in (('sa', (j_sa, k_sa)), ('root0', (j0, k0)),
                        ('root1', (j1, k1))):
        h2 = j - 1.5 * k
        print(f'[{name}] h2[{tag}] |h2+h2T|max={np.abs(h2 + h2.transpose(0,2,1)).max():.3e}'
              f'  norm={np.linalg.norm(h2):.6e}')
        split(h1 + h2, f'somf-{tag}')

    # AMFI variant: one-center exchange integrals only
    from pyscf.prop.gtensor.uhf import get_jk_amfi
    vj, vk = get_jk_amfi(mol, (0.5 * dm_sa, 0.5 * dm_sa))
    h2_amfi = (vj[0] + vj[1]) - 1.5 * (vk[0] + vk[1])
    split(h1 + h2_amfi, 'somf-sa-amfi')


if __name__ == '__main__':
    for name in ('oh', 'of'):
        run(name)
        print()
