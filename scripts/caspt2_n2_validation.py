#!/usr/bin/env python
"""Third-system validation of the contracted CASPT2 engine: N2/6-31G CAS(6,6).

Reference chain (the same one used for the H2O and CO anchors):
RHF -> CASSCF -> state-specific CASCI -> semicanonical orbitals.
Two variants of the contracted engine are evaluated on that reference:

* ``default`` -- :func:`caspt2_contracted_e2`, diagonal-Fock
                 (CASPT2-D level) denominators;
* ``coupled`` -- :func:`caspt2_contracted_e2_coupled`, the state-specific
                 generalized Fock with the inter-class coupling retained
                 (full-FIC level).

The stages are split so the reference can be built once and the two
expensive evaluations can run as independent processes:

    python caspt2_n2_validation.py --stage ref
    python caspt2_n2_validation.py --stage default
    python caspt2_n2_validation.py --stage coupled

``--stage all`` runs them sequentially in one process.  Per-class tables
are printed together with their ratios to the ORCA CASPT2-D and full-FIC
tables when those are supplied.

ORCA reference (6-31G, CAS(6,6), RHF -108.867763375101, CASSCF
-109.01554685351947, CASPT2-D -0.071520733, full-FIC -0.07301141705494;
the CASSCF must be driven with the NR orbital step and a tight GTol,
otherwise ORCA stops at a higher stationary point).

Usage:
    python caspt2_n2_validation.py --stage ref --outdir <dir>
    python caspt2_n2_validation.py --stage default --outdir <dir> \
        --orca-d X --orca-full Y --orca-classes-json <json>
"""
import argparse
import json
import os
import signal
import time

import numpy as np

import faulthandler
faulthandler.register(signal.SIGUSR1)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

CLASSES = ['V0_ijab', 'Vm1_iab', 'Vm2_ab', 'V1_ija', 'V2_ij', 'V0_ia',
           'Vm1_a', 'V1_i']

# N2 experimental bond length, 1.0977 Angstrom.
ATOM = 'N 0.0 0.0 0.0; N 0.0 0.0 1.0977'
BASIS = '6-31g'
NCORE, NACT, NELE = 4, 6, 6


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def mo_ints(C, h1e, eri):
    h1 = jnp.einsum('ai,ab,bj->ij', C, h1e, C)
    z = jnp.einsum('pi,pqrs->iqrs', C, eri)
    z = jnp.einsum('qj,iqrs->ijrs', C, z)
    z = jnp.einsum('rk,ijrs->ijks', C, z)
    return h1, jnp.einsum('sl,ijks->ijkl', C, z)


def build_reference(outdir):
    from pyscf import gto, scf, mcscf

    from diffmrc.core.casci import cs_casci
    from diffmrc.core.nevpt2.impl import canonicalize_jax, cas_dets_for2

    t0 = time.time()
    mol = gto.M(atom=ATOM, basis=BASIS, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    assert mf.converged
    log('RHF done %.12f (%.1f s)' % (mf.e_tot, time.time() - t0))

    t0 = time.time()
    mc = mcscf.CASSCF(mf, NACT, NELE)
    mc.kernel()
    assert mc.converged
    log('CASSCF done %.12f converged=%s (%.1f s)'
        % (mc.e_tot, mc.converged, time.time() - t0))

    t0 = time.time()
    C = jnp.asarray(mc.mo_coeff)
    h1e = jnp.asarray(mol.intor('int1e_kin') + mol.intor('int1e_nuc'))
    eri = jnp.asarray(mol.intor('int2e', aosym='s1'))
    nocc = NCORE + NACT
    h1_mo, eri_mo = mo_ints(C, h1e, eri)
    _, ci = cs_casci(h1_mo[:nocc, :nocc],
                     eri_mo[:nocc, :nocc, :nocc, :nocc],
                     ncore=NCORE, nact=NACT, nele_act=NELE, nroots=1,
                     return_ci=True)
    C2, ci2, eps = canonicalize_jax(C, h1e, eri, ci, ncore=NCORE,
                                    nact=NACT, nele_act=NELE)
    h1_mo2, eri_mo2 = mo_ints(C2, h1e, eri)
    dets = cas_dets_for2(NCORE, NACT, NELE)
    log('canonicalization done (%.1f s)' % (time.time() - t0))

    np.savez(os.path.join(outdir, 'n2_ref.npz'),
             h1_mo=np.asarray(h1_mo2), eri_mo=np.asarray(eri_mo2),
             ci=np.asarray(ci2), dets=np.asarray(dets),
             eps=np.asarray(eps), ncore=NCORE, nact=NACT, nele=NELE,
             casscf=np.asarray(mc.e_tot), rhf=np.asarray(mf.e_tot))
    log('reference saved: rhf %.12f casscf %.12f'
        % (mf.e_tot, mc.e_tot))


def load_reference(outdir):
    d = np.load(os.path.join(outdir, 'n2_ref.npz'))
    return (d['h1_mo'], d['eri_mo'], d['ci'], list(d['dets']), d['eps'],
            float(d['rhf']), float(d['casscf']))


def print_table(name, e2, bd, ref, args):
    print('=' * 78)
    print('%s  total %+18.12f' % (name, e2))
    print('%-9s %18s %18s' % ('class', 'value', 'ratio'))
    for k in CLASSES:
        ratio = ''
        if ref is not None and ref.get(k):
            ratio = '%10.5f' % (bd[k] / ref[k])
        print('%-9s %+18.12f %10s' % (k, bd[k], ratio))
    if args.orca_d is not None:
        print('total / ORCA-D    %.5f' % (e2 / args.orca_d))
    if args.orca_full is not None:
        print('total / ORCA-full %.5f' % (e2 / args.orca_full))


def stage_default(outdir, args):
    from diffmrc.core.caspt2 import caspt2_contracted_e2

    h1_mo, eri_mo, ci, dets, eps, rhf, casscf = load_reference(outdir)
    t0 = time.time()
    log('default path start')
    e2, bd = caspt2_contracted_e2(h1_mo, eri_mo, ci, dets, ncore=NCORE,
                                  nact=NACT, nele=NELE, eps=eps)
    log('default path done %.12f (%.1f s)' % (e2, time.time() - t0))
    ref = load_orca_classes(args, 'd')
    print_table('default (diagonal Fock)', e2, bd, ref, args)
    out = {'system': 'N2/6-31G CAS(6,6)', 'casscf': casscf, 'rhf': rhf,
           'total': float(e2),
           'classes': {k: float(bd[k]) for k in CLASSES}}
    with open(os.path.join(outdir, 'n2_default.json'), 'w') as fh:
        json.dump(out, fh, indent=2)


def stage_coupled(outdir, args):
    from diffmrc.core.caspt2 import caspt2_contracted_e2_coupled

    h1_mo, eri_mo, ci, dets, eps, rhf, casscf = load_reference(outdir)
    t0 = time.time()
    log('coupled path start (function assembly + pair matrices)')
    e2, bd, system = caspt2_contracted_e2_coupled(
        h1_mo, eri_mo, ci, dets, ncore=NCORE, nact=NACT, nele=NELE,
        return_system=True)
    log('coupled path done %.12f (%.1f s)' % (e2, time.time() - t0))

    f = system['fock']
    core = list(range(NCORE))
    act = list(range(NCORE, NCORE + NACT))
    vir = list(range(NCORE + NACT, h1_mo.shape[0]))

    def bmax(a, b):
        return float(np.abs(f[np.ix_(a, b)]).max())

    blocks = {'core_active': bmax(core, act),
              'active_virtual': bmax(act, vir),
              'core_virtual': bmax(core, vir)}
    print('Fock off-diagonal blocks: core-active %.3e  active-virtual %.3e '
          'core-virtual %.3e' % (blocks['core_active'],
                                 blocks['active_virtual'],
                                 blocks['core_virtual']))
    ref = load_orca_classes(args, 'full')
    print_table('coupled (full generalized Fock)', e2, bd, ref, args)
    out = {'system': 'N2/6-31G CAS(6,6)', 'casscf': casscf, 'rhf': rhf,
           'total': float(e2),
           'classes': {k: float(bd[k]) for k in CLASSES},
           'nfn': len(system['labels']), 'fock_blocks': blocks}
    with open(os.path.join(outdir, 'n2_coupled.json'), 'w') as fh:
        json.dump(out, fh, indent=2)


def load_orca_classes(args, key):
    if args.orca_classes_json:
        with open(args.orca_classes_json) as fh:
            return json.load(fh)[key]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', default='all',
                        choices=['ref', 'default', 'coupled', 'all'])
    parser.add_argument('--outdir', default='.')
    parser.add_argument('--orca-d', type=float, default=None)
    parser.add_argument('--orca-full', type=float, default=None)
    parser.add_argument('--orca-classes-json', default=None)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.stage in ('ref', 'all'):
        build_reference(args.outdir)
    if args.stage in ('default', 'all'):
        stage_default(args.outdir, args)
    if args.stage in ('coupled', 'all'):
        stage_coupled(args.outdir, args)


if __name__ == '__main__':
    main()
