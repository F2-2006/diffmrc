#!/usr/bin/env python
"""Acceptance run for the differentiable SOMF layer on OH and OF.

Reproduces the value-layer setup of ``validate_oh_of.py`` (ROHF +
state-averaged CASSCF, 6-31G, two roots), then:

1. compares the JAX SOMF splitting with the numpy value layer and with the
   ORCA 6.1.1 reference (SOMF and one-electron);
2. differentiates the splitting with respect to the reference density,
   using three parameterizations:
   - a global density scale ``P -> (1 + s) P``;
   - a symmetric density direction ``P -> P + s dP``;
   - an active-orbital rotation ``C -> C exp(theta A)`` together with the
     rotated active density ``P -> P_inactive + C(theta) D_act C(theta)^T``;
3. compares every reverse-mode derivative with a central finite difference
   at several step sizes.

The density scale changes the electron count and is a synthetic
parameterization used to exercise the response contraction; the orbital
rotation preserves ``trace(P S)``.

Run (from this directory):
  PYTHONPATH=<pkg>/src JAX_PLATFORMS=cpu python validate_oh_of_jax.py \
      --out results_somf_jax.json
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ.setdefault('MKL_CBWR', 'COMPATIBLE')

import argparse
import json
import os as _os
import sys
import time

import numpy as np
from pyscf import gto, mcscf, scf

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.scipy.linalg import expm

from diffmrc.core import somf as dsomf
from diffmrc.core import somf_jax as jsomf

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import validate_oh_of as ref  # noqa: E402

SYSTEMS = dict(ref.SYSTEMS)
EXPERIMENT_CM = {'oh': 139.8, 'of': None}


def build_reference(spec):
    """ROHF + SA-CASSCF data shared by the numpy and JAX paths."""
    mol = gto.M(atom=spec['atom'], basis='6-31g', unit='Bohr',
                spin=spec['spin'], verbose=0)
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('ROHF did not converge')
    nstates = 2
    mc = mcscf.CASSCF(mf, spec['nact'], mol.nelectron - 2 * spec['ncore'])
    mc.verbose = 0
    mc.conv_tol = 1e-10
    mc.max_cycle = 200
    mc.state_average_([1.0 / nstates] * nstates)
    mc.fcisolver.nroots = nstates
    mc.kernel()
    if not mc.converged:
        raise RuntimeError('CASSCF did not converge')

    es = np.asarray(mc.fcisolver.e_states)[:nstates]
    c = np.asarray(mc.mo_coeff)
    act = slice(spec['ncore'], spec['ncore'] + spec['nact'])
    ca = c[:, act]
    ci = mc.ci
    dm_sa = np.asarray(mc.make_rdm1())

    d_dict = {}
    for i in range(nstates):
        for j in range(nstates):
            d_dict[i, j] = np.asarray(mc.fcisolver.trans_rdm1(
                ci[i], ci[j], spec['nact'], mc.nelecas))
    d_arr = np.asarray([[d_dict[i, j] for j in range(nstates)]
                        for i in range(nstates)])
    return dict(mol=mol, mc=mc, es=es, ca=ca, dm_sa=dm_sa, d=d_arr,
                nstates=nstates)


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1.0)


def _fd_table(f, ad, steps):
    out = {}
    for h in steps:
        fd = float((f(h) - f(-h)) / (2.0 * h))
        out[f'{h:.0e}'] = dict(fd=fd, abs_dev=abs(ad - fd),
                               rel_dev=_rel(ad, fd))
    return out


def run(name, spec):
    t0 = time.time()
    data = build_reference(spec)
    mol = data['mol']
    es = data['es']
    ca = data['ca']
    dm_sa = data['dm_sa']
    d_arr = data['d']
    nstates = data['nstates']
    nact = ca.shape[1]
    ncore = int(spec['ncore'])
    nao = mol.nao_nr()

    # ---- numpy value layer ----
    h1_mo = dsomf.somf_matrix(mol, ca, None, mode='one_electron')
    h_mo = dsomf.somf_matrix(mol, ca, dm_sa, mode='somf')
    t_np = ref.transition_tensor(
        {(i, j): d_arr[i, j] for i in range(nstates)
         for j in range(nstates)}, h_mo, nstates)
    t1_np = ref.transition_tensor(
        {(i, j): d_arr[i, j] for i in range(nstates)
         for j in range(nstates)}, h1_mo, nstates)
    s_somf_np = ref.si_splitting(es, t_np, nstates)[0]
    s_1e_np = ref.si_splitting(es, t1_np, nstates)[0]

    # ---- JAX value layer ----
    ints = jsomf.somf_integrals(mol)
    ca_j = jnp.asarray(ca)
    es_j = jnp.asarray(es)
    d_j = jnp.asarray(d_arr)
    p_j = jnp.asarray(dm_sa)

    def splitting(p, c):
        return jsomf.somf_splitting_jax(p, ints.h1, ints.g, c, es_j, d_j)

    s_somf_jax = float(splitting(p_j, ca_j))
    h1_mo_jax = jsomf.somf_matrix_jax(None, ints.h1, ints.g,
                                      C=ca_j, mode='one_electron')
    t1_jax = jsomf.transition_tensor_jax(d_j, h1_mo_jax)
    s_1e_jax = float(jsomf.si_splitting_jax(es_j, t1_jax))

    value_check = dict(
        numpy_somf_cm=s_somf_np,
        jax_somf_cm=s_somf_jax,
        abs_dev_cm=abs(s_somf_jax - s_somf_np),
        rel_dev=abs(s_somf_jax - s_somf_np) / max(abs(s_somf_np), 1.0),
        numpy_one_electron_cm=s_1e_np,
        jax_one_electron_cm=s_1e_jax,
        one_electron_abs_dev_cm=abs(s_1e_jax - s_1e_np),
        t_component_max=[float(np.abs(t_np[:, :, k]).max())
                         for k in range(3)],
    )

    # ---- AD versus FD: global density scale ----
    f_scale = jax.jit(lambda s: splitting((1.0 + s) * p_j, ca_j))
    ad_scale = float(jax.grad(f_scale)(0.0))
    fd_scale = _fd_table(f_scale, ad_scale, (1e-3, 1e-4, 1e-5))

    # ---- AD versus FD: symmetric density direction ----
    rng = np.random.default_rng(20261026)
    dp = rng.normal(size=(nao, nao))
    dp = 0.5 * (dp + dp.T)
    dp_j = jnp.asarray(dp)
    f_dir = jax.jit(lambda s: splitting(p_j + s * dp_j, ca_j))
    ad_dir = float(jax.grad(f_dir)(0.0))
    fd_dir = _fd_table(f_dir, ad_dir, (1e-3, 1e-4, 1e-5))

    # ---- AD versus FD: active-orbital rotation ----
    d_act = ca.T @ dm_sa @ ca
    p_inactive = dm_sa - ca @ d_act @ ca.T
    g = rng.normal(size=(nact, nact))
    a_rot = (g - g.T) / np.linalg.norm(g - g.T)
    a_j = jnp.asarray(a_rot)
    p_inact_j = jnp.asarray(p_inactive)
    d_act_j = jnp.asarray(d_act)

    def f_rot(theta):
        u = expm(theta * a_j)
        cr = ca_j @ u
        pr = p_inact_j + cr @ d_act_j @ cr.T
        return jsomf.somf_splitting_jax(pr, ints.h1, ints.g, cr, es_j, d_j)

    f_rot = jax.jit(f_rot)
    ad_rot = float(jax.grad(f_rot)(0.0))
    fd_rot = _fd_table(f_rot, ad_rot, (1e-3, 1e-4, 1e-5))

    s_ovlp = jnp.asarray(mol.intor('int1e_ovlp'))
    trace_before = float(jnp.einsum('pq,pq->', p_j, s_ovlp))
    u_probe = expm(0.3 * a_j)
    cr_probe = ca_j @ u_probe
    trace_rot = float(jnp.einsum(
        'pq,pq->', p_inact_j + cr_probe @ d_act_j @ cr_probe.T, s_ovlp))

    out = dict(
        system=name,
        nao=int(nao), nelec=int(mol.nelectron),
        ncas=int(nact), ncore=ncore, nstates=nstates,
        cas_energy=float(data['mc'].e_tot),
        scalar_gap_cm=float((es[1] - es[0]) * ref.CM),
        value=value_check,
        orca=dict(one_electron=spec['orca_1e'], somf=spec['orca_somf']),
        experiment_cm=EXPERIMENT_CM.get(name),
        deviation=dict(
            one_electron_cm=s_1e_jax - spec['orca_1e'],
            one_electron_rel=(s_1e_jax - spec['orca_1e']) / spec['orca_1e'],
            somf_cm=s_somf_jax - spec['orca_somf'],
            somf_rel=(s_somf_jax - spec['orca_somf']) / spec['orca_somf'],
        ),
        ratio=dict(
            this=s_somf_jax / s_1e_jax,
            orca=spec['orca_somf'] / spec['orca_1e'],
            rel=(s_somf_jax / s_1e_jax)
            / (spec['orca_somf'] / spec['orca_1e']) - 1.0,
        ),
        ad_fd=dict(
            density_scale=dict(ad=ad_scale, fd=fd_scale),
            density_direction=dict(ad=ad_dir, fd=fd_dir),
            orbital_rotation=dict(ad=ad_rot, fd=fd_rot),
        ),
        density_trace=dict(before=trace_before, at_rotation_zero=trace_rot),
        wall_s=float(time.time() - t0),
    )
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', default='oh,of')
    ap.add_argument('--out', default='results_somf_jax.json')
    args = ap.parse_args()
    results = {}
    for name in args.systems.split(','):
        name = name.strip()
        print(f'===== {name} =====')
        results[name] = run(name, SYSTEMS[name])
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2)
    print('written', args.out)


if __name__ == '__main__':
    main()
