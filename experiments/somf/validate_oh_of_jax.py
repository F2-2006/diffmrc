#!/usr/bin/env python
"""Acceptance run for the differentiable SOMF layer on OH and OF.

WARNING (2026-11-02): deprecated as a value source for absolute splittings.
This script inherits ``validate_oh_of.py`` and therefore builds every
state-interaction matrix from pyscf ``trans_rdm1`` transition densities, which
are spin-summed charge densities (D_alpha + D_beta).  The Breit--Pauli z
component requires the spin difference (D_alpha - D_beta)/2, so the printed
one-electron and SOMF splittings are systematically high by a system-dependent
factor (0.3% for OH, 0.4% for OF).  The numbers in
``results_somf_jax.json`` are kept as history and must not be cited as primary
values.  The numpy-versus-JAX and AD-versus-FD comparisons remain internally
valid because both sides use the same spin-summed density, but they do not
certify the absolute splitting.  Correct route: the determinant-space
spin-orbital CI of ``core/spinor_ci.py``, driven by the Wave 18 SOC benchmark
(``code/soc_bench.py``).  The sibling ``experiments/somf/validate_oh_of.py``
and ``experiments/somf_ri/validate_oh_of_ri.py`` share the same transition
density and inherit the offset.

Reproduces the value-layer setup of ``validate_oh_of.py`` (state-averaged
CASSCF, 6-31G, two roots) and adds the differentiable path.

Two reference generators are run:

* ``reference``: exact pyscf ROHF followed by exact pyscf CASSCF.  This is
  the path that matches ``validate_oh_of.py`` and the Wave 7 numbers.
* ``rijonx``: ``driver.ri_scf_driver.RIJONX_ROHF`` (RI-J Coulomb, exact
  exchange) followed by the same exact pyscf CASSCF, started from the
  RIJONX orbitals.  The CASSCF and the SOMF operator remain four-center;
  SOMF has no RI variant and no RI approximation is used in it.

For both paths:

1. compares the JAX SOMF splitting with the numpy value layer and with the
   ORCA 6.1.1 reference (SOMF and one-electron);
2. differentiates the splitting with respect to the reference density,
   using a global density scale, a symmetric density direction, and an
   active-orbital rotation, and compares every reverse-mode derivative with
   a central finite difference.

SOMF integral storage is guarded at 20 GiB.  The SI assembly and every
AD-FD function are jitted; a persistent compilation cache is used when
``DIFFMRC_JAX_CACHE`` is set.  The reference CASSCF uses pyscf's FCI
solver: the diffmrc blocked-CI / ``ci_solver='auto'`` machinery belongs to
the diffmrc CASSCF/NEVPT2 chain, not to this pyscf reference path.

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
jax.config.update("jax_enable_x64", True)
_jax_cache = _os.environ.get('DIFFMRC_JAX_CACHE')
if _jax_cache:
    jax.config.update('jax_compilation_cache_dir', _jax_cache)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)
import jax.numpy as jnp
from jax.scipy.linalg import expm

from diffmrc.core import somf as dsomf
from diffmrc.core import somf_jax as jsomf

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import validate_oh_of as ref  # noqa: E402

SYSTEMS = dict(ref.SYSTEMS)
EXPERIMENT_CM = {'oh': 139.8, 'of': None}

# Memory guard for the four-center explicit SOMF kernel.
MEMORY_GUARD = 20 * 2 ** 30


def build_reference(spec, mo_coeff=None, mo_occ=None, scf_energy=None):
    """Exact pyscf CASSCF from either a fresh ROHF or supplied orbitals."""
    mol = gto.M(atom=spec['atom'], basis='6-31g', unit='Bohr',
                spin=spec['spin'], verbose=0)
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    if mo_coeff is None:
        mf.kernel()
        if not mf.converged:
            raise RuntimeError('ROHF did not converge')
    else:
        mf.mo_coeff = np.asarray(mo_coeff)
        mf.mo_occ = np.asarray(mo_occ)
        mf.converged = True
        if scf_energy is not None:
            mf.e_tot = float(scf_energy)
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


def rijonx_orbitals(spec):
    """Converged RIJONX (RI-J Coulomb, exact K) ROHF orbitals."""
    from pyscfad import gto as adgto
    from diffmrc.driver.ri_scf_driver import RIJONX_ROHF
    mol_ad = adgto.Mole(atom=spec['atom'], basis='6-31g', unit='Bohr',
                        spin=spec['spin'], verbose=0)
    mf = RIJONX_ROHF(mol_ad)
    mf.conv_tol = 1e-12
    mf.verbose = 0
    t0 = time.time()
    mf.kernel()
    wall = time.time() - t0
    if not mf.converged:
        raise RuntimeError('RIJONX SCF did not converge')
    return dict(mo_coeff=np.asarray(mf.mo_coeff),
                mo_occ=np.asarray(mf.mo_occ),
                e_tot=float(mf.e_tot),
                naux=int(mf.ri_state().auxmol.nao_nr()),
                wall_s=wall)


def evaluate(mol, data, spec):
    """Value layer and JAX SOMF splitting for one reference data set."""
    es = data['es']
    ca = data['ca']
    dm_sa = data['dm_sa']
    d_arr = data['d']
    nstates = data['nstates']
    nao = mol.nao_nr()

    h1_mo = dsomf.somf_matrix(mol, ca, None, mode='one_electron')
    h_mo = dsomf.somf_matrix(mol, ca, dm_sa, mode='somf')
    d_dict = {(i, j): d_arr[i, j] for i in range(nstates)
              for j in range(nstates)}
    t_np = ref.transition_tensor(d_dict, h_mo, nstates)
    t1_np = ref.transition_tensor(d_dict, h1_mo, nstates)
    s_somf_np = ref.si_splitting(es, t_np, nstates)[0]
    s_1e_np = ref.si_splitting(es, t1_np, nstates)[0]

    ints = jsomf.somf_integrals(mol, memory_limit=MEMORY_GUARD)
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

    kernel_bytes = jsomf.explicit_kernel_bytes(nao)
    assert kernel_bytes <= MEMORY_GUARD

    return dict(
        mol=mol, ints=ints, p_j=p_j, ca_j=ca_j, es_j=es_j, d_j=d_j,
        splitting=splitting,
        value=dict(
            numpy_somf_cm=s_somf_np,
            jax_somf_cm=s_somf_jax,
            abs_dev_cm=abs(s_somf_jax - s_somf_np),
            rel_dev=abs(s_somf_jax - s_somf_np) / max(abs(s_somf_np), 1.0),
            numpy_one_electron_cm=s_1e_np,
            jax_one_electron_cm=s_1e_jax,
            one_electron_abs_dev_cm=abs(s_1e_jax - s_1e_np),
            t_component_max=[float(np.abs(t_np[:, :, k]).max())
                             for k in range(3)],
        ),
        somf_cm=s_somf_jax,
        one_electron_cm=s_1e_jax,
        kernel_bytes=kernel_bytes,
        memory_guard_bytes=MEMORY_GUARD,
    )


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1.0)


def _fd_table(f, ad, steps):
    out = {}
    for h in steps:
        fd = float((f(h) - f(-h)) / (2.0 * h))
        out[f'{h:.0e}'] = dict(fd=fd, abs_dev=abs(ad - fd),
                               rel_dev=_rel(ad, fd))
    return out


def ad_fd_tests(ev, data, spec, seed=20261026):
    """Density scale, density direction, and active-orbital rotation."""
    mol = ev['mol']
    splitting = ev['splitting']
    p_j = ev['p_j']
    ca_j = ev['ca_j']
    es_j = ev['es_j']
    d_j = ev['d_j']
    ints = ev['ints']
    ca = data['ca']
    dm_sa = data['dm_sa']
    nao = mol.nao_nr()
    nact = ca.shape[1]

    f_scale = jax.jit(lambda s: splitting((1.0 + s) * p_j, ca_j))
    ad_scale = float(jax.grad(f_scale)(0.0))
    fd_scale = _fd_table(f_scale, ad_scale, (1e-3, 1e-4, 1e-5))

    rng = np.random.default_rng(seed)
    dp = rng.normal(size=(nao, nao))
    dp = 0.5 * (dp + dp.T)
    dp_j = jnp.asarray(dp)
    f_dir = jax.jit(lambda s: splitting(p_j + s * dp_j, ca_j))
    ad_dir = float(jax.grad(f_dir)(0.0))
    fd_dir = _fd_table(f_dir, ad_dir, (1e-3, 1e-4, 1e-5))

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

    return dict(
        density_scale=dict(ad=ad_scale, fd=fd_scale),
        density_direction=dict(ad=ad_dir, fd=fd_dir),
        orbital_rotation=dict(ad=ad_rot, fd=fd_rot),
        density_trace=dict(before=trace_before, at_rotation_zero=trace_rot),
    )


def run(name, spec):
    t0 = time.time()
    data = build_reference(spec)
    ev = evaluate(data['mol'], data, spec)
    mol = data['mol']
    es = data['es']
    adfd = ad_fd_tests(ev, data, spec)

    ri = rijonx_orbitals(spec)
    data_ri = build_reference(spec, mo_coeff=ri['mo_coeff'],
                              mo_occ=ri['mo_occ'], scf_energy=ri['e_tot'])
    ev_ri = evaluate(data_ri['mol'], data_ri, spec)
    adfd_ri = ad_fd_tests(ev_ri, data_ri, spec)

    s_1e = ev['one_electron_cm']
    s_somf = ev['somf_cm']
    out = dict(
        system=name,
        nao=int(mol.nao_nr()), nelec=int(mol.nelectron),
        ncas=int(spec['nact']), ncore=int(spec['ncore']), nstates=2,
        cas_energy=float(data['mc'].e_tot),
        scalar_gap_cm=float((es[1] - es[0]) * ref.CM),
        value=ev['value'],
        orca=dict(one_electron=spec['orca_1e'], somf=spec['orca_somf']),
        experiment_cm=EXPERIMENT_CM.get(name),
        deviation=dict(
            one_electron_cm=s_1e - spec['orca_1e'],
            one_electron_rel=(s_1e - spec['orca_1e']) / spec['orca_1e'],
            somf_cm=s_somf - spec['orca_somf'],
            somf_rel=(s_somf - spec['orca_somf']) / spec['orca_somf'],
        ),
        ratio=dict(
            this=s_somf / s_1e,
            orca=spec['orca_somf'] / spec['orca_1e'],
            rel=(s_somf / s_1e)
            / (spec['orca_somf'] / spec['orca_1e']) - 1.0,
        ),
        ad_fd=adfd,
        memory=dict(guard_bytes=MEMORY_GUARD,
                    kernel_bytes=ev['kernel_bytes'],
                    kernel_gib=ev['kernel_bytes'] / 2 ** 30),
        accelerated=dict(
            scf=dict(route='RIJONX (RI-J Coulomb, exact exchange)',
                     naux=ri['naux'], energy=ri['e_tot'],
                     wall_s=ri['wall_s']),
            casscf=dict(energy=float(data_ri['mc'].e_tot),
                        energy_shift=float(data_ri['mc'].e_tot)
                        - float(data['mc'].e_tot)),
            value=ev_ri['value'],
            somf_shift_cm=ev_ri['somf_cm'] - s_somf,
            ad_fd=adfd_ri,
        ),
        accelerations=dict(
            scf='RIJONX for the ROHF stage; the CASSCF and SOMF integrals '
                'stay four-center',
            somf='four-center explicit; no RI-SOMF',
            si='jax.jit',
            ad_fd='jax.jit',
            jax_cache=bool(_jax_cache),
            reference_ci='pyscf FCI; diffmrc blocked CI and '
                         "ci_solver='auto' are not on this path",
        ),
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
