"""Cost benchmark for the CASSCF-layer nuclear Hessian.

Three routes are compared on the same joint energy and the same
stationary orbital point:

    dense       core.hessian.implicit_hessian        (vectorized AD)
    columnwise  core.hessian.implicit_hessian_columnwise (one tangent/column)
    hvp         core.hessian.implicit_hessian_hvp    (matrix-free, LSMR)

and the unrolled route (a fixed number of Newton steps expanded into the
graph) is available as a fourth mode.  Each route runs in its own
process, so ``ru_maxrss`` is that route's peak rather than the maximum
over the whole benchmark.

The orbital point is prepared once with a theta-only energy whose SCF
reference is evaluated outside the differentiable function; the Hessian
routes use ``make_joint_energy_fn``, whose SCF is inside the graph.  The
two describe the same CASCI surface at the reference geometry.

Examples
--------
    python hessian_cost_benchmark.py prep --system h2o
    python hessian_cost_benchmark.py run --system h2o --route dense
    python hessian_cost_benchmark.py run --system h2o --route columnwise
    python hessian_cost_benchmark.py run --system h2o --route hvp
    python hessian_cost_benchmark.py run --system h2o --route unrolled --steps 3
    python hessian_cost_benchmark.py compare --system h2o
"""
import argparse
import gc
import json
import os
import resource
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from pyscfad import gto as pgto  # noqa: E402
from pyscfad import scf as pscf  # noqa: E402

from diffmrc.core.casci import cs_casci_blocked  # noqa: E402
from diffmrc.core.hessian import (implicit_hessian,  # noqa: E402
                                  implicit_hessian_columnwise,
                                  implicit_hessian_hvp, make_joint_energy_fn,
                                  make_theta_energy_fn)
from diffmrc.core.orblib import orbital_pairs  # noqa: E402
from diffmrc.core.rotation import block_rotation  # noqa: E402
from diffmrc.driver.orbitals import converge_orbital_point  # noqa: E402

SYSTEMS = {
    'h2o': dict(
        atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
        basis='6-31g', ncore=4, nact=2, nele=2),
    'co': dict(
        atom='C 0 0 -0.5643; O 0 0 0.5643',
        basis='6-31g', ncore=6, nact=2, nele=2),
    'lih': dict(
        atom='Li 0 0 0; H 0 0 1.6',
        basis='sto-3g', ncore=1, nact=2, nele=2),
}


def build_mol(system):
    spec = SYSTEMS[system]
    mol = pgto.Mole()
    mol.atom = spec['atom']
    mol.basis = spec['basis']
    mol.verbose = 0
    mol.build()
    return mol


def system_info(mol, system):
    spec = SYSTEMS[system]
    nmo = int(mol.nao_nr())
    pairs = orbital_pairs(spec['ncore'], spec['nact'], nmo,
                          include_cv=True)
    return dict(system=system, basis=spec['basis'], nmo=nmo,
                natm=int(mol.natm), ncore=spec['ncore'],
                nact=spec['nact'], nele=spec['nele'],
                npar=len(pairs), pairs=pairs)


def theta_energy(mol, info, conv_tol=1e-12):
    """Theta-only CASCI energy with the SCF reference evaluated once.

    The AO integrals travel as runtime arguments rather than through the
    closure: ``jax.jit`` embeds closed-over arrays in the traced graph,
    and the ``nao**4`` ERI is exactly the object the optimizer's
    docstring warns about.
    """
    mf = pscf.RHF(mol)
    mf.conv_tol = conv_tol
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('reference RHF did not converge')
    c = mf.mo_coeff
    h = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    e = mol.intor('int2e', aosym='s1')
    en = mol.energy_nuc()
    ncore, nact, nele = info['ncore'], info['nact'], info['nele']
    nmo, pairs = info['nmo'], info['pairs']
    nm = ncore + nact

    def E(theta, c, h, e, en):
        c2 = c @ block_rotation(theta, nmo, pairs)
        h1 = jnp.einsum('ai,ab,bj->ij', c2, h, c2)[:nm, :nm]
        e1 = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', c2, c2, c2, c2, e)
        e1 = e1[:nm, :nm, :nm, :nm]
        return cs_casci_blocked(h1, e1, ncore=ncore, nact=nact,
                                nele_act=nele)[0] + en

    return E, (c, h, e, en)


def theta_path(system):
    return os.path.join('results', f'hessian_theta_{system}.json')


def hessian_path(system, route):
    return os.path.join('results', f'hessian_{system}_{route}.npy')


def summary_path(system, route):
    return os.path.join('results', f'hessian_cost_{system}_{route}.json')


def peak_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2


def cmd_prep(args):
    mol = build_mol(args.system)
    info = system_info(mol, args.system)
    E_theta, arrays = theta_energy(mol, info)
    npar = info['npar']
    print(f"prep {args.system}: nmo={info['nmo']} natm={info['natm']} "
          f"npar={npar}", flush=True)
    t0 = time.time()
    point = converge_orbital_point(E_theta, npar,
                                   theta0=np.zeros(npar), arrays=arrays)
    wall = time.time() - t0
    print(f"  E = {point.energy:.12f}  |g| = {point.grad_norm:.3e}  "
          f"polished={getattr(point, 'polished', None)}  [{wall:.0f}s]",
          flush=True)
    out = dict(info, theta=[float(x) for x in np.asarray(point.theta)],
               energy=float(point.energy),
               grad_norm=float(point.grad_norm),
               prep_wall=float(wall))
    os.makedirs('results', exist_ok=True)
    with open(theta_path(args.system), 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"  wrote {theta_path(args.system)}", flush=True)


def load_theta(system):
    with open(theta_path(system)) as fh:
        data = json.load(fh)
    return np.asarray(data['theta'], dtype=float), data


def cmd_run(args):
    mol = build_mol(args.system)
    info = system_info(mol, args.system)
    theta, saved = load_theta(args.system)
    if int(theta.size) != info['npar']:
        raise RuntimeError(
            f"theta has {theta.size} parameters, system has "
            f"{info['npar']}")
    npar, natm = info['npar'], info['natm']
    print(f"run {args.system} route={args.route}: npar={npar} "
          f"natm={natm} theta|g|={saved['grad_norm']:.2e}", flush=True)

    def scf_tight(m):
        mf = pscf.RHF(m)
        mf.conv_tol = 1e-12
        return mf

    args_system = (info['ncore'], info['nact'], info['nele'], info['nmo'],
                   info['pairs'])
    t_setup = time.time()
    E_joint = make_joint_energy_fn(*args_system, pscf, scf=scf_tight)
    E_tt = None
    if args.route in ('columnwise', 'hvp'):
        E_tt = make_theta_energy_fn(mol, *args_system, pscf,
                                    scf=scf_tight)
    setup_s = time.time() - t_setup

    t0 = time.time()
    meta = {}
    if args.route == 'dense':
        res = implicit_hessian(mol, theta, E_joint)
        eig = np.linalg.eigvalsh(0.5 * (res.H_tt + res.H_tt.T))
        meta = dict(htt_eig_min=float(eig[0]), htt_eig_max=float(eig[-1]),
                    htt_n_negative=int(np.sum(eig < 0.0)),
                    dtheta_dR_max=float(res.dtheta_dR_max),
                    response_max=float(np.abs(res.H_response).max()))
    elif args.route == 'columnwise':
        res = implicit_hessian_columnwise(mol, theta, E_joint, E_tt=E_tt)
        eig = np.linalg.eigvalsh(0.5 * (res.H_tt + res.H_tt.T))
        meta = dict(htt_eig_min=float(eig[0]), htt_eig_max=float(eig[-1]),
                    htt_n_negative=int(np.sum(eig < 0.0)),
                    dtheta_dR_max=float(res.dtheta_dR_max),
                    response_max=float(np.abs(res.H_response).max()))
    elif args.route == 'hvp':
        res = implicit_hessian_hvp(
            mol, theta, E_joint, E_tt=E_tt,
            atol=args.atol, btol=args.btol, maxiter=args.maxiter,
            deflate=not args.no_deflate,
            deflate_cond_target=args.deflate_cond_target,
            deflate_tol_rel=args.deflate_tol_rel,
            stall_iters=args.stall_iters,
            time_limit=args.time_limit)
        meta = dict(lsmr_iterations=res.lsmr_iterations.tolist(),
                    lsmr_istop=res.lsmr_istop.tolist(),
                    lsmr_residual=res.lsmr_residual.tolist(),
                    lsmr_total_iterations=int(res.total_iterations),
                    lsmr_maxiter=res.maxiter,
                    lsmr_atol=float(res.atol),
                    lsmr_btol=float(res.btol),
                    solver_reason=res.solver_reason,
                    deflation=res.deflation,
                    dtheta_dR_max=res.dtheta_dR_max)
    elif args.route == 'unrolled':
        def E_unrolled(m, n_steps):
            E_of, arrays = theta_energy(m, info)
            th_sg = jnp.asarray(theta)

            def body(_, t):
                g = jax.grad(E_of, argnums=0)(t, *arrays)
                H = jax.hessian(E_of, argnums=0)(t, *arrays)
                H = 0.5 * (H + H.T) + args.reg * jnp.eye(npar)
                return t - jnp.linalg.solve(H, g)

            th_f = jax.lax.fori_loop(0, n_steps, body, th_sg)
            return E_of(th_f, *arrays)

        H = jax.hessian(lambda m: E_unrolled(m, args.steps))(mol)
        H = np.asarray(H.coords.coords).reshape(3 * natm, 3 * natm)
        H = 0.5 * (H + H.T)
        meta = dict(steps=int(args.steps), reg=float(args.reg))

        class _Result:
            H_total = H
        res = _Result()
    else:
        raise ValueError(f'unknown route {args.route}')

    wall = time.time() - t0
    H = np.asarray(res.H_total, dtype=float)
    peak = peak_gib()
    print(f"  wall = {wall:.1f}s  peak_rss = {peak:.2f} GiB  "
          f"|H|max = {np.abs(H).max():.6e}", flush=True)
    if args.route == 'hvp':
        print(f"  LSMR iterations = {res.lsmr_iterations.tolist()}", flush=True)

    os.makedirs('results', exist_ok=True)
    np.save(hessian_path(args.system, args.route), H)
    out = dict(system=args.system, route=args.route, npar=npar, natm=natm,
               wall_s=float(wall), setup_s=float(setup_s),
               peak_rss_gib=float(peak),
               hessian_path=hessian_path(args.system, args.route),
               theta_path=theta_path(args.system),
               grad_norm_theta=float(saved['grad_norm']), **meta)
    with open(summary_path(args.system, args.route), 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"  wrote {summary_path(args.system, args.route)}", flush=True)
    del res, H
    gc.collect()


def cmd_compare(args):
    from diffmrc.analysis.vibrations import wavenumbers

    mol = build_mol(args.system)
    coords = np.asarray(mol.atom_coords())
    masses = np.asarray(mol.atom_mass_list(isotope_avg=True))
    rows = []
    ref = None
    ref_freqs = None
    for route in args.routes:
        path = hessian_path(args.system, route)
        if not os.path.exists(path):
            continue
        H = np.load(path)
        asym = float(np.abs(H - H.T).max())
        if ref is None:
            ref = H
            diff = 0.0
        else:
            diff = float(np.abs(H - ref).max())
        scale = max(float(np.abs(ref).max()), 1.0)
        nu = wavenumbers(H, coords, masses)
        freqs = np.asarray(nu.freqs_cm)
        if ref_freqs is None:
            ref_freqs = freqs
            dfreq = 0.0
        else:
            n = min(freqs.size, ref_freqs.size)
            dfreq = float(np.abs(freqs[:n] - ref_freqs[:n]).max())
        rows.append(dict(route=route, n=int(H.shape[0]), asymmetry=asym,
                         max_abs_diff_vs_first=diff,
                         relative_diff_vs_first=diff / scale,
                         n_imag=int(nu.n_imag), dfreq_cm_vs_first=dfreq,
                         freqs_cm=[round(float(x), 3) for x in freqs],
                         zero_modes=[float(z) for z in
                                     nu.zero_mode_magnitudes]))
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prep')
    p.add_argument('--system', choices=sorted(SYSTEMS), required=True)
    p.set_defaults(func=cmd_prep)

    p = sub.add_parser('run')
    p.add_argument('--system', choices=sorted(SYSTEMS), required=True)
    p.add_argument('--route', required=True,
                   choices=['dense', 'columnwise', 'hvp', 'unrolled'])
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--reg', type=float, default=1.0e-6)
    p.add_argument('--atol', type=float, default=1.0e-12)
    p.add_argument('--btol', type=float, default=1.0e-12)
    p.add_argument('--maxiter', type=int, default=None)
    p.add_argument('--no-deflate', action='store_true',
                   help='solve on the full operator (old behaviour)')
    p.add_argument('--deflate-cond-target', type=float, default=100.0)
    p.add_argument('--deflate-tol-rel', type=float, default=1e-8)
    p.add_argument('--stall-iters', type=int, default=40)
    p.add_argument('--time-limit', type=float, default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('compare')
    p.add_argument('--system', choices=sorted(SYSTEMS), required=True)
    p.add_argument('--routes', nargs='+',
                   default=['dense', 'columnwise', 'hvp', 'unrolled'])
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
