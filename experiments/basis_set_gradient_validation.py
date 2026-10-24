"""Basis-set independence of the relaxed NEVPT2 nuclear gradient.

System: H2O/cc-pVDZ CAS(2,2), ncore=4, along the symmetric stretch.
The same staged protocol that validated H2O/6-31G and LiH/6-31G:

    prep      tight SCF reference, CASSCF-stationary theta* (multi-start)
    si        Si-class channel invariants at theta* (masked vs exact)
    response  H_tt by central differences of the theta gradient; M along
              the mode by central differences of the aligned theta gradient
    base      aligned frozen AD, Z-vector response, relaxed AD
    fd:<d>    aligned central differences: frozen and relaxed paths, the
              latter with a linear-response prediction and a tight
              L-BFGS-B theta solve at each displaced geometry
    summary   relative deviations against the AD values

The chain is the exact-Si one (``nevpt2_zvector``): the release value mask
in the Si class is a discontinuity when a channel norm crosses 1e-12, and
the derivative at the base point is not the masked slope.  The Si stage
records whether such channels exist at this basis.

Stages checkpoint into one JSON; re-running a finished stage is a no-op.
Environment: BSV_SYS (default H2O_DZ22), BSV_OUT (default results/...),
DIFFMRC_SRC (path to src), the usual thread variables, JAX_PLATFORMS=cpu.
"""
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["MKL_CBWR"] = "COMPATIBLE"
_src = os.environ.get('DIFFMRC_SRC')
if _src:
    sys.path.insert(0, _src)

import numpy as np                                                # noqa: E402
import jax                                                        # noqa: E402
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp                                      # noqa: E402
from pyscfad import gto as pgto, scf as pscf                      # noqa: E402

import diffmrc.core.nevpt2.impl as impl_mod                       # noqa: E402
import diffmrc.driver.nevpt2 as drv                               # noqa: E402
from diffmrc.core.eigh import eigh_cluster_ift as _EIGH_ORIG      # noqa: E402
from diffmrc.core.nevpt2.core import (e2_core, make_a22_full,     # noqa: E402
                                      make_a23, make_a25)
from diffmrc.core.nevpt2.impl import (cas_dets_for2,              # noqa: E402
                                      nevpt2_e2_full2)
from diffmrc.core.orblib import orbital_pairs                     # noqa: E402
from diffmrc.driver.nevpt2 import (_casscf_solution,              # noqa: E402
                                   _components_from, _spin_links)
from diffmrc.driver.nevpt2_zvector import (                       # noqa: E402
    _components_from_si_exact, _core_blocks, _e2_core_exact_si,
    align_reference_blocks, make_nevpt2_joint_fn_si_exact,
    nevpt2_e2_full2_si_exact, orbital_zvector, si_channel_table,
    si_energy_from_invariants)

_TOL = 1e-4
impl_mod.eigh_cluster_ift = lambda a, tol=_TOL: _EIGH_ORIG(a, tol=tol)
drv.eigh_cluster_ift = lambda a, tol=_TOL: _EIGH_ORIG(a, tol=tol)

_H2O = "O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880"
_SYSTEMS = {
    'H2O_DZ22': dict(atom=_H2O, basis='cc-pvdz', nact=2, nele=2),
    'H2O_631G22': dict(atom=_H2O, basis='6-31g', nact=2, nele=2),
}
SYS = os.environ.get('BSV_SYS', 'H2O_DZ22')
ATOM = _SYSTEMS[SYS]['atom']
BASIS = _SYSTEMS[SYS]['basis']
NACT = _SYSTEMS[SYS]['nact']
NELE = _SYSTEMS[SYS]['nele']
OUT = os.environ.get('BSV_OUT',
                     'results/basis_set_gradient_{}.json'.format(SYS))
STAGE = sys.argv[1] if len(sys.argv) > 1 else 'prep'
OPT = dict(maxiter=800, gtol=1e-10, ftol=1e-18, maxls=60)
GRAD_ACCEPT = 1e-3
GRAD_RESTART = 1e-8
D_HTT = 1e-5
D_M = 3e-6
D_FD = (3e-4, 3e-5)
RCOND = 1e-7
DO_RELEASE = os.environ.get('BSV_RELEASE', '1') == '1'


def rhf_tight(m):
    mf = pscf.RHF(m)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    return mf


def align_blocks_np(C_new, S, C_ref, bounds):
    """SVD polar alignment, block by block; numpy reference path."""
    C = np.array(C_new, dtype=float, copy=True)
    for lo, hi in bounds:
        if hi - lo <= 0:
            continue
        A = C[:, lo:hi].T @ S @ C_ref[:, lo:hi]
        U, _, Vt = np.linalg.svd(A)
        C[:, lo:hi] = C[:, lo:hi] @ (U @ Vt)
    return C


def displace_mode(mol, base_bohr, v, delta):
    c = base_bohr + delta * v
    m = pgto.Mole()
    m.unit = 'B'
    m.atom = [[mol.atom_symbol(k), tuple(c[k])] for k in range(mol.natm)]
    m.basis = BASIS
    m.build()
    return m


def stretch_mode(mol):
    c = np.asarray(mol.atom_coords())
    v = np.zeros_like(c)
    for k in range(1, mol.natm):
        d = c[k] - c[0]
        v[k] = d / np.linalg.norm(d)
    v[0] = -v[1:].sum(axis=0)
    return v


def emit(out):
    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=1, default=str)


def main():
    t_start = time.time()
    mol = pgto.Mole()
    mol.atom = ATOM
    mol.basis = BASIS
    mol.build()
    nmo = mol.nao_nr()
    ncore = mol.nelectron // 2 - NELE // 2
    nocc = ncore + NACT
    natm = mol.natm
    pairs = orbital_pairs(ncore, NACT, nmo, include_cv=True)
    npar = len(pairs)
    base_bohr = np.asarray(mol.atom_coords())
    v_mode = stretch_mode(mol)
    bounds = [(0, ncore), (ncore, ncore + NACT), (ncore + NACT, nmo)]
    dets = cas_dets_for2(ncore, NACT, NELE)
    la, lb = _spin_links(NACT, NELE)
    kw = dict(ncore=ncore, nact=NACT, nele=NELE, nmo=nmo, pairs=pairs,
              dets=dets, links_a=la, links_b=lb, ci_solver='blocked',
              eig_solver='lanczos')

    def E_of(C0_, h_, e_, enuc_, th_, scale):
        E_cas, e2 = _components_from_si_exact(
            C0_, h_, e_, enuc_, th_, want_e2=bool(scale != 0.0), **kw)
        return E_cas + scale * e2

    def E_cas_args(C0_, h_, e_, enuc_, th_):
        return E_of(C0_, h_, e_, enuc_, th_, 0.0)

    def E_full_args(C0_, h_, e_, enuc_, th_):
        return E_of(C0_, h_, e_, enuc_, th_, 1.0)

    CAS_E = jax.jit(E_cas_args)
    FULL_E = jax.jit(E_full_args)
    CAS_G = jax.jit(jax.grad(E_cas_args, argnums=4))
    FULL_G = jax.jit(jax.grad(E_full_args, argnums=4))
    CAS_VG = jax.jit(jax.value_and_grad(E_cas_args, argnums=4))

    def scf_ref(m_d):
        mf_ = rhf_tight(m_d)
        mf_.kernel()
        return dict(C_raw=np.asarray(mf_.mo_coeff),
                    S=np.asarray(m_d.intor('int1e_ovlp')),
                    h=np.asarray(mf_.get_hcore()),
                    e=np.asarray(m_d.intor('int2e', aosym='s1')),
                    enuc=float(m_d.energy_nuc()),
                    e_scf=float(mf_.e_tot),
                    converged=bool(mf_.converged))

    def align_info(ref, align_ref):
        """Block SVD spectra and the SVD-vs-Newton map discrepancy."""
        out_ = {}
        for lo, hi in bounds:
            if hi - lo <= 0:
                continue
            A = ref['C_raw'][:, lo:hi].T @ ref['S'] @ align_ref[:, lo:hi]
            sv = np.linalg.svd(A, compute_uv=False)
            out_['{}_{}'.format(lo, hi)] = dict(
                sv_min=float(sv.min()), sv_max=float(sv.max()),
                sv_spread=float(sv.max() - sv.min()))
        C_al = align_blocks_np(ref['C_raw'], ref['S'], align_ref, bounds)
        C_al_jax = np.asarray(align_reference_blocks(
            jnp.asarray(ref['C_raw']), jnp.asarray(ref['S']),
            jnp.asarray(align_ref), bounds))
        return C_al, dict(blocks=out_,
                          svd_newton_max=float(np.abs(C_al - C_al_jax).max()))

    def refs_at(m_d, align_ref):
        ref = scf_ref(m_d)
        C_al, info = align_info(ref, align_ref)
        ref['C_al'] = C_al
        ref['align'] = info
        return ref

    C0, S0, h0, e0 = (None,) * 4
    enuc0 = None

    # ---- prep ---------------------------------------------------------
    out = {}
    if os.path.exists(OUT):
        out = json.load(open(OUT, encoding='utf-8'))
    out.setdefault('stages', [])
    out.update(dict(system=SYS, basis=BASIS, ncore=ncore, nact=NACT,
                    nele=NELE, nmo=nmo, npar=npar, natm=natm,
                    mode=v_mode.tolist(),
                    protocol=dict(d_htt=D_HTT, d_m=D_M, d_fd=list(D_FD),
                                  rcond=RCOND,
                                  scf=dict(conv_tol=1e-13,
                                           conv_tol_grad=1e-10))))

    base_ref = scf_ref(mol)
    C_ref = base_ref['C_raw']
    _, base_ref['align'] = align_info(base_ref, C_ref)
    ref0 = (jnp.asarray(C_ref), jnp.asarray(base_ref['h']),
            jnp.asarray(base_ref['e']), base_ref['enuc'])

    if STAGE == 'prep' and 'prep' not in out['stages']:
        from scipy.optimize import minimize

        def run_opt(x0):
            x = np.asarray(x0, dtype=float)
            for _ in range(6):
                res = minimize(
                    lambda v: float(CAS_VG(*ref0, jnp.asarray(v))[0]), x,
                    jac=lambda v: np.asarray(CAS_VG(*ref0, jnp.asarray(v))[1],
                                             dtype=float),
                    method='L-BFGS-B', options=OPT)
                x = np.asarray(res.x, dtype=float)
                if float(np.abs(CAS_VG(*ref0, jnp.asarray(x))[1]).max()) \
                        <= GRAD_RESTART:
                    break
            e = float(CAS_E(*ref0, jnp.asarray(x)))
            g = np.asarray(CAS_G(*ref0, jnp.asarray(x)))
            return e, x, float(np.abs(g).max())

        rng = np.random.default_rng(11)
        starts = [('zeros', np.zeros(npar)),
                  ('rand+', 0.05 * rng.normal(size=npar)),
                  ('rand-', -0.05 * rng.normal(size=npar)),
                  ('rand2', 0.05 * rng.normal(size=npar))]
        sols, lock = [], {}
        t0 = time.time()
        for name, x0 in starts:
            e_r, x_r, gn = run_opt(x0)
            lock[name] = dict(E=e_r, gnorm=gn,
                              theta=[float(v) for v in x_r])
            print('  start %-6s E = %.12f  |g| = %.2e  [%.0fs]'
                  % (name, e_r, gn, time.time() - t0), flush=True)
            if np.isfinite(e_r) and gn <= GRAD_ACCEPT:
                sols.append((e_r, x_r, gn, name))
        if not sols:
            raise RuntimeError('no stationary theta')
        e_star, x_star, gn_star, src = min(sols, key=lambda t: t[0])
        e_list = sorted(s[0] for s in sols)
        n_distinct = 1 + sum(1 for a, b in zip(e_list, e_list[1:])
                             if b - a > 1e-8)
        cand = []
        for e_r, x_r, gn, name in sols:
            th_c = jnp.asarray(x_r)
            e_c = float(CAS_E(*ref0, th_c))
            e_f = float(FULL_E(*ref0, th_c))
            cand.append(dict(start=name, E_casscf=e_c, E_nevpt2=e_f,
                             e2=e_f - e_c, gnorm=gn))
        out['prep'] = dict(theta=[float(x) for x in x_star],
                           E_casscf=e_star, gnorm=gn_star, selected=src,
                           lock=lock, n_distinct=n_distinct,
                           spread=float(e_list[-1] - e_list[0]),
                           solutions_sorted=sorted(
                               [dict(start=name, E=e) for e, _, _, name
                                in sols], key=lambda r: r['E']),
                           candidates=cand)
        out['stages'].append('prep')
        emit(out)
        print('  theta* from %s: E = %.12f |g| = %.2e distinct = %d [%.0fs]'
              % (src, e_star, gn_star, n_distinct, time.time() - t0),
              flush=True)
        return

    if 'prep' not in out['stages']:
        raise RuntimeError('run prep first')
    th0 = jnp.asarray(np.asarray(out['prep']['theta']))

    # ---- si: channel invariants at theta* -----------------------------
    if STAGE == 'si' and 'si' not in out['stages']:
        t0 = time.time()
        th_si = jnp.asarray(np.asarray(
            out.get('base', {}).get('theta_base', out['prep']['theta'])))
        E0c, C_can, ci2, eps, h1c, eric = _casscf_solution(
            *ref0[:3], th_si, ncore=ncore, nact=NACT, nele=NELE, nmo=nmo,
            pairs=pairs, ci_solver='blocked', eig_solver='lanczos')
        ci_j = jnp.asarray(ci2)
        blocks = _core_blocks(h1c, eric, ci_j, dets, ncore, NACT, nocc,
                              eps, la, lb)
        t = jnp.einsum('prqs->pqrs', eric)
        h2e_v_si = t[ncore:nocc, ncore:nocc, :ncore, ncore:nocc]
        h1e_v_si = blocks['h1eff'][ncore:nocc, :ncore]
        tab = si_channel_table(blocks['h1e_act'], blocks['eri_act'],
                               h2e_v_si, h1e_v_si, blocks['dm1'],
                               blocks['dm2'], blocks['dm3'], blocks['f3ca'],
                               blocks['f3ac'], eps[:ncore])
        e2_masked_core = float(e2_core(
            blocks['h1e_act'], blocks['eri_act'], blocks['dm1'],
            blocks['dm2'], blocks['dm3'], eps, ncore, NACT, blocks['f3ca'],
            blocks['f3ac'], eri_mo=eric, h1eff=blocks['h1eff']))
        e2_exact_core = float(_e2_core_exact_si(blocks, eps, ncore, NACT,
                                                eric))
        e2_rel, _ = nevpt2_e2_full2(
            h1c, eric, ci_j, dets, E0c, mo_energy=eps, ncore=ncore,
            nact=NACT, nocc=nocc, links_a=la, links_b=lb)
        e2_ex, _ = nevpt2_e2_full2_si_exact(
            h1c, eric, ci_j, dets, E0c, mo_energy=eps, ncore=ncore,
            nact=NACT, nocc=nocc, links_a=la, links_b=lb)
        e2_rel = float(e2_rel)
        e2_ex = float(e2_ex)

        def si_invariants(ci):
            b = _core_blocks(h1c, eric, ci, dets, ncore, NACT, nocc,
                             eps, la, lb)
            delta = jnp.eye(NACT)
            a22 = make_a22_full(b['h1e_act'], b['eri_act'], b['dm2'],
                                b['dm3'], b['f3ca'], b['f3ac'])
            a23 = make_a23(b['h1e_act'], b['eri_act'], b['dm1'], b['dm2'],
                           b['dm3'])
            a25 = make_a25(b['h1e_act'], b['eri_act'], b['dm1'], b['dm2'])
            dm3_h = (jnp.einsum('abef,cd->abcdef', b['dm2'], delta) * 2
                     - b['dm3'].transpose(0, 1, 3, 2, 4, 5))
            dm2_h = (jnp.einsum('ab,cd->abcd', b['dm1'], delta) * 2
                     - b['dm2'].transpose(0, 1, 3, 2))
            dm1_h = 2 * delta - b['dm1'].transpose(1, 0)
            ener = (jnp.einsum('qpir,pqrabc,baic->i', h2e_v_si, a22,
                               h2e_v_si)
                    + jnp.einsum('qpir,pqra,ai->i', h2e_v_si, a23,
                                 h1e_v_si) * 2.0
                    + jnp.einsum('pi,pa,ai->i', h1e_v_si, a25, h1e_v_si))
            norm = (jnp.einsum('qpir,rpqbac,baic->i', h2e_v_si, dm3_h,
                               h2e_v_si)
                    + jnp.einsum('qpir,rpqa,ai->i', h2e_v_si, dm2_h,
                                 h1e_v_si) * 2.0
                    + jnp.einsum('pi,pa,ai->i', h1e_v_si, dm1_h, h1e_v_si))
            return norm, ener

        eps_core = -jnp.asarray(eps[:ncore])

        def si_val(ci, mode):
            n_, e_ = si_invariants(ci)
            return si_energy_from_invariants(n_, e_, eps_core, mode=mode)

        deriv = []
        bvec = np.asarray(ci2)
        if bvec.ndim > 1:
            bvec = bvec.reshape(-1)
        for seed in (5, 7):
            rng = np.random.default_rng(seed)
            v = rng.normal(size=bvec.size)
            v = v / np.abs(v).max()
            vj = jnp.asarray(v)
            ci_flat = jnp.asarray(bvec)
            f_masked = lambda c: si_val(c, 'masked')
            f_exact = lambda c: si_val(c, 'exact')
            ad_m = float(jax.jvp(f_masked, (ci_flat,), (vj,))[1])
            ad_e = float(jax.jvp(f_exact, (ci_flat,), (vj,))[1])
            deriv.append(dict(seed=seed, ad_masked=ad_m, ad_exact=ad_e,
                              diff=ad_m - ad_e))
            print('  seed %d: dSi/dc masked %+.6e  exact %+.6e  diff %.3e'
                  % (seed, ad_m, ad_e, ad_m - ad_e), flush=True)
        out['si'] = dict(
            channels={k: [float(x) for x in v]
                      for k, v in tab.items()},
            masked_flags=[bool(abs(x) > 1e-12) for x in tab['norm']],
            min_abs_norm=float(np.abs(tab['norm']).min()),
            min_abs_den=float(np.abs(tab['denominator']).min()),
            max_abs_dfdn=float(np.abs(tab['dfdn']).max()),
            e2_core_masked=e2_masked_core, e2_core_exact=e2_exact_core,
            e2_core_diff=e2_exact_core - e2_masked_core,
            e2_total_release=e2_rel, e2_total_exact=e2_ex,
            e2_total_diff=e2_ex - e2_rel,
            derivative_probe=deriv)
        out['stages'].append('si')
        emit(out)
        print('  Si: min|n| %.3e  min|den| %.3e  max|df/dn| %.3e  '
              'core masked %.12e exact %.12e diff %.3e  total diff %.3e  '
              '[%.0fs]'
              % (out['si']['min_abs_norm'], out['si']['min_abs_den'],
                 out['si']['max_abs_dfdn'], e2_masked_core, e2_exact_core,
                 e2_exact_core - e2_masked_core, e2_ex - e2_rel,
                 time.time() - t0), flush=True)
        return

    if 'prep' not in out['stages']:
        raise RuntimeError('run prep first')
    if STAGE == 'si' and 'si' in out['stages']:
        print('si already done')
        return

    # ---- response: H_tt and M along the mode --------------------------
    if STAGE == 'response' and 'response' not in out['stages']:
        t0 = time.time()
        Mx = np.zeros((npar, npar))
        for j in range(npar):
            ej = np.zeros(npar)
            ej[j] = D_HTT
            gp = np.asarray(CAS_G(*ref0, th0 + jnp.asarray(ej)))
            gm = np.asarray(CAS_G(*ref0, th0 - jnp.asarray(ej)))
            Mx[:, j] = (gp - gm) / (2 * D_HTT)
        H_tt = 0.5 * (Mx + Mx.T)
        eig = np.linalg.eigvalsh(H_tt)
        n_neg = int((eig < -1e-6 * max(1.0, np.abs(eig).max())).sum())
        m_p = displace_mode(mol, base_bohr, v_mode, +D_M)
        m_m = displace_mode(mol, base_bohr, v_mode, -D_M)
        rp = refs_at(m_p, C_ref)
        rm = refs_at(m_m, C_ref)
        Mv = (np.asarray(CAS_G(*ref_args(rp), th0))
              - np.asarray(CAS_G(*ref_args(rm), th0))) / (2 * D_M)
        out['response'] = dict(H_tt=H_tt.tolist(),
                               eig=[float(x) for x in eig], n_neg=n_neg,
                               kappa=float(np.linalg.cond(H_tt)),
                               Mv=[float(x) for x in Mv], d_M=D_M,
                               d_htt=D_HTT,
                               align_plus=rp['align'], align_minus=rm['align'],
                               scf_plus={k: rp[k] for k in
                                         ('e_scf', 'converged')},
                               scf_minus={k: rm[k] for k in
                                          ('e_scf', 'converged')})
        out['stages'].append('response')
        emit(out)
        print('  H_tt kappa %.3e  n_neg %d  |Mv| %.3e  [%.0fs]'
              % (out['response']['kappa'], n_neg,
                 float(np.linalg.norm(Mv)), time.time() - t0), flush=True)
        return

    if 'response' not in out['stages']:
        raise RuntimeError('run response first')
    H_tt = np.asarray(out['response']['H_tt'])
    Mv = np.asarray(out['response']['Mv'])

    # ---- base ---------------------------------------------------------
    if STAGE == 'base' and 'base' not in out['stages']:
        t0 = time.time()
        # Newton polish of the stationary theta with the finite-difference
        # orbital Hessian.  A residual of 7e-7 (L-BFGS-B floor) would enter
        # the envelope-convention response term through g_cas . D; the
        # polish is accepted only if the energy does not rise and the
        # gradient falls, and the step is capped at 1e-2.
        th_cur = np.asarray(th0, dtype=float)
        E_cur = float(CAS_E(*ref0, jnp.asarray(th_cur)))
        g_cur = np.asarray(CAS_G(*ref0, jnp.asarray(th_cur)))
        polish = [dict(k=0, E=E_cur, gnorm=float(np.abs(g_cur).max()))]
        pinvH = np.linalg.pinv(H_tt, rcond=RCOND)
        for k in range(5):
            gn = float(np.abs(g_cur).max())
            if gn <= 1e-10:
                break
            step = -pinvH @ g_cur
            sn = float(np.linalg.norm(step))
            if sn > 1e-2:
                step *= 1e-2 / sn
            th_t = th_cur + step
            E_t = float(CAS_E(*ref0, jnp.asarray(th_t)))
            g_t = np.asarray(CAS_G(*ref0, jnp.asarray(th_t)))
            gn_t = float(np.abs(g_t).max())
            polish.append(dict(k=k + 1, E=E_t, gnorm=gn_t,
                               step_norm=sn,
                               accepted=bool(E_t <= E_cur + 1e-13
                                             and gn_t < gn)))
            if E_t <= E_cur + 1e-13 and gn_t < gn:
                th_cur, E_cur, g_cur = th_t, E_t, g_t
            else:
                break
        th_base = jnp.asarray(th_cur)
        print('  theta polish: |g| %.2e -> %.2e  E %.12f'
              % (polish[0]['gnorm'], polish[-1]['gnorm'], E_cur),
              flush=True)
        f1_aligned = make_nevpt2_joint_fn_si_exact(
            ncore, NACT, NELE, nmo, pairs, rhf_tight, e2_scale=1.0,
            ci_solver='blocked', eig_solver='lanczos',
            align_ref=jnp.asarray(C_ref), align_bounds=bounds)
        f0_aligned = make_nevpt2_joint_fn_si_exact(
            ncore, NACT, NELE, nmo, pairs, rhf_tight, e2_scale=0.0,
            ci_solver='blocked', eig_solver='lanczos',
            align_ref=jnp.asarray(C_ref), align_bounds=bounds)
        g_aligned = np.asarray(jax.grad(
            lambda m_: f1_aligned(m_, th_base))(mol).coords)
        g_cas = np.asarray(jax.grad(
            lambda m_: f0_aligned(m_, th_base))(mol).coords)
        E_cas0 = float(CAS_E(*ref0, th_base))
        E_full0 = float(FULL_E(*ref0, th_base))
        G1 = np.asarray(FULL_G(*ref0, th_base))
        G0 = np.asarray(CAS_G(*ref0, th_base))
        g_corr = G1 - G0
        z, zinfo = orbital_zvector(H_tt, g_corr, rcond=RCOND)
        Dv = -np.linalg.pinv(H_tt, rcond=RCOND) @ Mv
        resp_z = float(-(Mv @ z))
        resp_direct = float(g_corr @ Dv)
        g_aligned_v = float(g_aligned.reshape(-1) @ v_mode.reshape(-1))
        g_cas_v = float(g_cas.reshape(-1) @ v_mode.reshape(-1))
        # source share in the discarded pseudo-inverse subspace
        w, V = np.linalg.eigh(0.5 * (H_tt + H_tt.T))
        keep = np.abs(w) > RCOND * max(1.0, np.abs(w).max())
        src_disc = float(np.linalg.norm(V[:, ~keep].T @ g_corr)
                         / max(np.linalg.norm(g_corr), 1e-30))
        rec = dict(g_aligned=g_aligned.tolist(),
                   g_cas_aligned=g_cas.tolist(),
                   g_aligned_v=g_aligned_v, g_cas_v=g_cas_v,
                   E_casscf=E_cas0, E_nevpt2=E_full0, e2=E_full0 - E_cas0,
                   g_corr=g_corr.tolist(), z=z.tolist(),
                   zinfo={k: v for k, v in zinfo.items()},
                   source_discarded_share=src_disc,
                   Dv=Dv.tolist(), resp_z=resp_z,
                   resp_direct=resp_direct,
                   relaxed_v=g_aligned_v + resp_z,
                   theta_base=[float(x) for x in np.asarray(th_base)],
                   theta_polish=polish, align_base=base_ref['align'])
        # response under a tighter pseudo-inverse cutoff
        z2, _ = orbital_zvector(H_tt, g_corr, rcond=1e-9)
        rec['resp_z_rcond1e-9'] = float(-(Mv @ z2))
        if DO_RELEASE:
            f1_release = make_release_joint(
                ncore, NACT, NELE, nmo, pairs, rhf_tight, C_ref, bounds,
                e2_scale=1.0)
            g_release = np.asarray(jax.grad(
                lambda m_: f1_release(m_, th_base))(mol).coords)
            rec['g_release'] = g_release.tolist()
            rec['g_release_v'] = float(
                g_release.reshape(-1) @ v_mode.reshape(-1))
        out['base'] = rec
        out['stages'].append('base')
        emit(out)
        print('  aligned frozen %.10e (cas %.10e)  resp %.10e (direct '
              '%.10e)  relaxed %.10e  releasediff %.3e  [%.0fs]'
              % (g_aligned_v, g_cas_v, resp_z, resp_direct,
                 rec['relaxed_v'],
                 (rec['g_release_v'] - g_aligned_v) if DO_RELEASE else 0.0,
                 time.time() - t0), flush=True)
        return

    if 'base' not in out['stages']:
        raise RuntimeError('run base first')
    base = out['base']
    th_base = jnp.asarray(np.asarray(base['theta_base']))
    Dv0 = np.asarray(base['Dv'])
    g_frozen_v = float(base['g_aligned_v'])
    g_corr = np.asarray(base['g_corr'])

    # ---- fd:<delta> ---------------------------------------------------
    if STAGE.startswith('fd:'):
        d = float(STAGE.split(':')[1])
        key = 'fd:{:g}'.format(d)
        if key in out['stages']:
            print('already done', key)
            return
        from scipy.optimize import minimize as _minimize
        t0 = time.time()
        # The relaxed-path theta solve is constrained to the subspace the
        # response operator keeps.  H_tt at this basis has one soft mode
        # (lambda ~ 2.8e-7, kappa ~ 3.4e8), and an unconstrained L-BFGS-B
        # exploits it: at delta = 3e-5 it moved 0.23 along the soft
        # direction, changing e2 by ~1e-7, which the 1/(2 delta) prefactor
        # turns into a 2e-2 gradient error.  orbital_zvector discards that
        # mode (rcond = 1e-7) because the finite-difference Hessian noise
        # is of the same order; the FD path must use the same convention.
        w_ht, V_ht = np.linalg.eigh(0.5 * (H_tt + H_tt.T))
        keep = np.abs(w_ht) > RCOND * max(1.0, np.abs(w_ht).max())
        V_keep = V_ht[:, keep]
        n_keep = int(keep.sum())

        def solve_theta(ref, th_start):
            th_start = np.asarray(th_start, dtype=float)

            def fun(u):
                th_v = jnp.asarray(th_start + V_keep @ np.asarray(u))
                val, grad = CAS_VG(*ref, th_v)
                return (float(val),
                        np.asarray(V_keep.T @ np.asarray(grad, dtype=float)))
            res = _minimize(fun, np.zeros(V_keep.shape[1]), jac=True,
                            method='L-BFGS-B',
                            options=dict(maxiter=800, gtol=1e-9,
                                         ftol=1e-20, maxls=60))
            th = np.asarray(th_start) + V_keep @ np.asarray(res.x)
            gn_kept = float(np.abs(np.asarray(res.jac)).max())
            gn_full = float(np.abs(np.asarray(
                CAS_G(*ref, jnp.asarray(th)))).max())
            gn_soft = float(np.abs(np.asarray(
                V_ht[:, ~keep].T @ np.asarray(
                    CAS_G(*ref, jnp.asarray(th))))
                ).max()) if (~keep).any() else 0.0
            return (jnp.asarray(th), gn_kept, gn_full, gn_soft,
                    int(res.nit))

        refs, info = {}, {}
        for s in (+1, -1):
            ref = refs_at(displace_mode(mol, base_bohr, v_mode, s * d),
                          C_ref)
            refs[s] = ref
            info['%+d' % s] = dict(
                e_scf=ref['e_scf'], converged=ref['converged'],
                align=ref['align'])
            gr = np.asarray(CAS_G(jnp.asarray(ref['C_raw']),
                                  jnp.asarray(ref['h']), jnp.asarray(ref['e']),
                                  ref['enuc'], th_base))
            ga = np.asarray(CAS_G(*ref_args(ref), th_base))
            info['%+d' % s]['gtheta_raw_max'] = float(np.abs(gr).max())
            info['%+d' % s]['gtheta_aligned_max'] = float(np.abs(ga).max())
        ra, rb = ref_args(refs[+1]), ref_args(refs[-1])
        e_p = float(FULL_E(*ra, th_base))
        e_m = float(FULL_E(*rb, th_base))
        fd_frozen = (e_p - e_m) / (2 * d)
        e0_p = float(CAS_E(*ra, th_base))
        e0_m = float(CAS_E(*rb, th_base))
        fd_cas = (e0_p - e0_m) / (2 * d)
        Mv_local = (np.asarray(CAS_G(*ra, th_base))
                    - np.asarray(CAS_G(*rb, th_base))) / (2 * d)
        Dv_local = -np.linalg.pinv(H_tt, rcond=RCOND) @ Mv_local
        resp_local = float(g_corr @ Dv_local)
        vals, vals_pred, th_opt = {}, {}, {}
        for s in (+1, -1):
            th_pred = np.asarray(th_base) + s * d * Dv0
            th_i, gn, gn_full, gn_soft, nit = solve_theta(
                ref_args(refs[s]), th_pred)
            th_opt[s] = np.asarray(th_i)
            vals[s] = float(FULL_E(*ref_args(refs[s]), th_i))
            vals_pred[s] = float(FULL_E(*ref_args(refs[s]),
                                        jnp.asarray(th_pred)))
            info['%+d' % s].update(
                gnorm_kept=gn, gnorm_full=gn_full, gnorm_soft=gn_soft,
                nit=nit,
                pred_step_max=float(np.abs(np.asarray(th_i) - th_pred).max()),
                total_step_max=float(np.abs(np.asarray(th_i)
                                            - np.asarray(th_base)).max()))
        fd_relaxed = (vals[+1] - vals[-1]) / (2 * d)
        dtheta_fd = (th_opt[+1] - th_opt[-1]) / (2 * d)
        dv_err = float(np.abs(dtheta_fd - Dv0).max())
        dv_rel = dv_err / max(float(np.abs(Dv0).max()), 1e-30)
        rec = dict(g_frozen_fd=fd_frozen, g_cas_fd=fd_cas,
                   g_e2_fd=fd_frozen - fd_cas,
                   g_e2_ad=base['g_aligned_v'] - base['g_cas_v'],
                   g_relaxed_fd=fd_relaxed,
                   resp_local=resp_local,
                   relaxed_local=g_frozen_v + resp_local,
                   Mv_local=Mv_local.tolist(), Dv_local=Dv_local.tolist(),
                   dtheta_fd=dtheta_fd.tolist(),
                   dtheta_vs_Dv_abs=dv_err, dtheta_vs_Dv_rel=dv_rel,
                   n_keep=n_keep, n_discarded=int((~keep).sum()),
                   refs=info, vals={str(k): v for k, v in vals.items()},
                   vals_pred={str(k): v for k, v in vals_pred.items()},
                   fd_relaxed_pred=(vals_pred[+1] - vals_pred[-1]) / (2 * d),
                   e_plus=e_p, e_minus=e_m)
        out.setdefault('fd', {})['{:g}'.format(d)] = rec
        out['stages'].append(key)
        emit(out)
        print('  d=%g frozen FD %.10e (AD %.10e rel %.3e) | relaxed FD '
              '%.10e (AD %.10e rel %.3e) | resp base %.10e local %.10e | '
              'e2 FD %.10e AD %.10e rel %.3e | dtheta rel %.3e '
              '[%.0fs]'
              % (d, fd_frozen, g_frozen_v,
                 abs(fd_frozen - g_frozen_v) / abs(fd_frozen),
                 fd_relaxed, base['relaxed_v'],
                 abs(fd_relaxed - base['relaxed_v']) / abs(fd_relaxed),
                 base['resp_z'], resp_local,
                 rec['g_e2_fd'], rec['g_e2_ad'],
                 abs(rec['g_e2_fd'] - rec['g_e2_ad']) / abs(rec['g_e2_fd']),
                 dv_rel, time.time() - t0), flush=True)
        return

    # ---- summary ------------------------------------------------------
    if STAGE == 'summary':
        rows = []
        for dk, r in sorted(out.get('fd', {}).items(),
                            key=lambda kv: -float(kv[0])):
            rows.append(dict(
                d=float(dk), frozen_fd=r['g_frozen_fd'],
                frozen_ad=g_frozen_v,
                frozen_rel=abs(r['g_frozen_fd'] - g_frozen_v)
                / abs(r['g_frozen_fd']),
                relaxed_fd=r['g_relaxed_fd'], relaxed_ad=base['relaxed_v'],
                relaxed_rel=abs(r['g_relaxed_fd'] - base['relaxed_v'])
                / abs(r['g_relaxed_fd']),
                local_rel=abs(r['g_relaxed_fd']
                              - (g_frozen_v + r['resp_local']))
                / abs(r['g_relaxed_fd']),
                dtheta_rel=r['dtheta_vs_Dv_rel']))
        out['summary'] = rows
        emit(out)
        for r in rows:
            print('d=%-8g frozen rel %.3e | relaxed rel %.3e | local-M rel '
                  '%.3e | dtheta rel %.3e'
                  % (r['d'], r['frozen_rel'], r['relaxed_rel'],
                     r['local_rel'], r['dtheta_rel']))
        print('SUMMARY_DONE')
        return

    raise RuntimeError('unknown stage {!r}'.format(STAGE))


def ref_args(ref):
    return (jnp.asarray(ref['C_al']), jnp.asarray(ref['h']),
            jnp.asarray(ref['e']), ref['enuc'])


def make_release_joint(ncore, nact, nele, nmo, pairs, scf_builder,
                       align_ref, align_bounds, e2_scale=1.0):
    """Release-chain joint function with the same alignment map.

    Used only to quantify how much the exact-Si correction moves the
    frozen gradient at this basis; it is not part of the validated chain.
    """
    dets = cas_dets_for2(ncore, nact, nele)
    links_a, links_b = _spin_links(nact, nele)
    kw = dict(ncore=ncore, nact=nact, nele=nele, nmo=nmo, pairs=pairs,
              dets=dets, links_a=links_a, links_b=links_b,
              ci_solver='blocked', eig_solver='lanczos')

    def components(m_, th_, want_e2=True):
        mf_ = scf_builder(m_)
        mf_.kernel()
        C = align_reference_blocks(mf_.mo_coeff, m_.intor('int1e_ovlp'),
                                   align_ref, align_bounds)
        return _components_from(C, mf_.get_hcore(),
                                m_.intor('int2e', aosym='s1'),
                                m_.energy_nuc(), th_, want_e2=want_e2, **kw)

    def E_joint(m_, th_):
        E_cas, e2 = components(m_, th_, want_e2=(e2_scale != 0.0))
        return E_cas + e2_scale * e2

    E_joint.components = components
    return E_joint


if __name__ == '__main__':
    main()
