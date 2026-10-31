"""Wave14: open-problem checks for the differentiable eigensolver stack.

Three modes, each self-contained:

    --local
        No pyscfad needed. Reproduces the container fix of ``eigh_safe``
        under ``grad(jit(f))``, the matrix-free Davidson route against the
        dense CI path on the H2 fixture of the test suite, the cluster
        rule on the recorded CO spectra, and a first-derivative AD-vs-FD
        check on the CI route.

    --n2
        Needs pyscfad (server). Builds N2/6-31G with the four 1s-derived
        core MOs frozen (ncore = 4, 14 = 2*4 + 6 electrons), CAS(6,6), and
        compares
          (a) cs_casci_blocked(eig_solver='davidson') against the dense
              blocked solver and the unblocked cs_casci (energy and CI
              vector);
          (b) AD against central differences of the same chain with
              respect to a rotation of the core+active MOs;
          (c) AD against central differences of an active-space 1-RDM
              functional along the same rotation.

    --spectra
        Needs pyscfad (server) and the theta* record of the CO/6-31G
        frequency run. Prints the active 1-RDM and Fock core/virtual
        spectra at theta*, the consecutive gaps, and the tolerance that
        ``core.eigh.cluster_tolerance`` selects from each spectrum.

Example (server, CO spectra):

    systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \
        nice -n 19 env JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
        python eigensolver_wave14.py --spectra --theta-json <theta.json>
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _peak_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _fd(f, point, direction, step):
    return (float(f(point + step * direction))
            - float(f(point - step * direction))) / (2.0 * step)


# ---------------------------------------------------------------------------
# --local
# ---------------------------------------------------------------------------

def local():
    from diffmrc.core.eigh import (cluster_tolerance, eigh_mask, eigh_safe,
                                   lowest_pair_auto)
    from diffmrc.core.iterative_ci import lowest_k_eigpairs

    print("== local: eigh container under grad(jit(f))")

    rng = np.random.default_rng(0)
    B = rng.normal(size=(5, 5))
    A = jnp.asarray(B + B.T + np.diag([0.0, 2.0, 5.0, 9.0, 14.0]))
    wgt = jnp.asarray(np.arange(1.0, 6.0))
    idxw = jnp.asarray(np.arange(0.0, 5.0))

    def loss(a, fn=eigh_safe):
        w, u = fn(a)
        return jnp.sum(w * wgt) + jnp.sum(u * idxw[:, None] * wgt[None, :])

    print("   primal container: %s (jnp.linalg.eigh: %s)"
          % (type(eigh_safe(A)).__name__, type(jnp.linalg.eigh(A)).__name__))
    g_eager = np.asarray(jax.grad(loss)(A))
    g_jit = np.asarray(jax.grad(lambda a: jax.jit(loss)(a))(A))
    g_jit_grad = np.asarray(jax.jit(jax.grad(loss))(A))
    print("   grad(jit(f)) finite: %s   max|grad(jit)-eager| = %.2e"
          % (np.isfinite(g_jit).all(), np.abs(g_jit - g_eager).max()))
    print("   jit(grad(f)) max|diff-eager| = %.2e"
          % np.abs(g_jit_grad - g_eager).max())

    B2 = rng.normal(size=(3, 3))
    B2 = B2 + B2.T
    A2 = np.zeros((6, 6))
    A2[:3, :3] = B2
    A2[3:, 3:] = B2
    A2 = jnp.asarray(A2)
    for name, fn in (('eigh_safe', eigh_safe), ('eigh_mask', eigh_mask)):
        def loss_fn(a, _fn=fn):
            w, u = _fn(a)
            n = a.shape[0]
            wf = jnp.arange(1.0, n + 1.0)
            iw = jnp.arange(0.0, n)
            return jnp.sum(w * wf) + jnp.sum(u * iw[:, None] * wf[None, :])
        g = np.asarray(jax.grad(lambda a: jax.jit(loss_fn)(a))(A2))
        print("   bit-exact pair, %s: finite = %s"
              % (name, np.isfinite(g).all()))

    print("\n== local: cluster rule on the recorded spectra")
    from tests.core.test_cluster_tolerance import (_DM1_SPECTRUM,
                                                   _FOCK_VIRTUAL_GAPS)
    dec = cluster_tolerance(_DM1_SPECTRUM)
    print("   dm1 spectrum: mode=%s tol=%.3e floor=%.3e physical=%.3e ratio=%.1f"
          % (dec.mode, dec.tol, dec.noise_floor, dec.physical_min, dec.ratio))
    spec = np.concatenate([[0.0], np.cumsum(_FOCK_VIRTUAL_GAPS)])
    dec = cluster_tolerance(spec, noise_scale=1.155e-5)
    print("   Fock virtual: mode=%s tol=%.3e floor=%.3e physical=%.3e ratio=%.1f"
          % (dec.mode, dec.tol, dec.noise_floor, dec.physical_min, dec.ratio))

    print("\n== local: Davidson against dense on the H2 fixture")
    from tests.core.test_iterative_ci import (_H2_ANCHOR_R, _h2_mo_integrals,
                                              _ci_matrix)
    h_mo, eri_mo = _h2_mo_integrals('6-31g', _H2_ANCHOR_R['6-31g'])
    h = jnp.asarray(_ci_matrix(np.asarray(h_mo), np.asarray(eri_mo), 0, 4, 2))
    w_ref = np.asarray(jnp.linalg.eigvalsh(h))
    ws, vs = lowest_k_eigpairs(h, k=1, tol=1e-13)
    print("   dim=%d  E_iter=%.12f  E_dense=%.12f  |dE|=%.2e"
          % (h.shape[0], float(ws[0]), w_ref[0], abs(float(ws[0]) - w_ref[0])))
    print("   |<v|v_dense>| = %.12f"
          % abs(float(vs[:, 0] @ jnp.linalg.eigh(h)[1][:, 0])))
    print("\nLOCAL_DONE")


# ---------------------------------------------------------------------------
# --n2
# ---------------------------------------------------------------------------

def _n2_integrals(ncore=4, nact=6):
    from pyscfad import gto, scf
    mol = gto.Mole()
    mol.atom = "N 0 0 0; N 0 0 1.0977"
    mol.basis = "6-31g"
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    C = jnp.asarray(mf.mo_coeff)
    h_ao = jnp.asarray(mf.get_hcore())
    eri_ao = jnp.asarray(mol.intor("int2e", aosym="s1"))
    nm = ncore + nact
    h1 = jnp.einsum("ai,ab,bj->ij", C[:, :nm], h_ao, C[:, :nm])
    z = jnp.einsum("pi,pqrs->iqrs", C[:, :nm], eri_ao)
    z = jnp.einsum("qj,iqrs->ijrs", C[:, :nm], z)
    z = jnp.einsum("rk,ijrs->ijks", C[:, :nm], z)
    eri1 = jnp.einsum("sl,ijks->ijkl", C[:, :nm], z)
    return mol, C, h_ao, eri_ao, h1, eri1, ncore, nact, nm


def _rotation_chain(mol, C, h_ao, eri_ao, ncore, nact, n_par=4, seed=7):
    """Return (h1(theta), eri1(theta)) and a deterministic set of directions."""
    from jax.scipy.linalg import expm
    nm = ncore + nact
    rng = np.random.default_rng(seed)
    dirs = []
    for _ in range(n_par):
        b = rng.normal(size=(nm, nm))
        b = b - b.T
        dirs.append(jnp.asarray(b / np.linalg.norm(b)))
    C0 = C[:, :nm]

    def integrals(theta):
        A = sum(theta[k] * dirs[k] for k in range(n_par))
        Ct = C0 @ expm(A)
        h1 = jnp.einsum("ai,ab,bj->ij", Ct, h_ao, Ct)
        z = jnp.einsum("pi,pqrs->iqrs", Ct, eri_ao)
        z = jnp.einsum("qj,iqrs->ijrs", Ct, z)
        z = jnp.einsum("rk,ijrs->ijks", Ct, z)
        eri1 = jnp.einsum("sl,ijks->ijkl", Ct, z)
        return h1, eri1

    return integrals, dirs


def n2():
    from diffmrc.core.casci import cs_casci, cs_casci_blocked
    from diffmrc.core.nevpt2.impl import cas_dets_for2
    from diffmrc.core.nevpt2.rdm import make_dm123_sf

    from math import comb
    mol, C, h_ao, eri_ao, h1, eri1, ncore, nact, nm = _n2_integrals()
    nele = 6
    assert 2 * ncore + nele == mol.nelectron, (ncore, nele, mol.nelectron)
    print("== N2/6-31G  ncore=%d  CAS(%d,%d)  total electrons=%d  "
          "full CI dim=%d  Sz=0 block=%d"
          % (ncore, nact, nele, mol.nelectron, comb(2 * nact, nele),
             comb(nact, nele // 2) ** 2))
    t0 = time.time()
    E_d, ci_d = cs_casci_blocked(h1, eri1, ncore=ncore, nact=nact,
                                 nele_act=6, eig_solver='eigh_safe')
    t_d = time.time() - t0
    t0 = time.time()
    E_m, ci_m = cs_casci_blocked(h1, eri1, ncore=ncore, nact=nact,
                                 nele_act=6, eig_solver='davidson')
    t_m = time.time() - t0
    E_p = cs_casci(h1, eri1, ncore=ncore, nact=nact, nele_act=6)
    overlap = abs(float(np.asarray(ci_d) @ np.asarray(ci_m)))
    print("   E(dense blocked) = %.12f  (%.1f s)" % (float(E_d), t_d))
    print("   E(matrix-free)   = %.12f  (%.1f s)" % (float(E_m), t_m))
    print("   E(plain cs_casci)= %.12f" % float(E_p))
    print("   |dE| matrix-free vs dense = %.2e   |<ci|ci>| = %.12f   peak=%.2f GB"
          % (abs(float(E_m) - float(E_d)), overlap, _peak_gb()))

    integrals, dirs = _rotation_chain(mol, C, h_ao, eri_ao, ncore, nact)
    dets = cas_dets_for2(ncore, nact, 6)
    rng = np.random.default_rng(33)
    W = jnp.asarray(0.5 * (rng.normal(size=(nact, nact))
                           + rng.normal(size=(nact, nact)).T))

    def energy_raw(theta, solver='davidson'):
        h_theta, e_theta = integrals(theta)
        return cs_casci_blocked(h_theta, e_theta, ncore=ncore, nact=nact,
                                nele_act=6, eig_solver=solver)[0]

    def density_raw(theta):
        h_theta, e_theta = integrals(theta)
        _, ci = cs_casci_blocked(h_theta, e_theta, ncore=ncore, nact=nact,
                                 nele_act=6, eig_solver='davidson')
        dm1, _, _ = make_dm123_sf(ci, dets, nact, ncore, max_rank=1)
        return jnp.sum(W * dm1)

    # jit both chains: the forward Davidson hosts a data-dependent
    # while_loop, one compilation is paid once and every FD point then
    # costs a dispatch instead of a fresh op-by-op trace.
    energy = jax.jit(lambda th: energy_raw(th, 'davidson'))
    energy_dense = jax.jit(lambda th: energy_raw(th, 'eigh_safe'))
    density = jax.jit(density_raw)
    print("\n== AD vs central FD along a core+active MO rotation")
    zero = jnp.zeros(len(dirs))
    grads = {}
    for name, fn in (('energy', energy), ('1-RDM functional', density)):
        t0 = time.time()
        g_jax = jax.grad(fn)(zero)
        g_jax.block_until_ready()
        g = np.asarray(g_jax)
        print("   %s: |AD| = %.6e  (grad+compile %.1f s)"
              % (name, np.linalg.norm(g), time.time() - t0))
        grads[name] = g
        for k in range(2):
            for step in (1e-4, 1e-5, 1e-6):
                fd = _fd(fn, zero, jnp.eye(len(dirs))[k], step)
                rel = abs(g[k] - fd) / max(abs(fd), 1e-300)
                print("      dir %d  step %.0e: AD=%+.10e FD=%+.10e rel=%.2e"
                      % (k, step, g[k], fd, rel), flush=True)

    # The dense route must give the same gradient: the two solvers solve
    # the same eigenproblem, so a difference would point at the mapping.
    t0 = time.time()
    g_d = np.asarray(jax.grad(energy_dense)(zero))
    print("\n   dense-route grad |AD| = %.6e  (grad+compile %.1f s)"
          % (np.linalg.norm(g_d), time.time() - t0))
    print("   max|grad(davidson) - grad(dense)| = %.2e"
          % np.abs(g_d - grads['energy']).max())
    print("   peak RSS = %.2f GB" % _peak_gb())
    print("\nN2_DONE")


# ---------------------------------------------------------------------------
# --spectra
# ---------------------------------------------------------------------------

def spectra(theta_json):
    from pyscfad import gto as pgto, scf as pscf
    from diffmrc.core.casci import cs_casci_blocked
    from diffmrc.core.eigh import cluster_tolerance, eigh_cluster_ift
    from diffmrc.core.nevpt2.impl import cas_dets_for2, make_dm123_sf
    from diffmrc.core.orblib import orbital_pairs
    from diffmrc.core.rotation import block_rotation

    atom, basis, nact, nele = 'C 0 0 0; O 0 0 1.1283', '6-31g', 6, 6
    th0 = jnp.asarray(np.asarray(json.load(open(theta_json))['theta'],
                                 dtype=float))
    mol = pgto.Mole()
    mol.atom = atom
    mol.basis = basis
    mol.build()
    nmo = mol.nao_nr()
    ncore = mol.nelectron // 2 - nele // 2
    pairs = orbital_pairs(ncore, nact, nmo, include_cv=True)
    mf = pscf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    C0 = jnp.asarray(np.asarray(mf.mo_coeff))
    h_ = jnp.asarray(np.asarray(mf.get_hcore()))
    e_ = jnp.asarray(mol.intor('int2e', aosym='s1'))
    C2 = C0 @ block_rotation(th0, nmo, pairs)
    h1 = jnp.einsum('ai,ab,bj->ij', C2, h_, C2)
    z = jnp.einsum('pi,pqrs->iqrs', C2, e_)
    z = jnp.einsum('qj,iqrs->ijrs', C2, z)
    z = jnp.einsum('rk,ijrs->ijks', C2, z)
    eri = jnp.einsum('sl,ijks->ijkl', C2, z)
    nm = ncore + nact
    print("CO/6-31G theta*: npar=%d (json npar=%d)  ncore=%d  nmo=%d"
          % (len(pairs), int(json.load(open(theta_json))['npar']), ncore, nmo),
          flush=True)
    dets = cas_dets_for2(ncore, nact, nele)

    def report(w, label):
        w = np.asarray(w)
        g = np.diff(w)
        dec = cluster_tolerance(w)
        dec_meas = cluster_tolerance(w, noise_scale=1.155e-5)
        print("== %s" % label)
        print("   spectrum: %s" % ['%+.9f' % x for x in w])
        print("   gaps    : %s" % ['%.3e' % x for x in g])
        print("   rule(default): mode=%s tol=%.3e floor=%.3e phys=%.3e ratio=%.1f"
              % (dec.mode, dec.tol, dec.noise_floor, dec.physical_min,
                 dec.ratio))
        print("   rule(noise=1.155e-5): mode=%s tol=%.3e"
              % (dec_meas.mode, dec_meas.tol))

    def fock_blocks(dm1, tag):
        w_nat, U_nat = eigh_cluster_ift(dm1)
        U_nat = U_nat[:, ::-1]
        D = jnp.zeros((nmo, nmo))
        D = D.at[:ncore, :ncore].set(2.0 * jnp.eye(ncore))
        D = D.at[ncore:nm, ncore:nm].set(dm1)
        F = h1 + jnp.einsum('pqrs,rs->pq', eri, D) \
            - 0.5 * jnp.einsum('psrq,rs->pq', eri, D)
        Uf = jnp.eye(nmo).at[ncore:nm, ncore:nm].set(U_nat)
        F = Uf.T @ F @ Uf
        report(np.linalg.eigvalsh(np.asarray(F[:ncore, :ncore])),
               'Fock core block (%s)' % tag)
        report(np.linalg.eigvalsh(np.asarray(F[nm:, nm:])),
               'Fock virtual block (%s)' % tag)

    # Both solvers: the derivative chain used the Lanczos vector, while
    # the exact blocked decomposition is the reference spectrum of the
    # same orbitals. Comparing the two shows whether a measured split is
    # numerical (solver residual) or physical.
    for solver in ('lanczos', 'eigh_safe'):
        E0, ci = cs_casci_blocked(h1[:nm, :nm], eri[:nm, :nm, :nm, :nm],
                                  ncore=ncore, nact=nact, nele_act=nele,
                                  eig_solver=solver)
        dm1, _, _ = make_dm123_sf(ci, dets, nact, ncore, max_rank=1)
        print("CO/6-31G theta* [%s]: E_CASCI = %.12f" % (solver, float(E0)),
              flush=True)
        report(np.linalg.eigvalsh(np.asarray(dm1)),
               'dm1 (active 1-RDM), %s' % solver)
        fock_blocks(dm1, solver)
    print("SPECTRA_DONE")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--local', action='store_true')
    ap.add_argument('--n2', action='store_true')
    ap.add_argument('--spectra', action='store_true')
    ap.add_argument('--theta-json', default=None,
                    help='theta* record (JSON) needed by --spectra')
    args = ap.parse_args(argv)
    ran = False
    if args.local:
        local()
        ran = True
    if args.n2:
        n2()
        ran = True
    if args.spectra:
        if args.theta_json is None:
            ap.error('--spectra needs --theta-json')
        spectra(args.theta_json)
        ran = True
    if not ran:
        ap.error('choose at least one mode')


if __name__ == '__main__':
    main()
