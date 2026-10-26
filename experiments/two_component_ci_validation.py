#!/usr/bin/env python
"""Value and first-derivative checks of the complex spinor CI.

Three independent blocks.

1. Non-relativistic spin-orbital limit.  A real spatial Hamiltonian is
   expanded into a spin-orbital basis and diagonalized by the complex
   spinor CI.  The unique levels are compared with pyscf's
   ``fci.direct_spin1``.  The same Hamiltonian is differentiated with
   respect to a one-electron perturbation, once with the Kramers-safe
   eigenvalue JVP and once with the installed JAX ``eigvalsh`` rule, and
   both are compared with a central difference.

2. Synthetic Kramers Hamiltonian.  An odd-electron, time-reversal-
   invariant Hamiltonian with T^2 = -1 has every level at least doubly
   degenerate.  The one-particle time-reversal matrix and the
   many-electron determinant matrix satisfy U^dag U = 1, U^* U = -1,
   T H^* T^T = H and T T^* = -I.

3. Physical X2C boundary.  The pyscf spinor ERI array and the X2C HF
   reference are examined for the conventions the complex CI needs.
   The measured statements are recorded in the JSON: the spinor ERI
   symmetry violation, the agreement of the ``c^* c c^* c`` transform
   with ``pyscf.ao2mo``, the Kramers pairing of the X2C HF orbitals, the
   failure of the many-electron time reversal to commute with the CI
   Hamiltonian built from that array, and the ``NotImplementedError``
   raised by pyscf for an ECP X2C reference.  The first two items
   validate the transform; the third and fourth identify the remaining
   interface that a production two-component CI needs.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import json
import sys
import time

import numpy as np
from pyscf import ao2mo, fci, gto

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from diffmrc.core import spinor_ci as sci  # noqa: E402


def spin_orbital_hamiltonian(mol, norb):
    """Real spatial Hamiltonian expanded into a spin-orbital basis."""
    h_sp = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri_sp = mol.intor("int2e", aosym="s1")
    h = h_sp[:norb, :norb]
    eri = eri_sp[:norb, :norb, :norb, :norb]
    nso = 2 * norb
    h1 = np.zeros((nso, nso), dtype=complex)
    h1[0::2, 0::2] = h
    h1[1::2, 1::2] = h
    eri_so = np.zeros((nso, nso, nso, nso), dtype=complex)
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    val = eri[p, q, r, s]
                    for sp in (0, 1):
                        for sq in (0, 1):
                            eri_so[2 * p + sp, 2 * q + sp,
                                   2 * r + sq, 2 * s + sq] = val
    return h, eri, h1, eri_so


def nonrelativistic_block():
    t0 = time.time()
    mol = gto.M(atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0",
                basis="6-31g", unit="Angstrom", verbose=0)
    norb, nelec = 4, 4
    h, eri, h1, eri_so = spin_orbital_hamiltonian(mol, norb)
    w_pyscf = np.sort(np.atleast_1d(
        fci.direct_spin1.kernel(h, eri, norb, nelec, nroots=5)[0]))
    H, dets, _ = sci.spinor_ci_matrix(h1, eri_so, 0, 2 * norb, nelec)
    w_all = np.linalg.eigvalsh(H)
    w_unique = np.unique(np.round(w_all, 10))[:5]
    value_error = float(np.max(np.abs(w_unique - w_pyscf)))

    rng = np.random.default_rng(17)
    v = rng.normal(size=h1.shape) + 1j * rng.normal(size=h1.shape)
    v = 0.5 * (v + v.conj().T)

    derivative = {}
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        structure = sci._ci_structure(0, 2 * norb, nelec)

        def energy_safe(lam):
            return sci.spinor_ci_energies_jax(
                h1 + lam * v, eri_so, 0, 2 * norb, nelec, nroots=1,
                structure=structure)[0]

        def energy_naive(lam):
            Hl, _ = sci.spinor_ci_matrix_jax(
                h1 + lam * v, eri_so, 0, 2 * norb, nelec,
                structure=structure)
            return jnp.real(jnp.linalg.eigvalsh(Hl)[0])

        g_safe = float(jax.grad(energy_safe)(0.0))
        try:
            g_naive = float(jax.grad(energy_naive)(0.0))
        except Exception:
            g_naive = float("nan")
        hstep = 1e-6
        fd = float((energy_safe(hstep) - energy_safe(-hstep)) / (2 * hstep))
        derivative = dict(ad_safe=g_safe, ad_naive_eigvalsh=g_naive,
                          finite_difference=fd,
                          error_safe=abs(g_safe - fd),
                          error_naive=(abs(g_naive - fd)
                                       if np.isfinite(g_naive) else None))
    except Exception as exc:  # pragma: no cover - environment dependent
        derivative = dict(error=f"{type(exc).__name__}: {exc}")

    return dict(
        system="h2o_nonrelativistic_spin_orbitals",
        norb=norb,
        nelec=nelec,
        ndet=int(len(dets)),
        pyscf_levels=[float(x) for x in w_pyscf],
        ci_unique_levels=[float(x) for x in w_unique],
        value_error=value_error,
        derivative=derivative,
        runtime_s=time.time() - t0,
    )


def synthetic_kramers_block():
    t0 = time.time()
    rng = np.random.default_rng(23)
    n_pairs = 3
    n = 2 * n_pairs
    u = np.zeros((n, n), dtype=complex)
    j2 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
    for p in range(n_pairs):
        u[2 * p:2 * p + 2, 2 * p:2 * p + 2] = j2
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = 0.5 * (h + h.conj().T)
    h = 0.5 * (h + u @ h.conj() @ u.T)
    e = rng.normal(size=(n,) * 4) + 1j * rng.normal(size=(n,) * 4)
    e = 0.5 * (e + e.transpose(1, 0, 2, 3).conj())
    e = 0.5 * (e + e.transpose(0, 1, 3, 2).conj())
    e = 0.5 * (e + e.transpose(2, 3, 0, 1).conj())
    e = 0.5 * (e + np.einsum("ia,jb,kc,ld,abcd->ijkl", u, u, u, u,
                            e.conj()))
    nele = 3
    H, dets, _ = sci.spinor_ci_matrix(h, e, 0, n, nele)
    w = np.linalg.eigvalsh(H)
    ok, gaps = sci.kramers_pairing(w, tol=1e-9)
    T = sci.time_reversal_determinant_matrix(u.T, dets)
    return dict(
        nspinor=n,
        nelec=nele,
        kramers_degenerate=bool(ok),
        max_within_pair_gap=float(np.max(gaps[0::2])),
        many_body_T_commutator=float(np.max(np.abs(T @ H.conj() @ T.T - H))),
        many_body_T_square=float(np.max(np.abs(T @ T.conj()
                                              + np.eye(len(dets))))),
        runtime_s=time.time() - t0,
    )


def physical_x2c_block():
    t0 = time.time()
    from diffmrc.core import x2c_df

    mol = gto.M(atom="O 0 0 0; H 0 0 1.8", basis="6-31g", spin=1,
                verbose=0)
    res = x2c_df.df_x2c_hf(mol, auxbasis="def2-universal-jkfit")
    c = res.mo_coeff
    s = mol.intor("int1e_ovlp_spinor")
    eri_ao = mol.intor("int2e_spinor")
    eri_mo = np.einsum("pa,qb,rc,sd,pqrs->abcd", c.conj(), c, c.conj(), c,
                       eri_ao, optimize=True)
    eri_a2m = ao2mo.full(mol, c, intor="int2e_spinor", aosym=1)
    eri_a2m = np.asarray(eri_a2m).reshape((mol.nao_2c(),) * 4)
    transform_error = float(np.max(np.abs(eri_mo - eri_a2m)))
    sym_ab = float(np.max(np.abs(eri_ao
                                 - eri_ao.transpose(1, 0, 2, 3).conj())))
    sym_scale = float(np.max(np.abs(eri_ao)))

    u = sci.kramers_unitary(mol)
    m = c.conj().T @ s @ u @ c.conj()
    occ = list(np.where(res.mo_occ > 0)[0])
    virt = [i for i in range(len(res.mo_energy)) if i not in occ]
    somo = None
    for i in occ:
        j = int(np.argmax(np.abs(m[:, i])))
        if j in virt:
            somo = (i, j)
            break
    pair_info = dict(somo=None)
    ci_info = {}
    if somo is not None:
        s_i, p_i = somo
        core = [i for i in occ if i != s_i]
        seen = {p_i}
        pairs = []
        for i in virt:
            if i in seen:
                continue
            j = int(np.argmax(np.abs(m[:, i])))
            if j in seen or j == i:
                continue
            pairs.append((i, j))
            seen.add(i)
            seen.add(j)
        pairs.sort(key=lambda x: res.mo_energy[x[0]])
        active = [s_i, p_i] + list(pairs[0]) + list(pairs[1])
        order = core + active + [i for i in range(len(res.mo_energy))
                                 if i not in set(core + active)]
        co = c[:, order]
        h1 = co.conj().T @ res.hcore @ co
        eri = np.einsum("pa,qb,rc,sd,pqrs->abcd", co.conj(), co, co.conj(),
                        co, eri_ao, optimize=True)
        nc, na, ne = len(core), len(active), 3
        w, _, dets, _ = sci.spinor_ci_solve(h1, eri, nc, na, ne, nroots=10)
        ok, gaps = sci.kramers_pairing(w, tol=1e-8)
        m_act = m[np.ix_(active, active)]
        H, _, _ = sci.spinor_ci_matrix(h1, eri, nc, na, ne)
        T = sci.time_reversal_determinant_matrix(m_act, dets)
        ci_info = dict(
            ncore=nc, nact=na, nele_act=ne,
            levels=[float(x) for x in w],
            kramers_degenerate=bool(ok),
            max_gap=float(np.max(gaps)),
            T_commutator=float(np.max(np.abs(T @ H.conj() @ T.T - H))),
            T_square=float(np.max(np.abs(T @ T.conj()
                                         + np.eye(len(dets))))),
        )
        pair_info = dict(somo=[int(s_i), int(p_i)], active=[int(x) for x in active])

    ecp_boundary = None
    try:
        from pyscf.x2c import x2c as pyscf_x2c
        ybf = gto.M(atom=[("Yb", (0, 0, 0)), ("F", (0, 0, 3.6))],
                    basis={"Yb": "stuttgart_rsc", "F": "6-31g"},
                    ecp={"Yb": "ecpds28mwbso"}, spin=1, unit="Bohr",
                    verbose=0)
        pyscf_x2c.SpinorX2CHelper(ybf).get_hcore(ybf)
        ecp_boundary = "no exception"
    except NotImplementedError:
        ecp_boundary = "NotImplementedError"
    except Exception as exc:  # pragma: no cover
        ecp_boundary = f"{type(exc).__name__}: {exc}"

    return dict(
        system="oh_x2c",
        nao=int(mol.nao_2c()),
        nelectron=int(mol.nelectron),
        spinor_eri_symmetry_violation=sym_ab,
        spinor_eri_scale=sym_scale,
        ao2mo_transform_error=transform_error,
        kramers_pairing=pair_info,
        ci=ci_info,
        ecp_x2c_boundary=ecp_boundary,
        runtime_s=time.time() - t0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = {
        "nonrelativistic": nonrelativistic_block(),
        "synthetic_kramers": synthetic_kramers_block(),
        "physical_x2c": physical_x2c_block(),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2)


if __name__ == "__main__":
    main()
