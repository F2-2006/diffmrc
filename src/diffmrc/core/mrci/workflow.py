"""End-to-end MRCI workflow: CASCI reference, uncontracted and contracted MRCISD.

PURPOSE
-------
Runs the full sequence on a reference object: builds the MRCISD
determinant space, assembles the dense Hamiltonian, solves CASCI in the
internal space, diagonalizes the uncontracted MRCISD space, builds and
orthonormalizes the internally contracted doubly-external basis, solves
the projected problem, and applies the Davidson correction to both
variants.

This module is named ``workflow`` rather than ``driver`` because the
architecture reserves that name for the top-level job layer; the core
layer must not reference the upper layers, and the layer-contract test
scans for the literal name.

PUBLIC INTERFACE
----------------
``run_mrci(ref, ms=0.0, roots=3, svd_tol=1e-8, verbose=True)``.

``ref`` is duck-typed: it must expose ``nact``, ``norb_ci``,
``nact_elec``, ``h1``, ``eri``, ``e_core`` and ``e_casci``.
``diffmrc.core.mrci.pyscf_iface.build_reference`` constructs one from a
pyscf RHF/ROHF plus CASCI run.  The return value is a plain dict of
energies and diagnostics; energies are in Eh and already include the
frozen-core energy and nuclear repulsion.

VALIDATED SCOPE
---------------
H2O/6-31G CAS(2,2), CH2/STO-3G and CH2/6-31G CAS(4,4), and the
non-interacting H2...H2 and He...He dimers in ``tests/core/test_mrci.py``.
The uncontracted MRCISD equals pyscf FCI to 1e-12 Eh in the two cases
where the MRCISD space is the full CI space.

BOUNDARIES
----------
* Single reference root (lowest CASCI root); Ms = 0 is the only tested
  spin sector.
* Dense diagonalization: memory is 8 N^2 bytes for N determinants, so the
  practical ceiling is of order 1e4 determinants.
"""

from __future__ import annotations

import numpy as np

from .davidson import davidson_correction
from .determinants import build_hamiltonian, s2_expectation
from .icmrci import build_contracted_matrix, orthonormalize, solve_mrci
from .space import build_space


def run_mrci(ref, ms: float = 0.0, roots: int = 3, svd_tol: float = 1e-8, verbose: bool = True):
    """Run CASCI, uncontracted MRCISD and internally contracted MRCISD."""
    nact = ref.nact
    next_ = ref.norb_ci - nact
    n_elec = ref.nact_elec

    internal, singles, doubles = build_space(nact, next_, n_elec, ms)
    dets = internal + singles + doubles
    index = {d: i for i, d in enumerate(dets)}
    n_int, n_sing, n_dbl = len(internal), len(singles), len(doubles)
    n_full = len(dets)

    if verbose:
        print(f"space: internal={n_int} singles={n_sing} doubles={n_dbl} total={n_full}")

    H = build_hamiltonian(dets, ref.norb_ci, ref.h1, ref.eri)

    # CASCI in the internal space
    H_int = H[:n_int, :n_int]
    w_int, v_int = np.linalg.eigh(H_int)
    e_casci = float(w_int[0] + ref.e_core)
    psi0 = {internal[k]: float(v_int[k, 0]) for k in range(n_int) if abs(v_int[k, 0]) > 1e-14}

    # Uncontracted MRCISD
    w_u, v_u = np.linalg.eigh(H)
    e_unc = float(w_u[0] + ref.e_core)
    ref_weight_unc = float(np.sum(v_u[:n_int, 0] ** 2))

    # Internally contracted MRCISD (WK88 Eq. 9-19)
    X, labels = build_contracted_matrix(psi0, nact, next_, index, n_full, svd_tol=svd_tol)
    B, rank, S = orthonormalize(X, tol=svd_tol)

    n_basis = n_int + n_sing + B.shape[1]
    C = np.zeros((n_full, n_basis))
    C[:n_int, :n_int] = np.eye(n_int)
    C[n_int:n_int + n_sing, n_int:n_int + n_sing] = np.eye(n_sing)
    if B.shape[1]:
        C[:, n_int + n_sing:] = B

    ic_results = solve_mrci(H, C, n_int, n_sing, dets, ref.norb_ci, nroots=roots)
    ic = ic_results[0]

    e_ic = float(ic.energy + ref.e_core)
    e_q_unc = e_unc + davidson_correction(e_unc, e_casci, ref_weight_unc)
    e_q_ic = e_ic + davidson_correction(e_ic, e_casci, ic.ref_weight)

    out = {
        "n_internal": n_int,
        "n_singles": n_sing,
        "n_doubles_uncontracted": n_dbl,
        "n_doubles_basis": int(B.shape[1]),
        "n_pair_functions": len(labels),
        "e_casci": e_casci,
        "e_pyscf_casci": ref.e_casci,
        "e_uncontracted": e_unc,
        "e_ic": e_ic,
        "e_uncontracted_q": float(e_q_unc),
        "e_ic_q": float(e_q_ic),
        "e_ic_minus_uncontracted": e_ic - e_unc,
        "ref_weight_uncontracted": ref_weight_unc,
        "ref_weight_ic": ic.ref_weight,
        "s2_uncontracted": _s2_of(v_u[:, 0], dets, ref.norb_ci),
        "s2_ic": ic.s2,
        "ic_roots": [(float(r.energy + ref.e_core), r.s2) for r in ic_results],
        "uncontracted_roots": [(float(w_u[k] + ref.e_core), _s2_of(v_u[:, k], dets, ref.norb_ci))
                               for k in range(min(roots, len(w_u)))],
    }
    if verbose:
        print(f"CASCI (this code)      {e_casci:.10f}")
        print(f"CASCI (pyscf)          {ref.e_casci:.10f}   diff {e_casci - ref.e_casci:.2e}")
        print(f"MRCISD uncontracted    {e_unc:.10f}   <S^2>={out['s2_uncontracted']:.4f}")
        print(f"MRCISD contracted      {e_ic:.10f}   <S^2>={out['s2_ic']:.4f}")
        print(f"E(ic) - E(uncontr)     {e_ic - e_unc:+.3e} Ha")
        print(f"contracted basis rank  {B.shape[1]} / pair functions {len(labels)}")
        print(f"MRCISD+Q uncontracted  {e_q_unc:.10f}")
        print(f"MRCISD+Q contracted    {e_q_ic:.10f}")
    return out


def _s2_of(vec, dets, norb):
    return float(s2_expectation(vec, dets, norb))
