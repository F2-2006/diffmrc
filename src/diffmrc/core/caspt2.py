#!/usr/bin/env python
"""Differentiable CASPT2: minimal core (P1 pilot, first class).

This module implements the active-to-double-virtual class of an
uncontracted CASPT2 in the semicanonical (Roos) zeroth order.  The
class coupling ``norm`` is the same physical quantity the NEVPT2
double-external class (``nevpt2.impl.srs_e2``) assembles -- what
differs between the two methods is the denominator:

    NEVPT2 (Dyall zeroth order):  D = eps_a + eps_b + h/norm
    CASPT2 (Roos zeroth order):   D = eps_a + eps_b - E0

with E0 = sum_p occ_p eps_p the diagonal-Fock reference expectation
(core doubly occupied, active natural occupations, virtual empty).
``eps`` must be the semicanonical Fock diagonal -- the
``canonicalize_jax`` output -- because both denominators are defined
only for that basis.

The class carries the whole H2 correlation energy (no core, no single
external by symmetry), which is the first anchor:
server/caspt2_anchor/h2_cas22.log, ORCA CASPT2 = -0.004043415 Eh.
"""
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp  # noqa: E402

import numpy as np  # noqa: E402

__all__ = ['caspt2_double_external_e2']


def caspt2_double_external_e2(h1e_mo, eri_mo, c_ci, dets, *,
                              ncore=0, nact=2, nocc=None, mo_energy=None):
    """Active-to-double-virtual CASPT2 class energy.

    Inputs follow ``nevpt2_e2_full2``: full-MO ``h1e_mo``/``eri_mo``,
    the CI vector in the orbital basis, the determinant bitmaps, the
    space definition.  ``mo_energy`` is the semicanonical Fock diagonal
    (``canonicalize_jax`` output).  Returns ``(e2, norm, diff)`` --
    the class energy and the coupling and denominator arrays for the
    record.
    """
    from .nevpt2.impl import (make_a7, make_dm123_sf,
                              make_gfock_mo_energy)

    if nocc is None:
        nocc = ncore + nact
    nmo = h1e_mo.shape[0]

    # Effective one-electron integrals including the frozen-core VHF,
    # exactly as the NEVPT2 chain slices them.
    if ncore > 0:
        vhf = (2.0 * jnp.einsum('pqcc->pq', eri_mo[:, :, :ncore, :ncore])
               - jnp.einsum('pccq->pq', eri_mo[:, :ncore, :ncore, :]))
    else:
        vhf = jnp.zeros_like(h1e_mo)
    h1eff = h1e_mo + vhf
    h1e_act = h1eff[ncore:nocc, ncore:nocc]
    eri_act = jnp.einsum(
        "ijkl->ikjl",
        eri_mo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc])
    h2e_v = jnp.einsum("aibj->abij", eri_mo[nocc:, ncore:nocc,
                                            nocc:, ncore:nocc])

    dm1, dm2, dm3 = make_dm123_sf(c_ci, dets, nact, ncore)
    eps = mo_energy if mo_energy is not None else make_gfock_mo_energy(
        h1e_mo, eri_mo, dm1, ncore, nact)

    # The class coupling: the same contraction srs_e2 assembles (squared
    # coupling per virtual pair, active space contracted in).
    rm2, _a7 = make_a7(h1e_act, eri_act, dm1, dm2, dm3)
    norm = 0.5 * jnp.einsum("rsqp,rsba,pqba->rs", h2e_v, h2e_v, rm2)

    # CASPT2 zeroth-order reference: the diagonal-Fock expectation.
    occ = jnp.zeros(nmo)
    occ = occ.at[:ncore].set(2.0)
    occ = occ.at[ncore:nocc].set(jnp.diag(dm1))
    e0_zero = jnp.sum(occ * eps)

    diff = eps[nocc:, None] + eps[None, nocc:] - e0_zero
    diff_safe = jnp.where(jnp.abs(diff) > 1e-12, diff, jnp.ones_like(diff))
    e2 = -jnp.sum(jnp.where(jnp.abs(diff) > 1e-12, norm / diff_safe, 0.0))
    return float(e2), np.asarray(norm), np.asarray(diff)
