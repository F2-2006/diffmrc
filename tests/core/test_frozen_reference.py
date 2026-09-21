"""Frozen reference values from the validated numerical kernel.

These numbers come from the P14 manuscript's validation runs. If a
migration changes any of them, the migration is wrong -- the kernel is
frozen and its numerics must not move.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pyscfad = pytest.importorskip("pyscfad")
from pyscfad import gto as pgto, scf as pscf  # noqa: E402

from diffmrc.core.rotation import block_rotation  # noqa: E402
from diffmrc.core.orblib import orbital_pairs  # noqa: E402
from diffmrc.core.casci import cs_casci_blocked  # noqa: E402
from diffmrc.core.optimizer import optimize_theta  # noqa: E402

# H2O / 6-31G, CAS(2,2). Reference energy from the manuscript's Table 1 run.
H2O_631G_CAS22_E = -75.9848760005
H2O_ATOM = 'O 0 0 0; H 0.7557 0 0.5880; H -0.7557 0 0.5880'


def _h2o_energy():
    mol = pgto.Mole(); mol.atom = H2O_ATOM; mol.basis = '6-31g'; mol.build()
    nmo = mol.nao_nr()
    mf = pscf.RHF(mol); mf.kernel()
    C0 = mf.mo_coeff
    h1e = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    eri = mol.intor('int2e', aosym='s1')
    enuc = mol.energy_nuc()
    ncore = mol.nelectron // 2 - 1
    pairs = orbital_pairs(ncore, 2, nmo, include_cv=True)

    def E_cas(C):
        Cj = jnp.asarray(C)
        h1m = jnp.einsum('ai,ab,bj->ij', Cj, h1e, Cj)[:ncore + 2, :ncore + 2]
        e1m = jnp.einsum('pi,qj,rk,sl,pqrs->ijkl', Cj, Cj, Cj, Cj,
                         eri)[:ncore + 2, :ncore + 2, :ncore + 2, :ncore + 2]
        return cs_casci_blocked(h1m, e1m, ncore=ncore, nact=2,
                                nele_act=2)[0] + enuc

    E_theta = lambda t: E_cas(C0 @ block_rotation(t, nmo, pairs))
    _, th, _ = optimize_theta(None, None, E_theta, len(pairs), steps=2000,
                              g_conv=1e-12, theta0=np.zeros(len(pairs)))
    return float(E_theta(jnp.asarray(np.asarray(th))))


def test_h2o_cas22_energy_matches_frozen_reference():
    E = _h2o_energy()
    assert abs(E - H2O_631G_CAS22_E) < 1e-6, (
        f'kernel numerics moved: {E} vs frozen {H2O_631G_CAS22_E}'
    )


def test_orbital_pairs_count():
    """Parameter count for H2O/6-31G CAS(2,2) is NPAR=50."""
    assert len(orbital_pairs(4, 2, 13, include_cv=True)) == 50
