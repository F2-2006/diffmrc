"""The state-averaged CASSCF + per-state NEVPT2 pipeline (driver layer).

All additive: the core exposes ``cs_casci(nroots>1, return_ci=True)`` (the
equal-weight mean energy and the per-root CI vectors) and the NEVPT2
contraction consumes one vector at a time; the only new construction is
the average-density canonicalization, pinned against the core routine by
a k=1 identity test.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def _symmetric_ints(n, seed=0):
    """Synthetic chemically-symmetric integrals on n MOs (test scaffolding,
    not physical): p<->q, r<->s and (pq)<->(rs) symmetries enforced."""
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n, n))
    h = 0.5 * (h + h.T)
    eri = rng.normal(size=(n, n, n, n)) * 0.1
    eri = 0.25 * (eri + eri.transpose(1, 0, 2, 3)
                  + eri.transpose(0, 1, 3, 2)
                  + eri.transpose(2, 3, 0, 1))
    return jnp.asarray(h), jnp.asarray(eri)


def test_sa_cas_solution_matches_the_core_state_average():
    from diffmrc.core.casci import cs_casci
    from diffmrc.driver.nevpt2 import sa_cas_solution

    h, eri = _symmetric_ints(4)
    ncore, nact, nele, nroots = 0, 2, 2, 2
    e_sa, cis = sa_cas_solution(h, eri, ncore=ncore, nact=nact, nele=nele,
                                nroots=nroots)
    e_ref = cs_casci(h, eri, ncore=ncore, nact=nact, nele_act=nele,
                     nroots=nroots)
    assert float(e_sa) == pytest.approx(float(e_ref), abs=1e-12)
    assert np.asarray(cis).shape[1] == nroots
    gram = np.asarray(cis).T @ np.asarray(cis)
    assert np.allclose(gram, np.eye(nroots), atol=1e-10)
