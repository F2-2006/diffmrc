"""The memory-bounded integral-derivative backend.

Three properties are pinned:

* the ``s4`` -> ``s1`` restore reproduces ``pyscf.ao2mo.restore`` exactly;
* the two ``pyscfad`` backends return the same coordinate derivative of
  the two-electron integral tensor, to machine precision;
* ``build_mole`` leaves the basis exponents and the contraction
  coefficients untraced, which is what the vjp backend requires.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")

pyscf = pytest.importorskip("pyscf")
pyscfad = pytest.importorskip("pyscfad")

from diffmrc.core.adint import (build_mole, int2e_s1,  # noqa: E402
                                use_vjp_integrals, vjp_integrals_enabled)

_ATOM = "O 0 0 0; H 0 0 0.757; H 0 0.586 -0.478"
_BASIS = "6-31g"


@pytest.fixture
def vjp_state():
    """Restore the process-global backend choice after the test.

    ``use_vjp_integrals`` writes ``pyscfad.config.moleintor_opt``, which
    is process-global; a test that leaves it set changes every later
    test in the same pytest session.
    """
    old = vjp_integrals_enabled()
    yield
    use_vjp_integrals(old)


def _pyscf_mol():
    from pyscf import gto

    return gto.M(atom=_ATOM, basis=_BASIS, unit="B", verbose=0)


def test_restore_matches_pyscf():
    from pyscf import ao2mo

    mol = _pyscf_mol()
    nao = mol.nao_nr()
    e4 = mol.intor("int2e", aosym="s4")
    ref = ao2mo.restore(1, e4, nao)

    from diffmrc.core.adint import _pair_index, _restore_s1

    got = np.asarray(_restore_s1(jnp.asarray(e4), jnp.asarray(_pair_index(nao))))
    assert got.shape == (nao, nao, nao, nao)
    assert np.abs(got - ref).max() == 0.0


def test_build_mole_leaves_basis_untraced():
    mol = build_mole(atom=_ATOM, basis=_BASIS)
    assert mol.exp is None
    assert mol.ctr_coeff is None
    assert mol.coords is not None
    assert mol.nao_nr() == _pyscf_mol().nao_nr()


def test_vjp_requires_untraced_basis(vjp_state):
    from pyscfad import gto

    m = gto.Mole()
    m.atom = _ATOM
    m.basis = _BASIS
    m.unit = "B"
    m.verbose = 0
    m.build()
    use_vjp_integrals(True)
    with pytest.raises(RuntimeError):
        int2e_s1(m)


def test_backends_agree_on_coordinate_derivative(vjp_state):
    def energy_and_grad(route):
        use_vjp_integrals(route)
        mol = build_mole(atom=_ATOM, basis=_BASIS)
        nao = mol.nao_rr() if hasattr(mol, "nao_rr") else mol.nao_nr()

        def f(m_):
            e = int2e_s1(m_)
            h = m_.intor("int1e_kin") + m_.intor("int1e_nuc")
            return jnp.sum(h) * 1e-3 + jnp.sum(e)

        g = jax.grad(f)(mol)
        return nao, float(f(mol)), np.asarray(g.coords)

    nao0, e0, g0 = energy_and_grad(False)
    nao1, e1, g1 = energy_and_grad(True)
    assert nao0 == nao1
    assert e0 == pytest.approx(e1, abs=1e-10)
    scale = max(np.abs(g0).max(), 1.0)
    assert np.abs(g0 - g1).max() / scale < 1e-12
