"""Memory-bounded AO integral derivatives for the differentiable chain.

Why this module exists
----------------------

The nuclear gradient of every energy in this package is a reverse-mode
derivative through ``pyscfad``'s integral objects.  ``pyscfad`` ships two
backends for the two-electron integrals:

``moleintor_opt = False`` (the upstream default)
    a ``custom_jvp``.  Its forward form differentiates the integral
    tensor elementwise and returns a tangent of the full ``nao**4``
    shape.  Reverse mode transposes that linear map by replaying it on
    one basis tangent per Cartesian direction, so the expression
    materialises ``nctr * nao**4`` doubles at once, with
    ``nctr = 3 * natm``.

``moleintor_opt = True``
    a ``custom_vjp`` whose backward kernel (``GTOnr2e_fill_r0_vjp``)
    contracts the ``nao**4`` cotangent against the derivative inside
    C and returns only ``(natm, 3)``.  The working set is the cotangent
    itself, not a factor of ``nctr`` times it.

The second backend is the one whose cost is independent of the molecule
size, and it is what this module switches on.  It is not a drop-in
replacement, for two reasons that the module handles:

* the vjp kernel only knows the ``s4`` packing, so a caller that wants
  the ``s1`` tensor has to ask for ``s4`` and restore it.  The restore
  is a pair of ``take`` operations on the ``(nao, nao)`` pair index, so
  it stays differentiable and never allocates an ``nao**4`` index map.
* the vjp kernel cannot differentiate the basis exponents or the
  contraction coefficients, and it raises rather than silently dropping
  them.  A ``pyscfad`` Mole built with the default
  ``trace_exp=True, trace_ctr_coeff=True`` carries both as traced
  leaves, so ``use_vjp_integrals()`` requires a molecule built with
  those two tracers off.  ``build_mole`` does that.

Neither switch changes the value of any integral or of any derivative:
the two backends compute the same analytic expression.  Measured
agreement on the ERI coordinate derivative is at the 1e-15 relative
level (``tests/core/test_adint.py``).

Usage
-----

    from diffmrc.core.adint import build_mole, int2e_s1, use_vjp_integrals

    use_vjp_integrals(True)
    mol = build_mole(atom='O 0 0 0; H 0 0 0.757; H 0 0.586 -0.478',
                     basis='6-31g')
    eri = int2e_s1(mol)          # s1 tensor, bounded working set

With the switch off, ``int2e_s1`` is exactly
``mol.intor('int2e', aosym='s1')``.
"""
import numpy as np

__all__ = ['use_vjp_integrals', 'vjp_integrals_enabled', 'int2e_s1',
           'build_mole']

_ENABLED = False
_PAIR_CACHE = {}


def use_vjp_integrals(enable=True):
    """Select the memory-bounded integral-derivative backend.

    Sets ``pyscfad.config.moleintor_opt`` to ``enable`` and records the
    choice for ``int2e_s1``.  Returns the new state.

    The setting is global to the ``pyscfad`` package, matching the
    upstream flag.  Molecules differentiated through this backend must
    be built with ``trace_exp=False, trace_ctr_coeff=False``; see
    ``build_mole``.

    Two-centre integrals keep the ``custom_jvp`` backend even when the
    switch is on: the ``custom_vjp`` two-centre kernel segfaults on
    ``ECPscalar`` (measured on HI/lanl2dz, both the bare integral
    gradient and the SCF gradient; ``int1e_kin``, ``int1e_nuc`` and
    ``int1e_ovlp`` are fine).  A two-centre tensor is ``nao**2``, so its
    ``custom_jvp`` transpose costs ``nctr * nao**2`` -- six orders of
    magnitude below the four-centre term at ``nao = 100`` and not worth
    the risk.  The rebinding acts on the module attribute that
    ``pyscfad.gto.moleintor.intor`` reads at call time.
    """
    global _ENABLED
    import pyscfad

    pyscfad.config.moleintor_opt = bool(enable)
    if enable:
        from pyscfad.gto import _moleintor_jvp, _moleintor_vjp

        _moleintor_vjp.intor2c = _moleintor_jvp.intor2c
    _ENABLED = bool(enable)
    return _ENABLED


def vjp_integrals_enabled():
    """Whether ``int2e_s1`` will use the s4-backed route."""
    return _ENABLED


def _pair_index(nao):
    """``(nao, nao)`` map from an AO pair to its ``s4`` packed index.

    ``P[i, j]`` is the row of the packed array that holds ``(i, j)`` in
    the pyscf ``s4`` convention: pairs are enumerated with ``i >= j`` in
    row-major order and mirrored.
    """
    P = _PAIR_CACHE.get(nao)
    if P is None:
        P = np.empty((nao, nao), dtype=np.int64)
        k = 0
        for i in range(nao):
            for j in range(i + 1):
                P[i, j] = k
                P[j, i] = k
                k += 1
        P.setflags(write=False)
        _PAIR_CACHE[nao] = P
    return P


def _restore_s1(e4, P):
    """Unpack an ``s4`` ERI tensor to ``(nao, nao, nao, nao)``.

    Two gathers instead of one: the first expands the bra pair index,
    the second the ket pair index.  The intermediate is
    ``(nao, nao, npair)``, smaller than the ``nao**4`` output, and no
    ``nao**4`` integer index map is ever built.
    """
    from jax import numpy as jnp

    return jnp.take(jnp.take(e4, P, axis=0), P, axis=2)


def int2e_s1(mol):
    """The ``s1`` two-electron integral tensor of ``mol``.

    Routes through the ``s4`` packing and a differentiable restore when
    ``use_vjp_integrals(True)`` has been called, and is the plain
    ``mol.intor('int2e', aosym='s1')`` otherwise.  Both routes return
    the same tensor.
    """
    if not _ENABLED:
        return mol.intor('int2e', aosym='s1')
    if getattr(mol, 'exp', None) is not None or \
            getattr(mol, 'ctr_coeff', None) is not None:
        raise RuntimeError(
            'the vjp integral backend cannot differentiate basis exponents '
            'or contraction coefficients; build the molecule with '
            'diffmrc.core.adint.build_mole (trace_exp=False, '
            'trace_ctr_coeff=False) or call use_vjp_integrals(False)')
    from jax import numpy as jnp

    nao = mol.nao_nr()
    P = jnp.asarray(_pair_index(nao))
    e4 = mol.intor('int2e', aosym='s4')
    return _restore_s1(e4, P)


def build_mole(atom=None, basis=None, ecp=None, unit='Bohr', verbose=0,
               **kwargs):
    """A ``pyscfad`` Mole whose only traced leaf is the nuclear geometry.

    ``pyscfad.gto.Mole.build`` traces the coordinates, the basis
    exponents and the contraction coefficients by default.  The last two
    are irrelevant to a nuclear gradient and cost a full ``nao**4``
    tangent per Cartesian direction in the JVP backend and a hard
    ``NotImplementedError`` in the VJP backend.  This constructor leaves
    them untraced.

    Parameters are the ones the chain builds with: ``atom``, ``basis``,
    ``ecp``, ``unit``, ``verbose``.  Remaining keyword arguments go to
    ``Mole.build``.
    """
    from pyscfad import gto

    m = gto.Mole()
    if atom is not None:
        m.atom = atom
    if basis is not None:
        m.basis = basis
    if ecp is not None:
        m.ecp = ecp
    m.unit = unit
    m.verbose = verbose
    kwargs.setdefault('trace_exp', False)
    kwargs.setdefault('trace_ctr_coeff', False)
    m.build(**kwargs)
    return m
