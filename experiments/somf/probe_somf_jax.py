#!/usr/bin/env python
"""Diagnostic probe for the differentiable SOMF layer.

Prints the complex SI spectrum, its real symmetric embedding, and compares
the first derivative of a two-state splitting from three routes with a
central finite difference:

* ``route='doubled'``: real embedding through ``eigh_safe``;
* ``route='complex'``: Hermitian matrix through the eigenvalue-only rule;
* plain ``jnp.linalg.eigvalsh`` for reference.

The transition tensor is a general physical one (all three Cartesian
components), so a phase gauge cannot make the two-state matrix real; the
doubled embedding is the route that preserves the spectrum.

Run on 101:
  PYTHONPATH=<pkg>/src JAX_PLATFORMS=cpu python probe_somf_jax.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

import numpy as np

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from diffmrc.core import somf_jax as jsomf


def physical_tensor(nstates=2, seed=3):
    rng = np.random.default_rng(seed)
    t = np.zeros((nstates, nstates, 3))
    for i in range(nstates):
        for j in range(i + 1, nstates):
            v = rng.normal(size=3)
            t[i, j] = v
            t[j, i] = -v
    return t


def main():
    es = np.asarray([0.0, 0.02])
    t = physical_tensor()
    h_c = np.asarray(jsomf.si_hamiltonian_jax(es, t))
    h_m = np.asarray(jsomf.si_hamiltonian_doubled_jax(es, t))
    print('T =')
    print(t)
    print('hermiticity complex', np.abs(h_c - h_c.conj().T).max())
    print('symmetry doubled   ', np.abs(h_m - h_m.T).max())
    print('eigvals complex', np.linalg.eigvalsh(h_c))
    print('eigvals doubled', np.linalg.eigvalsh(h_m))
    print('splitting doubled', float(jsomf.si_splitting_jax(
        es, t, route='doubled')))
    print('splitting complex', float(jsomf.si_splitting_jax(
        es, t, route='complex')))

    def f_doubled(c):
        return jsomf.si_splitting_jax(es, c * t, route='doubled')

    def f_complex(c):
        return jsomf.si_splitting_jax(es, c * t, route='complex')

    def f_plain(c):
        w = jnp.linalg.eigvalsh(jsomf.si_hamiltonian_jax(es, c * t))
        return (w[2] - w[0]) * jsomf.CM_PER_EH

    def fd(f, c, h):
        return float((f(c + h) - f(c - h)) / (2.0 * h))

    print('AD doubled', float(jax.grad(f_doubled)(1.0)))
    print('AD complex', float(jax.grad(f_complex)(1.0)))
    print('AD plain  ', float(jax.grad(f_plain)(1.0)))
    for h in (1e-3, 1e-4, 1e-5, 1e-6):
        print('h=%.0e  FD doubled %.12f  FD complex %.12f  FD plain %.12f'
              % (h, fd(f_doubled, 1.0, h), fd(f_complex, 1.0, h),
                 fd(f_plain, 1.0, h)))
    probe_eigh_safe_jit()


def probe_eigh_safe_jit():
    """Why the SI path does not call ``core.eigh.eigh_safe``.

    ``eigh_safe`` returns ``jnp.linalg.eigh(a)``; in JAX 0.10.2 that is an
    ``EighResult`` namedtuple, while its custom JVP rule returns a plain
    tuple, so grad-of-jit raises a structure error.  The frozen core
    module is not modified; the eigenvalue-only rule in ``somf_jax``
    covers the SI use case.
    """
    from diffmrc.core.eigh import eigh_safe
    a = jnp.asarray([[1.0, 0.1], [0.1, 2.0]])

    def f(x):
        return jnp.sum(eigh_safe(x)[0])

    print('eigh_safe eager grad', jax.grad(f)(a))
    for label, g in (('jit(grad)', jax.jit(jax.grad(f))),
                     ('grad(jit)', jax.grad(jax.jit(f)))):
        try:
            print('eigh_safe', label, g(a))
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            print('eigh_safe', label, 'FAILED:',
                  type(exc).__name__, str(exc).splitlines()[0])


if __name__ == '__main__':
    main()
