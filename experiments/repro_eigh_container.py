"""Reproduce the pre-fix ``EighResult`` container error and the fixed path.

The fix in ``core/eigh.py`` rebuilds the tangent from the primal output's
pytree structure. This script keeps a verbatim copy of the *pre-fix* JVP
rule (a hard-coded ``(dw, du)`` tangent) next to the fixed ``eigh_safe`` so
the failure and the repair can be checked on any jax version without
reverting the release code.

Run:
    PYTHONPATH=src python experiments/repro_eigh_container.py
"""
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


@jax.custom_jvp
def eigh_prefix(a):
    """The rule as it stood before the container fix."""
    return jnp.linalg.eigh(a)


@eigh_prefix.defjvp
def _eigh_prefix_jvp(primals, tangents):
    (a,), (da,) = primals, tangents
    w, u = jnp.linalg.eigh(a)
    L = u.T @ (0.5 * (da + da.T)) @ u
    diff = w[None, :] - w[:, None]
    tau = 1e-4
    F = diff / (diff ** 2 + tau ** 2)
    n = w.shape[0]
    C = L * F * (1.0 - jnp.eye(n))
    dw = jnp.diag(L)
    du = u @ C
    return (w, u), (dw, du)          # hard-coded tuple: the bug


def _loss(fn, n):
    wgt = jnp.arange(1.0, n + 1.0)
    idxw = jnp.arange(0.0, n)

    def f(a):
        w, u = fn(a)
        return jnp.sum(w * wgt) + jnp.sum(u * idxw[:, None] * wgt[None, :])

    return f


def _probe(label, grad_fn, a):
    try:
        g = grad_fn(a)
        jax.block_until_ready(g)
        finite = bool(np.isfinite(np.asarray(g)).all())
        print("%-28s OK      finite=%s  |g|=%.6e"
              % (label, finite, float(jnp.linalg.norm(g))))
    except Exception as exc:                        # noqa: BLE001
        first = str(exc).splitlines()[0]
        print("%-28s FAILED  %s: %s" % (label, type(exc).__name__, first))


def main():
    from diffmrc.core.eigh import eigh_safe

    rng = np.random.default_rng(0)
    B = rng.normal(size=(5, 5))
    A = jnp.asarray(B + B.T + np.diag([0.0, 2.0, 5.0, 9.0, 14.0]))
    print("jax", jax.__version__)
    print("eager container:", type(jnp.linalg.eigh(A)).__name__)
    for name, fn in (("pre-fix", eigh_prefix), ("fixed eigh_safe", eigh_safe)):
        loss = _loss(fn, 5)
        print("--", name)
        _probe("eager grad", jax.grad(loss), A)
        _probe("jit(grad)", jax.jit(jax.grad(loss)), A)
        _probe("grad(jit)",
               lambda a, _l=loss: jax.grad(lambda x: jax.jit(_l)(x))(a), A)
    print("REPRO_DONE")


if __name__ == "__main__":
    main()
