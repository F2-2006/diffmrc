"""Robust orbital optimizer, replacing a hand-written L-BFGS.

WHY THIS MODULE EXISTS

A hand-written L-BFGS frequently failed to converge when started from a
perturbed point: a perturbation of 1e-2 produced a displacement of 2.78 in the
parameter vector, two orders of magnitude larger, and terminated at a gradient
norm of 2.4e-1, far from convergence. In one scan, none of five perturbed
starting points converged. Results computed with that optimizer and read as
"lower solutions" were in fact unconverged intermediate values.

ROOT CAUSES, from reading the implementation in casscf_lih.optimize_theta

  Defect 1 - the line-search fallback branch is dead code.

      for _ in range(20):
          if en < e + 1e-4*lr*float(g@d): break
          lr *= 0.5
      if lr < 1e-8:          # after 20 halvings the smallest lr is
          d = -g; lr = 1e-3  # 2^-20, about 9.5e-7, never below 1e-8,
                             # so this branch never executes

  Defect 2 - the initial Hessian scaling is fixed at gamma = 1.0, whereas the
  standard choice is (s.y)/(y.y). With an empty memory the direction is -g at
  an uncontrolled magnitude, and along flat directions gamma can become very
  large, giving an oversized step that the Armijo test rejects, leaving only a
  microscopic move.

Together these make the iteration either overstep or be rejected into
microscopic steps, so that after 300 iterations the gradient norm is still
large.

WHAT THIS MODULE DOES

The primary path uses scipy's L-BFGS-B, whose safeguards are well tested, with
gradients still supplied by JAX. This is a numerical substitution only: the
optimizer itself is never differentiated, so differentiability is unaffected.
The hand-written version is retained as a fallback with both defects fixed.

USAGE

    from optim_theta import optimize_theta
    E_jit, theta, gnorm = optimize_theta(mol, C0, E_cas, npar,
                                         steps=400, g_conv=1e-8,
                                         label='N2', theta0=None)
"""
import logging

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

logger = logging.getLogger(__name__)

__all__ = ['optimize_theta', 'optimize_theta_scipy', 'optimize_theta_hand']

# Step-size floor for the handwritten fallback. The original used 1e-8, which
# twenty halvings never reach; 1e-6 is attainable.
_LR_FLOOR = 1e-6


def optimize_theta_scipy(E_of, npar, steps=400, g_conv=1e-8,
                         theta0=None, label='', log_interval=0, arrays=()):
    """Primary path: scipy L-BFGS-B.

    ``arrays`` are extra runtime arguments appended to every call of
    ``E_of(theta, *arrays)``: large reference arrays (AO integrals) must
    travel this way rather than through closure, or ``jax.jit`` embeds
    them in the traced graph as constants and the compiles explode
    (nao^4 tensors; the pattern of driver.kernel.energy_function, and
    the lesson of a 20 GB compile at benzene/cc-pVDZ, 2026-10-13).

    Returns (E_jit, theta, gnorm, info).
    """
    import scipy.optimize as sopt

    E_jit = jax.jit(E_of)
    g_jit = jax.jit(jax.grad(E_of))

    def fun(x):
        return float(E_jit(jnp.asarray(x), *arrays))

    def jac(x):
        return np.asarray(g_jit(jnp.asarray(x), *arrays), dtype=float)

    x0 = (np.zeros(npar) if theta0 is None
          else np.asarray(theta0, dtype=float))
    try:
        res = sopt.minimize(
            fun, x0, jac=jac, method='L-BFGS-B',
            options=dict(maxiter=int(steps), gtol=float(g_conv),
                         ftol=1e-14, maxls=50))
        x = np.asarray(res.x, dtype=float)
        ok = bool(res.success) or res.status in (0, 1)
        msg = str(res.message)[:60]
        n_it = int(getattr(res, 'nit', -1))
    except Exception as ex:
        return None, jnp.asarray(x0), float('nan'), dict(
            ok=False, msg=f'{type(ex).__name__}: {str(ex)[:50]}', nit=-1)

    theta = jnp.asarray(x)
    gnorm = float(jnp.linalg.norm(g_jit(theta, *arrays)))
    return E_jit, theta, gnorm, dict(ok=ok, msg=msg, nit=n_it,
                                     E=float(E_jit(theta, *arrays)))


def optimize_theta_hand(E_of, npar, steps=400, g_conv=1e-9, theta0=None,
                        label='', log_interval=0, arrays=()):
    """Fallback path: hand-written L-BFGS with both defects fixed.

    ``arrays`` as in ``optimize_theta_scipy``.

    Returns (E_jit, theta, gnorm, info).
    """
    E_jit = jax.jit(E_of)
    g_jit = jax.jit(jax.grad(E_of))

    theta = (jnp.zeros(npar) if theta0 is None else jnp.asarray(theta0))
    g = g_jit(theta, *arrays)
    gnorm = float(jnp.linalg.norm(g))
    s_list, y_list = [], []
    m = 8
    n_stall = 0

    for it in range(steps):
        if gnorm < g_conv:
            break
        if log_interval and it % log_interval == 0:
            try:
                rss = (open("/proc/self/status").read().split("VmRSS:")[1]
                       .split()[0])
                rss_s = f'  RSS={int(rss)/1048576:.1f}GB'
            except Exception:
                rss_s = ''
            logger.info('[%s iter %d] E = %.10f  |g| = %.2e%s',
                        label, it, float(E_jit(theta, *arrays)), gnorm, rss_s)

        # ---- L-BFGS two-loop recursion
        ss, ys = s_list, y_list
        q = g
        alphas = []
        for s, y in reversed(list(zip(ss, ys))):
            rho = 1.0 / (y @ s + 1e-30)
            alpha = rho * (s @ q)
            alphas.append(alpha)
            q = q - alpha * y
        # Fix for defect 2: with an empty memory, scale by the gradient norm
        # rather than a fixed 1.0; with a populated memory, use the standard
        # (s.y)/(y.y) clamped to a reasonable range.
        if ys:
            gamma = float((ys[-1] @ ss[-1]) / (ys[-1] @ ys[-1] + 1e-30))
            gamma = min(max(gamma, 1e-3), 1e3)
        else:
            gamma = 1.0 / (gnorm + 1e-12)
        r = gamma * q
        for s, y, alpha in zip(ss, ys, reversed(alphas)):
            rho = 1.0 / (y @ s + 1e-30)
            beta = rho * (y @ r)
            r = r + s * (alpha - beta)
        d = -r
        # Cap the step norm, so that flat directions cannot blow up.
        dn = float(jnp.linalg.norm(d))
        if dn > 10.0:
            d = d * (10.0 / dn)
            dn = 10.0

        # ---- Armijo backtracking
        e = float(E_jit(theta, *arrays))
        gd = float(g @ d)
        lr = 1.0
        accepted = False
        for _ in range(30):
            theta_new = theta + lr * d
            if float(E_jit(theta_new, *arrays)) < e + 1e-4 * lr * gd:
                accepted = True
                break
            lr *= 0.5
        # Fix for defect 1: the fallback is now reachable, since 30 halvings
        # bring lr to about 9.3e-10, below the floor.
        if not accepted or lr < _LR_FLOOR:
            d = -g
            lr = 1e-2 / (gnorm + 1e-12)
            theta_new = theta + lr * d
            s_list, y_list = [], []      # clear the memory; the original kept
                                         # it, letting bad curvature persist
            n_stall += 1
        else:
            n_stall = 0

        s = theta_new - theta
        y = g_jit(theta_new, *arrays) - g
        if float(s @ y) > 1e-12:
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        theta = theta_new
        g = g_jit(theta, *arrays)
        gnorm = float(jnp.linalg.norm(g))
        if n_stall > 20:
            break

    return E_jit, theta, gnorm, dict(ok=(gnorm < 1e-6), msg='hand',
                                     nit=it + 1, E=float(E_jit(theta, *arrays)))


def optimize_theta(mol, C0, E_of, npar, steps=400, g_conv=1e-8,
                   log_interval=0, label='', theta0=None):
    """Unified entry point, keeping the return signature of the original.

    Uses scipy L-BFGS-B; if that fails, falls back to the handwritten version,
    and if scipy reports non-convergence, retries the handwritten version from
    scipy's endpoint.

    Returns (E_jit, theta, gnorm).
    """
    E_jit, theta, gnorm, info = optimize_theta_scipy(
        E_of, npar, steps=steps, g_conv=g_conv, theta0=theta0,
        label=label, log_interval=log_interval)
    if E_jit is None:
        _, theta, gnorm, info = optimize_theta_hand(
            E_of, npar, steps=steps, g_conv=g_conv, theta0=theta0,
            label=label, log_interval=log_interval)
        E_jit = jax.jit(E_of)
        return E_jit, theta, gnorm

    if gnorm > max(g_conv * 100, 1e-6) and theta0 is None:
        E2_, th2, gn2, info2 = optimize_theta_hand(
            E_of, npar, steps=steps, g_conv=g_conv, theta0=theta,
            label=label, log_interval=log_interval)
        if gn2 < gnorm:
            theta, gnorm = th2, gn2

    return E_jit, theta, gnorm
