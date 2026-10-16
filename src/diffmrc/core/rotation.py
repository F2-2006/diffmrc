#!/usr/bin/env python
"""Orbital rotation parametrization and a differentiable CASSCF chain, validated on LiH.

LiH carries a core orbital (the Li 1s), which makes it the transferability test
of the framework to systems with both core and active spaces:

  - the general CI solver with a fully occupied core and Condon-Slater rules
    over the active space, where core-active cross terms are handled
    automatically;
  - the outer rotation pairs, core<->active for STO-3G and core<->active plus
    core<->virtual plus active<->virtual for 6-31G;
  - a quasi-Newton orbital optimizer, and the envelope-theorem nuclear gradient.

Verification: energy against an independent CASSCF implementation, and the
gradient against both a central difference and the analytic reference.

The `block_rotation` function defined here is the parametrization used
throughout the package; the production optimizer lives in optim_theta.py.

Usage: python casscf_lih.py [all|sto3g|631g]
"""
import logging
import os
os.environ.setdefault("JAX_PLATFORMS", "")

import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

logger = logging.getLogger(__name__)


def block_rotation(theta, nmo, pairs):
    """Exponential-map orbital rotation, C = R(theta), as a product of Givens rotations.

    Builds an nmo x nmo identity and accumulates one plane rotation per entry of
    `theta`, in the order given by `pairs`. The result multiplies the reference
    orbital set on the right, so the orbitals become a differentiable function
    of a fixed-size parameter vector.

    The accumulation runs as a `lax.scan` with the pair indices as runtime
    values, so the traced graph carries one scan node instead of one node per
    pair: its size is independent of the pair count.  The earlier unrolled
    Python loop made the compile cost grow with the window -- at the ~10^3-pair
    windows of the aromatic QUESTDB set, single jit compiles took tens of
    minutes and ~20 GB (benzene/cc-pVDZ, 2026-10-13).  Each step applies the
    same arithmetic as ``R @ Rt`` through two column updates (columns i and j
    mix as col_i' = c*col_i - s*col_j, col_j' = s*col_i + c*col_j; all other
    columns are untouched), and the step order -- hence the parametrization --
    is unchanged.

    `pairs` must span every non-redundant inter-block rotation; see
    orblib.orbital_pairs.
    """
    if len(pairs) == 0:
        return jnp.eye(nmo)
    idx_i = jnp.asarray([p[0] for p in pairs], dtype=jnp.int32)
    idx_j = jnp.asarray([p[1] for p in pairs], dtype=jnp.int32)

    def _step(R, xs):
        t, i, j = xs
        c, s = jnp.cos(t), jnp.sin(t)
        ci = R[:, i]
        cj = R[:, j]
        R = R.at[:, i].set(c * ci - s * cj)
        R = R.at[:, j].set(s * ci + c * cj)
        return R, None

    R, _ = jax.lax.scan(_step, jnp.eye(nmo),
                        (jnp.asarray(theta), idx_i, idx_j))
    return R

def optimize_theta(mol, C0, E_of, npar, steps=200, g_conv=1e-9,
                   log_interval=0, label="", theta0=None):
    """L-BFGS with two-loop recursion and backtracking line search.

    First-order only, which keeps the JIT compilation cheap, and superlinear
    convergence, reaching the tolerance in roughly 20 to 50 steps for about ten
    parameters.

    log_interval > 0 logs energy, gradient norm, and resident memory every N
    steps (at INFO; raise the logging level, e.g. the CLI's -v), which is how
    an earlier out-of-memory failure in a state-averaged calculation was
    diagnosed.

    NOTE: this implementation is retained for the validation case. The
    production optimizer is optim_theta.optimize_theta, which replaces it with
    a safeguarded L-BFGS-B; the two defects of the version here are a line-search
    fallback branch that is unreachable (twenty halvings leave the step at
    2^-20, above the 1e-8 floor) and an initial Hessian scaling fixed at unity
    rather than (s.y)/(y.y), which lets the step explode along flat directions.

    theta0 allows a warm start. Without it the optimization begins from the RHF
    orbitals, which on systems with shallow barriers can settle on a suboptimal
    CASSCF stationary point; supplying a theta0 closer to the solution of a
    reference calculation improves the second-order correction substantially.
    The reported energy and gradient are always computed by this chain.
    """
    import resource
    E_jit = jax.jit(E_of)
    g_jit = jax.jit(jax.grad(E_of))
    theta = (jnp.zeros(npar) if theta0 is None
             else jnp.asarray(theta0))
    g = g_jit(theta)
    gnorm = float(jnp.linalg.norm(g))
    s_list, y_list = [], []
    m = 8  # memory length

    def lbfgs_dir(g_, ss, ys):
        q = g_
        alphas = []
        for s, y in reversed(list(zip(ss, ys))):
            rho = 1.0 / (y @ s + 1e-30)
            alpha = rho * (s @ q)
            alphas.append(alpha)
            q = q - alpha * y
        gamma = (ys[-1] @ ss[-1]) / (ys[-1] @ ys[-1] + 1e-30) if ys else 1.0
        r = gamma * q
        for s, y, alpha in zip(ss, ys, reversed(alphas)):
            rho = 1.0 / (y @ s + 1e-30)
            beta = rho * (y @ r)
            r = r + s * (alpha - beta)
        return -r

    for it in range(steps):
        if gnorm < g_conv:
            break
        if log_interval and it % log_interval == 0:
            rss = open("/proc/self/status").read().split("VmRSS:")[1].split()[0]
            logger.info('[%s iter %d] E = %.10f  |g| = %.2e  RSS = %.1fGB',
                        label, it, float(E_jit(theta)), gnorm,
                        int(rss) / 1048576)
        d = lbfgs_dir(g, s_list, y_list)
        # Backtracking line search (Armijo condition)
        e = float(E_jit(theta))
        lr = 1.0
        for _ in range(20):
            theta_new = theta + lr * d
            en = float(E_jit(theta_new))
            if en < e + 1e-4 * lr * float(g @ d):
                break
            lr *= 0.5
        if lr < 1e-8:  # line search failed; fall back to a gradient step
            d = -g
            lr = 1e-3
            theta_new = theta + lr * d
        s = theta_new - theta
        y = g_jit(theta_new) - g
        if float(s @ y) > 1e-12:  # store only if the curvature condition holds
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        theta = theta_new
        g = g_jit(theta)
        gnorm = float(jnp.linalg.norm(g))
    E = float(E_jit(theta))
    return E, theta, gnorm
