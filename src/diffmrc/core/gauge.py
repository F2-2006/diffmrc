"""Deterministic phase convention for canonical molecular orbitals.

WHY THIS EXISTS

The canonical orbitals returned by an SCF are eigenvectors of the Fock
matrix, and an eigenvector is defined only up to a sign.  The
eigensolver does not fix that sign: the branch it takes depends on the
code path, and the code path of a threaded eigen-decomposition depends
on the thread count.  On this stack the control is OMP_NUM_THREADS
(the pyscfad SCF diagonalizes through JAX/XLA; MKL_NUM_THREADS alone
changes nothing).  Measured on
H2O/6-31G with the release stack (2026-11-02, 101): the SCF coefficient
matrices at OMP_NUM_THREADS = 1, 4 and 8 satisfy

    C_a^T S C_b = diag(+1, -1, ...)

to 1e-12 off the diagonal, with the SCF energy identical to 1e-14 --
the same density and the same orbitals, differing only in the phase
gauge.

That gauge is harmless for an energy evaluated on the orbitals
themselves, and it is what the CI and the density contractions consume.
It is NOT harmless for the orbital-rotation chart C(theta) = C0 @
R(theta): the parameter theta is defined relative to the columns of C0,
so flipping the sign of column j changes which physical rotation a given
theta describes.  The orbital optimizer absorbs the flip when it
re-optimizes theta, but any fixed-theta evaluation -- a frozen-theta
nuclear gradient, a frequency scan, a restarted optimization, or an
evaluation of a theta taken from another run -- then depends on the
thread count.  On H2O/6-31G CAS(2,2) at one fixed theta the NEVPT2 e2
moved by 2.5e-3 Ha between OMP_NUM_THREADS = 1 and 8 solely through this
gauge.

THE CONVENTION

Each column is flipped so that the first coefficient inside a relative
band of the largest absolute coefficient is positive.  The band is
taken relative to the column maximum so that symmetry-equivalent
coefficients (the two hydrogen 1s amplitudes of a symmetric MO, equal to
roundoff and therefore not stably ordered) do not decide the sign: the
lowest index inside the band decides, and that index is stable.

The returned sign is stop-gradient: it is piecewise constant, its
derivative is zero almost everywhere, and differentiating through
``jnp.sign`` at a zero pivot would produce a NaN.

This is a gauge choice, not a rotation: the orbitals and every energy
computed from them are unchanged.  Only the meaning of a rotation
parameter measured from the returned columns changes, and it changes to
a convention that no longer depends on the thread count.
"""
from jax import lax
from jax import numpy as jnp

__all__ = ['canonical_mo_phases']

# Relative half-width of the band around the column maximum that is
# treated as tied.  The SCF coefficients carry roundoff at 1e-14 relative,
# so any value well above that and well below the separation between
# physically distinct amplitudes works; 1e-6 leaves six orders of margin
# on both sides.
BAND_REL = 1e-6


def canonical_mo_phases(C, band=BAND_REL):
    """Return ``C`` with a deterministic, thread-independent column sign.

    Parameters
    ----------
    C : array_like, shape (nao, nmo)
        Canonical orbital coefficients, orthonormal in the AO metric.
    band : float
        Relative band width used for the tie-break.  See the module
        docstring.

    Returns
    -------
    jax.Array
        ``C @ diag(s)`` with ``s`` in {+1, -1} chosen per column.  The
        columns span the same orbitals; only the phase convention is
        fixed.

    Raises
    ------
    ValueError
        If ``C`` is not two-dimensional.
    """
    C = jnp.asarray(C)
    if C.ndim != 2:
        raise ValueError(
            f'canonical_mo_phases needs a 2-D (nao, nmo) array, '
            f'got shape {C.shape}')
    if C.shape[1] == 0:
        return C
    mag = jnp.abs(C)
    mx = jnp.max(mag, axis=0)
    tied = mag >= mx[None, :] * (1.0 - band)
    # argmax on an integer mask returns the first maximal entry, i.e. the
    # lowest AO index inside the band.  Deterministic by construction.
    row = jnp.argmax(tied.astype(jnp.int32), axis=0)
    pivot = C[row, jnp.arange(C.shape[1])]
    sign = jnp.where(pivot < 0.0, -1.0, 1.0)
    return C * lax.stop_gradient(sign)[None, :]
