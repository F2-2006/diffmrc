"""Vibrational frequencies from a nuclear Hessian.

The Hessian is mass-weighted, projected onto the vibrational subspace,
and diagonalized. The projection removes the translations and rotations
that would otherwise appear as zero or imaginary frequencies; how many
there are depends on whether the molecule is linear, which is decided
from the geometry rather than assumed.

The conversion factor is written once, as the product of two defined
constants, rather than as a second literal. An earlier script in this
project carried both forms; they agreed, but nothing would have caught
them diverging.

The module also carries the diatomic-scan side: parabola fits of E(r)
scans with the audit numbers (span, vertex-in-window, three independent
estimates) the f-block scans relied on. A local three-point check cannot
self-detect its failures -- three points always lie on *some* parabola
-- so the audit exists to make a soft glitch (one point in a different
solution basin inflates the curvature) and a displaced window (vertex
outside the sampled range) visible instead of silent.

References
----------
The mass-weighting and projection follow the standard treatment; see
E. B. Wilson, J. C. Decius, P. C. Cross, *Molecular Vibrations* (1955).
"""
from dataclasses import dataclass

import numpy as np

__all__ = ['Projector', 'VibrationalResult', 'trans_rot_projector',
           'mass_weighted_hessian', 'wavenumbers', 'HA2CM', 'AMU2ME',
           'ANG2BOHR', 'SCAN_SPAN_MAX', 'ScanFit', 'omega_of_curvature',
           'fit_scan_parabola', 'multipoint_scan_estimates']

# Hartree to wavenumber, and atomic mass unit to electron mass.
HA2CM = 219474.6313632
AMU2ME = 1822.8884862
_LINEAR_TOL = 1e-6

#: Bohr per Angstrom, for the scan fits (the geometry is differentiated
#: in Angstrom, the curvature arrives in Eh/Angstrom^2).
ANG2BOHR = 1.8897259886

#: Above this neighbouring-point step (Hartree) a three-point parabola
#: is presumed to mix solution basins. Physical steps measured across
#: six f-block systems (2026-10-11): 2e-4 .. 1.2e-3 Ha; a crossed basin
#: showed 2.7e-2 (CeF) or 6.7e-1 (TmF).
SCAN_SPAN_MAX = 5e-3


@dataclass
class Projector:
    """Projection onto the vibrational subspace, and the mode count."""

    P: np.ndarray
    n_trans_rot: int


@dataclass
class VibrationalResult:
    """Frequencies and the diagnostics that say whether to trust them."""

    freqs_cm: np.ndarray
    n_imag: int
    zero_mode_magnitudes: np.ndarray
    n_trans_rot: int


def trans_rot_projector(coords, masses):
    """Projector onto the vibrational subspace, in mass-weighted coordinates.

    Parameters
    ----------
    coords : array_like
        Nuclear coordinates, shape ``(natm, 3)``.
    masses : array_like
        Atomic masses in amu, shape ``(natm,)``.

    Returns
    -------
    Projector
        ``P`` has shape ``(3 natm, 3 natm)``; ``n_trans_rot`` is 5 for a
        linear molecule and 6 otherwise.

    Raises
    ------
    ValueError
        If the shapes are inconsistent.
    """
    coords = np.asarray(coords, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f'coords must be (natm, 3), got {coords.shape}')
    if masses.shape != (coords.shape[0],):
        raise ValueError(
            f'masses shape {masses.shape} does not match {coords.shape[0]} '
            f'atoms'
        )

    n = len(masses)
    m3 = np.repeat(masses, 3)
    sm = np.sqrt(m3)

    # Translations.
    basis = []
    for axis in range(3):
        v = np.zeros(3 * n)
        v[axis::3] = sm[axis::3]
        basis.append(v)

    # Rotations about the centre of mass. A linear molecule has only two
    # independent ones; the third is numerically null and is dropped by
    # the norm test below.
    com = (masses[:, None] * coords).sum(0) / masses.sum()
    rc = coords - com
    for axis in range(3):
        e = np.zeros(3)
        e[axis] = 1.0
        v = np.zeros(3 * n)
        for i in range(n):
            v[3 * i:3 * i + 3] = sm[3 * i] * np.cross(e, rc[i])
        basis.append(v)

    B = np.array(basis).T                      # (3n, 6)
    norms = np.linalg.norm(B[:, 3:], axis=0)
    keep = norms > _LINEAR_TOL
    n_trans_rot = 3 + int(keep.sum())
    B = np.hstack([B[:, :3], B[:, 3:][:, keep]])
    Q, _ = np.linalg.qr(B)
    return Projector(P=np.eye(3 * n) - Q @ Q.T, n_trans_rot=n_trans_rot)


def mass_weighted_hessian(H, masses):
    """Divide the Hessian elementwise by sqrt(m_i m_j).

    Parameters
    ----------
    H : array_like
        Hessian in Eh/Bohr^2, shape ``(3 natm, 3 natm)``.
    masses : array_like
        Atomic masses in amu.

    Returns
    -------
    numpy.ndarray
        The mass-weighted Hessian.

    Raises
    ------
    ValueError
        If the shapes are inconsistent.
    """
    H = np.asarray(H, dtype=float)
    masses = np.asarray(masses, dtype=float)
    m3 = np.repeat(masses, 3)
    if H.shape != (m3.size, m3.size):
        raise ValueError(
            f'H shape {H.shape} does not match {m3.size} mass-weighted '
            f'coordinates'
        )
    return H / np.sqrt(np.outer(m3, m3))


def wavenumbers(H, coords, masses):
    """Vibrational wavenumbers in cm^-1.

    Parameters
    ----------
    H : array_like
        Hessian in Eh/Bohr^2.
    coords : array_like
        Nuclear coordinates, shape ``(natm, 3)``, in the same length unit
        the Hessian was differentiated with (Bohr).
    masses : array_like
        Atomic masses in amu.

    Returns
    -------
    VibrationalResult
        ``freqs_cm`` holds the ``3 natm - n_trans_rot`` vibrational
        wavenumbers, signed: a negative entry is an imaginary frequency.
        ``zero_mode_magnitudes`` holds the magnitudes of the projected-out
        eigenvalues, which should be numerically zero; reporting them lets
        a caller confirm the projection actually worked rather than trust
        it.
    """
    H = np.asarray(H, dtype=float)
    proj = trans_rot_projector(coords, masses)
    Hmw = mass_weighted_hessian(H, masses)
    Hp = proj.P @ Hmw @ proj.P
    Hp = 0.5 * (Hp + Hp.T)
    w = np.linalg.eigvalsh(Hp)
    order = np.argsort(np.abs(w))
    zero_modes = np.abs(w[order[:proj.n_trans_rot]])
    lam = w[order[proj.n_trans_rot:]]
    nu = np.sign(lam) * np.sqrt(np.abs(lam)) * (HA2CM / np.sqrt(AMU2ME))
    return VibrationalResult(
        freqs_cm=np.sort(nu), n_imag=int((nu < 0).sum()),
        zero_mode_magnitudes=zero_modes, n_trans_rot=proj.n_trans_rot,
    )


@dataclass
class ScanFit:
    """A parabola fitted to a diatomic bond-length scan, with the audit
    numbers that decide whether to trust it."""

    vertex: float            # fitted stationary point, Angstrom
    k: float                 # spring constant d2E/dr2, Eh/Angstrom^2
    omega: float             # harmonic wavenumber, cm^-1 (nan if k <= 0)
    n_points: int
    span: float              # max |neighbour step|, Eh
    vertex_in_window: bool


def omega_of_curvature(k_ha_ang2, mu_amu):
    """The diatomic scan unit chain: curvature (Eh/Angstrom^2, spring
    constant) and reduced mass (amu) to omega_e (cm^-1)."""
    k_au = float(k_ha_ang2) / ANG2BOHR ** 2
    mu_au = float(mu_amu) * AMU2ME
    if not k_au > 0 or not mu_au > 0:
        return float('nan')
    return float(np.sqrt(k_au / mu_au) * HA2CM)


def fit_scan_parabola(grid, energies, mu_amu):
    """Quadratic fit of a bond-length scan.

    Exact through three points; a least-squares quadratic beyond (robust
    to one corrupted point, at the price of an anharmonicity bias over a
    wide span). ``span`` is the largest neighbouring-point energy step,
    to compare against :data:`SCAN_SPAN_MAX`.
    """
    x = np.asarray(grid, dtype=float)
    y = np.asarray(energies, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x.size < 3:
        raise ValueError(f'a parabola needs at least 3 points, got {x.size}')
    c = np.polyfit(x, y, 2)
    vertex = float(-c[1] / (2 * c[0]))
    k = float(2 * c[0])
    steps = np.abs(np.diff(y))
    return ScanFit(
        vertex=vertex, k=k, omega=omega_of_curvature(k, mu_amu),
        n_points=int(x.size),
        span=float(steps.max()) if steps.size else 0.0,
        vertex_in_window=bool(x[0] <= vertex <= x[-1]),
    )


def multipoint_scan_estimates(grid, energies, mu_amu, inner_tol=0.0301):
    """The three-estimate scan audit.

    Two failure modes survive a local three-point check: a soft glitch
    inflates the curvature (CeO 2026-10-12: apparent 1536 vs ~820
    cm^-1), and a scan centred on a wrong reference pushes the vertex
    outside the window (LuF: 0.107 Angstrom beyond the grid). Three
    estimates -- the inner triplet, a least-squares quadratic over all
    points, and a quartic evaluated at its own vertex -- plus the span
    and window flags make the disagreements visible.

    Returns ``{'q3': ScanFit, 'q5': ScanFit, 'q4': ScanFit}``; the
    quartic vertex search is confined to the sampled window padded by
    0.05 Angstrom, and its fields are nan when no stationary point
    exists there.
    """
    x = np.asarray(grid, dtype=float)
    y = np.asarray(energies, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x.size < 5:
        raise ValueError(f'five points expected, got {x.size}')
    centre = float(np.median(x))
    inner = np.abs(x - centre) <= inner_tol
    span = float(np.abs(np.diff(y)).max()) if y.size > 1 else 0.0
    out = {'q3': fit_scan_parabola(x[inner], y[inner], mu_amu),
           'q5': fit_scan_parabola(x, y, mu_amu)}
    c4 = np.polyfit(x, y, 4)
    roots = np.roots(np.polyder(c4))
    cand = [r.real for r in roots
            if abs(r.imag) < 1e-9
            and x.min() - 0.05 <= r.real <= x.max() + 0.05]
    if cand:
        xv = min(cand, key=lambda r: float(np.polyval(c4, r)))
        k4 = float(np.polyval(np.polyder(c4, 2), xv))
        out['q4'] = ScanFit(vertex=float(xv), k=k4,
                            omega=omega_of_curvature(k4, mu_amu),
                            n_points=int(x.size), span=span,
                            vertex_in_window=True)
    else:
        out['q4'] = ScanFit(vertex=float('nan'), k=float('nan'),
                            omega=float('nan'), n_points=int(x.size),
                            span=span, vertex_in_window=False)
    return out
