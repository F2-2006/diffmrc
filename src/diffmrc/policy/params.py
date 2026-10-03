"""Every parameter the program has, declared once.

The experiment scripts that preceded this package carried three different
values -- 1e-8, 1e-12, 1e-13 -- for the same physical quantity, the
convergence tolerance on the orbital gradient. Nothing was wrong with any
one of them in its own context; the problem is that a reader could not
tell which run used which, and a user could not ask.

Declaring each parameter once, with its unit, its range and a one-line
description, is what makes ``--show-defaults`` possible at all. It also
gives the report a way to say which values the user chose and which the
program supplied -- the distinction a reader needs to tell a surprising
result from a surprising input.

The values themselves are not invented here. Each one is either measured
in this project (the refinement and geometry tolerances), fixed by the
protocol being implemented (the entropy thresholds), or a documented
engineering limit.
"""
from dataclasses import dataclass, field

__all__ = ['Param', 'EffectiveParams', 'PARAMS', 'defaults', 'validate',
           'effective']

LAYERS = ('core', 'analysis', 'policy', 'driver', 'report')


@dataclass(frozen=True)
class Param:
    """One parameter: its default, its bounds, and what it means."""

    key: str
    default: object
    help: str
    layer: str
    unit: str = ''
    lo: float = None
    hi: float = None
    choices: tuple = ()


@dataclass
class EffectiveParams:
    """The values a run actually uses, and where each came from."""

    values: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.values[key]

    def __contains__(self, key):
        return key in self.values

    def __iter__(self):
        return iter(self.values)

    def items(self):
        return self.values.items()


_PARAM_LIST = [
    # ---- driver: the self-check thresholds (spec section 5.2) ----
    Param('gradient_tol', 1e-8,
          'orbital gradient norm below which the point counts as stationary',
          'driver', unit='dimensionless', lo=0.0, hi=1e-2),
    Param('energy_identity_tol', 1e-8,
          'tolerance on the identity CASSCF <= starting ROHF',
          'driver', unit='Eh', lo=0.0, hi=1e-4),
    Param('geom_grad_tol', 1e-5,
          'force norm below which a geometry counts as optimized',
          'driver', unit='Eh/Bohr', lo=0.0, hi=1e-1),
    Param('geom_max_steps', 200,
          'iteration cap for the geometry optimizer',
          'driver', unit='steps', lo=1.0, hi=5000.0),

    # ---- core: refinement and linear algebra ----
    Param('refine_target', 1e-11,
          'gradient norm the Newton refinement aims for',
          'core', unit='dimensionless', lo=0.0, hi=1e-6),
    Param('optimizer_grad_tol', 1e-6,
          'orbital gradient target for the L-BFGS-B stage of the orbital '
          'point; its measured floor is 1.5e-5, so tighter targets report '
          'false non-convergence. The Newton refinement continues from '
          'where this stage stops',
          'driver', unit='dimensionless', lo=0.0, hi=1e-2),
    Param('basis', '6-31g',
          'basis set for the end-to-end path; the released list is what '
          'this package was validated on, and extending it is a one-line '
          'change once the new basis has a reference number',
          'driver', choices=('sto-3g', '6-31g', 'cc-pvdz', 'ano')),
    Param('method', 'casci',
          'reported energy: the CASCI/CASSCF value, or the on-top tPBE '
          'evaluation at the CASSCF-converged orbitals -- the one-step '
          'MC-PDFT protocol of the DVS paper (A. Wardzala, B. King, '
          'L. Gagliardi, JPCA 2025, doi 10.1021/acs.jpca.4c08597)',
          'driver', choices=('casci', 'mcpdft')),
    Param('multistart_tries', 4,
          'starting points tried at the robust level; the variationally '
          'lowest point is kept. A single start from the reference '
          'orbitals lands on a stationary saddle on H2O/6-31g, and SCF '
          'multi-solution can put the correct solution in another basin',
          'driver', unit='starts', lo=1.0, hi=32.0),
    Param('refine_max_iter', 80,
          'iteration cap for the Newton refinement',
          'core', unit='steps', lo=1.0, hi=1000.0),
    Param('rcond', 1e-9,
          'relative cutoff for the pseudo-inverse of the orbital Hessian',
          'core', unit='relative', lo=0.0, hi=1e-3),

    # ---- policy: active-space selection ----
    Param('plateau_values', 10,
          'consecutive constant steps required to call a stretch a plateau',
          'policy', unit='sweep points', lo=1.0, hi=100.0),
    Param('threshold_step', 0.01,
          'step of the threshold sweep in the plateau detection',
          'policy', unit='relative', lo=0.0, hi=1.0),
    Param('min_entropy', 0.14,
          'largest single-orbital entropy below which the system is single '
          'reference (Stein & Reiher, JCTC 2016)',
          'policy', unit='dimensionless', lo=0.0, hi=1.0),
    Param('localization', 'pm',
          'orbital localization used before the entanglement analysis',
          'policy', choices=('pm', 'boys', 'ibo')),
    Param('max_super_space', 50,
          'largest exploratory space the entanglement analysis will build',
          'policy', unit='orbitals', lo=4.0, hi=200.0),
    Param('weight_min_ratio', 2.0,
          'smallest acceptable ratio between the weakest kept and the '
          'strongest dropped candidate in a weight ranking',
          'policy', unit='ratio', lo=1.0, hi=100.0),
    Param('overlap_min_sigma', 1.1e-6,
          'smallest active-space overlap singular value that still counts '
          'as safe (King & Gagliardi 2021, following Sayfutyarova; '
          'significant only at the extreme low end)',
          'policy', unit='dimensionless', lo=0.0, hi=1.0),
    Param('noon_threshold', 0.05,
          'half-width of the ASS1ST quasi-natural-orbital band [T, 2 - T] '
          '(Khedkar & Roemelt; their examples span T = 0.02 to 0.05)',
          'policy', unit='occupation', lo=0.0, hi=1.0),
]

PARAMS = {p.key: p for p in _PARAM_LIST}


def defaults():
    """The default value of every parameter."""
    return {k: p.default for k, p in PARAMS.items()}


def validate(values):
    """Check a set of parameter values.

    Parameters
    ----------
    values : mapping
        Keys to check. Keys not in `PARAMS` are reported rather than
        ignored, because a misspelled override that is silently dropped
        is worse than an error.

    Returns
    -------
    list of str
        One message per problem, empty when everything is acceptable.
        Every problem is reported, not just the first: a user fixing
        errors one retry at a time is a user wasting time.
    """
    problems = []
    for key, value in values.items():
        p = PARAMS.get(key)
        if p is None:
            problems.append(f'{key}: unknown parameter')
            continue
        if p.choices:
            if value not in p.choices:
                problems.append(
                    f'{key}: {value!r} is not one of {list(p.choices)}'
                )
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            problems.append(f'{key}: {value!r} is not a number')
            continue
        if p.lo is not None and numeric < p.lo:
            problems.append(
                f'{key}: {numeric:g} is below the minimum {p.lo:g}'
            )
        if p.hi is not None and numeric > p.hi:
            problems.append(
                f'{key}: {numeric:g} is above the maximum {p.hi:g}'
            )
    return problems


def effective(overrides=None):
    """Merge user overrides over the defaults, recording each source.

    Parameters
    ----------
    overrides : mapping, optional
        Values the user supplied. Keys that are not parameters, or that
        fall outside their range, raise.

    Returns
    -------
    EffectiveParams
        ``values`` holds the merged configuration; ``sources`` maps each
        key to ``'user'`` or ``'default'``.

    Raises
    ------
    ValueError
        If any override is unknown or out of range.
    """
    overrides = dict(overrides or {})
    problems = validate(overrides)
    if problems:
        raise ValueError('; '.join(problems))
    values = defaults()
    sources = {k: 'default' for k in values}
    for key, value in overrides.items():
        values[key] = value
        sources[key] = 'user'
    return EffectiveParams(values=values, sources=sources)


def parse_active_space(text):
    """Parse '2-4' or '2,3,4' into a list of orbital indices.

    Returns None when the text does not parse, so callers can report the
    offending string rather than raising. The parser lives here, not in
    the command line, because the driver needs it too and the driver may
    not import the command line -- the import contract runs the other
    way.
    """
    text = text.strip()
    if not text:
        return None
    try:
        if '-' in text:
            lo, hi = text.split('-', 1)
            lo, hi = int(lo), int(hi)
            if hi < lo:
                return None
            return list(range(lo, hi + 1))
        parts = text.split(',')
        if any(not part.strip() for part in parts):
            return None
        return [int(part) for part in parts]
    except ValueError:
        return None
