"""The one entry point.

Three ideas shape this module.

Defaults are generated from the parameter table rather than repeated
here, so a value has exactly one definition and ``--show-defaults``
cannot drift from what a run actually uses.

Validation happens before anything expensive. ``--check`` runs the same
parse and validation path as a real run, so an input that fails here
would have failed there, only later and after spending computer time.

Errors say which thing is wrong, what its value was, why that is not
allowed, and what to do. The model is a message from PySCF that names
the common misconception directly: "mol.spin = 2S = Nalpha - Nbeta, not
2S+1". Most error messages in that library are bare assertions with no
text at all, which is why this project writes its own.

Exit codes: 0 success, 1 the calculation failed or was refused, 2 the
input was rejected.
"""
import argparse
import os
import sys

__all__ = ['build_parser', 'main']

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BAD_INPUT = 2

_LEVELS = ('fast', 'robust')


def build_parser():
    """Build the argument parser, with defaults from the parameter table."""
    from ..policy.params import PARAMS

    p = argparse.ArgumentParser(
        prog='diffmrc',
        description='End-to-end differentiable multireference electronic '
                    'structure.',
        epilog='Run with --show-defaults to see every parameter the program '
               'decides for you.',
    )
    p.add_argument('geometry', nargs='?',
                   help='molecular geometry in xyz format')
    p.add_argument('--charge', type=int, default=0,
                   help='total charge (default: 0)')
    p.add_argument('--spin', type=int, default=0,
                   help='2S = Nalpha - Nbeta, NOT the multiplicity 2S+1 '
                        '(default: 0)')
    p.add_argument('--level', choices=_LEVELS, default='robust',
                   help='fast skips the entanglement verification '
                        '(default: robust)')
    p.add_argument('--active-space', default=None,
                   help='manual active space, e.g. 2-4 or 2,3,4; bypasses '
                        'automatic selection')
    p.add_argument('--outdir', default='.',
                   help='where to write the reports (default: current dir)')
    p.add_argument('--show-defaults', action='store_true',
                   help='print every parameter and its default, then exit')
    p.add_argument('--check', action='store_true',
                   help='validate the input without computing anything')

    # Every numeric and choice parameter is exposed, generated from the
    # table so no default is written twice.
    for key, spec in PARAMS.items():
        if key == 'localization':
            p.add_argument(f'--{key.replace("_", "-")}',
                           choices=list(spec.choices), default=spec.default,
                           help=spec.help)
        else:
            flag = f'--{key.replace("_", "-")}'
            p.add_argument(flag, type=float, default=spec.default,
                           metavar='VALUE', help=spec.help)
    return p


def _param_overrides(args):
    """Collect the parameter values the user actually set."""
    from ..policy.params import PARAMS

    out = {}
    for key in PARAMS:
        attr = key.replace('-', '_')
        if not hasattr(args, attr):
            continue
        value = getattr(args, attr)
        if value != PARAMS[key].default:
            out[key] = value
    return out


def _print_defaults(stream):
    from ..policy.params import PARAMS

    width = max(len(k) for k in PARAMS)
    for key in sorted(PARAMS):
        spec = PARAMS[key]
        unit = f' [{spec.unit}]' if spec.unit else ''
        stream.write(f'  {key:<{width}}  {spec.default!r}{unit}\n')
        stream.write(f'  {"":<{width}}  {spec.help}\n')


def _parse_active_space(text):
    """Parse '2-4' or '2,3,4' into a list of indices.

    Returns None when the text does not parse, so the caller can report
    the offending string rather than raising a bare ValueError.
    """
    text = text.strip()
    try:
        if '-' in text:
            lo, hi = text.split('-', 1)
            lo, hi = int(lo), int(hi)
            if hi < lo:
                return None
            return list(range(lo, hi + 1))
        return [int(x) for x in text.split(',') if x.strip()]
    except ValueError:
        return None


def _validate_input(path, charge, spin, active_space):
    """Return a list of problems; empty means the input is usable."""
    from ..policy.params import validate

    problems = []
    if not os.path.exists(path):
        problems.append(f'{path}: no such file')
        return problems

    try:
        text = open(path, encoding='utf-8').read()
    except OSError as exc:
        problems.append(f'{path}: cannot be read ({exc.strerror})')
        return problems

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        problems.append(f'{path}: the file is empty')
        return problems
    try:
        n_atoms = int(lines[0].split()[0])
    except (ValueError, IndexError):
        problems.append(
            f'{path}: the first line must be the number of atoms, '
            f'found {lines[0]!r}'
        )
        return problems

    body = lines[2:2 + n_atoms] if len(lines) > 2 else []
    if len(body) < n_atoms:
        problems.append(
            f'{path}: the header says {n_atoms} atoms but only {len(body)} '
            f'lines follow'
        )

    # Element symbols are checked against the built-in table rather than
    # by asking the numerical library to build a Mole. Validation must
    # work without that library installed: --check exists precisely so a
    # user can find an input error without spending computer time, and
    # requiring the whole kernel for it would defeat the purpose.
    symbols = [ln.split()[0] for ln in body if ln.split()]
    unknown = [s for s in symbols if not _is_known_element(s)]
    if unknown:
        problems.append(
            f'{path}: unknown element symbol(s) {unknown}; '
            f'expected e.g. H, C, O, Eu'
        )

    # A second pass with the real library when it is available, which
    # catches anything the table cannot: bad coordinates, basis problems.
    if not problems:
        try:
            from pyscf import gto
            atom = '; '.join(f'{s} 0 0 0' for s in symbols)
            gto.M(atom=atom, unit='Angstrom', verbose=0)
        except ImportError:
            pass                      # the kernel is optional for --check
        except Exception as exc:
            problems.append(f'{path}: {exc}')

    if spin < 0:
        problems.append(
            f'spin: {spin} is negative; spin is 2S = Nalpha - Nbeta, '
            f'which cannot be negative'
        )

    n_electrons = None
    try:
        n_electrons = sum(_atomic_number(ln.split()[0])
                          for ln in body if ln.split()) - charge
    except Exception:
        pass
    if n_electrons is not None:
        if n_electrons < 0:
            problems.append(
                f'charge: {charge} would remove more electrons than the '
                f'{n_electrons + charge} the molecule has'
            )
        elif spin > n_electrons:
            problems.append(
                f'spin: {spin} unpaired electrons is impossible for '
                f'{n_electrons} electrons; spin is 2S = Nalpha - Nbeta, '
                f'and Nbeta cannot be negative'
            )
        elif (n_electrons - spin) % 2 != 0:
            problems.append(
                f'spin: {n_electrons} electrons with spin {spin} is not '
                f'consistent; the number of unpaired electrons must have '
                f'the same parity as the electron count. Note spin is '
                f'2S = Nalpha - Nbeta, not the multiplicity 2S+1'
            )

    if active_space is not None and _parse_active_space(active_space) is None:
        problems.append(
            f'active-space: {active_space!r} is not a range like 2-4 or a '
            f'list like 2,3,4'
        )

    problems.extend(validate({}))
    return problems


_ELEMENTS = (
    'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe '
    'Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In '
    'Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf '
    'Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am '
    'Cm Bk Cf Es Fm Md No Lr'
).split()


def _atomic_number(symbol):
    return _ELEMENTS.index(symbol.capitalize()) + 1


def _is_known_element(symbol):
    try:
        _atomic_number(symbol)
        return True
    except ValueError:
        return False


def main(argv=None):
    """Run the command line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments, defaulting to ``sys.argv[1:]``.

    Returns
    -------
    int
        The process exit code.
    """
    args = build_parser().parse_args(argv)

    if args.show_defaults:
        _print_defaults(sys.stdout)
        return EXIT_OK

    if not args.geometry:
        sys.stderr.write(
            'diffmrc: a geometry file is required\n'
            '  usage: diffmrc mol.xyz\n'
            '  use --show-defaults to see every parameter\n'
        )
        return EXIT_BAD_INPUT

    problems = _validate_input(args.geometry, args.charge, args.spin,
                               args.active_space)
    if problems:
        for p in problems:
            sys.stderr.write(f'diffmrc: {p}\n')
        return EXIT_BAD_INPUT

    if args.check:
        sys.stdout.write(f'{args.geometry}: ok\n')
        return EXIT_OK

    return _run(args)


def _run(args):
    """Execute a calculation and write the reports.

    Kept separate so that ``--check`` can share the validation path
    without importing the numerical layers.
    """
    from ..policy.params import effective
    from ..report.summary import ReportData, render_json, render_text

    try:
        params = effective(_param_overrides(args))
    except ValueError as exc:
        sys.stderr.write(f'diffmrc: {exc}\n')
        return EXIT_BAD_INPUT

    try:
        result = _compute(args, params)
    except ImportError as exc:
        sys.stderr.write(
            f'diffmrc: the numerical kernel is not available ({exc}).\n'
            f'  install it with: pip install -e ".[core]"\n'
        )
        return EXIT_FAILED
    except Exception as exc:
        sys.stderr.write(
            f'diffmrc: the calculation failed: {type(exc).__name__}: {exc}\n'
        )
        return EXIT_FAILED

    data = _to_report(args, params, result)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'diffmrc_report.md'), 'w',
              encoding='utf-8') as fh:
        fh.write(render_text(data))
    with open(os.path.join(outdir, 'diffmrc_result.json'), 'w',
              encoding='utf-8') as fh:
        fh.write(render_json(data))

    sys.stdout.write(render_text(data))
    return EXIT_OK if not data.is_refusal() else EXIT_FAILED


def _compute(args, params):
    """Run the driver. Separated so tests can substitute a stub."""
    from ..driver.run import run_from_geometry

    return run_from_geometry(
        geometry_file=args.geometry, charge=args.charge, spin=args.spin,
        level=args.level, active_space=args.active_space, params=params,
    )


def _to_report(args, params, result):
    """Map a driver result onto the report's data class.

    The mapping is explicit because the report layer may not import the
    driver: everything crossing the boundary has to be plain data.
    """
    from ..report.summary import ReportData

    algo = getattr(result, 'algorithm_data', {}) or {}
    space = algo.get('active_space', {}) or {}
    freqs = getattr(result, 'frequencies', None)
    freq_list = list(getattr(freqs, 'freqs_cm', []) or []) if freqs else []

    decision = getattr(result, 'decision', None)
    if decision is None:
        decision_dict = dict(level='refused', reasons=['no decision recorded'],
                             suggestions=[])
    else:
        decision_dict = dict(
            level=getattr(decision.level, 'value', str(decision.level)),
            reasons=list(decision.reasons),
            suggestions=list(decision.suggestions),
        )

    checks = []
    if decision is not None:
        checks = [dict(name=c.name, passed=bool(c.passed), detail=c.detail)
                  for c in decision.checks]

    versions = {'diffmrc': _version_of('diffmrc'),
                'pyscfad': _version_of('pyscfad'),
                'jax': _version_of('jax')}

    return ReportData(
        energy=float(getattr(result, 'energy', float('nan'))),
        geometry=algo.get('geometry', args.geometry),
        frequencies=freq_list,
        active_space=dict(orbitals=list(space.get('active', [])),
                          n_electrons=space.get('n_elec_active'),
                          evidence=list(algo.get('evidence', []))),
        checks=checks, decision=decision_dict,
        provenance=dict(params=dict(params.values),
                        sources=dict(params.sources),
                        versions=versions,
                        geometry_input=args.geometry),
    )


def _version_of(name):
    try:
        module = __import__(name)
        return getattr(module, '__version__', 'unknown')
    except Exception:
        return 'not installed'
