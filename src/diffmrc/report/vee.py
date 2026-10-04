"""Render a vertical-excitation table for a person and for a machine.

The companion of ``summary.py`` for the state-averaged path.  The
grouped states come first, with their excitation energies and the
active window that produced them, followed by the frontier report that
suggested the window -- the suggestion is a suggestion, and a reader
must be able to check it rather than trust it.

Excitation energies are in eV above the reference (the lowest grouped
state) -- the report contract is eV, and the caller converts from the
tables' Hartree at the boundary.  The columns are the CASSCF and the
per-state NEVPT2 vertical energies, plus the one-step MC-PDFT column
when the caller supplies it (``dE_mcpdft`` on the states).
"""
import json
from dataclasses import dataclass, field

__all__ = ['VeeReportData', 'render_vee_text', 'render_vee_json']


@dataclass
class VeeReportData:
    """Everything a VEE report needs, as plain data.

    No field may hold an object from another layer: the report is
    rendered from this and nothing else.
    """

    geometry: str = ''
    family: str = ''
    window: dict = field(default_factory=dict)
    grad_norm: float = float('nan')
    states: list = field(default_factory=list)
    frontier: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


def _fmt(value, spec='.6f'):
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _render_header(data):
    w = data.window or {}
    return [
        f'vertical excitations ({data.family})   {data.geometry}',
        'active window: ncore={}  nact={}  nele={}    '
        'orbital-gradient norm: {}'.format(
            w.get('ncore'), w.get('nact'), w.get('nele'),
            _fmt(data.grad_norm, '.2e')),
        '',
    ]


def _render_states(data):
    has_mcpdft = any(s.get('dE_mcpdft') is not None for s in data.states)
    lines = [
        ' state  multiplicity   dE(CASSCF)/eV   dE(NEVPT2)/eV   '
        + ('dE(MC-PDFT)/eV   ' if has_mcpdft else '') + 'Sz components',
        '------  ------------   --------------   --------------   '
        + ('----------------   ' if has_mcpdft else '') + '-------------',
    ]
    for s in data.states:
        row = '{:>6}  {:>12}   {:>14}   {:>14}   '.format(
            s.get('index', ''),
            s.get('multiplicity', ''),
            _fmt(s.get('dE_casscf'), '+.6f'),
            _fmt(s.get('dE_nevpt2'), '+.6f'))
        if has_mcpdft:
            row += '{:>16}   '.format(_fmt(s.get('dE_mcpdft'), '+.6f'))
        row += '{:>13}'.format(','.join(str(x) for x in s.get('szs', []))
                               or '-')
        lines.append(row)
    lines.append('')
    return lines


def _render_frontier(data):
    f = data.frontier or {}
    lines = ['frontier identity (what suggested the window):']
    for key in ('homo_minus_1', 'homo', 'lumo'):
        ident = f.get(key) or {}
        lines.append('  {:12s} kind={:5s} axis={} centre={}'.format(
            key + ':', str(ident.get('kind')), ident.get('axis'),
            ident.get('centre')))
    lines.append('  cross_component={}   lone_below_homo={}'.format(
        f.get('cross_component'), f.get('lone_below_homo')))
    lines.append('  occupied_degenerate_pairs={}'.format(
        f.get('occupied_degenerate_pairs')))
    lines.append('')
    lines.append('The window is a suggestion from these identities, not a '
                 'decision: check the frontier line above before quoting '
                 'the table.')
    lines.append('')
    return lines


def render_vee_text(data):
    """Render a vertical-excitation run as a human-readable report.

    Parameters
    ----------
    data : VeeReportData

    Returns
    -------
    str
    """
    lines = []
    lines.extend(_render_header(data))
    lines.extend(_render_states(data))
    lines.extend(_render_frontier(data))
    return '\n'.join(lines)


def render_vee_json(data, indent=1):
    """Render the same content as machine-readable JSON.

    Parameters
    ----------
    data : VeeReportData
    indent : int
        Indentation passed to ``json.dumps``.

    Returns
    -------
    str
    """
    payload = dict(
        geometry=data.geometry,
        family=data.family,
        window=dict(data.window or {}),
        grad_norm=data.grad_norm,
        states=[dict(s) for s in (data.states or [])],
        frontier=dict(data.frontier or {}),
        provenance=dict(data.provenance or {}),
    )
    return json.dumps(payload, indent=indent, ensure_ascii=False, default=str)
