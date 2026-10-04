"""Render a finished run for a person and for a machine.

Two audiences, one source. The text report is ordered so that the
verdict comes first: a reader must not have to reach the end to learn
that the number above is not to be trusted. The JSON report carries the
same content so a later script can read it back.

Every self-check is listed with its measured value, its threshold and a
yes/no. Collapsing four criteria into one flag loses the information a
reader needs to judge how close a run came to failing, which is the
difference between a result to trust and one to re-run.
"""
import json
from dataclasses import dataclass, field

__all__ = ['ReportData', 'render_text', 'render_json']

_REFUSED = 'refused'


@dataclass
class ReportData:
    """Everything a report needs, as plain data.

    No field may hold an object from another layer: the report is
    rendered from this and nothing else.
    """

    energy: float = float('nan')
    geometry: str = ''
    frequencies: list = field(default_factory=list)
    active_space: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)
    decision: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    on_top: dict = field(default_factory=dict)

    def is_refusal(self):
        return str(self.decision.get('level', '')).lower() == _REFUSED


def _fmt(value, spec='.6f'):
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _render_verdict(data):
    lines = []
    if data.is_refusal():
        lines.append('=' * 72)
        lines.append('REFUSED  --  no trustworthy result is reported below')
        lines.append('=' * 72)
        for reason in data.decision.get('reasons', []):
            lines.append(f'  blocked by: {reason}')
        suggestions = data.decision.get('suggestions', [])
        if suggestions:
            lines.append('')
            lines.append('  what to try:')
            for s in suggestions:
                lines.append(f'    - {s}')
        lines.append('')
    else:
        level = data.decision.get('level', 'unknown')
        lines.append('=' * 72)
        lines.append(f'diffmrc result  --  level: {level}')
        lines.append('=' * 72)
    return lines


def _render_result(data):
    lines = ['', 'Result', '-' * 72]
    lines.append(f'  energy          {_fmt(data.energy)} Eh')
    if data.geometry:
        lines.append(f'  geometry        {data.geometry}')
    freqs = list(data.frequencies or [])
    if not freqs:
        lines.append('  frequencies     not computed')
    else:
        n_imag = sum(1 for f in freqs if float(f) < 0)
        shown = ', '.join(_fmt(f, '.2f') for f in freqs)
        lines.append(f'  frequencies     [{shown}] cm^-1')
        if n_imag:
            lines.append(f'                  {n_imag} imaginary '
                         f'(the structure is not a minimum)')
    return lines


def _render_active_space(data):
    lines = ['', 'Active space', '-' * 72]
    space = data.active_space or {}
    orbitals = space.get('orbitals', [])
    if not orbitals:
        lines.append('  not selected')
        return lines
    lines.append(f'  orbitals        {list(orbitals)}')
    lines.append(f'  electrons       {space.get("n_electrons", "?")}')
    evidence = space.get('evidence', [])
    if evidence:
        lines.append('  why:')
        for item in evidence:
            lines.append(f'    - {item}')
    return lines


def _render_checks(data):
    lines = ['', 'Self-checks', '-' * 72]
    checks = list(data.checks or [])
    if not checks:
        lines.append('  no self-checks were run')
        return lines
    for c in checks:
        mark = 'YES' if c.get('passed') else 'NO '
        lines.append(f'  [{mark}] {c.get("name", "?")}')
        detail = c.get('detail', '')
        if detail:
            lines.append(f'          {detail}')
    return lines


def _render_params(data):
    lines = ['', 'Parameters', '-' * 72]
    prov = data.provenance or {}
    params = prov.get('params', {})
    sources = prov.get('sources', {})
    if not params:
        lines.append('  none recorded')
        return lines
    width = max(len(str(k)) for k in params)
    for key in sorted(params):
        src = sources.get(key, '?')
        lines.append(f'  {str(key):<{width}}  {params[key]!r:<18}  ({src})')
    return lines


def _render_provenance(data):
    lines = ['', 'Reproduce', '-' * 72]
    prov = data.provenance or {}
    versions = prov.get('versions', {})
    for name in sorted(versions):
        lines.append(f'  {name:<16}{versions[name]}')
    if prov.get('geometry_input'):
        lines.append(f'  {"input":<16}{prov["geometry_input"]}')
    return lines


def _render_on_top(data):
    if not data.on_top:
        return []
    o = data.on_top
    lines = ['on-top decomposition:',
             '  method            {}'.format(o.get('method')),
             '  protocol          {}'.format(o.get('protocol')),
             '  E_mcwfn / Eh      {}'.format(_fmt(o.get('e_mcwfn'),
                                                 '+.10f')),
             '  E_ot    / Eh      {}'.format(_fmt(o.get('e_ot'), '+.10f')),
             '  E_CASSCF / Eh     {}'.format(_fmt(o.get('e_casscf'),
                                                 '+.10f'))]
    if o.get('hyb'):
        lines.append('  hybrid lambda     {}'.format(o.get('hyb')))
    lines.append('')
    return lines


def render_text(data):
    """Render a run as a human-readable report.

    Parameters
    ----------
    data : ReportData

    Returns
    -------
    str
    """
    lines = []
    lines.extend(_render_verdict(data))
    if not data.is_refusal():
        lines.extend(_render_result(data))
    lines.extend(_render_on_top(data))
    lines.extend(_render_active_space(data))
    lines.extend(_render_checks(data))
    lines.extend(_render_params(data))
    lines.extend(_render_provenance(data))
    lines.append('')
    return '\n'.join(lines)


def render_json(data, indent=1):
    """Render the same content as machine-readable JSON.

    Parameters
    ----------
    data : ReportData
    indent : int
        Indentation passed to ``json.dumps``.

    Returns
    -------
    str
    """
    payload = dict(
        energy=data.energy,
        geometry=data.geometry,
        frequencies=list(data.frequencies or []),
        active_space=dict(data.active_space or {}),
        checks=[dict(c) for c in (data.checks or [])],
        decision=dict(data.decision or {}),
        provenance=dict(data.provenance or {}),
        on_top=dict(data.on_top or {}),
    )
    return json.dumps(payload, indent=indent, ensure_ascii=False, default=str)
