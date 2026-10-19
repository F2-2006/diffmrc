"""t160: Collect frequency results from the experiment logs into a single summary.

The frequency study writes its JSON only at the end of the script, so a run
that is stopped partway leaves its finished results unrecorded. This utility
parses the logs instead, so that a partially completed run still yields whatever
it did produce.

Both the English output of the current scripts and the Chinese-language logs
written by earlier versions are recognized, since the parser is meant to work on
logs that already exist as well as on new ones.

Usage: python collect_frequency_results.py
"""
import os
import re
import json
import glob

RES = 'results'
OUT = 'results/t160_freq_summary.json'

EXP = {'H2O': [1594.7, 3656.7, 3755.8], 'CO': [2143.3], 'N2': [2330.0]}

# Plan labels as they appear in the logs, mapped to normalized keys. Each
# mapping lists the English and the Chinese spelling.
TAGS = [('A (theta frozen)', 'A_frozen'),
        ('A (frozen theta)', 'A_frozen'),
        ('frozen expansion N=2', 'unrolled_N2'),
        ('unrolled N=2', 'unrolled_N2'),
        ('frozen expansion N=3', 'unrolled_N3'),
        ('unrolled N=3', 'unrolled_N3'),
        ('full Newton N=3', 'newton_N3'),
        ('full Newton N=3', 'newton_N3'),
        ('implicit', 'implicit'),
        ('implicit', 'implicit')]

pat_freq = re.compile(r'frequencies \[cm-1\] = \[([^\]]+)\]')
pat_imag = re.compile(r'imaginary(?: modes)? = (\d+)')
TITLE_MARK = ('frequencies',)

summary = {}
for sysname in ('H2O', 'CO', 'N2'):
    files = sorted(glob.glob(f'{RES}/t153_{sysname.lower()}_*.log'))
    if not files:
        continue
    latest = files[-1]                       # most recent run
    txt = open(latest, encoding='utf-8', errors='ignore').read()
    lines = txt.splitlines()
    entry = dict(log=os.path.basename(latest), plans={})
    cur = None
    for line in lines:
        for tname, tkey in TAGS:
            if tname in line and not any(m in line for m in TITLE_MARK):
                # Plan header line, of the form "  A (theta frozen)   (65s)"
                if re.search(r'\(\d+s\)', line) or line.strip().endswith(tname):
                    cur = tkey
        m = pat_freq.search(line)
        if m and cur:
            vals = [float(x.strip().strip("'\""))
                    for x in m.group(1).split(',')]
            entry['plans'][cur] = vals
            cur = None
        mi = pat_imag.search(line)
        if mi and entry['plans']:
            k = list(entry['plans'])[-1]
            entry['plans'][k + '_imag'] = int(mi.group(1))
    # Deviation from the experimental fundamentals
    if sysname in EXP and entry['plans']:
        exp = EXP[sysname]
        for k in [x for x in entry['plans'] if not x.endswith('_imag')]:
            v = entry['plans'][k]
            if len(v) == len(exp):
                entry[f'dev_{k}'] = float(
                    sum(abs(a - b) for a, b in zip(sorted(v), sorted(exp))))
        entry['exp'] = exp
    summary[sysname] = entry

json.dump(summary, open(OUT, 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

print('=' * 90)
print('Frequency summary, parsed from the frequency-study logs')
print('=' * 90)
for s, e in summary.items():
    print(f'\n  {s}  ({e["log"]})')
    exp = e.get('exp')
    if exp:
        print(f'    experiment   = {exp}')
    for k, v in e['plans'].items():
        if k.endswith('_imag'):
            continue
        dev = e.get(f'dev_{k}')
        devs = f'   sum of deviations = {dev:.1f}' if dev is not None else ''
        print(f'    {k:<14} = {[f"{x:.1f}" for x in v]}{devs}')
    print('    imaginary: ' + ', '.join(
        f'{k[:-5]}={v}' for k, v in e['plans'].items() if k.endswith('_imag')))

print(f'\n>>> {OUT}')
print('\nt160_DONE')
