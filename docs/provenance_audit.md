# Provenance audit of the kernel layer (2026-10-10)

Scope: every file under `src/diffmrc/core/` (18 files) plus a targeted
scan of `driver/` and `src/diffmrc/` at large. Question asked of each
file: does it contain third-party code (copied source), or is the
relationship to upstream projects (PySCF, pyscfad, OpenMolcas, libxc)
one of interface use, convention alignment, or semantic
reimplementation?

Method: (1) feature-word scan (`adapted`, `taken`, `borrowed`,
`derived`, `copy`, `verbatim`); (2) for every mention of PySCF/Molcas/
ORCA in the kernel, read the surrounding statement and classify it
(interface call vs convention note vs provenance claim); (3) spot-read
files whose docstrings declare a reference relationship.

## Per-file conclusions

| File | Relation to upstream | Verdict |
|---|---|---|
| `casci.py` | calls pyscfad (`import scf`); CI machinery (cs_casci / cs_casci_blocked) is own work | no third-party code |
| `caspt2.py` | zeroth-order conventions follow OpenMolcas (declared in-file, `caspt2.py:225-234`: "established by line-by-line source comparison; OpenMolcas v26.06 src/caspt2"); implementation is a distinct full-CI-space route | convention alignment, declared |
| `eigh.py` | own derivative rules (Lorentzian soft rule, cluster IFT); pyscfad's `eigh_gen` is cited as an upstream reference for the same problem, not used | no third-party code |
| `grad.py` | pyscfad injected as a callable to avoid a hard dependency; no algorithm taken | no third-party code |
| `hessian.py` | same injection pattern as `grad.py` | no third-party code |
| `lanczos.py` | own bordered-IFT Lanczos kernel | no third-party code |
| `mcpdft.py` | translation convention follows Li Manni et al. 2014 and the pyscf engine (validated against it); the jnp PBE kernel is own, validated pointwise against libxc | convention alignment, declared |
| `optimizer.py`, `orblib.py`, `rotation.py`, `stationary.py`, `safe_math.py` | own work | no third-party code |
| `nevpt2/core.py` | transposition tables annotated against PySCF's internal conventions (comments of the form `（PySCF: ppaa[act,core].T(0,2,1,3)）`) | convention notes |
| `nevpt2/f3ca.py` | semantic reimplementation of PySCF's C kernels (`libmc`/`libfci`), reverse-engineered by direct ctypes calls plus least-squares layout fitting, checked element-by-element to machine precision | reimplementation, declared (statement tightened in this audit) |
| `nevpt2/impl.py` | validated against the PySCF reference numerically ("reproduces the reference to 0.00") | numerical target, declared |
| `nevpt2/rdm.py` | "spin-free and in the norm order of PySCF fci.rdm" | convention alignment, declared |
| `nevpt2/__init__.py`, `core/__init__.py` | package docstrings only | n/a |

## Findings and dispositions

1. **`core/lanczos.py:112`** cited an internal project memory name
   (`memory \`where-untaken-branch-nan\``) -- meaningless to outside
   readers and a workflow leak. Fixed: the comment now states the rule
   itself (0/0 in the untaken branch reaches the reverse pass as
   0 * inf).
2. **`core/nevpt2/f3ca.py`** referred to "the reference C implementation"
   without naming it. Fixed: the docstring now names PySCF's
   `libmc`/`libfci` kernels and states explicitly that this is a
   semantic reimplementation, not a copy of their sources.
3. No file was found to contain copied third-party source. All
   PySCF/OpenMolcas relationships are interface calls, convention
   alignments, numerical targets, or declared reimplementations.
4. Recorded, not changed: internal experiment tags (`t237`, `t384d`,
   ...) appear in kernel docstrings. These are research-phase
   experiment numbers; the README's "early development" status note
   and the `experiments/` index cover them by design.

## Statement for the release notes

The kernel was written against PySCF/pyscfad as a library (interface
use). Where it follows upstream conventions (CASPT2 zeroth order, MC-PDFT
translation, NEVPT2 RDM ordering) or reimplements upstream semantics
(the f3ca contraction), the relationship is declared in the source and
validated numerically against the upstream implementation; no upstream
source code is copied. The project is MIT-licensed and depends on
pyscf/pyscfad (Apache-2.0) at runtime.
