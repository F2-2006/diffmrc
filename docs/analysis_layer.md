# The analysis layer

Wavefunction analysis lives apart from the numerical kernel so that it can
be tested on hand-constructed numbers. Nothing here knows how a
wavefunction was obtained: every function takes numpy arrays and returns
numpy arrays or scalars.

## Why the separation matters

The kernel is validated against reference energies, which requires a full
electronic-structure calculation to test anything. Functions in this layer
have analytic limits -- an empty orbital has zero entropy, four equally
weighted occupations give exactly ln 4 -- so they can be checked in
milliseconds and their failure modes isolated.

The import contract enforces this: `diffmrc.analysis` may not import
`diffmrc.core`, or any higher layer. Run `lint-imports` to check.

## What is here

| Module | Purpose |
|---|---|
| `entropy.py` | Single-orbital entanglement entropy from occupation numbers |
| `rdm.py` | Extract those occupation numbers from reduced density matrices |
| `plateau.py` | Locate the gap between strongly and weakly correlated orbitals |
| `diagnostics.py` | Zs(1) and the single-reference test |
| `composition.py` | How much of an element and shell each orbital carries |

## The spin-summed two-particle density

`rdm.py` is the one module where a convention error is easy to make and
invisible afterwards. The kernel and PySCF both store the two-particle
density matrix **spin-summed**, so the same-orbital element counts both
orderings of the pair:

    rdm2[i, i, i, i] = 2 * <n_{i,alpha} n_{i,beta}>

The entropy formula needs `<n_a n_b>`, so `entropy_from_rdms` halves that
diagonal by default. Skipping the halving gives a doubly occupied orbital
`p2 = 2`, which is not a probability; the weight vector then collapses to a
half-and-half mixture of the empty and doubly occupied states and reports
ln 2 for an orbital that is a closed shell. The test
`test_halving_is_what_keeps_a_doubly_occupied_orbital_unentangled` pins
this down.

Note that the one-particle matrices are the opposite case: they must be
**spin-resolved**, because the entropy needs the two spin occupations
separately. A calculation that returns a spin-resolved one-particle matrix
and a spin-summed two-particle matrix is the normal situation.

## The absolute floor before the plateau sweep

`detect_plateaus` normalizes the spectrum by its largest value before
sweeping the threshold. The normalization is what makes the staircase
comparable between systems, but it has a failure mode: a spectrum of pure
numerical noise acquires a relative structure once divided by its own
maximum. Ten orbitals at 1e-06 with three at 2e-06 normalize to a
staircase that holds flat over a wide range and reports a plateau,
inviting an active space to be selected from round-off.

The guard is the protocol's own criterion. Stein and Reiher assign
multiconfigurational character only if at least one single-orbital entropy
exceeds 0.14, so `detect_plateaus` returns no plateau when the largest
entropy falls below that value. Callers that want the unguarded sweep can
pass `min_entropy=0.0`; the normalized spectrum is returned either way, so
a caller can inspect what the guard suppressed.

Related, from the same protocol: when a spectrum shows *no* plateau, the
prescribed fallback is to exclude orbitals whose entropy is below one to
two percent of the maximum, rather than to report a plateau height anyway.
That fallback belongs to the selection policy, not to this function.

## A single determinant is a free regression test

Any single determinant -- closed shell, open shell, any filling -- has zero
entanglement on every orbital. That is an exact statement with no external
dependency, and it catches both the convention error above and any future
change to the entropy formula. Prefer it over comparisons against another
program, which can agree while both are wrong.

## Protocol provenance

The entropy and plateau protocols follow Stein and Reiher:

- C. J. Stein, M. Reiher, J. Chem. Theory Comput. 12, 1760 (2016)
- C. J. Stein, M. Reiher, J. Comput. Chem. 40, 2216 (2019)
- C. J. Stein, M. Reiher, Mol. Phys. 115, 2110 (2017)

The implementations here are written from the published method and operate
on this project's own wavefunctions.
