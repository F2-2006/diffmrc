"""Chemical rules and active-space selection.

This layer answers one question: which orbitals belong in the active
space, and how confident are we? It reads the occupation numbers and the
orbital composition that a calculation produced and returns a decision
with its evidence attached. It does not run any electronic-structure
calculation itself, so its rules can be tested on hand-built vectors.

Selection follows a two-source protocol: a rule-based candidate from
chemical structure, and an entanglement-based verification from the
wavefunction. Agreement of the two licenses an automatic answer.
Disagreement is reported rather than resolved silently, and every rule is
allowed to refuse.

When the rules refuse outright -- a composition outside their scope, a
mixed orbital they will not label -- ``entropy_selection`` builds the
candidate from the wavefunction's own spectrum (the autoCAS plateau
step, or a growth along the mutual-information matrix), so the layer
covers the decision in both directions: from structure to a candidate,
and from the spectrum to a candidate.
"""
