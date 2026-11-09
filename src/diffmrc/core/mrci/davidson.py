"""Davidson a posteriori correction.

PURPOSE
-------
Applies the Langhoff-Davidson size-consistency correction to an MRCI
energy.

ORIGINAL DEFINITION
-------------------
Langhoff & Davidson, Int. J. Quantum Chem. 1974, 8, 61, Eq. (2):

    Delta E_quad = (1 - C0^2) Delta E_DE

For MRCI the single reference coefficient C0^2 is replaced by the summed
weight of the reference (internal) configurations and Delta E_DE by the
correlation energy obtained at the MRCI level.  That replacement is a direct
generalization of Eq. (2); no single numbered equation in the retrieved
sources states it, so the generalization is documented here rather than
attributed.

VALIDATED SCOPE
---------------
Used in ``tests/core/test_mrci.py`` for the CH2/6-31G and the two
non-interacting-fragment benchmarks; the fragment test shows that the
correction reduces the He...He deviation from 1.07e-4 to 2.02e-6 Eh and
over-corrects H2...H2 to -5.28e-5 Eh.

BOUNDARIES
----------
* This is a single-reference correction generalized by reference weight;
  it is not the DS77 renormalized variant and has no multi-reference
  weighting.
* The corrected energy is not variational.  It is an error indicator, not
  a quantitative result, particularly for dissociating or strongly
  multi-reference states.
"""

from __future__ import annotations


def davidson_correction(e_mrci: float, e_ref: float, ref_weight: float) -> float:
    """Return the Davidson correction (LD74 Eq. 2 generalized to MRCI).

    Parameters
    ----------
    e_mrci : float
        MRCI total energy.
    e_ref : float
        Reference (CASCI/CASSCF) total energy.
    ref_weight : float
        Sum of squared coefficients of the reference configurations in the
        normalized MRCI vector.
    """
    return (1.0 - ref_weight) * (e_mrci - e_ref)
