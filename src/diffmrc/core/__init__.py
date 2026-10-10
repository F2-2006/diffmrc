"""Numerical kernel: differentiable CI, CASSCF, gradients, and NEVPT2.

This layer is frozen. It is validated against reference values and must not
be modified except for import-path changes during packaging, and for
diagnostic hooks that provably do not alter any return value or derivative
(see ``core.eigh.set_trace_sink`` for the pattern and its tests).
"""
