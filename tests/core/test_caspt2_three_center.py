"""Three-center contracted CASPT2: parity with the materialized path.

The three-center routine replaces only the coupling assembly (W from a
three-center factor ``U`` with ``eri = U U^T``); classes, overlaps,
denominators and the V0_ia block solve are shared with
``caspt2_contracted_e2``.  These tests pin that the two paths agree on
the same reference and that the VM component keeps the relative weight
-1 (not ``-sqrt(3)``).
"""
import ast
from pathlib import Path

import numpy as np
import pytest

from diffmrc.core.caspt2 import caspt2_contracted_e2
from diffmrc.core.caspt2_three_center import (
    _w_three_center,
    caspt2_contracted_e2_three_center,
)
from tests.core.test_caspt2 import _cas_dets, _double_class_reference


def _three_center_case(seed=11, ncore=2, nact=2, nele=2, nmo=6, naux=9):
    """A deterministic pseudo-system with an exact three-center factor.

    ``u`` is symmetrized in its first two indices so that
    ``eri[a, b, c, d] = sum_P u[a, b, P] u[c, d, P]`` has the full
    chemist symmetry.  No physical meaning; the two paths consume the
    same tensor, so the comparison is exact up to roundoff.
    """
    rng = np.random.default_rng(seed)
    h1 = rng.normal(size=(nmo, nmo))
    h1 = 0.5 * (h1 + h1.T)
    w = rng.normal(size=(nmo, nmo, naux))
    u = 0.5 * (w + w.transpose(1, 0, 2))
    uf = u.reshape(nmo * nmo, naux)
    eri = (uf @ uf.T).reshape(nmo, nmo, nmo, nmo)
    # Well-separated positive orbital energies keep the denominator
    # guards inactive, as in the materialized-path tests.
    eps = np.linspace(1.0, 2.0, nmo)
    dets = _cas_dets(ncore, nact, nele)
    amask = sum(1 << so for so in range(0, 2 * (ncore + nact), 2))
    ci = np.zeros(len(dets))
    for i, d in enumerate(dets):
        if bin(d & amask).count('1') == (2 * ncore + nele) // 2:
            ci[i] = rng.normal()
    ci = ci / np.linalg.norm(ci)
    return h1, u, eri, eps, dets, ci


def test_three_center_matches_materialized_classes():
    """Same reference, same operator content: every class and the total
    agree with the materialized path to roundoff."""
    h1, u, eri, eps, dets, ci = _three_center_case()
    e_3c, bd_3c = caspt2_contracted_e2_three_center(
        h1, u, ci, dets, ncore=2, nact=2, nele=2, eps=eps)
    e_mat, bd_mat = caspt2_contracted_e2(
        h1, eri, ci, dets, ncore=2, nact=2, nele=2, eps=eps)

    assert e_3c == pytest.approx(e_mat, abs=1e-10)
    for name in bd_mat:
        assert bd_3c[name] == pytest.approx(bd_mat[name], abs=1e-10), name
    assert sum(bd_3c.values()) == pytest.approx(e_3c, abs=1e-10)


def test_three_center_coupling_matches_dense_terms():
    """The term scatter reproduces ``sum_D K[D] g(D)`` built from the
    Slater-Condon kernel and the dense four-center tensor."""
    from diffmrc.core.caspt2 import _sc_kernel

    h1, u, eri, eps, dets, ci = _three_center_case(seed=5)
    ref = [(int(d), float(c)) for d, c in zip(dets, ci) if abs(c) > 1e-14]
    kacc = {int(dets[0]): 0.37, int(dets[1]): -0.62}
    w_3c = _w_three_center(kacc, h1, u, {}, ref)

    h_elem, _ = _sc_kernel(h1, eri)
    w_dense = sum(kc * sum(cJ * h_elem(J, dd) for J, cJ in ref)
                  for dd, kc in kacc.items())
    assert w_3c == pytest.approx(w_dense, abs=1e-10)


def test_three_center_vm_relative_weight_is_minus_one():
    """The VM component keeps ``cross - direct`` (relative weight -1).

    The synthetic case has no core orbitals, so ``Vm2_ab`` carries both
    orderings and the two candidate weights give visibly different
    values (0.21 Eh here).  The independent reconstruction fixes the
    expected number, so a reintroduced ``-sqrt(3)`` fails this test even
    if both paths were changed together.
    """
    h1, u, eri, eps, dets, ci = _three_center_case(
        seed=7, ncore=0, nact=2, nele=2, nmo=4, naux=6)
    qs_list = [(0, 0), (0, 1)]
    ps_list = [(2, 2), (2, 3), (3, 3)]

    e_minus_one = _double_class_reference(-1.0, h1, eri, eps, dets, ci,
                                          qs_list, ps_list)
    e_minus_sqrt3 = _double_class_reference(-np.sqrt(3.0), h1, eri, eps,
                                            dets, ci, qs_list, ps_list)
    assert abs(e_minus_one - e_minus_sqrt3) > 1e-6

    _, bd = caspt2_contracted_e2_three_center(
        h1, u, ci, dets, ncore=0, nact=2, nele=2, eps=eps)
    assert bd['Vm2_ab'] == pytest.approx(e_minus_one, abs=1e-10)


def test_three_center_module_has_no_sqrt3_weight():
    """Guard the source against a literal ``sqrt(3)`` weight.

    The docstring may mention the OpenMolcas overall scale; only a call
    to ``sqrt`` with the constant 3 is rejected.
    """
    import diffmrc.core.caspt2_three_center as mod

    tree = ast.parse(Path(mod.__file__).read_text(encoding='utf-8'))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value,
                                                          ast.Name):
            name = f'{func.value.id}.{func.attr}'
        elif isinstance(func, ast.Name):
            name = func.id
        else:
            continue
        if name not in ('np.sqrt', 'numpy.sqrt', 'math.sqrt'):
            continue
        if (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == 3):
            offenders.append(node.lineno)
    assert not offenders, f'sqrt(3) weight at lines {offenders}'
