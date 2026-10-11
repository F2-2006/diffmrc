"""The native trace sink: kernel eigh calls recorded on request.

Added for the t397 lesson -- the cluster-member scan had to monkey-patch
this frozen module from outside, missed two other bindings of the same
functions, and silently recorded nothing (a 1.7-hour rerun to discover).
The sink lives inside the module instead, and the discipline test
(``test_kernel_binding.py``) keeps the bindings predictable.
"""
import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from diffmrc.core import eigh as eigh_mod


def test_trace_sink_records_a_call(tmp_path):
    path = tmp_path / "trace.jsonl"
    eigh_mod.set_trace_sink(str(path))
    try:
        a = jnp.diag(jnp.array([1.0, 2.0, 4.0]))
        eigh_mod.eigh_safe(a)
    finally:
        eigh_mod.set_trace_sink(None)
    rows = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]['solver'] == 'eigh_safe'
    assert rows[0]['n'] == 3
    assert abs(rows[0]['min_gap'] - 1.0) < 1e-12
    assert rows[0]['n_near'] == 0


def test_near_pair_is_counted(tmp_path):
    path = tmp_path / "trace.jsonl"
    eigh_mod.set_trace_sink(str(path))
    try:
        # a bit-exact pair: the t397 target case
        a = jnp.diag(jnp.array([1.0, 1.0 + 1e-30, 5.0]))
        eigh_mod.eigh_safe(a)
    finally:
        eigh_mod.set_trace_sink(None)
    rows = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert rows[0]['n_near'] == 1


def test_off_by_default_writes_nothing(tmp_path):
    eigh_mod.set_trace_sink(None)
    out = eigh_mod.eigh_safe(jnp.eye(2))
    w, _ = out
    assert np.isfinite(np.asarray(w)).all()
    assert not list(tmp_path.iterdir())      # nothing was written anywhere


def test_trace_does_not_alter_values(tmp_path):
    a = jnp.array([[2.0, 0.3], [0.3, 1.0]])
    w_plain, u_plain = eigh_mod.eigh_safe(a)
    eigh_mod.set_trace_sink(str(tmp_path / "t.jsonl"))
    try:
        w_traced, u_traced = eigh_mod.eigh_safe(a)
    finally:
        eigh_mod.set_trace_sink(None)
    assert np.array_equal(np.asarray(w_plain), np.asarray(w_traced))
    assert np.array_equal(np.asarray(u_plain), np.asarray(u_traced))


def test_cluster_ift_and_mask_are_covered(tmp_path):
    path = tmp_path / "trace.jsonl"
    eigh_mod.set_trace_sink(str(path))
    try:
        a = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        eigh_mod.eigh_cluster_ift(a)
        eigh_mod.eigh_mask(a)
    finally:
        eigh_mod.set_trace_sink(None)
    names = [json.loads(ln)['solver'] for ln in path.read_text().splitlines()]
    assert names == ['eigh_cluster_ift', 'eigh_mask']
