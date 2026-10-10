# Developer notes

Four conventions of this codebase that were previously implicit, written
down after the 2026-10 P1 hardening batch. Each one exists because its
absence cost a real incident; the tests that enforce them are named.

## 1. Kernel symbols are imported at module level

The numerical kernel (`diffmrc.core.eigh`, `diffmrc.core.safe_math`)
must be imported at module top level, never inside a function. Heavy
optional dependencies (pyscf, pyscfad, jax) keep the opposite style --
deferred imports so `--check` works without the kernel installed; the
rule covers the kernel only.

Why: the t397 trace tool patched `core.nevpt2.impl`'s namespace and
silently recorded nothing, because the same functions were reachable
through a function-scoped import in `driver.nevpt2`. The seven-geometry
rerun it caused is in the commit log. With module-level imports only,
`grep` finds every kernel use.

Enforced by `tests/core/test_kernel_binding.py` (AST scan; offenders
are listed with file:line).

## 2. Guard the reverse pass, not just the value

Prefer `diffmrc.core.safe_math` primitives (`positive_part`,
`safe_divide`, `guarded_ratio`) over open-coded `jnp.maximum` / raw
division / two-step where-floor patterns in new code. Existing call
sites migrate gradually; no numeric path may change silently.

Why: three incidents share one mechanism -- a harmless singularity in
the values becomes NaN in the derivatives: negative noise feeding a
fractional power (t384d), 0/0 in the *untaken* branch of a `where`
reaching the reverse pass as 0 * inf (commit 5e640c9), roundoff
eigenvalue gaps in the raw eigh rule (`core/eigh.py` docstring). The
primitives' docstrings state the value and gradient semantics (the
pyscfad `safe_sqrt` convention).

Enforced by `tests/core/test_safe_math.py`.

## 3. Trace the kernel, do not patch it

To record kernel eigh calls (cluster members, minimum gaps), use

```python
from diffmrc.core.eigh import set_trace_sink
set_trace_sink('/path/trace.jsonl')   # or env DIFFMRC_EIGH_TRACE
```

The sink is diagnostics only: it never alters a return value, swallows
its own failures, and skips tracing contexts. Do not monkey-patch the
kernel from outside for diagnosis -- that is the pattern that failed in
t397 (see note 1). The sink's records carry `solver`, `n`, `min_gap`,
`n_near` per call.

Enforced by `tests/core/test_eigh_trace.py`.

## 4. Methods are declared once, in `policy.methods`

A new reported-energy method is one entry in `METHODS`
(`MethodSpec`: `hyb`, `on_top`, `ground_protocol`, `vee_protocol`),
plus the factories the driver needs. The parameter choices, the ground
state dispatch (`driver.kernel`), the vertical-excitation dispatch
(`driver.run`) and the report fields (`cli.main`) all read this table;
none of them carries its own literal `if method == ...` any more.

Why the table lives in the policy layer: `policy.params` declares the
choice list and the import contract forbids policy -> driver; the
dispatch *behaviour* stays in the layer that owns it and reads the
data. Adding a method with new behaviour (a new factory) is a driver
change; adding a method that reuses an existing factory shape is one
table entry.

Enforced by `tests/policy/test_method_registry.py` (the texts are
asserted character-for-character against the pre-registry literals --
the refactor must not change a single rendered character).
