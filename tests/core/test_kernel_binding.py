"""Import discipline for kernel symbols: module-level imports only.

Background (t397): ``eigh_safe`` was reachable through bindings in three
different ways -- module-level in ``core.nevpt2.impl``, module-level in
``core.casci``, and function-scoped in ``driver.nevpt2``. A tracing tool
that patched one namespace therefore missed the others and silently
recorded nothing, which cost a full 1.7-hour rerun of a seven-geometry
scan to discover. The rule enforced here is the one that makes kernel
use statically auditable: the numerical kernel (``core.eigh`` and
future kernel modules) may only be imported at module level.

Heavy optional dependencies (pyscf, pyscfad, jax) are deliberately not
covered by this rule -- the deferred-import style for those is a
supported pattern (``run.py``'s docstring explains why).
"""
import ast
import pathlib

KERNEL_MODULES = (
    'diffmrc.core.eigh',
    'diffmrc.core.safe_math',  # added with the safe-primitives module
)
SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / 'src'
PKG_ROOT = SRC_ROOT / 'diffmrc'


def _resolve_from(path, node):
    """Absolute module name of an ImportFrom node, relative or absolute."""
    if node.level == 0:
        return node.module or ''
    parts = list(path.relative_to(SRC_ROOT).with_suffix('').parts)
    parts = parts[:-1]                      # drop the module itself
    up = node.level - 1
    base = parts[:len(parts) - up] if up else parts
    if node.module:
        base = base + node.module.split('.')
    return '.'.join(base)


def _kernel_names(path, node):
    """The kernel symbols this import statement pulls in (may be empty)."""
    if isinstance(node, ast.ImportFrom):
        if _resolve_from(path, node) in KERNEL_MODULES:
            return [alias.name for alias in node.names]
    elif isinstance(node, ast.Import):
        return [alias.name for alias in node.names
                if alias.name in KERNEL_MODULES]
    return []


def _function_scoped_imports(tree):
    """Every import statement that sits inside a function body."""
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = fn.end_lineno or fn.lineno
            for child in ast.walk(fn):
                if (isinstance(child, (ast.Import, ast.ImportFrom))
                        and fn.lineno <= child.lineno <= end):
                    yield child


def test_no_function_scoped_kernel_imports():
    offenders = []
    for path in sorted(PKG_ROOT.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in _function_scoped_imports(tree):
            names = _kernel_names(path, node)
            if names:
                offenders.append(
                    f'{path.relative_to(PKG_ROOT)}:{node.lineno} {names}')
    assert not offenders, (
        'kernel symbols may only be imported at module level; offenders:\n'
        + '\n'.join(offenders))
