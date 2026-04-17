"""Static AST guard: confirm no CPU-binding third-party imports in src/torch_dfo/.

Walks every .py file under src/torch_dfo/ and asserts that no Import or
ImportFrom node references a CPU-bound library (numpy, scipy, pandas, etc.)
at any scope — including inside TYPE_CHECKING blocks and lazy-import branches.

This complements the existing runtime check in test_phased.py and catches
violations before the code is ever executed.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src" / "torch_dfo"

_FORBIDDEN = frozenset(
    {
        "numpy",
        "scipy",
        "pandas",
        "numba",
        "cython",
        "jax",
        "sklearn",
        "joblib",
    }
)


def _root_package(name: str) -> str:
    return name.split(".")[0]


def _forbidden_imports(tree: ast.AST) -> list[tuple[int, str]]:
    """Return list of (lineno, module_name) for any forbidden import."""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits.extend(
                (node.lineno, alias.name)
                for alias in node.names
                if _root_package(alias.name) in _FORBIDDEN
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and _root_package(node.module) in _FORBIDDEN
        ):
            hits.append((node.lineno, node.module))
    return hits


def test_no_cpu_binding_imports() -> None:
    """No file under src/torch_dfo/ may import a CPU-binding library."""
    violations: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{path}: SyntaxError — {exc}")
            continue
        hits = _forbidden_imports(tree)
        for lineno, module in hits:
            rel = path.relative_to(_SRC.parent.parent)
            violations.append(f"{rel}:{lineno}: forbidden import '{module}'")

    assert not violations, (
        "CPU-binding imports found in src/torch_dfo/:\n"
        + "\n".join(f"  {v}" for v in violations)
        + "\n\nAll numpy/scipy/pandas/numba/jax/sklearn must stay in "
        "[dev] / [benchmarks] extras, never in src/torch_dfo/."
    )
