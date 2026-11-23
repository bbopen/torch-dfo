"""Static tokenize guard: prohibit non-directive inline comments in src/torch_dfo/.

Inline comments in library code drift from implementation over time. Prefer clear
names, small helpers, and docstrings for public contracts; keep inline comments
limited to required tool directives (``# type: ignore``, ``# noqa``, ``# pragma:``).
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src" / "torch_dfo"
_DIRECTIVE_RE = re.compile(r"^#\s*(type:\s*ignore|noqa|pragma:\s*no\s*cover)\b", re.IGNORECASE)


def test_no_inline_comments_in_src() -> None:
    """No file under src/torch_dfo/ may contain non-directive inline comments."""
    violations: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT and not _DIRECTIVE_RE.match(tok.string):
                rel = path.relative_to(_SRC.parent.parent)
                violations.append(f"{rel}:{tok.start[0]}: {tok.string}")

    assert not violations, (
        "Non-directive inline comments found in src/torch_dfo/:\n"
        + "\n".join(f"  {v}" for v in violations)
        + "\n\nKeep src/torch_dfo/ self-documenting via clear names and docstrings."
    )
