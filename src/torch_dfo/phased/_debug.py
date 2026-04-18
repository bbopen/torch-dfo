"""Contributor-only debug machinery for the phased orchestrator.

This module is intentionally private (prefixed ``_``) and is NOT re-exported
from :mod:`torch_dfo.phased`.  It hosts the ``TORCH_DFO_DEBUG_DISABLE``
environment-variable handling used during regression testing to selectively
disable individual dim>=40 scheduling branches.

Tag names and semantics may change across minor versions.  Shipped users
should not depend on any symbol defined here.
"""

from __future__ import annotations

import os
from typing import Final

# ---------------------------------------------------------------------------
# Debug-only mechanism toggle (contributor tool; not a public API).
#
# Set TORCH_DFO_DEBUG_DISABLE to a comma-separated list of tags to disable
# individual dim>=40 scheduling branches during regression testing.
# Recognised tags: dim40_valley_branch, dim40_line_sampling,
# dim40_budget_transfer, dim40_focus_cycle, dim40_adaptive_burst,
# dim40_parity_bounds, dim40_terminal_focus, or `all`.
# Tag names and semantics may change across minor versions.
# ---------------------------------------------------------------------------
TAG_DIM40_VALLEY_BRANCH: Final[str] = "dim40_valley_branch"
TAG_DIM40_LINE_SAMPLING: Final[str] = "dim40_line_sampling"
TAG_DIM40_BUDGET_TRANSFER: Final[str] = "dim40_budget_transfer"
TAG_DIM40_FOCUS_CYCLE: Final[str] = "dim40_focus_cycle"
TAG_DIM40_ADAPTIVE_BURST: Final[str] = "dim40_adaptive_burst"
TAG_DIM40_PARITY_BOUNDS: Final[str] = "dim40_parity_bounds"
TAG_DIM40_TERMINAL_FOCUS: Final[str] = "dim40_terminal_focus"

_DEBUG_MECHANISM_TAGS: Final[frozenset[str]] = frozenset(
    {
        TAG_DIM40_VALLEY_BRANCH,
        TAG_DIM40_LINE_SAMPLING,
        TAG_DIM40_BUDGET_TRANSFER,
        TAG_DIM40_FOCUS_CYCLE,
        TAG_DIM40_ADAPTIVE_BURST,
        TAG_DIM40_PARITY_BOUNDS,
        TAG_DIM40_TERMINAL_FOCUS,
    }
)


def _debug_is_disabled(tag: str) -> bool:
    """Return True when the given dim>=40 mechanism tag is disabled via env."""
    raw = os.environ.get("TORCH_DFO_DEBUG_DISABLE", "").strip().lower()
    if not raw:
        return False
    tokens = {t.strip() for t in raw.split(",") if t.strip()}
    if "all" in tokens:
        return True
    return tag in tokens
