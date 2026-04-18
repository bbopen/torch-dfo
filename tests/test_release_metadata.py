"""Release metadata consistency checks."""

from __future__ import annotations

import re
from pathlib import Path

import torch_dfo

ROOT = Path(__file__).resolve().parents[1]


def _project_version() -> str:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_release_metadata_versions_are_consistent() -> None:
    version = _project_version()

    assert torch_dfo.__version__ == version
    assert f'release = "{version}"' in (ROOT / "docs/conf.py").read_text(encoding="utf-8")
    assert f"version = {{{version}}}" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"## {version} —" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def test_unreleased_changelog_heading_is_not_versioned() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(r"^## \[Unreleased\]$", changelog, flags=re.MULTILINE)
    assert not re.search(r"^## \[Unreleased\].*\d+\.\d+\.\d+", changelog, flags=re.MULTILINE)
