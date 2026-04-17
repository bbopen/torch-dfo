"""Sphinx configuration for torch-dfo documentation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

project = "torch-dfo"
copyright = "2026, Brett G. Bonner"
author = "Brett G. Bonner"
release = "0.9.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
]

# Napoleon settings (NumPy-style docstrings)
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True  # Attributes render inline; no duplicate-object pages.

# Intersphinx: link to PyTorch and Python docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# Sphinx Gallery
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/\d+_",
    "plot_gallery": False,  # no matplotlib required; examples use torch only
    "download_all_examples": False,
}

# Autodoc
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autosummary_generate = False

# Theme
html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/bbopen/torch-dfo/",
    "source_branch": "master",
    "source_directory": "docs/",
}
html_title = "torch-dfo"

# MyST
myst_enable_extensions = ["colon_fence", "deflist", "tasklist"]

# Misc
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
