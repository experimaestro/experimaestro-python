# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Mock mkdocs module if not installed (mkdocs is only needed for mkdocs builds)
import sys
from unittest.mock import MagicMock

try:
    import mkdocs  # noqa: F401
except ImportError:
    sys.modules["mkdocs"] = MagicMock()
    sys.modules["mkdocs.config"] = MagicMock()
    sys.modules["mkdocs.config.config_options"] = MagicMock()
    sys.modules["mkdocs.structure"] = MagicMock()
    sys.modules["mkdocs.structure.pages"] = MagicMock()
    sys.modules["mkdocs.plugins"] = MagicMock()

import experimaestro

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Experimaestro"
copyright = "2024, Benjamin Piwowarski"
author = "Benjamin Piwowarski"
release = experimaestro.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Custom experimaestro extension for Config/Task documentation
    "experimaestro.sphinx",
    # Markdown support
    "myst_parser",
    # Theme
    "sphinx_rtd_theme",
    # API documentation
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # Cross-project linking
    "sphinx.ext.intersphinx",
    # Google/NumPy style docstrings
    "sphinx.ext.napoleon",
    # Auto-link code in code blocks to documentation
    "sphinx_codeautolink",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The root document.
root_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- MyST Parser configuration -----------------------------------------------
# Enable colon fence for compatibility with MkDocs admonitions
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Enable heading anchors for cross-reference support
myst_heading_anchors = 3
