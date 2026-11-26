# Configuration file for the Sphinx documentation builder.
import sys
from pathlib import Path


# Get absolute path to the project root directory
docs_dir = Path(__file__).parent.parent.absolute()  # /path/to/metatopia/docs
project_root = docs_dir.parent  # /path/to/metatopia

# Add project root and package directories to Python path
sys.path.insert(0, str(project_root))

# Add any additional paths if needed
pkg_dir = project_root / "metatopia"

if pkg_dir.exists():
    sys.path.insert(0, str(pkg_dir))


# -- Project information -----------------------------------------------------
project = "metatopia"
copyright = "2026, Bessa Research Group"
author = "Igor Kuszczak"

# Single source of truth: read the version from the package itself so the
# docs cannot drift from VERSION / metatopia.__version__.
from metatopia.__version__ import __version__ as _version  # noqa: E402


version = _version
release = _version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_book_theme",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

# Logo configuration
html_logo = "_static/metatopia_logo.svg"

html_theme_options = {
    "repository_url": "https://github.com/bessagroup/metatopia",
    "use_repository_button": True,
    "show_toc_level": 2,
}


# Autosummary settings
autosummary_generate = False

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

autodoc_default_options = {
    "member-order": "bysource",  # Documents members in source code order
    "exclude-members": "__repr__, __str__, __weakref__",
}

myst_enable_extensions = ["amsmath", "dollarmath"]
nb_execution_mode = "off"
