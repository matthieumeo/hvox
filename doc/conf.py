import configparser
import datetime
import pathlib
import re
from typing import Mapping

import alabaster


def setup_config() -> configparser.ConfigParser:
    """
    Load information contained in `setup.cfg`.
    """
    sphinx_src_dir = pathlib.Path(__file__).parent
    setup_path = sphinx_src_dir / ".." / "setup.cfg"
    setup_path = setup_path.resolve(strict=True)

    with setup_path.open(mode="r") as f:
        cfg = configparser.ConfigParser()
        cfg.read_file(f)
    return cfg


def pkg_info() -> Mapping:
    """
    Load information contained in `PKG-INFO`.
    """
    sphinx_src_dir = pathlib.Path(__file__).parent
    info_path = sphinx_src_dir / ".." / "src" / "hvox.egg-info" / "PKG-INFO"
    info_path = info_path.resolve(strict=True)

    # Pattern definitions
    pat_version = r"Version: (.+)$"

    with info_path.open(mode="r") as f:
        info = dict(version=None)
        for line in f:
            m = re.match(pat_version, line)
            if m is not None:
                info["version"] = m.group(1)
    return info


# -- Project information -----------------------------------------------------
cfg, info = setup_config(), pkg_info()
project = cfg.get("metadata", "name")
author = cfg.get("metadata", "author")
copyright = f"{datetime.date.today().year}, {author}"
version = release = info["version"]

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "alabaster",
]

templates_path = ["_templates"]
master_doc = "index"
exclude_patterns = []
pygments_style = "sphinx"
add_module_names = False
plot_include_source = True

# -- Options for HTML output -------------------------------------------------
html_static_path = ["_static"]
html_theme_path = [alabaster.get_path()]
html_theme = "alabaster"
html_theme_options = {
    #'nosidebar': True, # Disable showing the sidebar. Defaults to 'false'
    "description": "Spline-based regression and decomposition of time series with seasonal and trend components.",
    "description_font_style": "italic",
    "github_banner": True,
    "github_button": True,
    "github_user": "matthieumeo",
    "github_repo": "hvox",
    "fixed_sidebar": True,
    "page_width": "70%",
    "logo": "hvox.png",
    "logo_name": True,
    "logo_text_align": "center",
    "body_text_align": "justify",
    "sidebar_collapse": False,
    "touch_icon": "hvox.png",
    "show_powered_by": True,
}

# -- Options for HTMLHelp output ---------------------------------------------
htmlhelp_basename = "HVOX"
html_context = {
    "menu_links_name": "Repository",
    "menu_links": [
        (
            '<i class="fa fa-github fa-fw"></i> Source Code',
            "https://github.com/matthieumeo/hvox",
        ),
    ],
    "doc_path": "docs/source",
    "github_project": "matthieumeo",
    "github_repo": "hvox",
    "github_version": "master",
}

# -- Extension configuration -------------------------------------------------
# -- Options for autosummary extension ---------------------------------------
autosummary_generate = True

# -- Options for autodoc extension -------------------------------------------
autodoc_member_order = "bysource"
autodoc_default_flags = [
    "members",
    # 'inherited-members',
    "show-inheritance",
]
autodoc_inherit_docstrings = True

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "NumPy [latest]": ("https://docs.scipy.org/doc/numpy/", None),
    "SciPy [latest]": ("https://docs.scipy.org/doc/scipy/reference", None),
    "Pycsou [latest]": ("https://matthieumeo.github.io/pycsou/html/index", None),
    "dask [latest]": ("https://docs.dask.org/en/latest/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
}

# -- Options for napoleon extension ------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True
