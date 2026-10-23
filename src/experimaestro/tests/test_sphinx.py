"""Tests for the experimaestro Sphinx extension"""

import re

import pytest

pytest.importorskip("sphinx")
pytest.importorskip("sphinx_codeautolink")

MODULE = '''
from experimaestro import Config, Param


class Sample(Config):
    """A sample configuration"""

    x: Param[int]
'''

CONF = """
extensions = ["experimaestro.sphinx", "sphinx_codeautolink"]
"""

INDEX = """
Sample
======

.. autoxpmconfig:: xpm_sphinx_sample.Sample

.. autoclass:: experimaestro.core.objects.ConfigMixin
   :members: instance

.. code-block:: python

    from xpm_sphinx_sample import Sample

    sample = Sample.C(x=1).instance()
"""


def test_codeautolink_config_usage(tmp_path, monkeypatch):
    """``Config.C(...)`` and ConfigMixin methods link to their documentation"""
    from sphinx.application import Sphinx

    srcdir = tmp_path / "src"
    srcdir.mkdir()
    (srcdir / "conf.py").write_text(CONF)
    (srcdir / "index.rst").write_text(INDEX)
    (tmp_path / "xpm_sphinx_sample.py").write_text(MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))

    outdir = tmp_path / "build"
    app = Sphinx(
        str(srcdir),
        str(srcdir),
        str(outdir),
        str(tmp_path / "doctrees"),
        "html",
        status=None,
        freshenv=True,
    )
    app.build()

    html = (outdir / "index.html").read_text()
    links = re.findall(r'class="sphinx-codeautolink-a" href="([^"]*)"', html)

    # Import + Sample.C usage
    assert links.count("index.html#xpm_sphinx_sample.Sample") == 2, links
    assert "index.html#experimaestro.core.objects.ConfigMixin.instance" in links, links
