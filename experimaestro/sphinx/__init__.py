# Generate a nice documentation for experimaestro Configuration
# objects

from inspect import Parameter, Signature
from pathlib import Path
from sphinx.util.fileutil import copy_asset
import experimaestro
from typing import Any, Dict, List, Optional, Tuple
from docutils import nodes
from sphinx.application import Sphinx
from sphinx import addnodes
from sphinx.ext.autodoc import ClassDocumenter
from sphinx.locale import _, __
from sphinx.util import inspect, logging
from sphinx.domains.python import PyClasslike, PyAttribute, directives, desc_signature
from sphinx.util.typing import OptionSpec
from docutils.statemachine import StringList
import logging
import re

from experimaestro import Config
from experimaestro.core.types import ObjectType, ArrayType

# See Sphinx documentation on developping extensions
# https://www.sphinx-doc.org/en/master/extdev/index.html#dev-extensions

logger = logging.getLogger(__name__)


class XPMConfigDirective(PyClasslike):
    pass


class XPMParamDirective(PyAttribute):
    """Description of an attribute."""

    option_spec: OptionSpec = PyAttribute.option_spec.copy()
    option_spec.update(
        {
            "constant": directives.unchanged,
            "generated": directives.unchanged,
        }
    )

    def handle_signature(self, sig: str, signode: desc_signature) -> Tuple[str, str]:
        fullname, prefix = super().handle_signature(sig, signode)
        if self.options.get("constant") is not None:
            signode += addnodes.desc_annotation("constant", nodes.Text("constant"))
        if self.options.get("generated") is not None:
            signode += addnodes.desc_annotation("generated", nodes.Text("generated"))
        return fullname, prefix


class ConfigDocumenter(ClassDocumenter):
    objtype = "xpmconfig"

    directivetype = "xpmconfig"
    # directivetype = ClassDocumenter.objtype

    priority = 10 + ClassDocumenter.priority

    option_spec = dict(ClassDocumenter.option_spec)

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return isinstance(member, Config)

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)

    def get_object_members(self, want_all: bool):  # -> Tuple[bool, ObjectMembers]:
        r = super().get_object_members(want_all)
        return r

    def _get_signature(
        self,
    ) -> Tuple[Optional[Any], Optional[str], Optional[Signature]]:
        r = super()._get_signature()

        config: Config = self.object
        xpminfo = config.__getxpmtype__()

        p = []
        for name, argument in xpminfo.arguments.items():
            if not (argument.constant or argument.generator is not None):
                p.append(Parameter(name, Parameter.KEYWORD_ONLY))
        s = Signature(p)

        return r[0], r[1], s

    def add_content(
        self, more_content: Optional[StringList], no_docstring: bool = False
    ) -> None:

        xpminfo = getxpminfo(self.object)
        source_name = self.get_sourcename()

        super().add_content(more_content, no_docstring)

        for name, argument in xpminfo.arguments.items():
            typestr = argument.type.name()

            self.add_line(f".. py:xpmparam:: {name}", source_name)
            self.add_line(f"   :type: {typestr}", source_name)

            if argument.default is not None:
                self.add_line(f"   :value: {argument.default}", source_name)
            if argument.generator:
                self.add_line(f"   :generated:", source_name)
            elif argument.constant:
                self.add_line(f"   :constant:", source_name)

            # self.add_line("", source_name)
            if argument.help:
                self.add_line("", source_name)
                self.add_line(argument.help, source_name)
            self.add_line("", source_name)


RE_ATTRIBUTE = re.compile(r"""^.. attribute:: (.*)""")


def getxpminfo(obj):
    config: Config = obj
    return config.__getxpmtype__()


def _process_docstring(
    app: Sphinx, what: str, name: str, obj: Any, options: Any, lines: List[str]
) -> None:
    # Removes attributes from docstring
    # Comes after napoleon (lesser priority) so benefit from a standardized
    # output
    if obj is not None and inspect.isclass(obj) and issubclass(obj, Config):
        xpminfo = getxpminfo(obj)
        names = set(xpminfo.arguments.keys())

        newlines = []
        skip = False
        for line in lines:
            if m := RE_ATTRIBUTE.match(line):
                if m.group(1) in names:
                    skip = True
            elif skip:
                if line.startswith(".."):
                    skip = False

            if not skip:
                newlines.append(line)

        lines[:] = newlines[:]


def copy_asset_files(app, exc):
    import os

    assetspath = Path(__file__).parent / "static"
    if exc is None:  # build succeeded
        for path in assetspath.glob("*.css"):
            copy_asset(str(path.absolute()), os.path.join(app.outdir, "_static"))


def setup(app: Sphinx) -> Dict[str, Any]:
    """Setup experimaestro for Sphinx documentation"""

    app.connect("build-finished", copy_asset_files)
    app.add_css_file("experimaestro.css")

    # We need autodoc
    app.setup_extension("sphinx.ext.autodoc")

    app.add_autodocumenter(ConfigDocumenter)

    app.add_directive("py:xpmconfig", XPMConfigDirective)
    app.add_directive("py:xpmparam", XPMParamDirective)
    app.connect("autodoc-process-docstring", _process_docstring, 600)

    return {"version": experimaestro.__version__, "parallel_read_safe": True}
