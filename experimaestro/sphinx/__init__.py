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
from sphinx.ext.autodoc import ClassDocumenter, Documenter, restify
from sphinx.locale import _, __
from sphinx.util import inspect, logging
from sphinx.domains.python import (
    PyClasslike,
    PyAttribute,
    PyObject,
    directives,
    desc_signature,
    _parse_annotation,
)
from sphinx.util.typing import OptionSpec
from docutils.statemachine import StringList
import logging
import re

from experimaestro import Config, Task

# See Sphinx documentation on developping extensions
# https://www.sphinx-doc.org/en/master/extdev/index.html#dev-extensions

logger = logging.getLogger(__name__)


class XPMConfigDirective(PyClasslike):
    """Directive for XPM configurations/tasks"""

    option_spec: OptionSpec = PyClasslike.option_spec.copy()

    option_spec.update(
        {
            "task": directives.unchanged,
        }
    )

    def get_signature_prefix(self, sig: str) -> List[nodes.Node]:
        """May return a prefix to put before the object name in the
        signature.
        """
        if self.options.get("task") is not None:
            return [nodes.Text("XPM Task")]
        return [nodes.Text("XPM Config")]


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


def getxpminfo(obj):
    """Returns the XPM type information for a configuration object"""
    config: Config = obj
    return config.__getxpmtype__()


class ConfigDocumenter(ClassDocumenter):
    """Documenter for experimaestro configurations and tasks"""

    objtype = "xpmconfig"
    directivetype = "xpmconfig"

    # We need to have a higher priority than class documenter to get called
    priority = 10 + ClassDocumenter.priority

    option_spec = dict(ClassDocumenter.option_spec)

    @classmethod
    def can_document_member(cls, member: Any, *args) -> bool:
        can_document = inspect.isclass(member) and issubclass(member, Config)
        return can_document

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)

    def get_object_members(self, want_all: bool):  # -> Tuple[bool, ObjectMembers]:
        r = super().get_object_members(want_all)
        return r

    def _get_signature(
        self,
    ) -> Tuple[Optional[Any], Optional[str], Optional[Signature]]:
        r = super()._get_signature()

        xpminfo = getxpminfo(self.object)

        p = []
        for name, argument in xpminfo.arguments.items():
            if not (argument.constant or argument.generator is not None):
                p.append(Parameter(name, Parameter.KEYWORD_ONLY))
        s = Signature(p)

        return r[0], r[1], s

    @staticmethod
    def formatDefault(value) -> str:
        if isinstance(value, Config):
            objecttype = value.__xpmtype__.objecttype
            params = ", ".join(
                [f"{key}={value}" for key, value in value.__xpm__.values.items()]
            )
            # It would be possible to do better... if not
            return f"{objecttype.__module__}.{objecttype.__qualname__}({params})"

        return str(value)

    def add_directive_header(self, sig: str) -> None:
        sourcename = self.get_sourcename()
        xpminfo = getxpminfo(self.object)

        # Mostly copied from ClassDocumenter but adapted (no other choice)
        # FIXME: would be better to be a plugin

        if self.doc_as_attr:
            self.directivetype = "attribute"

        # Skip ClassDocumenter
        Documenter.add_directive_header(self, sig)

        if self.analyzer and ".".join(self.objpath) in self.analyzer.finals:
            self.add_line("   :final:", sourcename)

        canonical_fullname = self.get_canonical_fullname()
        if (
            not self.doc_as_attr
            and canonical_fullname
            and self.fullname != canonical_fullname
        ):
            self.add_line("   :canonical: %s" % canonical_fullname, sourcename)

        # Our specific code
        if issubclass(self.object, Task):
            self.add_line(f"   :task:", sourcename)

        # add inheritance info, if wanted
        if not self.doc_as_attr and self.options.show_inheritance:
            if inspect.getorigbases(self.object):
                # A subclass of generic types
                # refs: PEP-560 <https://www.python.org/dev/peps/pep-0560/>
                bases = list(self.object.__orig_bases__)
            elif hasattr(self.object, "__bases__") and len(self.object.__bases__):
                # A normal class
                bases = list(self.object.__bases__)
            else:
                bases = []

            self.env.events.emit(
                "autodoc-process-bases", self.fullname, self.object, self.options, bases
            )

            if self.config.autodoc_typehints_format == "short":
                base_classes = [restify(cls, "smart") for cls in bases]
            else:
                base_classes = [restify(cls) for cls in bases]

            sourcename = self.get_sourcename()
            self.add_line("", sourcename)
            self.add_line("   " + _("Bases: %s") % ", ".join(base_classes), sourcename)

        # Adds return type if different
        if xpminfo.returntype != xpminfo.objecttype:
            self.add_line("", sourcename)
            self.add_line(
                "   " + _("Submit type: %s") % restify(xpminfo.returntype), sourcename
            )

            # annotations = _parse_annotation(str(xpminfo.returntype), self.env)

    def add_content(
        self, more_content: Optional[StringList], no_docstring: bool = False
    ) -> None:

        xpminfo = getxpminfo(self.object)
        source_name = self.get_sourcename()

        super().add_content(more_content)

        for name, argument in xpminfo.arguments.items():
            typestr = argument.type.name()

            self.add_line(f".. py:xpmparam:: {name}", source_name)
            self.add_line(f"   :type: {typestr}", source_name)

            if argument.default is not None:
                self.add_line(
                    f"   :value: {ConfigDocumenter.formatDefault(argument.default)}",
                    source_name,
                )
            if argument.generator:
                self.add_line(f"   :generated:", source_name)
            elif argument.constant:
                self.add_line(f"   :constant:", source_name)

            # self.add_line("", source_name)
            if argument.help:
                self.add_line("", source_name)
                self.add_line("    " + argument.help, source_name)
            self.add_line("", source_name)


RE_ATTRIBUTE = re.compile(r"""^.. attribute:: (.*)""")


def _process_docstring(
    app: Sphinx, what: str, name: str, obj: Any, options: Any, lines: List[str]
) -> None:
    """Removes already documented (by XPM) attributes from docstring

    Comes after napoleon (lesser priority) so benefit from a standardized
    output
    """
    if obj is not None and ConfigDocumenter.can_document_member(obj):
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

    # Adds our stylesheet
    app.connect("build-finished", copy_asset_files)
    app.add_css_file("experimaestro.css")

    # We need autodoc
    app.setup_extension("sphinx.ext.autodoc")

    # Defines our extensions (documenter, domains)
    app.add_autodocumenter(ConfigDocumenter)
    app.add_directive("py:xpmconfig", XPMConfigDirective)

    pydomain = app.registry.domains["py"]
    pydomain.object_types["xpmconfig"] = pydomain.object_types["class"]
    # pydomain.add_object_type('py:xpmconfig', pydomain.object_types["class"])

    app.add_directive("py:xpmparam", XPMParamDirective)
    app.connect("autodoc-process-docstring", _process_docstring, 600)

    return {"version": experimaestro.__version__, "parallel_read_safe": True}
