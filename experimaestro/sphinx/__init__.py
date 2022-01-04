import experimaestro
from typing import Any, Dict, Optional
from sphinx.application import Sphinx
from sphinx.ext.autodoc import ClassDocumenter, bool_option
from sphinx.locale import _, __
from sphinx.util import inspect, logging
from docutils.statemachine import StringList
import logging
import re

from experimaestro import Config
from experimaestro.core.types import ObjectType

# See Sphinx documentation on developping extensions
# https://www.sphinx-doc.org/en/master/extdev/index.html#dev-extensions

logger = logging.getLogger(__name__)


class ConfigDocumenter(ClassDocumenter):
    objtype = "xpmconfig"

    directivetype = "class"

    priority = 10 + ClassDocumenter.priority

    option_spec = dict(ClassDocumenter.option_spec)

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        return isinstance(member, Config)

    def format_signature(self, **kwargs: Any) -> str:
        """Format the signature (arguments and return annotation) of the object.

        Let the user process it via the ``autodoc-process-signature`` event.
        """
        if self.args is not None:
            # signature given explicitly
            args = "(%s)" % self.args
            retann = self.retann
        else:
            # try to introspect the signature
            try:
                retann = None
                args = self._call_format_args(**kwargs)
                if args:
                    matched = re.match(r"^(\(.*\))\s+->\s+(.*)$", args)
                    if matched:
                        args = matched.group(1)
                        retann = matched.group(2)
                args = None
            except Exception as exc:
                logger.warning(
                    __("error while formatting arguments for %s: %s"),
                    self.fullname,
                    exc,
                    type="autodoc",
                )
                args = None

        result = self.env.events.emit_firstresult(
            "autodoc-process-signature",
            self.objtype,
            self.fullname,
            self.object,
            self.options,
            args,
            retann,
        )
        if result:
            args, retann = result

        if args is not None:
            return args + ((" -> %s" % retann) if retann else "")
        else:
            return ""

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)

    def add_content(
        self, more_content: Optional[StringList], no_docstring: bool = False
    ) -> None:

        super().add_content(more_content, no_docstring)
        source_name = self.get_sourcename()

        config: Config = self.object
        xpminfo = config.__getxpmtype__()

        for name, argument in xpminfo.arguments.items():

            if isinstance(argument.type, ObjectType):
                basetype = argument.type.basetype
                typestr = f":ref:`{basetype.__module__}.{basetype.__qualname__}"
            else:
                typestr = argument.type.name()

            s = ""
            if argument.generator:
                s += f" [*generated*]"
            elif argument.constant:
                s += f" [*constant*] "

            self.add_line(f"**{name}** ({typestr})", source_name)
            if argument.help:
                self.add_line(argument.help, source_name)
            self.add_line("", source_name)


def skip_non_undoc(app, what, name, obj, skip, options):
    assert False
    return True
    # if undoc-members is set, show only undocumented members
    if "undoc-members" in options and obj.__doc__ is not None:
        # skip member that have a __doc__
        return True
    else:
        return None


def setup(app: Sphinx) -> Dict[str, Any]:
    """Setup experimaestro for Sphinx documentation"""

    # We need autodoc
    app.setup_extension("sphinx.ext.autodoc")

    app.add_autodocumenter(ConfigDocumenter)
    app.connect("autodoc-skip-member", skip_non_undoc)

    return {"version": experimaestro.__version__, "parallel_read_safe": True}
