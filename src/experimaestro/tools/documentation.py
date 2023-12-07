import importlib
import inspect
import logging
import pkgutil
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Set, Tuple, Union, Dict, Type
from dataclasses import dataclass

import docutils.nodes as nodes
import sphobjinv
from docutils.frontend import OptionParser
from docutils.nodes import document
from docutils.parsers.rst import Directive, Parser, directives
from docutils.utils import new_document
from sphinx.directives.other import TocTree
from sphinx.domains.python import PyCurrentModule
from termcolor import cprint

from experimaestro import Config
from experimaestro.sphinx import PyObject


def documented_from_objects(objects_inv: Path) -> Set[str]:
    inv = sphobjinv.Inventory(objects_inv)
    return set(
        obj.name
        for obj in inv.objects
        if obj.domain == "py" and obj.role == "xpmconfig"
    )


@dataclass
class DocumentationReport:
    #: Errors while parsing
    errors: List[str]

    #: Undocumented
    undocumented: List[Config]

    #: Documented but non existing
    falsy_documented: List[Config]


def analyze(
    packages: Union[str, List[str]], documented: Set[str], skip_packages: Set[str]
) -> Tuple[bool, List[Config]]:
    """Returns the list of undocumented configurations

    :param package: package(s) to explore
    :param documented: documented symbols (fully qualified names)
    :param skip_packages: List of sub-packages that should be skipped
    :return: a tuple containing whether errors where detected when importing
        files and the list of undocumented configurations
    """

    def process(mod: ModuleType, configurations: Dict[str, Type[Config]]):
        errors = []
        for info in pkgutil.iter_modules(mod.__path__, prefix=f"{mod.__name__}."):
            try:
                logging.info("Processing %s...", info.name)
                mod = importlib.import_module(info.name)
                for x in mod.__dict__.values():
                    if inspect.isclass(x) and issubclass(x, Config):
                        configurations[f"{x.__module__}.{x.__qualname__}"] = x

            except Exception:
                logging.exception(f"Module {info.name} could not be loaded")
                errors.append(f"Module {info.name} could not be loaded")

            # Process sub-modules
            if info.ispkg:
                errors.extend(process(mod, configurations))

        return errors

    configurations = {}
    for package in list(packages):
        errors = process(import_module(package), configurations)

    # Search for undocumented
    undocumented = []
    for name, configuration in configurations.items():
        if name.startswith(f"{package}.") and name not in documented:
            if all(not name.startswith(f"{x}.") for x in skip_packages):
                undocumented.append(configuration)

    # Search for falsy documented
    falsy = [name for name in documented if name not in configurations]

    return DocumentationReport(errors, undocumented, falsy)


class autodoc(nodes.Node):
    def __init__(self, content) -> None:
        super().__init__()
        self.content = content
        self.children = []


class AutotocDirective(Directive):
    has_content = True
    optional_arguments = TocTree.optional_arguments
    option_spec = TocTree.option_spec
    required_arguments = TocTree.required_arguments

    def run(self):
        return [autodoc(self.content)]


class autoxpmconfig(nodes.Node):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.children = []


class AutoXPMDirective(Directive):
    has_content = PyObject.has_content
    optional_arguments = PyObject.optional_arguments
    required_arguments = PyObject.required_arguments
    option_spec = PyObject.option_spec.copy()
    option_spec.update({"members": directives.unchanged})

    def run(self):
        return [autoxpmconfig(self.arguments[0].strip())]


class currentmodule(nodes.Node):
    def __init__(self, modname: Optional[str]) -> None:
        super().__init__()
        self.modname = modname
        self.children = []


class CurrentModuleDirective(Directive):
    has_content = PyCurrentModule.has_content
    optional_arguments = PyCurrentModule.optional_arguments
    required_arguments = PyCurrentModule.required_arguments
    option_spec = PyCurrentModule.option_spec

    def run(self):
        modname = self.arguments[0].strip()
        if modname == "None":
            modname = None
        return [currentmodule(modname)]


@lru_cache
def get_parser():
    directives.register_directive("toctree", AutotocDirective)
    directives.register_directive("autoxpmconfig", AutoXPMDirective)
    directives.register_directive("currentmodule", CurrentModuleDirective)
    settings = OptionParser(components=(Parser,)).get_default_values()

    return Parser(), settings


class MyVisitor(nodes.NodeVisitor):
    def __init__(self, document: document):
        super().__init__(document)
        self.toc: Set[str] = set()
        self.currentmodule = None
        self.documented = set()

    def visit_autodoc(self, node: autodoc) -> None:
        self.toc.update(node.content)

    def visit_autoxpmconfig(self, node: autoxpmconfig) -> None:
        """Called for all other node types."""
        if node.name.find(".") == -1:
            name = f"{self.currentmodule}.{node.name}"
        else:
            name = node.name

        self.documented.add(name)
        logging.debug("[autoxpmconfig] %s / %s", node.name, name)

    def visit_currentmodule(self, node: currentmodule) -> None:
        """Called for all other node types."""
        self.currentmodule = node.modname
        logging.debug("[current module] %s", self.currentmodule)

    def unknown_visit(self, node: nodes.Node) -> None:
        """Called for all other node types."""
        pass


class DocumentVisitor:
    def __init__(self) -> None:
        self.processed = set()
        self.documented = set()
        self.errors = []

    def parse_rst(self, doc_path: Path):
        logging.info("Parsing %s", doc_path)
        input_str = doc_path.read_text()
        parser, settings = get_parser()
        document = new_document(str(doc_path.absolute()), settings)
        parser.parse(input_str, document)
        visitor = MyVisitor(document)
        document.walk(visitor)
        self.documented.update(visitor.documented)

        for to_visit in visitor.toc:
            path = (doc_path.parent / f"{to_visit}.rst").absolute()
            if path not in self.processed:
                if path.is_file():
                    self.processed.add(path)
                    self.parse_rst(path)
                else:
                    self.errors.append(f"Could not find {to_visit} in file {doc_path}")


class DocumentationAnalyzer:
    """Returns a report on the documented files"""

    def __init__(self, doc_path: Path, modules: List[str], excluded: Set[str]):
        assert doc_path.is_file()

        self.doc_path = doc_path
        self.modules = modules
        self.excluded = excluded

        # Report
        self.documentation_errors = []
        self.parsing_errors = []
        self.undocumented = []
        self.falsy_documented = []  #: Documented but not in code

    def analyze(self):
        visitor = DocumentVisitor()
        visitor.parse_rst(self.doc_path)
        self.documentation_errors = visitor.errors

        report = analyze(self.modules, visitor.documented, self.excluded)
        self.parsing_errors = report.errors
        self.undocumented = [
            f"{config.__module__}.{config.__qualname__}"
            for config in report.undocumented
        ]
        self.falsy_documented = report.falsy_documented

    def report(self):
        cprint(
            f"{len(self.documentation_errors)} errors were "
            "encountered while parsing documentation",
            "red" if self.documentation_errors else "green",
        )
        for error in self.documentation_errors:
            cprint(f"  [documentation error] {error}", "red")

        for error in self.falsy_documented:
            cprint(f"  [falsy documented] {error}", "red")

        cprint(
            f"{len(self.parsing_errors)} errors were encountered while parsing modules",
            "red" if self.parsing_errors else "green",
        )

        for error in self.parsing_errors:
            cprint(f"  [import error] {error}", "red")

        cprint(f"{len(self.undocumented)} undocumented configurations")
        for error in sorted(self.undocumented):
            cprint(f"  [undocumented] {error}", "red")

    def assert_no_undocumented(self):
        assert (
            len(self.documentation_errors) == 0
            and len(self.parsing_errors) == 0
            and len(self.undocumented) == 0
        )

    def assert_valid_documentation(self):
        """Asserts that there are no falsy documented"""
        self.assert_no_undocumented()
        assert (
            len(self.falsy_documented) == 0
        ), f"{self.falsy_documented} falsy documented members"
