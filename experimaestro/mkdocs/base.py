"""mkdocs plugin for documentation generation

See https://www.mkdocs.org/user-guide/plugins/ for plugin API documentation
"""

from collections import defaultdict
import itertools
import re
from experimaestro.core.types import ObjectType
import mkdocs
from pathlib import Path
import pkgutil
from typing import DefaultDict, List, Dict
import importlib
import logging
import inspect
import mkdocs.config.config_options as config_options
from mkdocs.structure.files import File as MkdocFile
from mkdocs.structure.pages import Page as MkdocPage
from mkdocs.structure.nav import Navigation as MkdocNavigation
from experimaestro.core.objects import Config

MODULEPATH = Path(__file__).parent


class Configurations:
    def __init__(self):
        self.tasks = set()
        self.configs = set()


class Documentation(mkdocs.plugins.BasePlugin):
    RE_SHOWCLASS = re.compile(r"::xpm::([^ \n]+)")

    config_scheme = (
        ("name", config_options.Type(str, default="Tasks and configurations")),
        ("modules", config_options.Type(list)),
        ("init", config_options.Type(list)),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # path to sets of XPM types
        self.configurations = defaultdict(lambda: Configurations())
        self.tasks = {}

        # Maps XPM types to markdown paths
        self.type2path = {}

    def on_config(self, config, **kwargs):
        # Import modules in init
        for module_name in self.config["init"]:
            importlib.import_module(module_name)

        # Include documentation pages in config
        parsed = set()

        for name_packagename in self.config["modules"]:
            module_name, md_path = next(iter(name_packagename.items()))
            path_cfgs = self.configurations[md_path]
            if md_path.endswith(".md"):
                md_path = md_path[:-3]

            package = importlib.import_module(module_name)
            basepath = Path(package.__path__[0])

            for path in basepath.rglob("*.py"):
                parts = list(path.relative_to(basepath).parts)
                if parts[-1] == "__init__.py":
                    parts = parts[:-1]
                elif parts[-1].endswith(".py"):
                    parts[-1] = parts[-1][:-3]

                fullname = (
                    f"""{module_name}.{".".join(parts)}""" if parts else module_name
                )

                # Avoid to re-parse
                if fullname in parsed:
                    continue
                parsed.add(fullname)

                try:
                    module = importlib.import_module(fullname)
                    for _, member in inspect.getmembers(
                        module, lambda t: inspect.isclass(t) and issubclass(t, Config)
                    ):
                        # Only include members of the module
                        if member.__module__ != fullname:
                            continue

                        d = (
                            path_cfgs.tasks
                            if getattr(member.__getxpmtype__(), "task", None)
                            is not None
                            else path_cfgs.configs
                        )

                        self.type2path[
                            f"{member.__module__}.{member.__qualname__}"
                        ] = md_path

                        d.add(member.__xpmtype__)
                except Exception as e:
                    logging.error(
                        "Error while reading definitions file %s: %s", path, e
                    )
        return config

    def showclass(self, m: re.Match):
        return self.getlink(m.group(1).strip())

    def getlink(self, qualname: str):
        md_path = self.type2path.get(qualname, None)
        if md_path is None:
            return qualname
        return f"[{qualname}](/{md_path}#{qualname})"

    def build_doc(self, lines: List[str], configs: Dict[str, Config]):
        """Build the documentation for a list of configurations"""
        configs = sorted(configs, key=lambda x: str(x.identifier))
        for xpminfo in configs:
            lines.extend(
                (
                    f"""### {xpminfo.identifier} <span id="{xpminfo.objecttype.__module__}.{xpminfo.objecttype.__qualname__}"> </span>\n\n""",
                    f"""```py3\nfrom {xpminfo.objecttype.__module__} import {xpminfo.objecttype.__name__}\n```\n\n""",
                )
            )
            if xpminfo.description:
                lines.extend(
                    f"""<div class="xpm-description">{xpminfo.description}</div>\n\n"""
                )
            for name, argument in xpminfo.arguments.items():

                if isinstance(argument.type, ObjectType):
                    basetype = argument.type.basetype
                    typestr = self.getlink(
                        f"{basetype.__module__}.{basetype.__qualname__}"
                    )
                else:
                    typestr = argument.type.name()

                lines.append(f"- **{name}** ({typestr})")
                if argument.help:
                    lines.append(f"\n  {argument.help}")
                lines.append("\n")

    def on_page_markdown(self, markdown, page: MkdocPage, **kwargs):
        """Generate markdown pages"""
        path = page.file.src_path

        markdown = Documentation.RE_SHOWCLASS.sub(self.showclass, markdown)

        cfgs = self.configurations.get(path, None)
        if cfgs is None:
            return markdown

        lines = [
            markdown,
            "<style>",
            (MODULEPATH / "style.css").read_text(),
            "</style>\n",
            "<div><hr></div>",
            "*Documentation generated by experimaestro*\n",
        ]

        lines.extend(["## Configurations\n\n"])
        self.build_doc(lines, cfgs.configs)

        lines.extend(["## Tasks\n\n"])
        self.build_doc(lines, cfgs.tasks)

        return "".join(lines)
