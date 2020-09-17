"""mkdocs plugin for documentation generation

See https://www.mkdocs.org/user-guide/plugins/ for plugin API documentation
"""

import itertools
import mkdocs
from pathlib import Path
import pkgutil
from typing import List, Dict
import importlib
import logging
import inspect
import mkdocs.config.config_options as config_options
from mkdocs.structure.files import File as MkdocFile
from mkdocs.structure.pages import Page as MkdocPage
from mkdocs.structure.nav import Navigation as MkdocNavigation
from experimaestro.core.objects import Config, Task

MODULEPATH = Path(__file__).parent


def build_doc(lines: List[str], configs: Dict[str, Config]):
    configs = sorted(configs, key=lambda x: str(x.identifier))
    for xpminfo in configs:
        lines.extend(
            (
                f"### {xpminfo.identifier}\n\n```python3\nfrom {xpminfo.objecttype.__module__} import {xpminfo.objecttype.__name__}\n\n```\n\n"
            )
        )
        if xpminfo.description:
            lines.extend(
                f"""<div class="xpm-description">{xpminfo.description}</div>\n\n"""
            )
        for name, argument in xpminfo.arguments.items():
            lines.append(f"- **{name}** ({argument.type.name()})")
            if argument.help:
                lines.append(f"\n  {argument.help}")
            lines.append("\n")


class Documentation(mkdocs.plugins.BasePlugin):
    config_scheme = (
        ("name", config_options.Type(str, default="Tasks and configurations")),
        ("path", config_options.Type(str, default="xpm.md")),
        ("modules", config_options.Type(list)),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configs = set()
        self.tasks = set()

    def on_config(self, config, **kwargs):
        # Include documentation pages in config
        for name_packagename in (names for names in self.config["modules"]):
            name, module_name = next(iter(name_packagename.items()))

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

                try:
                    module = importlib.import_module(fullname)
                    for _, member in inspect.getmembers(
                        module, lambda t: inspect.isclass(t) and issubclass(t, Config)
                    ):
                        d = self.tasks if issubclass(member, Task) else self.configs
                        d.add(member.__xpm__)
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    logging.error(
                        "Error while reading definitions file %s: %s", path, e
                    )
        return config

    def on_files(self, files, config):
        """Called when list of files has been read"""
        files.append(MkdocFile(self.config["path"], "", config["site_dir"], False))

    def on_page_read_source(self, config, page: MkdocPage, **kwargs):
        """Generate markdown pages"""
        path = page.file.src_path

        if path == self.config["path"]:
            logging.warning("Path is %s", path)

            lines = []

            lines.extend(
                ["<style>", (MODULEPATH / "style.css").read_text(), "</style>"]
            )
            lines.extend(["## Configurations\n\n"])
            build_doc(lines, self.configs)

            lines.extend(["## Tasks\n\n"])
            build_doc(lines, self.tasks)

            return "".join(lines)
