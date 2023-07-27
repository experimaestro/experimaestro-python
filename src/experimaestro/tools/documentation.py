import logging
from pathlib import Path
import pkgutil
from types import ModuleType
from typing import List, Set, Tuple
import sphobjinv
import inspect
from experimaestro import Config
from importlib import import_module


def documented_from_objects(objects_inv: Path) -> Set[str]:
    inv = sphobjinv.Inventory(objects_inv)
    return set(
        obj.name
        for obj in inv.objects
        if obj.domain == "py" and obj.role == "xpmconfig"
    )


def undocumented(
    package: str, documented: Set[str], skip_packages: Set[str]
) -> Tuple[bool, List[Config]]:
    """Returns the list of undocumented configurations

    :param package: package to explore
    :param documented: documented symbols (fully qualified names)
    :param skip_packages: List of sub-packages that should be skipped
    :return: a tuple containing whether errors where detected when importing
        files and the list of undocumented configurations
    """

    def process(mod: ModuleType, configurations: Set):
        ok = True
        for info in pkgutil.iter_modules(mod.__path__, prefix=f"{mod.__name__}."):
            try:
                logging.info("Processing %s...", info.name)
                mod = info.module_finder.find_module(info.name).load_module(info.name)
                configurations.update(
                    x
                    for x in mod.__dict__.values()
                    if inspect.isclass(x) and issubclass(x, Config)
                )
            except Exception:
                logging.exception("Module %s could not be loaded", info.name)

                ok = False

            # Process sub-modules
            if info.ispkg:
                ok = process(mod, configurations) & ok

        return ok

    configurations = set()
    ok = process(import_module(package), configurations)

    configs = []
    for configuration in configurations:
        name = f"{configuration.__module__}.{configuration.__qualname__}"
        if name.startswith(f"{package}.") and name not in documented:
            if all(not name.startswith(f"{x}.") for x in skip_packages):
                configs.append(configuration)

    return ok, configs
