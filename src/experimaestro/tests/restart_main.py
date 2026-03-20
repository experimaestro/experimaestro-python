# ruff: noqa: T201 - Test helper script
import importlib
import sys
from experimaestro import experiment

if __name__ == "__main__":
    wspath, module, functionname = sys.argv[1:]
    print("Importing", module)
    f = getattr(importlib.import_module(module), functionname)
    with experiment(wspath, "restart", no_environmental_impact=True) as xp:
        f(xp)
