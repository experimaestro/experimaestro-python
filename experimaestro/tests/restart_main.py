import importlib
import sys
from experimaestro import experiment

if __name__ == "__main__":
    wspath, module, functionname = sys.argv[1:]
    print("Importing", module)
    f = getattr(importlib.import_module(module), functionname)
    with experiment(wspath, "restart") as xp:
        f(xp)
