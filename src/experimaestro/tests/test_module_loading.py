"""Loading objects whose type is defined in a top-level module (issue #293)

Top-level modules (``__package__ == ""``) are serialized with their source
file path, which should only be used as a fallback when the module cannot be
imported by name.
"""

import importlib
import shutil
import sys
from pathlib import Path

import pytest

from experimaestro.core.context import SerializationContext
from experimaestro.core.serialization import from_state_dict, state_dict

MODULE_SOURCE = """
from experimaestro import Config, Param

EXECUTIONS.append(__name__)

class Leaf(Config):
    x: Param[int]

class Node(Config):
    a: Param[Leaf]
    b: Param[Leaf]
"""


@pytest.fixture
def module_dir(tmp_path):
    """A directory with a top-level module, cleaned from sys.modules/sys.path"""
    names = set()

    def write(name: str) -> Path:
        names.add(name)
        path = tmp_path / f"{name}.py"
        path.write_text(f"EXECUTIONS = []\n{MODULE_SOURCE}")
        return path

    old_path = sys.path[:]
    yield write
    sys.path[:] = old_path
    for name in names:
        sys.modules.pop(name, None)


def serialize(module):
    leaf = module.Leaf.C(x=1)
    node = module.Node.C(a=leaf, b=module.Leaf.C(x=2))
    return state_dict(SerializationContext(save_directory=None), node)


def serialize_then_forget(name: str, path: Path):
    """Serializes objects from a module, then makes it non importable (e.g. a
    module next to the experiment script, whose directory is not in the job
    Python path)"""
    sys.path.insert(0, str(path.parent))
    state = serialize(importlib.import_module(name))
    sys.path.remove(str(path.parent))
    del sys.modules[name]
    return state


def test_top_level_module_records_file(module_dir):
    path = module_dir("xpm_toplevel_a")
    sys.path.insert(0, str(path.parent))
    module = importlib.import_module("xpm_toplevel_a")
    assert module.__package__ == ""

    state = serialize(module)
    assert all(d["file"] == str(path) for d in state["objects"])

    loaded = from_state_dict(state)
    assert type(loaded) is module.Node.XPMConfig
    assert loaded.a.x == 1 and loaded.b.x == 2
    # Already imported: no re-execution
    assert module.EXECUTIONS == ["xpm_toplevel_a"]


def test_top_level_module_moved(module_dir, tmp_path):
    """The recorded file no longer exists, but the module is importable"""
    path = module_dir("xpm_toplevel_b")
    sys.path.insert(0, str(path.parent))
    state = serialize(importlib.import_module("xpm_toplevel_b"))

    # Simulates loading on another host: new location, fresh interpreter state
    new_dir = tmp_path / "elsewhere"
    new_dir.mkdir()
    shutil.move(path, new_dir / path.name)
    sys.path[:] = [str(new_dir)] + [p for p in sys.path if p != str(path.parent)]
    del sys.modules["xpm_toplevel_b"]

    loaded = from_state_dict(state)
    module = sys.modules["xpm_toplevel_b"]
    assert module.__file__ == str(new_dir / path.name)
    assert type(loaded) is module.Node.XPMConfig
    assert type(loaded.a) is type(loaded.b) is module.Leaf.XPMConfig
    assert module.EXECUTIONS == ["xpm_toplevel_b"]


def test_file_module_executed_once(module_dir):
    """A module that is not importable is loaded from its file only once"""
    path = module_dir("xpm_toplevel_c")
    state = serialize_then_forget("xpm_toplevel_c", path)
    loaded = from_state_dict(state)

    loaded_module = sys.modules["xpm_toplevel_c"]
    assert loaded_module.EXECUTIONS == ["xpm_toplevel_c"]
    assert type(loaded) is loaded_module.Node.XPMConfig
    assert type(loaded.a) is type(loaded.b) is loaded_module.Leaf.XPMConfig


def test_missing_module_and_file(module_dir):
    path = module_dir("xpm_toplevel_d")
    state = serialize_then_forget("xpm_toplevel_d", path)

    path.unlink()
    with pytest.raises(ModuleNotFoundError, match="does not exist"):
        from_state_dict(state)
