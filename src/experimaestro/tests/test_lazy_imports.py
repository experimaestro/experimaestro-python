"""Heavy dependencies are not loaded when starting the command line

Every imported module costs file system lookups, which are slow on shared
file systems (clusters): ``experimaestro version`` or the remote state server
should not pay for SSH, documentation or validation libraries.
"""

import subprocess
import sys

# Not loaded whatever the Python version
ALWAYS_LAZY = ["fabric", "paramiko", "mkdocs", "requests"]

# Deferred with ``__lazy_modules__`` (PEP 810)
PEP810_LAZY = [
    "pydantic",
    "yaml",
    "psutil",
    "humanfriendly",
    "tqdm",
    "experimaestro.scheduler",
    "experimaestro.core.objects",
]


def loaded_modules(statement: str) -> set[str]:
    code = f"import sys\n{statement}\nprint('\\n'.join(sys.modules))"
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    return set(output.split())


def test_cli_lazy_imports():
    loaded = loaded_modules("import experimaestro.cli")
    expected = ALWAYS_LAZY + (PEP810_LAZY if sys.version_info >= (3, 15) else [])
    assert sorted(name for name in expected if name in loaded) == []


def test_lazy_reexports():
    """Lazily imported names are still reachable"""
    loaded = loaded_modules(
        "from experimaestro.experiments import GridSearch, mock_modules\n"
        "from experimaestro.mkdocs import MKDOCS_AVAILABLE\n"
        "from experimaestro.launchers import SshPath, SshConnector"
    )
    assert "experimaestro.experiments.grid" in loaded
    assert "experimaestro.connectors.ssh" in loaded


def test_lazy_names_collide_with_submodules():
    """Lazy names keep their value when a submodule with the same name is
    imported (it would replace them)"""
    loaded = loaded_modules(
        "import experimaestro.progress, experimaestro.notifications\n"
        "import experimaestro.scheduler.experiment\n"
        "from experimaestro import progress, experiment\n"
        "from experimaestro.scheduler import experiment as scheduler_experiment\n"
        "assert callable(progress) and not isinstance(progress, type(sys))\n"
        "assert isinstance(experiment, type), experiment\n"
        "assert scheduler_experiment is experiment"
    )
    assert "experimaestro.notifications" in loaded
