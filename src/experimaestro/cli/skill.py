"""Install experimaestro agent skill for LLM coding assistants."""

import shutil
from importlib.resources import files
from pathlib import Path

import click
from termcolor import cprint

# Map of known targets to their skill directories
TARGETS: dict[str, Path] = {
    "agents": Path.home() / ".agents" / "skills",
    "claude": Path.home() / ".claude" / "skills",
    "cursor": Path.home() / ".cursor" / "skills",
}

SKILL_NAME = "experimaestro"


def _get_source_dir() -> Path:
    """Get the skill source directory from the installed package or repo."""
    # Try the repo-level .agents/skills/ first (for editable installs / dev)
    pkg_root = Path(__file__).resolve().parents[2]
    repo_skill = pkg_root.parent / ".agents" / "skills" / SKILL_NAME
    if (repo_skill / "SKILL.md").is_file():
        return repo_skill

    # Fallback: shipped inside the package
    pkg_skill = files("experimaestro").joinpath(".agents", "skills", SKILL_NAME)
    p = Path(str(pkg_skill))
    if (p / "SKILL.md").is_file():
        return p

    raise click.ClickException(
        "Cannot find the experimaestro skill source. "
        "Is the package installed correctly?"
    )


def _detect_targets() -> list[str]:
    """Auto-detect which LLM tool config directories exist."""
    found = []
    for name, skills_dir in TARGETS.items():
        # Check if the parent config dir exists (e.g. ~/.claude/)
        config_dir = skills_dir.parent
        if config_dir.is_dir():
            found.append(name)
    return found


def _install_to(target: str, source: Path) -> None:
    """Copy the skill to the target directory."""
    dest = TARGETS[target] / SKILL_NAME
    dest.mkdir(parents=True, exist_ok=True)

    for src_file in source.iterdir():
        if src_file.is_file():
            shutil.copy2(src_file, dest / src_file.name)

    cprint(f"  Installed to {dest}", "green")


@click.command("install-skill")
@click.argument("targets", nargs=-1)
@click.option("--list", "list_targets", is_flag=True, help="List available targets")
def install_skill(targets: tuple[str, ...], list_targets: bool):
    """Install the experimaestro agent skill for LLM coding assistants.

    By default, installs to ~/.agents/skills/ (cross-client open standard).
    Specify targets to install to specific tools:

    \b
      experimaestro install-skill            # default: .agents
      experimaestro install-skill claude     # ~/.claude/skills/
      experimaestro install-skill agents claude  # both

    Available targets: agents, claude, cursor
    """
    if list_targets:
        detected = _detect_targets()
        for name, path in TARGETS.items():
            marker = " (detected)" if name in detected else ""
            installed = (
                " [installed]" if (path / SKILL_NAME / "SKILL.md").is_file() else ""
            )
            cprint(f"  {name:10s} {path}{marker}{installed}")
        return

    source = _get_source_dir()

    if not targets:
        targets = ("agents",)

    for target in targets:
        if target not in TARGETS:
            cprint(f"  Unknown target: {target}", "red")
            cprint(f"  Available: {', '.join(TARGETS)}", "yellow")
            raise SystemExit(1)
        _install_to(target, source)
