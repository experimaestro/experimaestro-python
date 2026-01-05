"""Refactoring commands for experimaestro codebase patterns"""

import ast
import re
import sys
from pathlib import Path
from typing import Iterator

import click
from termcolor import cprint


class DefaultValueFinder(ast.NodeVisitor):
    """AST visitor to find class definitions with Param/Meta/Option annotations"""

    def __init__(self, source_lines: list[str]):
        self.source_lines = source_lines
        self.findings: list[dict] = []
        self.current_class: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef):
        old_class = self.current_class
        self.current_class = node.name

        # Check if this class might be a Config/Task (has annotations)
        for item in node.body:
            if isinstance(item, ast.AnnAssign):
                self._check_annotation(item, node)

        self.generic_visit(node)
        self.current_class = old_class

    def _check_annotation(self, node: ast.AnnAssign, class_node: ast.ClassDef):
        """Check if an annotated assignment uses Param/Meta/Option with bare default"""
        if not isinstance(node.target, ast.Name):
            return

        param_name = node.target.id

        # Check if annotation is Param[...], Meta[...], or Option[...]
        annotation = node.annotation
        is_param_type = False

        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                if annotation.value.id in ("Param", "Meta", "Option"):
                    is_param_type = True

        if not is_param_type:
            return

        # Check if there's a default value
        if node.value is None:
            return

        # Check if the default is already wrapped in field()
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                if node.value.func.id == "field":
                    return  # Already using field()

        # Found a bare default value
        self.findings.append(
            {
                "class_name": self.current_class,
                "param_name": param_name,
                "line": node.lineno,
                "col_offset": node.col_offset,
                "end_line": node.end_lineno,
                "end_col_offset": node.end_col_offset,
                "value_line": node.value.lineno,
                "value_col": node.value.col_offset,
                "value_end_line": node.value.end_lineno,
                "value_end_col": node.value.end_col_offset,
            }
        )


def find_bare_defaults(file_path: Path) -> list[dict]:
    """Find all bare default values in a Python file"""
    try:
        source = file_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    source_lines = source.splitlines()
    finder = DefaultValueFinder(source_lines)
    finder.visit(tree)
    return finder.findings


def refactor_file(file_path: Path, perform: bool) -> int:
    """Refactor a single file, returns number of changes made/found"""
    findings = find_bare_defaults(file_path)
    if not findings:
        return 0

    source = file_path.read_text()
    source_lines = source.splitlines(keepends=True)

    # Sort findings by line number in reverse order (to not mess up offsets)
    findings.sort(key=lambda f: (f["line"], f["col_offset"]), reverse=True)

    changes_made = 0
    for finding in findings:
        class_name = finding["class_name"]
        param_name = finding["param_name"]
        line_num = finding["line"]

        # Get the line content
        line_idx = line_num - 1
        if line_idx >= len(source_lines):
            continue

        line = source_lines[line_idx]

        # Try to find and replace the pattern on this line
        # Pattern: `param_name: Param[...] = value` -> `param_name: Param[...] = field(ignore_default=value)`
        # We need to be careful with multi-line values

        # Simple case: value is on the same line
        if finding["value_line"] == finding["value_end_line"] == line_num:
            # Extract the value part
            value_start = finding["value_col"]
            value_end = finding["value_end_col"]

            # Get the original value string
            original_value = line[value_start:value_end]

            # Create the replacement
            new_value = f"field(ignore_default={original_value})"

            # Replace in the line
            new_line = line[:value_start] + new_value + line[value_end:]
            source_lines[line_idx] = new_line

            if perform:
                cprint(
                    f"  {file_path}:{line_num}: {class_name}.{param_name} = {original_value} "
                    f"-> field(ignore_default={original_value})",
                    "green",
                )
            else:
                cprint(
                    f"  {file_path}:{line_num}: {class_name}.{param_name} = {original_value} "
                    f"-> field(ignore_default={original_value})",
                    "yellow",
                )

            changes_made += 1
        else:
            # Multi-line value - more complex handling needed
            # For now, just report it
            cprint(
                f"  {file_path}:{line_num}: {class_name}.{param_name} has multi-line default (manual fix required)",
                "red",
            )
            changes_made += 1

    if perform and changes_made > 0:
        # Check if we need to add 'field' import
        new_source = "".join(source_lines)

        # Simple check for field import
        if "from experimaestro" in new_source or "import experimaestro" in new_source:
            # Check if field is already imported
            if not re.search(
                r"from\s+experimaestro[^\n]*\bfield\b", new_source
            ) and not re.search(
                r"from\s+experimaestro\.core\.arguments[^\n]*\bfield\b", new_source
            ):
                # Try to add field to existing import
                new_source = re.sub(
                    r"(from\s+experimaestro\s+import\s+)([^\n]+)",
                    r"\1field, \2",
                    new_source,
                    count=1,
                )

        file_path.write_text(new_source)

    return changes_made


def find_python_files(path: Path) -> Iterator[Path]:
    """Find all Python files in a directory"""
    if path.is_file():
        if path.suffix == ".py":
            yield path
    else:
        for py_file in path.rglob("*.py"):
            # Skip common directories
            parts = py_file.parts
            if any(
                p in parts
                for p in ("__pycache__", ".git", ".venv", "venv", "node_modules")
            ):
                continue
            yield py_file


@click.group()
def refactor():
    """Refactor codebase patterns"""
    pass


@refactor.command(name="default-values")
@click.option(
    "--perform",
    is_flag=True,
    help="Perform the refactoring (default is dry-run)",
)
@click.argument("path", type=click.Path(exists=True, path_type=Path), default=".")
def default_values(path: Path, perform: bool):
    """Fix ambiguous default values in configuration files.

    Converts `x: Param[int] = 23` to `x: Param[int] = field(ignore_default=23)`
    to make the behavior explicit.

    By default runs in dry-run mode. Use --perform to apply changes.
    """
    if not perform:
        cprint("DRY RUN MODE: No changes will be written", "yellow")
        cprint("Use --perform to apply changes\n", "yellow")

    total_changes = 0
    files_with_changes = 0

    for py_file in find_python_files(path):
        changes = refactor_file(py_file, perform)
        if changes > 0:
            total_changes += changes
            files_with_changes += 1

    if total_changes == 0:
        cprint("\nNo bare default values found.", "green")
    else:
        action = "Fixed" if perform else "Found"
        cprint(
            f"\n{action} {total_changes} bare default value(s) in {files_with_changes} file(s).",
            "green" if perform else "yellow",
        )
        if not perform:
            cprint("Run with --perform to apply changes.", "yellow")

    sys.exit(0 if perform or total_changes == 0 else 1)
