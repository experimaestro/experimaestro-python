# Documenting Configurations

There are two documentation plugins, one for for Sphinx (recommended) and mkdocs (deprecated).

## Sphinx

Just configure Sphinx to use the `experimaestro.sphinx` extension.

Sphinx allows to link documentations (through the extension `sphinx.ext.intersphinx`).

You can then use the directive `::autoxpmconfig QUALITIFIED_CLASSNAME` (which extends
`autoclass`) to build the documentation of a configuration or class.

## mkdocs

An example configuration

```yaml
plugins:
  - experimaestro:
    init:
      # Allows to cope with uninstalled modules when generating documentation
      - mymodule.mkdocs_init
    modules:
      # Learning to rank modul
      - mymodule.letor: letor.md
      # Neural module
      - mymodule.neural: neural.md
      # All the rest
      - mymodule: misc.md
    external:
      # Maps datatype
      - datamaestro: "https://datamaestro.rtfd.io/en/latest/experimaestro-mapping.json"
```

## Including methods

```python
class CSV:
    @documentation
    def data(self) -> Tuple[List[str], "numpy.array"]:
        """Returns the list of fields and the numeric data as a Numpy array

        Returns: List of fields
        """

```

## Writing subpackage documentation

It is possible to write a structured documentation

```md
---
title: Learning to rank
---

# Learning to rank

## Scores

Scorers are able to give a score to a (query, document) pair. Among the
scorers, some are have learnable parameters.

::xpm::xpmir.rankers:Scorer

## Trainers

Trainers are responsible for defining the loss (given a learnable scorer)

::xpm::xpmir.letor.trainers:Trainer

## Sampler

How to sample learning batches.

::xpm::xpmir.letor.samplers:Sampler
```

## Some tricks

### Avoiding to load some modules

When building the documentation, it might be useful not to have to install
all the dependencies.

```python
import re
from experimaestro.mkdocs.metaloader import DependencyInjectorFinder

DependencyInjectorFinder.install(
    re.compile(r"^(torch|pandas|bs4|pytorch_transformers|pytrec_eval|apex)($|\.)")
)
```

## Testing Documentation Coverage

You can verify that all configurations in your package are documented using the `DocumentationAnalyzer` tool. This is particularly useful as a pytest test to ensure documentation stays up to date with code changes.

### Basic Usage

```python
from pathlib import Path
from experimaestro.tools.documentation import DocumentationAnalyzer


def test_documented():
    """Test if every configuration is documented"""
    # Path to your Sphinx documentation entry point (index.rst)
    doc_path = Path(__file__).parents[3] / "docs" / "source" / "index.rst"

    # Create the analyzer with:
    # - doc_path: path to the main RST file
    # - modules: set of module names to check for Config subclasses
    # - excluded: set of module prefixes to skip (e.g., test modules)
    analyzer = DocumentationAnalyzer(
        doc_path,
        modules={"mypackage"},
        excluded={"mypackage.test", "mypackage.tests"}
    )

    analyzer.analyze()
    analyzer.report()  # Prints a colored report to stdout
    analyzer.assert_valid_documentation()
```

### What It Checks

The analyzer performs the following checks:

1. **Undocumented configurations**: Finds all `Config` subclasses in your modules that are not documented with `autoxpmconfig` directives
2. **Falsy documented**: Detects documentation entries that reference non-existent configurations (e.g., after renaming or removing a class)
3. **Documentation parsing errors**: Reports issues when parsing RST files (e.g., missing referenced files in toctree)
4. **Module import errors**: Reports modules that couldn't be imported during analysis

### Assertion Methods

**`analyzer.assert_no_undocumented()`**

Raises `AssertionError` if any of the following conditions are detected:

- **Documentation parsing errors**: Issues when parsing RST files (e.g., a toctree references a file that doesn't exist)
- **Module import errors**: Python modules that couldn't be imported during analysis (e.g., missing dependencies)
- **Undocumented configurations**: `Config` subclasses found in your modules that have no corresponding `autoxpmconfig` directive in the documentation

**`analyzer.assert_valid_documentation()`**

Performs all checks from `assert_no_undocumented()`, plus:

- **Falsy documented configurations**: Documentation entries (`autoxpmconfig` directives) that reference configurations which don't exist in the code. This can happen when:
  - A class was renamed or removed but documentation wasn't updated
  - A typo in the qualified class name in the documentation
  - The module path changed

Both methods provide detailed error messages listing all issues found, making it easy to identify and fix problems.

### Integration with CI

Add the test to your test suite to catch documentation drift:

```python
# tests/test_documentation.py
import pytest
from pathlib import Path
from experimaestro.tools.documentation import DocumentationAnalyzer


@pytest.mark.documentation
def test_all_configs_documented():
    doc_path = Path(__file__).parents[2] / "docs" / "source" / "index.rst"
    analyzer = DocumentationAnalyzer(
        doc_path,
        modules={"mypackage"},
        excluded={"mypackage.tests"}
    )
    analyzer.analyze()
    analyzer.report()
    analyzer.assert_valid_documentation()
```

You can then run documentation tests specifically with:

```bash
pytest -m documentation
```
