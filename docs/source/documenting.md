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

```py3
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

```py3
import re
from experimaestro.mkdocs.metaloader import DependencyInjectorFinder

DependencyInjectorFinder.install(
    re.compile(r"^(torch|pandas|bs4|pytorch_transformers|pytrec_eval|apex)($|\.)")
)
```
