# Launchers

Launchers, together with the [Connector](../connectors/index.md), specify how a task should be launched.
There exist two types of launchers at the moment, [direct launcher](#direct) (starting
a new process) or through [slurm](#slurm)

## Types

(direct)=
### Direct

By default, jobs are launched directly by the scheduler using python scripts.

::: experimaestro.launchers.direct.DirectLauncher

(slurm)=
### Slurm

The [Slurm](https://slurm.schedmd.com/documentation.html) workload manager launcher is supported.
It is possible to use different settings for different jobs by using the `config`
method of the launcher

```python

from experimaestro.launchers.slurm import SlurmLauncher

launcher = SlurmLauncher(nodes=1)
gpulauncher = launcher.config(gpu_per_node=1)

with experiment(launcher=launcher):
    # Default
    mytask().submit()

    # If needed, options can be used
    mytask().submit(launcher=gpulauncher)
```

::: experimaestro.launchers.slurm.SlurmOptions

::: experimaestro.launchers.slurm.SlurmLauncher

## Launcher file

The `launchers.py` file dictates how a given *requirement* (e.g., 2 CPU with
64Go of memory) is mapped to a given `Launcher` configuration.

### Requirements

::: experimaestro.launcherfinder.specs.HostRequirement
::: experimaestro.launcherfinder.specs.HostSimpleRequirement
::: experimaestro.launcherfinder.specs.CudaSpecification
::: experimaestro.launcherfinder.specs.CPUSpecification

#### Parsing requirements

::: experimaestro.launcherfinder.parser.parse

```python
from experimaestro.launcherfinder.parser import parse

req = parse("""duration=40h & cpu(mem=700GiB) & cuda(mem=32GiB) * 8 | duration=50h & cpu(mem=700GiB) & cuda(mem=32GiB) * 4""")
```

Requirements can be manipulated:

- duration can be multiplied by a given coefficient using
  `req.multiply_duration`. For instance,  `req.multiply_duration(2)` multiplies
  all the duration by 2.



### Example

To construct launchers given a specification, you have to use a `launchers.py`
file within the configuration directory.

```python
from typing import Set
from experimaestro.launcherfinder import (
    HostRequirement,
    HostSpecification,
    CudaSpecification,
    CPUSpecification,
)
from experimaestro.launchers.slurm import SlurmLauncher, SlurmOptions
from experimaestro.launchers.direct import DirectLauncher
from experimaestro.connectors.local import LocalConnector


def find_launcher(requirements: HostRequirement, tags: Set[str] = set()):
    """Find a launcher"""

    if match := requirements.match(HostSpecification(cuda=[])):
        # No GPU: run directly
        return DirectLauncher(connector=LocalConnector.instance())

    if match := requirements.match(
        HostSpecification(
            max_duration=100 * 3600,
            cpu=CPUSpecification(cores=32, memory=129 * (1024**3)),
            cuda=[CudaSpecification(memory=24 * (1024**3)) for _ in range(8)],
        )
    ):
        if len(match.requirement.cuda_gpus) > 0:
            return SlurmLauncher(
                connector=LocalConnector.instance(),
                options=SlurmOptions(gpus_per_node=len(match.requirement.cuda_gpus)),
            )

    # Could not find a host
    return None
```

## Tags

Tags can be used to filter out some launchers

```python
from experimaestro.launcherfinder import find_launcher

find_launcher("""duration=4 days & cuda(mem=4G) * 2 & cpu(mem=400M, cores=4)""", tags=["slurm"])
```
will search for a launcher that has the tag `slurm` (see example below).
