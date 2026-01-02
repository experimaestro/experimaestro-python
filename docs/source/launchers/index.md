# Launchers

Launchers, together with the [Connector](../connectors/index.md), specify how a task should be launched.
There exist two types of launchers at the moment, [direct launcher](#direct) (starting
a new process) or through [slurm](#slurm)

## Types

(direct)=
### Direct

By default, jobs are launched directly by the scheduler using python scripts.

{py:class}`~experimaestro.launchers.direct.DirectLauncher` - Launcher that runs tasks directly as local processes without any job scheduler.

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

{py:class}`~experimaestro.launchers.slurm.SlurmOptions` - Configuration options for SLURM jobs (nodes, time, partition, GPUs, etc.).

{py:class}`~experimaestro.launchers.slurm.SlurmLauncher` - Launcher that submits tasks to the SLURM workload manager.

## Launcher file

The `launchers.py` file dictates how a given *requirement* (e.g., 2 CPU with
64Go of memory) is mapped to a given `Launcher` configuration.

### Requirements

- {py:class}`~experimaestro.launcherfinder.specs.HostRequirement` - Abstract base representing a disjunction of host requirements (alternatives).
- {py:class}`~experimaestro.launcherfinder.specs.HostSimpleRequirement` - A single host requirement specifying CPU, GPU, and duration constraints.
- {py:class}`~experimaestro.launcherfinder.specs.CudaSpecification` - Specifies CUDA GPU requirements (memory, model).
- {py:class}`~experimaestro.launcherfinder.specs.CPUSpecification` - Specifies CPU requirements (cores, memory).

#### Parsing requirements

{py:func}`~experimaestro.launcherfinder.parser.parse` - Parses a requirement specification string into a {py:class}`~experimaestro.launcherfinder.specs.HostRequirement` object.

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
