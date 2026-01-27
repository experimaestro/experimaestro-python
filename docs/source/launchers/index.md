# Launchers

Launchers, together with the [Connector](../connectors/index.md), specify how a task should be launched.
There exist two types of launchers at the moment, [direct launcher](#direct) (starting
a new process) or through [slurm](#slurm)

## Priority

All launchers have a `priority` property (default: 0) that determines their preference
when used with a {py:class}`~experimaestro.launchers.DynamicLauncher`. Higher priority
values indicate more preferred launchers.

```python
from experimaestro.launchers.slurm import SlurmLauncher

# Create launchers with different priorities
fast_launcher = SlurmLauncher(options=SlurmOptions(partition="fast"), priority=10)
slow_launcher = SlurmLauncher(options=SlurmOptions(partition="slow"), priority=1)
```

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

(dynamic)=
### Dynamic Launcher

The {py:class}`~experimaestro.launchers.DynamicLauncher` allows dynamic selection
from a list of launchers based on their priorities. This is useful when you have
multiple execution options (e.g., different SLURM partitions or clusters) and want
to select the best one at runtime.

```python
from experimaestro.launchers import DynamicLauncher
from experimaestro.launchers.slurm import SlurmLauncher, SlurmOptions

# Create launchers with different priorities
fast_partition = SlurmLauncher(
    options=SlurmOptions(partition="fast", time="1:00:00"),
    priority=10
)
slow_partition = SlurmLauncher(
    options=SlurmOptions(partition="slow", time="24:00:00"),
    priority=1
)

# By default, selects highest priority (fast_partition)
# If priorities tie, samples uniformly among tied launchers
dynamic = DynamicLauncher([fast_partition, slow_partition])

# With sample=True, samples proportionally to priority
# fast_partition has 10/(10+1) ≈ 91% chance of being selected
dynamic_sampled = DynamicLauncher(
    [fast_partition, slow_partition],
    sample=True
)

with experiment(launcher=dynamic):
    mytask().submit()
```

#### Selection Modes

- **Default mode** (`sample=False`): Selects the launcher with the highest priority.
  If multiple launchers share the highest priority, one is chosen uniformly at random.

- **Sampling mode** (`sample=True`): Samples a launcher with probability proportional
  to its priority. All priorities must be positive in this mode.

#### Extending DynamicLauncher

You can subclass `DynamicLauncher` and override the `update()` method to refresh
the launcher list before each job submission. This is useful for checking cluster
availability or queue status:

```python
class ClusterAwareLauncher(DynamicLauncher):
    def update(self):
        # Check cluster availability and update priorities
        for launcher in self._launchers:
            if is_cluster_available(launcher):
                launcher.priority = 10
            else:
                launcher.priority = 0
```

{py:class}`~experimaestro.launchers.DynamicLauncher` - Launcher that dynamically selects from a list of launchers based on priority.

## Launcher file

The `launchers.py` file dictates how a given *requirement* (e.g., 2 CPU with
64Go of memory) is mapped to a given `Launcher` configuration.

### Requirements

- {py:class}`~experimaestro.launcherfinder.specs.HostRequirement` - Abstract base representing a disjunction of host requirements (alternatives).
- {py:class}`~experimaestro.launcherfinder.specs.HostSimpleRequirement` - A single host requirement specifying CPU, GPU, and duration constraints.
- {py:class}`~experimaestro.launcherfinder.specs.AcceleratorSpecification` - Generic accelerator (GPU) specification that matches any accelerator type.
- {py:class}`~experimaestro.launcherfinder.specs.CudaSpecification` - Specifies NVIDIA CUDA GPU requirements (dedicated memory).
- {py:class}`~experimaestro.launcherfinder.specs.MPSSpecification` - Specifies Apple Metal Performance Shaders requirements (unified memory).
- {py:class}`~experimaestro.launcherfinder.specs.CPUSpecification` - Specifies CPU requirements (cores, memory).

#### Accelerator Types

The launcher finder supports multiple accelerator (GPU) types:

- **CUDA** (`cuda()`): NVIDIA GPUs with dedicated memory. Use when you specifically need CUDA support.
- **MPS** (`mps()`): Apple Silicon GPUs with unified memory (shared with CPU). Use for macOS Metal support.
- **Generic** (`gpu()`): Matches any accelerator type. Use for cross-platform compatibility.

```{note}
MPS uses unified memory - the GPU shares RAM with the CPU. When matching MPS requirements,
the combined CPU + GPU memory request must not exceed the total system memory.
```

#### Parsing requirements

{py:func}`~experimaestro.launcherfinder.parser.parse` - Parses a requirement specification string into a {py:class}`~experimaestro.launcherfinder.specs.HostRequirement` object.

**Syntax elements:**

- `duration=<N><unit>`: Job duration (units: h/hours, d/days, m/mins)
- `cpu(mem=<size>, cores=<N>)`: CPU requirements
- `cuda(mem=<size>) * <N>`: NVIDIA CUDA GPU requirements (memory and count)
- `mps(mem=<size>) * <N>`: Apple MPS GPU requirements (unified memory)
- `gpu(mem=<size>) * <N>`: Generic GPU requirements (matches any accelerator)
- Memory sizes: `<N>G`, `<N>GiB`, `<N>M`, `<N>MiB`

**Examples:**

```python
from experimaestro.launcherfinder.parser import parse

# Request 8 NVIDIA GPUs with 32GB each
req = parse("duration=40h & cpu(mem=700GiB) & cuda(mem=32GiB) * 8")

# Cross-platform: CUDA on Linux/Windows OR MPS on macOS
req = parse("duration=4h & cuda(mem=8GiB) | duration=4h & mps(mem=8GiB)")

# Generic GPU requirement (matches any accelerator)
req = parse("duration=2h & gpu(mem=4GiB)")
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
    AcceleratorSpecification,
    CudaSpecification,
    MPSSpecification,
    CPUSpecification,
)
from experimaestro.launchers.slurm import SlurmLauncher, SlurmOptions
from experimaestro.launchers.direct import DirectLauncher
from experimaestro.connectors.local import LocalConnector


def find_launcher(requirements: HostRequirement, tags: Set[str] = set()):
    """Find a launcher"""

    if match := requirements.match(HostSpecification(accelerators=[])):
        # No GPU: run directly
        return DirectLauncher(connector=LocalConnector.instance())

    # CUDA cluster with SLURM
    if match := requirements.match(
        HostSpecification(
            max_duration=100 * 3600,
            cpu=CPUSpecification(cores=32, memory=129 * (1024**3)),
            accelerators=[CudaSpecification(memory=24 * (1024**3)) for _ in range(8)],
        )
    ):
        if len(match.requirement.accelerators) > 0:
            return SlurmLauncher(
                connector=LocalConnector.instance(),
                options=SlurmOptions(gpus_per_node=len(match.requirement.accelerators)),
            )

    # Apple Silicon with MPS (unified memory)
    if match := requirements.match(
        HostSpecification(
            cpu=CPUSpecification(cores=8, memory=32 * (1024**3)),
            accelerators=[MPSSpecification(memory=32 * (1024**3))],
        )
    ):
        return DirectLauncher(connector=LocalConnector.instance())

    # Could not find a host
    return None
```

```{note}
The `cuda=` parameter is still supported for backwards compatibility but
`accelerators=` is preferred for new code as it supports all accelerator types.
```

## Tags

Tags can be used to filter out some launchers

```python
from experimaestro.launcherfinder import find_launcher

find_launcher("""duration=4 days & cuda(mem=4G) * 2 & cpu(mem=400M, cores=4)""", tags=["slurm"])
```
will search for a launcher that has the tag `slurm` (see example below).
