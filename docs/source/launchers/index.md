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

(slurm-pilots)=
#### Pilot allocations (beta)

```{warning}
This feature is in beta: the options and the files it writes may change.
```

```{important}
Pilots are coordinated through files in the `pilots/` folder of the workspace:
the workspace must be on a file system **shared by the submission host and the
compute nodes** (e.g. NFS or Lustre), and the experiment must submit to SLURM
from a cluster host (local connector, not an SSH connector). The pilots run the
same Python interpreter as the experiment (`XPM_PILOT_PYTHON` overrides it),
which must thus be reachable from the compute nodes.
```

When many short jobs request the same resources, waiting in the SLURM queue
for each of them can take longer than running them. With the `pilot` option, a
SLURM launcher submits *pilot* jobs instead: each pilot holds an allocation for
up to `time` and runs queued jobs one after another, until not enough time is
left or it has nothing to do (see *Stopping* below).

```python
from experimaestro.launchers.slurm import PilotOptions, SlurmLauncher, SlurmOptions

launcher = SlurmLauncher(
    options=SlurmOptions(partition="gpu", gpus_per_node=1, time="1:00:00"),
    pilot=PilotOptions(time="20:00:00", idle_timeout=600, max_pilots=4),
)
```

- **Job duration**: the `time` SLURM option of the job's launcher is its time
  limit. A pilot only takes a job if it can finish in time (plus a `margin`), and
  stops the job (`SIGTERM`, then `SIGKILL` after `kill_wait` seconds) when the
  time limit is reached: the job then fails with a *timeout*, so a
  {ref}`resumable task <resumable-tasks>` is retried and
  {py:meth}`~experimaestro.ResumableTask.remaining_time` gives the time left
  before the job's own time limit (or the pilot's end, if earlier).
- **Pools**: the pilots that can run a job form a *pool*, stored in the
  `pilots/<pool>` folder of the workspace. `<pool>` is the first 16 hexadecimal
  digits of a SHA-256 hash of:

  - the path of the SLURM commands (`binpath` of the launcher),
  - the SLURM options *except* `time` (options that are not set are left out),
  - all the {py:class}`~experimaestro.launchers.slurm.PilotOptions` (`time`,
    `idle_timeout`, `max_pilots`, `margin`, `kill_wait`, `poll_interval`,
    `linger`).

  Two launchers that only differ by `time` (e.g. built by `launchers.py` for
  requirements with different durations) thus share a pool, while changing any
  other SLURM or pilot option gives a different pool. The output of a pilot is
  in `pilots/<pool>/pilots/<pilot>.{out,err}`.
- **Sharing between experiments**: a pool belongs to a workspace, not to an
  experiment. Experiments running in the same workspace with the same options
  use the same pool, and any of its pilots can run any of their jobs (whichever
  experiment submitted them, and whichever submitted the pilot). Experiments in
  different workspaces never share pilots. Jobs without a `time` option, or
  longer than a pilot (`time` + `margin` above the pilot `time`), do not use
  pilots and are submitted with `sbatch` as usual.
- **Starting pilots**: experiments waiting for jobs of a pool check it every
  `poll_interval` seconds. The waiting jobs are matched (longest first) with the
  pilots that can take a job: pilots still pending in the SLURM queue (they will
  have their full `time`) and idle running pilots whose remaining time covers
  the job's `time` plus `margin`. A new pilot is submitted for each job left
  without one, so a job that does not fit in what remains of a running pilot
  gets a new pilot, and so do jobs that become ready while all pilots are busy.
  `max_pilots` caps the number of pilots of the pool (pending or running): above
  it, jobs wait until a pilot is free — or stops, in which case a new one is
  submitted. A pilot takes the oldest waiting job that fits in its remaining
  time, by exclusively creating a claim file with a hard link (reliable on NFS),
  so that a job is run only once even when several pilots look at the queue.
- **Stopping**: an idle pilot stops after `idle_timeout` seconds, or `linger`
  seconds (1 minute by default) after no experiment has jobs in its pool (each
  experiment renews a lease while it has jobs queued or running in the pool), so
  that allocations are released soon after the experiments end.
  `experimaestro workspace pilots --stop` asks all the pilots to stop taking jobs.
- **Failures**: a job is marked as failed if its pilot ends while running it
  (e.g. the pilot is cancelled, or a node fails); a timeout of the pilot is
  reported as a timeout of the job. Killing a job only stops this job, not the
  pilot.

In `launchers.py`, the policy is plain Python, e.g. to only pack short jobs:

```python
PILOT = PilotOptions(time="20:00:00")


def find_launcher(requirements: HostRequirement, tags=set()):
    if match := requirements.match(HostSpecification(...)):
        duration = match.requirement.duration
        return SlurmLauncher(
            options=SlurmOptions(time=SlurmOptions.format_time(duration)),
            pilot=PILOT if duration <= 4 * 3600 else None,
        )
```

The job details (TUI and web UI) show where a job runs (e.g. *running in
pilot SLURM job 1234 on node12, time limit at 2026-11-05 14:00*) or that it
waits for a pilot, and `experimaestro workspace pilots` lists the pilots of a
workspace (pending, running with their current job, idle) and the waiting jobs
(`--all` also shows the stopped pilots).

{py:class}`~experimaestro.launchers.slurm.PilotOptions` - Options of the pilot allocations.

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
    options=SlurmOptions(partition="fast", time="1:00:00"), priority=10
)
slow_partition = SlurmLauncher(
    options=SlurmOptions(partition="slow", time="24:00:00"), priority=1
)

# By default, selects highest priority (fast_partition)
# If priorities tie, samples uniformly among tied launchers
dynamic = DynamicLauncher([fast_partition, slow_partition])

# With sample=True, samples proportionally to priority
# fast_partition has 10/(10+1) ≈ 91% chance of being selected
dynamic_sampled = DynamicLauncher([fast_partition, slow_partition], sample=True)

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

(accelerator-types)=
#### Accelerator Types

The launcher finder supports multiple accelerator (GPU) types:

- **CUDA** (`cuda()`): NVIDIA GPUs with dedicated memory. Use when you specifically need CUDA support.
- **MPS** (`mps()`): Apple Silicon GPUs with unified memory (shared with CPU). Use for macOS Metal support.
- **Generic** (`gpu()`): Matches any accelerator type. Use for cross-platform compatibility.

```{tip}
If your code runs on any accelerator (e.g., PyTorch code that works on both
CUDA and MPS backends), prefer the generic `gpu(...)` over `cuda(...)`: the
same requirement will then match an NVIDIA GPU on a Linux cluster and an
Apple Silicon GPU on a laptop. Reserve `cuda(...)` for code that genuinely
requires CUDA (e.g., custom CUDA kernels or CUDA-only libraries).
```

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

```{note}
Memory sizes require an explicit unit: `cuda(mem=32)` is rejected with a
`ValueError` (use `cuda(mem=32G)`).
```

**Examples:**

```python
from experimaestro.launcherfinder.parser import parse

# Request 8 NVIDIA GPUs with 32GB each
req = parse("duration=40h & cpu(mem=700GiB) & cuda(mem=32GiB) * 8")

# Cross-platform code: generic GPU requirement (matches any accelerator)
req = parse("duration=2h & gpu(mem=4GiB)")

# Equivalent explicit alternative: CUDA on Linux/Windows OR MPS on macOS
req = parse("duration=4h & cuda(mem=8GiB) | duration=4h & mps(mem=8GiB)")
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

(long-jobs)=
### Jobs longer than the partition time limit

A cluster usually caps job duration (e.g. SLURM `MaxTime`), while a training
task may legitimately request more (`duration=120h`). Rather than lowering the
requirement in every experiment, `launchers.py` can accept the total duration
and run {ref}`resumable tasks <resumable-tasks>` in slices: the job is
submitted with the partition limit and, on timeout, resubmitted by the
scheduler (up to `max_retries`, 3 by default) and restarts from its
checkpoints.

Whether a job is resumable is only known at submission, so this is done with a
submit listener ({py:meth}`~experimaestro.launchers.Launcher.addListener`),
which receives the {py:class}`~experimaestro.scheduler.jobs.Job` and can
replace its launcher. Non-resumable tasks are refused when the experiment
starts, instead of being rejected by SLURM later.

```python
from experimaestro import Job
from experimaestro.launcherfinder import HostRequirement, HostSpecification
from experimaestro.launchers.slurm import SlurmLauncher, SlurmOptions

# Hard limit of the SLURM partitions
PARTITION_MAX_DURATION = 48 * 3600
# Total duration accepted for resumable tasks (several 48h runs)
MAX_DURATION = 196 * 3600


def cap_duration(job: Job):
    if not job.resumable:
        raise ValueError(
            f"{job.config.__xpmtype__.identifier} requests more than"
            f" {PARTITION_MAX_DURATION // 3600}h but is not a ResumableTask"
        )
    job.launcher = job.launcher.config(
        time=SlurmOptions.format_time(PARTITION_MAX_DURATION)
    )


def find_launcher(requirements: HostRequirement, tags=set()):
    if match := requirements.match(
        # (with the other limits of the host: CPUs, memory, accelerators)
        HostSpecification(max_duration=MAX_DURATION)
    ):
        duration = match.requirement.duration
        launcher = SlurmLauncher(
            options=SlurmOptions(
                time=SlurmOptions.format_time(min(duration, PARTITION_MAX_DURATION)),
                # ... and the other SLURM options (partition, GPUs, ...)
            )
        )
        if duration > PARTITION_MAX_DURATION:
            launcher.addListener(cap_duration)
        return launcher
```

```{note}
The listener replaces `job.launcher` (using
{py:meth}`~experimaestro.launchers.slurm.SlurmLauncher.config`) instead of
modifying the launcher options in place: the same launcher instance is
usually shared by many jobs, some of which may not be resumable.
```

## Tags

Tags can be used to filter out some launchers

```python
from experimaestro.launcherfinder import find_launcher

find_launcher(
    """duration=4 days & cuda(mem=4G) * 2 & cpu(mem=400M, cores=4)""", tags=["slurm"]
)
```
will search for a launcher that has the tag `slurm` (see example below).

## Debugging the launcher file

You can query the launcher finder directly — without running an experiment — to
check which launcher a given requirement resolves to.

### From the command line

The `find-launchers` command loads `launchers.py` from the configuration
directory (`~/.config/experimaestro` by default, or the directory given with
`--config`) and prints the launcher returned for a requirement specification:

```bash
experimaestro tools find-launchers 'duration=1h & cuda(mem=32G) * 1 & cpu(cores=8)'

# With an explicit configuration directory
experimaestro tools find-launchers --config /path/to/config-dir 'duration=1h & cuda(mem=32G) * 1'
```

### From Python

The same query can be run from a REPL, which is convenient for inspecting
intermediate objects:

```python
from pathlib import Path
from experimaestro.launcherfinder import LauncherRegistry, find_launcher

# Optional: use a non-default configuration directory
LauncherRegistry.set_config_dir(Path("/path/to/config-dir"))

launcher = find_launcher("duration=1h & cuda(mem=32G) * 1 & cpu(cores=8)")
print(launcher)
```

`find_launcher` raises a `LauncherNotFoundError` when no launcher matches; the
lower-level `LauncherRegistry.instance().find(...)` returns `None` instead.

### Checking how a specification is parsed

If the returned launcher is not the one you expect, first check that the
specification string is parsed the way you intend:

```python
from experimaestro.launcherfinder.parser import parse

req = parse("duration=1h & cuda(mem=32G) * 1 & cpu(cores=8)")
for alternative in req.requirements:
    print(alternative)
```

Note that memory sizes require an explicit unit: `cuda(mem=32)` raises a
`ValueError` instead of being silently interpreted as 32 bytes. Use
`cuda(mem=32G)` instead.

### Tracing the matching process

Since `launchers.py` is plain Python, you can add `logging.debug` calls in your
`find_launcher` function (e.g., to print every matching host specification and
its score) and enable debug logging when querying:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```
