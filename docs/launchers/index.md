# Launchers

## Types

### direct (default)

By default, jobs are launched directly by the scheduler using python scripts.

### Slurm (since 0.8.7)

The [Slurm](https://slurm.schedmd.com/documentation.html) workload manager launcher is supported.
It is possible to use different settings for different jobs by using the `config`
method of the launcher

```py3

from experimaestro.launchers.slurm import SlurmLauncher

launcher = SlurmLauncher(nodes=1)
gpulauncher = launcher.config(gpu_per_node=1)

with experiment(launcher=launcher):
    # Default
    mytask().submit()

    # If needed, options can be used
    mytask().submit(launcher=gpulauncher)
```

The Slurm launcher constructor and the `config` method can take the following parameters:

- `nodes`: Number of requested nodes
- `time`: Maximum job time (in seconds)
- `gpus`: Total number of GPUs
- `gpus_per_node`: Number of GPUs per node
- `account`: The slurm account for launching the job
- `qos`: The requested Quality of Service

To use launcher configuration files, one can use an automatic convertion tool

```sh
scontrol show nodes | experimaestro launchers slurm convert
```

## Launcher configuration file (since 0.11)

In order to automate the process of choosing the right launcher, a `launchers.yaml`
configuration file can be written.

```py
# Finds a launcher so that we get 2 CUDA GPUs with 14G of memory (at least) on each
from experimaestro.launcherfinder import cuda_gpu, find_launcher
gpulauncher = find_launcher(cuda_gpu(mem="14G") * 2)
```

Simple strings can also be parsed (for configuration files)

```py

from experimaestro.launcherfinder import find_launcher

find_launcher("""duration=4 days & cuda(mem=4G) * 2 & cpu(mem=400M, cores=4)""")
```

## Tags

Tags can be used to filter out some launchers

```py

from experimaestro.launcherfinder import find_launcher

find_launcher("""duration=4 days & cuda(mem=4G) * 2 & cpu(mem=400M, cores=4)""", tags=["slurm"])
```
will search for a launcher that has the tag `slurm` (see example below).

## Search process

Launcher groups are sorted by decreasing weights and filtered by group before the search.
Then, for each launcher group, experimaestro searches for the first matching launcher (details
are type-specific).

## Example of a configuration

This configurations contains four launcher groups (two local, two through slurm).


```yaml
# --- Local launchers

local:
  - # Standard launcher for small tasks
    connector: local
    weight: 5

    # Describes the available CPUs
    cpu: { cores: 40, memory: 1G }

  - # Intensive launcher with more memory and GPU
    connector: local
    weight: 4

    # Use a token to avoid running too many tasks
    tokens:
      localtoken: 1

    cpu: { cores: 40, memory: 8G }

    gpus:
      - model: GTX1080
        count: 1
        memory: 8116MiB

# --- Slurm launchers

slurm:
  # We can use fully manual SLURM configuration
  - id: manual
    connector: local
    tags: [slurm]
    weight: 3

    # Describes the GPU features and link them to the two
    # possible properties (memory and number of GPUs)
    features_regex:
      - GPU(?P<cuda_count>\d+)
      - GPUM(?P<cuda_memory>\d+G)

    options:
      gpu:
        # At least 70% of the memory should be requested
        # (from version 0.11.8)
        # For instance, if the GPU has 64G, we won't target it
        # if we request less than 44.8G (= 70% of 64G)
        min_mem_ratio: 0.7

    partitions:
      # Partition "big GPUs"
      biggpus:
        # has two types of nodes
        nodes:
          - # Nodes yep/yop
            hosts: [yop, yep]
            # Associated features
            features: [GPU3, GPUM48G]
          - hosts: [yip, yup, yap]
            features: [GPU2, GPUM24G]

      # Partition "Small GPUs"
      smallgpus:
        nodes:
          - hosts: [alpha, beta, gamma, delta]
            features: [GPU2, GPUM24G]

  # We can also use SLURM for semi-automatic configuration
  - id: auto
    connector: local
    tags: [slurm]

    # Describes the GPU features and link them to the two
    # possible properties (memory and number of GPUs)
    features_regex:
      - GPU(?P<cuda_count>\d+)
      - GPUM(?P<cuda_memory>\d+G)

    partitions:
      # Disable the "heavy" partition
      heavy: { disabled: true }

    # Use `sinfo` to ask partition/node details (e.g. name and features)
    query_slurm: true
```
