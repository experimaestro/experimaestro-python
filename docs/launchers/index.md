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

## Launcher configuration file (beta)

In order to automate the process of choosing the right launcher, a `launchers.yaml`
configuration file can be written.

```py
# Finds a launcher so that we get 2 CUDA GPUs with 12G of memory (at least) on each
gpulauncher_mem48 = launcher.find_launcher(LauncherSpec(cuda_memory=["12G", "12G"]))
```
