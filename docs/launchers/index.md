# Launchers

## bash script (default)

## Slurm (since 0.8.7)

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
